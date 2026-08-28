"""Reconstruct KETOD processed text and structural feature CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import FEATURE_COLUMNS, INSTRUCTION, bool_text, ketod_feature_rows, write_csv, write_json


EXPECTED_ROWS = {"train": 41939, "test": 4964}


def _single_path(root: Path, filename: str) -> Path:
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    if not matches:
        if filename in {"train_ketod.json", "test_ketod.json"}:
            archives = sorted(
                path
                for path in root.rglob("*.zip")
                if path.is_file()
                and "ketod" in "/".join(path.relative_to(root).parts).lower()
            )
            archive_hint = (
                f" A KETOD ZIP appears to still be compressed: {archives[0]}."
                if archives
                else ""
            )
            raise FileNotFoundError(
                "KETOD annotations were not found."
                f"{archive_hint} Extract the upstream KETOD release so that "
                "train_ketod.json and test_ketod.json are visible below --raw-root."
            )
        raise FileNotFoundError(f"expected exactly one {filename!r} below {root}, found 0")
    if len(matches) > 1:
        candidates = ", ".join(str(path) for path in matches)
        raise RuntimeError(f"ambiguous {filename!r} below {root}; candidates: {candidates}")
    return matches[0]


def _sgd_root(raw_root: Path) -> Path:
    candidates = sorted(
        path.parent.parent
        for path in raw_root.rglob("dialogues_001.json")
        if path.is_file() and path.parent.name in {"train", "dev", "test"}
    )
    if not candidates:
        raise FileNotFoundError(f"cannot locate SGD train/dev/test directories below {raw_root}")
    roots = sorted({candidate for candidate in candidates if (candidate / "train").is_dir()})
    if len(roots) != 1:
        raise RuntimeError(f"ambiguous SGD roots: {roots}")
    return roots[0]


def _load_sgd_turn_index(
    sgd_root: Path,
    split: str,
    needed_ids: set[str],
    *,
    require_all: bool = True,
) -> dict[str, list[dict[str, str]]]:
    split_dir = sgd_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"missing SGD split directory: {split_dir}")
    index: dict[str, list[dict[str, str]]] = {}
    shard_paths = sorted(split_dir.glob("dialogues_*.json"))
    if not shard_paths:
        raise FileNotFoundError(f"no dialogue shards under {split_dir}")
    for shard_path in shard_paths:
        with shard_path.open("r", encoding="utf-8") as handle:
            shard = json.load(handle)
        if not isinstance(shard, list):
            raise ValueError(f"{shard_path}: expected top-level list")
        for dialogue in shard:
            dialogue_id = dialogue.get("dialogue_id")
            if dialogue_id not in needed_ids:
                continue
            if dialogue_id in index:
                raise ValueError(f"duplicate SGD dialogue_id {dialogue_id!r} in {split}")
            turns: list[dict[str, str]] = []
            for turn_index, turn in enumerate(dialogue.get("turns", [])):
                speaker = turn.get("speaker")
                utterance = turn.get("utterance")
                if speaker not in {"USER", "SYSTEM"} or not isinstance(utterance, str):
                    raise ValueError(f"{shard_path} dialogue {dialogue_id} turn {turn_index}: malformed turn")
                turns.append({"speaker": speaker, "utterance": utterance})
            if not turns:
                raise ValueError(f"{shard_path} dialogue {dialogue_id}: no turns")
            index[dialogue_id] = turns
    missing = sorted(needed_ids - set(index))
    if missing and require_all:
        raise KeyError(f"missing SGD {split} dialogue IDs (first 10): {missing[:10]}")
    return index


def _load_annotations(path: Path) -> list[tuple[str, list[dict[str, Any]]]]:
    with path.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    if not isinstance(annotations, list):
        raise ValueError(f"{path}: expected top-level list")
    result: list[tuple[str, list[dict[str, Any]]]] = []
    seen: set[str] = set()
    for item in annotations:
        dialogue_id = item.get("dialogue_id")
        turns = item.get("turns")
        if not isinstance(dialogue_id, str) or not isinstance(turns, list):
            raise ValueError(f"{path}: malformed annotation record")
        if dialogue_id in seen:
            raise ValueError(f"{path}: duplicate dialogue_id {dialogue_id!r}")
        seen.add(dialogue_id)
        result.append((dialogue_id, turns))
    return result


def _merge_turns(
    dialogue_id: str,
    sgd_turns: list[dict[str, str]],
    annotation_turns: list[dict[str, Any]],
    alignment_audit: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    if len(sgd_turns) != len(annotation_turns):
        # This is the explicit behavior of the released gen_ketod_data.py: zip()
        # truncates the longer SGD dialogue to the annotation length. Record every
        # such alignment rather than silently discarding source turns.
        alignment_audit.append(
            {
                "split": split,
                "dialogue_id": dialogue_id,
                "sgd_turn_count": len(sgd_turns),
                "annotation_turn_count": len(annotation_turns),
                "policy": "official_zip_truncation_to_annotation_length",
            }
        )
    merged: list[dict[str, Any]] = []
    for index, (sgd_turn, annotation_turn) in enumerate(zip(sgd_turns, annotation_turns)):
        if "enrich" not in annotation_turn:
            raise ValueError(f"KETOD {dialogue_id} turn {index}: missing enrich")
        row = dict(sgd_turn)
        row["enrich"] = bool_text(annotation_turn["enrich"]) == "True"
        merged.append(row)
    return merged


def _rows_for_dialogue(dialogue_id: str, turns: list[dict[str, Any]]) -> list[dict[str, str]]:
    context: list[str] = []
    rows: list[dict[str, str]] = []
    for index, turn in enumerate(turns):
        speaker = turn["speaker"]
        # Historical KETOD text serialization strips each utterance before adding
        # the speaker prefix and joining turns with one ASCII space.
        context.append(f"{speaker}: {turn['utterance'].strip()}")
        if speaker == "USER" and index + 1 < len(turns) and turns[index + 1]["speaker"] == "SYSTEM":
            rows.append(
                {
                    "instruction": INSTRUCTION,
                    "input": " ".join(context),
                    "output": bool_text(turns[index + 1]["enrich"]),
                }
            )
    if not rows:
        raise ValueError(f"KETOD {dialogue_id}: no USER→SYSTEM examples")
    return rows


def prepare(raw_root: Path, output_root: Path) -> dict[str, int]:
    annotation_train = _single_path(raw_root, "train_ketod.json")
    annotation_test = _single_path(raw_root, "test_ketod.json")
    sgd_root = _sgd_root(raw_root)
    train_annotations = _load_annotations(annotation_train)
    test_annotations = _load_annotations(annotation_test)
    train_ids = {dialogue_id for dialogue_id, _ in train_annotations}
    test_ids = {dialogue_id for dialogue_id, _ in test_annotations}

    # The official generator builds a train index and uses it as the fallback for test IDs.
    # The upstream generator uses the train map as a fallback for test IDs absent from SGD test.
    sgd_train = _load_sgd_turn_index(sgd_root, "train", train_ids | test_ids, require_all=False)
    missing_train = train_ids - set(sgd_train)
    if missing_train:
        raise KeyError(f"missing SGD train dialogue IDs (first 10): {sorted(missing_train)[:10]}")
    sgd_test = _load_sgd_turn_index(sgd_root, "test", test_ids, require_all=False) if test_ids else {}

    output_dir = output_root / "ketod"
    text_rows: dict[str, list[dict[str, str]]] = {"train": [], "test": []}
    feature_rows: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    alignment_audit: list[dict[str, Any]] = []
    for split, annotations in (("train", train_annotations), ("test", test_annotations)):
        for dialogue_id, annotation_turns in annotations:
            sgd_turns = sgd_test.get(dialogue_id) if split == "test" else None
            if sgd_turns is None:
                if split == "test":
                    alignment_audit.append(
                        {
                            "split": split,
                            "dialogue_id": dialogue_id,
                            "policy": "official_test_to_train_fallback",
                        }
                    )
                sgd_turns = sgd_train.get(dialogue_id)
            if sgd_turns is None:
                raise KeyError(f"KETOD {split}: no SGD dialogue for {dialogue_id}")
            merged = _merge_turns(dialogue_id, sgd_turns, annotation_turns, alignment_audit, split)
            dialogue = {"dialogue_id": dialogue_id, "turns": merged}
            text_rows[split].extend(_rows_for_dialogue(dialogue_id, merged))
            feature_rows[split].extend(ketod_feature_rows(dialogue))

    counts: dict[str, int] = {}
    for split, filename in (("train", "train_full.csv"), ("test", "test_full.csv")):
        text_path = output_dir / filename
        feature_path = output_dir / ("train_features.csv" if split == "train" else "test_features.csv")
        text_count = len(text_rows[split])
        feature_count = len(feature_rows[split])
        if text_count != feature_count:
            raise AssertionError(f"KETOD {split}: text/features count mismatch {text_count} != {feature_count}")
        if text_count != EXPECTED_ROWS[split]:
            raise AssertionError(f"KETOD {split}: expected {EXPECTED_ROWS[split]} rows, got {text_count}")
        write_csv(text_path, ["instruction", "input", "output"], text_rows[split])
        write_csv(feature_path, FEATURE_COLUMNS, feature_rows[split])
        counts[split] = text_count
    write_json(output_dir / "alignment_audit.json", alignment_audit)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    counts = prepare(args.raw_root, args.output_root)
    print(f"KETOD prepared: train={counts['train']} test={counts['test']}")


if __name__ == "__main__":
    main()
