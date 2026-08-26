"""Reconstruct DSTC11 processed text and structural feature CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import FEATURE_COLUMNS, load_json, processed_feature_row, serialize_turns, write_csv


EXPECTED_ROWS = {"train": 28431, "val": 4173}
SPEAKER_MAP = {"U": "USER", "S": "SYSTEM"}


def _raw_file(raw_root: Path, split: str, name: str) -> Path:
    suffix = Path("data") / split / name
    matches = sorted(
        path
        for path in raw_root.rglob(name)
        if path.is_file()
        and "dstc11" in {part.lower() for part in path.relative_to(raw_root).parts}
        and path.as_posix().endswith(suffix.as_posix())
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one DSTC11 {suffix}, found {len(matches)}")
    return matches[0]


def prepare(raw_root: Path, output_root: Path) -> dict[str, int]:
    output_dir = output_root / "dstc11"
    counts: dict[str, int] = {}
    for split, output_name, feature_name in (
        ("train", "train.csv", "train_features.csv"),
        ("val", "val.csv", "test_features.csv"),
    ):
        logs_path = _raw_file(raw_root, split, "logs.json")
        labels_path = _raw_file(raw_root, split, "labels.json")
        logs = load_json(logs_path)
        labels = load_json(labels_path)
        if not isinstance(logs, list) or not isinstance(labels, list):
            raise ValueError(f"DSTC11 {split}: logs and labels must be top-level lists")
        if len(logs) != len(labels):
            raise ValueError(f"DSTC11 {split}: logs/labels length mismatch {len(logs)} != {len(labels)}")
        if len(logs) != EXPECTED_ROWS[split]:
            raise AssertionError(f"DSTC11 {split}: expected {EXPECTED_ROWS[split]} rows, got {len(logs)}")

        text_rows: list[dict[str, str]] = []
        feature_rows = []
        for index, (log, label) in enumerate(zip(logs, labels)):
            if not isinstance(log, list) or not isinstance(label, dict) or "target" not in label:
                raise ValueError(f"DSTC11 {split} row {index}: malformed log or label")
            target = label["target"]
            if not isinstance(target, bool):
                raise ValueError(f"DSTC11 {split} row {index}: target must be boolean")
            input_text = serialize_turns(log, SPEAKER_MAP)
            text_rows.append({"input": input_text, "output": "True" if target else "False"})
            feature_rows.append(processed_feature_row(input_text, target, index))

        text_count = write_csv(output_dir / output_name, ["input", "output"], text_rows)
        feature_count = write_csv(output_dir / feature_name, FEATURE_COLUMNS, feature_rows)
        if text_count != feature_count:
            raise AssertionError(f"DSTC11 {split}: text/features count mismatch")
        counts[split] = text_count
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    counts = prepare(args.raw_root, args.output_root)
    print(f"DSTC11 prepared: train={counts['train']} held-out={counts['val']}")


if __name__ == "__main__":
    main()
