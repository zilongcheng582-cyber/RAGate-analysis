"""Reconstruct DSTC9 processed text and structural feature CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from common import FEATURE_COLUMNS, dstc9_feature_row, load_json, serialize_turns, write_csv


EXPECTED_ROWS = {"train": 71348, "val": 9663}
SPEAKER_MAP = {"U": "USER", "S": "SYSTEM"}
DSTC9_COMPONENT = re.compile(r"(?:^|[^a-z0-9])dstc9(?:[^a-z0-9]|$)", re.IGNORECASE)


def _raw_file(raw_root: Path, split: str, name: str) -> Path:
    suffix = Path("data") / split / name
    matches = sorted(
        path
        for path in raw_root.rglob(name)
        if path.is_file()
        and any(DSTC9_COMPONENT.search(part) for part in path.relative_to(raw_root).parts)
        and path.relative_to(raw_root).parts[-3:] == suffix.parts
    )
    if not matches:
        raise FileNotFoundError(
            f"DSTC9 raw file not found: expected data/{split}/{name} below {raw_root} "
            "under a directory component containing the token 'dstc9'"
        )
    if len(matches) > 1:
        candidates = ", ".join(str(path) for path in matches)
        raise RuntimeError(f"ambiguous DSTC9 data/{split}/{name}; candidates: {candidates}")
    return matches[0]


def prepare(raw_root: Path, output_root: Path) -> dict[str, int]:
    output_dir = output_root / "dstc9"
    counts: dict[str, int] = {}
    for split, output_name, feature_name in (
        ("train", "train_dstc9.csv", "train_features.csv"),
        ("val", "test_dstc9.csv", "test_features.csv"),
    ):
        logs_path = _raw_file(raw_root, split, "logs.json")
        labels_path = _raw_file(raw_root, split, "labels.json")
        logs = load_json(logs_path)
        labels = load_json(labels_path)
        if not isinstance(logs, list) or not isinstance(labels, list):
            raise ValueError(f"DSTC9 {split}: logs and labels must be top-level lists")
        if len(logs) != len(labels):
            raise ValueError(f"DSTC9 {split}: logs/labels length mismatch {len(logs)} != {len(labels)}")
        if len(logs) != EXPECTED_ROWS[split]:
            raise AssertionError(f"DSTC9 {split}: expected {EXPECTED_ROWS[split]} rows, got {len(logs)}")

        text_rows: list[dict[str, str]] = []
        feature_rows = []
        for index, (log, label) in enumerate(zip(logs, labels)):
            if not isinstance(log, list) or not isinstance(label, dict) or "target" not in label:
                raise ValueError(f"DSTC9 {split} row {index}: malformed log or label")
            text_rows.append({"input": serialize_turns(log, SPEAKER_MAP), "output": "True" if label["target"] is True else "False" if label["target"] is False else None})
            if text_rows[-1]["output"] is None:
                raise ValueError(f"DSTC9 {split} row {index}: target must be boolean")
            feature_rows.append(dstc9_feature_row(log, label, index))

        text_count = write_csv(output_dir / output_name, ["input", "output"], text_rows)
        feature_count = write_csv(output_dir / feature_name, FEATURE_COLUMNS, feature_rows)
        if text_count != feature_count:
            raise AssertionError(f"DSTC9 {split}: text/features count mismatch")
        counts[split] = text_count
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    counts = prepare(args.raw_root, args.output_root)
    print(f"DSTC9 prepared: train={counts['train']} held-out={counts['val']}")


if __name__ == "__main__":
    main()
