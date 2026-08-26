"""Inventory staged raw files without modifying them."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_hint(path: Path) -> str:
    if path.suffix.lower() != ".json":
        return "not-json"
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        prefix = handle.read(8192).lstrip()
    top = "array" if prefix.startswith("[") else "object" if prefix.startswith("{") else "unknown"
    keys = sorted(set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', prefix)))
    return f"top_level={top}; sample_keys={','.join(keys[:20])}"


def classify(relative_path: str) -> tuple[str, str]:
    p = relative_path.lower().replace("\\", "/")
    if p.startswith("dstc9/") or "/dstc9/" in p:
        split = "train" if "/data/train/" in p else "validation" if "/data/val/" in p else "other"
        return "DSTC9", split
    if p.startswith("dstc11/") or "/dstc11/" in p:
        split = "train" if "/data/train/" in p else "validation" if "/data/val/" in p else "test/other"
        return "DSTC11", split
    if p.startswith("ketod_release/") or "/ketod_release/" in p:
        if "train_ketod" in p:
            return "KETOD", "train-annotation"
        if "test_ketod" in p:
            return "KETOD", "test-annotation"
        if "dev_ketod" in p:
            return "KETOD", "dev-annotation"
        return "KETOD", "release-support"
    if p.startswith("sgd/") or "/sgd/" in p:
        split = "train" if "/train/" in p else "dev" if "/dev/" in p else "test" if "/test/" in p else "other"
        return "KETOD upstream SGD", split
    if p.startswith("ketod_outer/") or "/ketod_outer/" in p:
        return "KETOD", "release-archive"
    return "unclassified", "unclassified"


def build(raw_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(raw_root).as_posix()
        dataset, split = classify(relative)
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "extension": path.suffix.lower(),
                "candidate_dataset": dataset,
                "candidate_split": split,
                "detected_schema": schema_hint(path),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "raw_inventory.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (output_dir / "raw_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["relative_path"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# RAW_DATA_MAP",
        "",
        "This map was generated from the staged read-only extraction. Archives and E-drive files were not modified.",
        "",
        "- KETOD train/test annotations are joined to the staged Google SGD train/test dialogue shards by `dialogue_id`.",
        "- DSTC9 train/validation use `data/train` and `data/val` logs/labels.",
        "- DSTC11 train/validation use `data/train` and `data/val` logs/labels.",
        "- Official test/evaluation material is retained for provenance but is not used as the paper's held-out split.",
        "",
    ]
    (output_dir / "RAW_DATA_MAP.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = build(args.raw_root, args.output_dir)
    print(f"Inventoried {len(rows)} files from {args.raw_root}")


if __name__ == "__main__":
    main()
