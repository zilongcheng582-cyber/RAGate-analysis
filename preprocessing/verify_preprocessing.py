"""Fail-closed byte/semantic verification for reconstructed benchmark files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


FILES = [
    ("KETOD", "train", "ketod/train_full.csv", "train_full.csv", "train_features.csv"),
    ("KETOD", "test", "ketod/test_full.csv", "test_full.csv", "test_features.csv"),
    ("DSTC9", "train", "dstc9/train_dstc9.csv", "train_dstc9.csv", "train_features.csv"),
    ("DSTC9", "held-out", "dstc9/test_dstc9.csv", "test_dstc9.csv", "test_features.csv"),
    ("DSTC11", "train", "dstc11/train.csv", "train.csv", "train_features.csv"),
    ("DSTC11", "held-out", "dstc11/val.csv", "val.csv", "test_features.csv"),
]
EXPECTED_ROWS = {
    "KETOD/train": 41939,
    "KETOD/test": 4964,
    "DSTC9/train": 71348,
    "DSTC9/held-out": 9663,
    "DSTC11/train": 28431,
    "DSTC11/held-out": 4173,
}
FEATURE_COLUMNS = [
    "dialogue_id",
    "turn_idx",
    "turn_position_ratio",
    "prev_sys_is_question",
    "user_has_question",
    "user_starts_question_word",
    "user_turn_len_log",
    "sys_turn_len_log",
    "dialogue_len_log",
    "consecutive_sys_turns",
    "turn_len_ratio",
    "turn_position_squared",
    "label",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        return list(reader.fieldnames), list(reader)


def decimal_equal(left: str, right: str) -> bool:
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return False


def compare_rows(
    dataset: str,
    split: str,
    generated: Path,
    canonical: Path,
    differences: list[dict[str, Any]],
) -> tuple[bool, int]:
    before = len(differences)
    gen_header, gen_rows = read_csv(generated)
    can_header, can_rows = read_csv(canonical)
    if gen_header != can_header:
        differences.append(
            {
                "dataset": dataset,
                "split": split,
                "row_index": "header",
                "column": "__columns__",
                "reconstructed_value": repr(gen_header),
                "canonical_value": repr(can_header),
                "source_raw_identifier": "",
            }
        )
    if len(gen_rows) != len(can_rows):
        differences.append(
            {
                "dataset": dataset,
                "split": split,
                "row_index": "count",
                "column": "__row_count__",
                "reconstructed_value": str(len(gen_rows)),
                "canonical_value": str(len(can_rows)),
                "source_raw_identifier": "",
            }
        )
    numeric_columns = set(FEATURE_COLUMNS) - {"dialogue_id"}
    for index, (gen, can) in enumerate(zip(gen_rows, can_rows)):
        for column in gen_header:
            left = gen.get(column, "")
            right = can.get(column, "")
            equal = decimal_equal(left, right) if column in numeric_columns else left == right
            if not equal:
                differences.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "row_index": index,
                        "column": column,
                        "reconstructed_value": left,
                        "canonical_value": right,
                        "source_raw_identifier": gen.get("dialogue_id", str(index)),
                    }
                )
    return len(differences) == before, len(gen_rows)


def canonical_path(args: argparse.Namespace, dataset: str, relative: str) -> Path | None:
    roots = {
        "KETOD": args.canonical_ketod_root,
        "DSTC9": args.canonical_dstc9_root,
        "DSTC11": args.canonical_dstc11_root,
    }
    root = roots[dataset]
    if root is None:
        return None
    filename = Path(relative).name
    if dataset == "KETOD":
        return root / filename
    if dataset == "DSTC9":
        return root / ("train" if filename.startswith("train") else "val") / filename
    return root / filename


def verify(args: argparse.Namespace) -> dict[str, Any]:
    generated_root = args.generated_root
    manifest_path = args.manifest
    manifest_rows = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                manifest_rows[row["relative_path"]] = row
    differences: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for dataset, split, relative, text_name, feature_name in FILES:
        # The generated root contains dataset directories directly; manifest paths
        # are data-relative (e.g. data/ketod/train_full.csv).
        generated = generated_root / Path(relative).parts[0] / Path(relative).parts[1]
        manifest = manifest_rows.get(f"data/{relative}", {})
        if not generated.exists():
            raise FileNotFoundError(generated)
        header, rows = read_csv(generated)
        expected = EXPECTED_ROWS[f"{dataset}/{split}"]
        if len(rows) != expected:
            differences.append({"dataset": dataset, "split": split, "row_index": "count", "column": "__row_count__", "reconstructed_value": str(len(rows)), "canonical_value": str(expected), "source_raw_identifier": ""})
        if manifest and int(manifest.get("rows", expected)) != len(rows):
            differences.append({"dataset": dataset, "split": split, "row_index": "manifest", "column": "__row_count__", "reconstructed_value": str(len(rows)), "canonical_value": str(manifest.get("rows")), "source_raw_identifier": ""})
        canonical = canonical_path(args, dataset, relative)
        byte_exact = None
        semantic_exact = None
        result = {"dataset": dataset, "split": split, "relative_path": relative, "canonical": "LOCAL-ONLY oracle supplied via CLI" if canonical else None, "byte_exact": byte_exact, "semantic_exact": semantic_exact, "rows": len(rows), "columns": header}
        if canonical is not None and canonical.exists():
            generated_hash = sha256(generated)
            canonical_hash = sha256(canonical)
            byte_exact = generated_hash.lower() == canonical_hash.lower()
            if byte_exact:
                semantic_exact = True
            else:
                before = len(differences)
                semantic_exact, _ = compare_rows(dataset, split, generated, canonical, differences)
                if len(differences) > before and semantic_exact:
                    semantic_exact = False
            result.update({"generated_sha256": generated_hash, "byte_exact": byte_exact, "semantic_exact": semantic_exact})
        results.append(result)

    # Cross-file label and feature invariants.
    for dataset, split, relative, text_name, feature_name in FILES:
        text_path = generated_root / Path(relative).parts[0] / Path(relative).parts[1]
        feature_path = generated_root / Path(relative).parts[0] / feature_name
        text_header, text_rows = read_csv(text_path)
        feature_header, feature_rows = read_csv(feature_path)
        if feature_header != FEATURE_COLUMNS:
            differences.append({"dataset": dataset, "split": split, "row_index": "header", "column": "feature_schema", "reconstructed_value": repr(feature_header), "canonical_value": repr(FEATURE_COLUMNS), "source_raw_identifier": ""})
        if len(text_rows) != len(feature_rows):
            differences.append({"dataset": dataset, "split": split, "row_index": "count", "column": "text_feature_alignment", "reconstructed_value": str(len(text_rows)), "canonical_value": str(len(feature_rows)), "source_raw_identifier": ""})
        for index, row in enumerate(feature_rows):
            if row.get("label") not in {"0", "1"}:
                differences.append({"dataset": dataset, "split": split, "row_index": index, "column": "label", "reconstructed_value": row.get("label", ""), "canonical_value": "0 or 1", "source_raw_identifier": row.get("dialogue_id", str(index))})
            try:
                ratio = float(row["turn_position_ratio"])
                squared = float(row["turn_position_squared"])
                if not math.isfinite(ratio) or not math.isfinite(squared) or not math.isclose(squared, ratio**2, rel_tol=0.0, abs_tol=1e-15):
                    raise ValueError
                for column in ("user_turn_len_log", "sys_turn_len_log", "dialogue_len_log", "turn_len_ratio"):
                    if not math.isfinite(float(row[column])):
                        raise ValueError
            except (KeyError, ValueError):
                differences.append({"dataset": dataset, "split": split, "row_index": index, "column": "feature_numeric", "reconstructed_value": repr(row), "canonical_value": "finite values and squared ratio", "source_raw_identifier": row.get("dialogue_id", str(index))})
            if index < len(text_rows):
                expected_label = "1" if text_rows[index].get("output") == "True" else "0" if text_rows[index].get("output") == "False" else "invalid"
                if row.get("label") != expected_label:
                    differences.append({"dataset": dataset, "split": split, "row_index": index, "column": "label", "reconstructed_value": row.get("label", ""), "canonical_value": expected_label, "source_raw_identifier": row.get("dialogue_id", str(index))})
                if not text_rows[index].get("input", "").strip():
                    differences.append({"dataset": dataset, "split": split, "row_index": index, "column": "input", "reconstructed_value": "", "canonical_value": "non-null", "source_raw_identifier": row.get("dialogue_id", str(index))})

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    diff_path = report_dir / "verification_differences.csv"
    fields = ["dataset", "split", "row_index", "column", "reconstructed_value", "canonical_value", "source_raw_identifier"]
    with diff_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(differences)
    summary = {"passed": not differences, "results": results, "difference_count": len(differences), "differences_path": diff_path.as_posix()}
    (report_dir / "verification_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if differences:
        raise SystemExit(f"verification failed with {len(differences)} differences; see {diff_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/preprocessing_verification"))
    parser.add_argument("--canonical-ketod-root", type=Path)
    parser.add_argument("--canonical-dstc9-root", type=Path)
    parser.add_argument("--canonical-dstc11-root", type=Path)
    args = parser.parse_args()
    verify(args)


if __name__ == "__main__":
    main()
