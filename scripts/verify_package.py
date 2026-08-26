#!/usr/bin/env python3
"""Dependency-free structural and protocol audit for the public artifact."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("KETOD", "DSTC9", "DSTC11")
EXPECTED_PAIRS = {(source, target) for source in DATASETS for target in DATASETS}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def verify_data(data_root: Path, failures: list[str]) -> int:
    manifest = read_rows(ROOT / "data" / "data_manifest.csv")
    for row in manifest:
        path = data_root / row["relative_path"]
        require(path.is_file(), f"missing data file: {row['relative_path']}", failures)
        if path.is_file():
            require(
                path.stat().st_size == int(row["bytes"]),
                f"data size mismatch: {row['relative_path']}",
                failures,
            )
            require(
                sha256(path) == row["sha256"],
                f"data hash mismatch: {row['relative_path']}",
                failures,
            )
    return len(manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-data", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    args = parser.parse_args()
    failures: list[str] = []

    required = (
        "README.md",
        "ARTIFACT_MANIFEST.md",
        "config_paths.json",
        "requirements-lightweight.txt",
        "requirements-minilm.txt",
        "requirements-bert.txt",
        "data/data_manifest.csv",
        "docs/PROTOCOL.md",
        "docs/PAPER_RESULTS_MAP.md",
        "lightweight/train_lr_ablation.py",
        "lightweight/feature_importance_spearman.py",
        "lightweight/run_transfer_controls.py",
        "lightweight/run_feature_controls.py",
        "minilm/minilm_input_audit.py",
        "minilm/minilm_transfer_ready.py",
        "bert/bert_input_audit.py",
        "bert/bert_transfer_ready.py",
        "bert/run_autodl.sh",
        "reference_results/minilm/minilm_results_ready.csv",
        "reference_results/bert/bert_results_ready.csv",
        "SHA256SUMS.txt",
    )
    for relative in required:
        require((ROOT / relative).is_file(), f"missing required file: {relative}", failures)

    # No removed experiment implementation, weights, datasets or caches.
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    forbidden_names = re.compile(r"(?:^|[_-])mha(?:[_-]|$)", re.IGNORECASE)
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".pyc"}
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        require(not forbidden_names.search(relative), f"removed experiment path: {relative}", failures)
        require(path.suffix.lower() not in forbidden_suffixes, f"binary payload: {relative}", failures)
        require("__pycache__" not in path.parts, f"Python cache: {relative}", failures)
        require(not path.name.startswith("pred_"), f"prediction payload: {relative}", failures)

    # All Python source must parse without importing optional ML dependencies.
    for path in ROOT.rglob("*.py"):
        try:
            compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures.append(f"Python syntax error in {path.relative_to(ROOT)}: {exc}")

    config = json.loads((ROOT / "config_paths.json").read_text(encoding="utf-8"))
    require(tuple(config) == DATASETS, "config dataset order/name mismatch", failures)
    for dataset, values in config.items():
        for role, value in values.items():
            path = Path(value)
            require(not path.is_absolute(), f"absolute config path: {dataset}.{role}", failures)
            require(value.replace("\\", "/").startswith("data/"), f"non-data config path: {value}", failures)

    searchable_suffixes = {".py", ".json", ".md", ".csv", ".sh", ".txt"}
    personal_patterns = ("C:\\Users\\", "E:\\", "E:/", "/root/", "程子龙", "gpt版本")
    removed_code_patterns = ("train_MHA", "mha_inference", "RAGate-MHA")
    for path in files:
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() not in searchable_suffixes:
            continue
        content = path.read_text(encoding="utf-8-sig", errors="replace")
        # JSON escapes Windows separators, so normalize doubled backslashes
        # before checking for personal absolute paths.
        path_scan_content = content.replace("\\\\", "\\")
        for pattern in personal_patterns:
            require(
                pattern not in content and pattern not in path_scan_content,
                f"personal/absolute path in {path.relative_to(ROOT)}",
                failures,
            )
        if path.suffix.lower() in {".py", ".sh", ".json"}:
            for pattern in removed_code_patterns:
                require(pattern not in content, f"removed experiment code in {path.relative_to(ROOT)}", failures)

    lightweight = (ROOT / "lightweight" / "run_transfer_controls.py").read_text(encoding="utf-8")
    require('C_GRID = [0.01, 0.1, 1.0, 10.0]' in lightweight, "lightweight C grid", failures)
    require('class_weight="balanced"' in lightweight, "lightweight class weight", failures)
    require("nested 3-fold OOF" in lightweight, "nested source-OOF calibration", failures)
    require("thresholded F1" not in lightweight or "roc_auc" in lightweight, "ranking metrics", failures)

    minilm = (ROOT / "minilm" / "minilm_transfer_ready.py").read_text(encoding="utf-8")
    for fragment in (
        'default="left"',
        'default=256',
        'class_weight="balanced"',
        'TEXT_COL = "input"',
        'average_precision_score',
        'tokenizer.truncation_side',
    ):
        require(fragment in minilm, f"MiniLM protocol fragment: {fragment}", failures)

    bert = (ROOT / "bert" / "bert_transfer_ready.py").read_text(encoding="utf-8")
    for fragment in (
        'MODEL_NAME = "bert-base-uncased"',
        'default=3',
        'default=32',
        'default=256',
        'default="left"',
        'CrossEntropyLoss(weight=',
        'average_precision_score',
    ):
        require(fragment in bert, f"BERT protocol fragment: {fragment}", failures)

    result_specs = (
        ("reference_results/minilm/minilm_results_ready.csv", "truncation_side", "left"),
        ("reference_results/bert/bert_results_ready.csv", "truncation_side", "left"),
    )
    for relative, side_col, expected_side in result_specs:
        rows = read_rows(ROOT / relative)
        pairs = {(row["train_on"], row["test_on"]) for row in rows}
        require(len(rows) == 9 and pairs == EXPECTED_PAIRS, f"incomplete 3x3 results: {relative}", failures)
        require({row[side_col] for row in rows} == {expected_side}, f"wrong truncation: {relative}", failures)
        require({float(row["threshold"]) for row in rows} == {0.5}, f"wrong threshold: {relative}", failures)
        require({int(float(row["max_len"])) for row in rows} == {256}, f"wrong max length: {relative}", failures)
        if "minilm" in relative:
            require(
                {row.get("model") for row in rows} == {"all-MiniLM-L6-v2+LR"},
                f"wrong MiniLM model label: {relative}",
                failures,
            )
        if "bert" in relative:
            require(
                {row.get("model") for row in rows} == {"BERT"},
                f"wrong BERT model label: {relative}",
                failures,
            )
        require(
            "average_precision" in rows[0],
            f"missing average_precision column: {relative}",
            failures,
        )

    manifest_path = ROOT / "SHA256SUMS.txt"
    hash_count = 0
    if manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            target = ROOT / relative
            require(target.is_file(), f"manifest target missing: {relative}", failures)
            if target.is_file():
                require(sha256(target) == expected, f"manifest hash mismatch: {relative}", failures)
            hash_count += 1

    data_count = 0
    if args.verify_data:
        data_count = verify_data(args.data_root.resolve(), failures)

    if failures:
        print("PACKAGE_AUDIT=FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("PACKAGE_AUDIT=PASS")
    print(f"package_files={len(files)}")
    print(f"hashed_files={hash_count}")
    print("python_syntax=PASS")
    print("no_removed_experiment_code=PASS")
    print("no_data_weights_predictions_or_caches=PASS")
    print("relative_paths=PASS")
    print("lightweight_protocol=PASS")
    print("minilm_protocol_and_3x3_results=PASS")
    print("bert_protocol_and_3x3_results=PASS")
    if args.verify_data:
        print(f"verified_data_files={data_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
