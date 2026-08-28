#!/usr/bin/env python3
"""Dependency-free sanity checker for the public reproducibility repository."""
from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("KETOD", "DSTC9", "DSTC11")
EXPECTED_PAIRS = {(source, target) for source in DATASETS for target in DATASETS}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def repository_files() -> list[Path]:
    """Return tracked and non-ignored files, or all files outside a Git checkout."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
    return [path for path in ROOT.rglob("*") if path.is_file()]


def main() -> int:
    failures: list[str] = []

    required = (
        "LICENSE",
        "README.md",
        "config_paths.json",
        "requirements-lightweight.txt",
        "requirements-minilm.txt",
        "requirements-bert.txt",
        "data/data_manifest.csv",
        "docs/PROTOCOL.md",
        "docs/PAPER_RESULTS_MAP.md",
        "docs/DATA_ACQUISITION.md",
        "figure2_spec.py",
        "lightweight/train_lr_ablation.py",
        "lightweight/feature_importance_spearman.py",
        "lightweight/run_transfer_controls.py",
        "lightweight/run_feature_controls.py",
        "minilm/minilm_input_audit.py",
        "minilm/minilm_transfer_ready.py",
        "bert/bert_input_audit.py",
        "bert/bert_transfer_ready.py",
        "bert/run_autodl.sh",
        "scripts/make_paper_figures.py",
        "tests/test_preprocessing_locators.py",
        "tests/test_paper_figures.py",
        "requirements-test.txt",
        "reference_results/lightweight/baseline_verification.csv",
        "reference_results/lightweight/feature_importance.csv",
        "reference_results/lightweight/lr_results.csv",
        "reference_results/lightweight/spearman_rho_results.csv",
        "reference_results/minilm/minilm_results_ready.csv",
        "reference_results/bert/bert_results_ready.csv",
        "reference_results/bert/run_metadata.json",
    )
    for relative in required:
        require((ROOT / relative).is_file(), f"missing required file: {relative}", failures)

    # No weights, datasets, generated predictions, archives, or caches.
    files = repository_files()
    forbidden_suffixes = {
        ".bin", ".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz",
        ".pyc", ".zip", ".7z", ".tar", ".gz",
    }
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        require(path.suffix.lower() not in forbidden_suffixes, f"binary payload: {relative}", failures)
        require(
            not ({"__pycache__", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints"} & set(path.parts)),
            f"cache path: {relative}",
            failures,
        )
        require(
            not re.match(r"(?:pred_|predictions|logits|embeddings)", path.name, re.IGNORECASE),
            f"prediction/array payload: {relative}",
            failures,
        )
        require("raw_data" not in path.parts, f"raw benchmark payload: {relative}", failures)
        require(
            not (
                len(path.parts) >= 3
                and path.parts[-3] == "data"
                and path.parts[-2] in {"ketod", "dstc9", "dstc11"}
            ),
            f"generated benchmark payload: {relative}",
            failures,
        )

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
    absolute_path_patterns = (
        re.compile(r"(?i)\b[A-Z]:[\\/]"),
        re.compile(r"/(?:root|home|Users|mnt/data)/"),
    )
    for path in files:
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() not in searchable_suffixes:
            continue
        content = path.read_text(encoding="utf-8-sig", errors="replace")
        # JSON escapes Windows separators, so normalize doubled backslashes
        # before checking for personal absolute paths.
        path_scan_content = content.replace("\\\\", "\\")
        for pattern in absolute_path_patterns:
            require(
                pattern.search(path_scan_content) is None,
                f"personal/absolute path in {path.relative_to(ROOT)}",
                failures,
            )

    lightweight = (ROOT / "lightweight" / "run_transfer_controls.py").read_text(encoding="utf-8")
    require('C_GRID = [0.01, 0.1, 1.0, 10.0]' in lightweight, "lightweight C grid", failures)
    require('class_weight="balanced"' in lightweight, "lightweight class weight", failures)
    require("nested 3-fold OOF" in lightweight, "nested source-OOF calibration", failures)
    require("thresholded F1" not in lightweight or "roc_auc" in lightweight, "ranking metrics", failures)

    importance = (ROOT / "lightweight" / "feature_importance_spearman.py").read_text(encoding="utf-8")
    require('method="min"' in importance, "feature rank tie policy", failures)
    require(
        'figure2_feature_importance.png' in importance
        and 'Normalized |coefficient|' in importance,
        "normalized paper Figure 2 output",
        failures,
    )

    paper_figures = (ROOT / "scripts" / "make_paper_figures.py").read_text(encoding="utf-8")
    for fragment in (
        'baseline_verification.csv',
        'minilm_results_ready.csv',
        'bert_results_ready.csv',
        'rerun_f1_pos',
        'figure3_representation_checks.png',
    ):
        require(fragment in paper_figures, f"paper figure source fragment: {fragment}", failures)
    figure2_spec = (ROOT / "figure2_spec.py").read_text(encoding="utf-8")
    for fragment in ('"turn_position_squared"', '"pos_sq"', '"user_q_word"', '"cons_sys"'):
        require(fragment in figure2_spec, f"paper Figure 2 display fragment: {fragment}", failures)

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

    if failures:
        print("PACKAGE_AUDIT=FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("PACKAGE_AUDIT=PASS")
    print(f"repository_files={len(files)}")
    print("python_syntax=PASS")
    print("repository_contents=PASS")
    print("no_data_weights_predictions_or_caches=PASS")
    print("relative_paths=PASS")
    print("lightweight_protocol=PASS")
    print("minilm_protocol_and_3x3_results=PASS")
    print("bert_protocol_and_3x3_results=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
