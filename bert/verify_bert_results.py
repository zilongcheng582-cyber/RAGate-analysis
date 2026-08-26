#!/usr/bin/env python3
"""Fail closed if the BERT rerun is incomplete or used the wrong protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_TARGET_ROWS = {"KETOD": 4964, "DSTC9": 9663, "DSTC11": 4173}
EXPECTED_SOURCES = tuple(EXPECTED_TARGET_ROWS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    table_path = result_dir / "bert_results_ready.csv"
    metadata_path = result_dir / "run_metadata.json"
    table = pd.read_csv(table_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    expected_pairs = {(s, t) for s in EXPECTED_SOURCES for t in EXPECTED_SOURCES}
    actual_pairs = set(zip(table["train_on"], table["test_on"]))
    if len(table) != 9 or actual_pairs != expected_pairs:
        raise AssertionError(f"Expected complete 3x3 matrix; rows={len(table)}, pairs={actual_pairs}")

    metric_cols = ["macro_f1", "pos_f1", "neg_f1", "roc_auc", "average_precision"]
    required_columns = set(metric_cols) | {
        "raw_exact_overlap_rows",
        "raw_exact_overlap_rate",
        "tokenized_input_overlap_rows",
        "tokenized_input_overlap_rate",
        "raw_nonoverlap_rows",
        "model_input_nonoverlap_rows",
    }
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise AssertionError(f"Missing result columns: {sorted(missing_columns)}")
    values = table[metric_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise AssertionError("Metrics must all be finite and in [0, 1]")

    for row in table.itertuples(index=False):
        if row.model != "BERT":
            raise AssertionError(f"Unexpected model label: {row.model!r}")
        if int(row.target_rows) != EXPECTED_TARGET_ROWS[row.test_on]:
            raise AssertionError(f"Wrong target row count for {row.train_on}->{row.test_on}")
        if float(row.threshold) != 0.5:
            raise AssertionError("BERT evaluation threshold must be 0.5")
        if str(row.truncation_side) != "left" or int(row.max_len) != 256:
            raise AssertionError("Expected left truncation at max_len=256")
        if int(row.epochs) != 3 or int(row.batch_size) != 32 or int(row.seed) != 42:
            raise AssertionError("Training protocol metadata mismatch")
        if int(row.tokenized_input_overlap_rows) < int(row.raw_exact_overlap_rows):
            raise AssertionError("Tokenized overlap cannot be smaller than raw exact overlap")
        if not 0 <= int(row.model_input_nonoverlap_rows) <= int(row.target_rows):
            raise AssertionError("Invalid model-input non-overlap row count")

    required_metadata = {
        "model_name": "bert-base-uncased",
        "tokenizer_truncation_side": "left",
        "max_len": 256,
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.10,
        "seed": 42,
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise AssertionError(f"Metadata mismatch: {key}={metadata.get(key)!r}, expected {expected!r}")

    def pair(source: str, target: str) -> dict[str, float]:
        row = table[(table.train_on == source) & (table.test_on == target)].iloc[0]
        return {k: float(row[k]) for k in metric_cols}

    summary = {
        "protocol_validation": "PASS",
        "paper_values_to_update": {
            "KETOD_to_KETOD": pair("KETOD", "KETOD"),
            "DSTC9_to_KETOD": pair("DSTC9", "KETOD"),
            "DSTC11_to_KETOD": pair("DSTC11", "KETOD"),
        },
        "interpretation_rule": (
            "Update the paper from these rerun values. The BERT capacity-check claim is supported "
            "only if both DSTC-to-KETOD directions remain weak in ranking (ROC-AUC), not merely "
            "at the fixed-threshold positive F1."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
