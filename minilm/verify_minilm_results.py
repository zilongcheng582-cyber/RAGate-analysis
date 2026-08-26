#!/usr/bin/env python3
"""Validate formal MiniLM metrics, predictions, and protocol metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

DATASET_ORDER = ("KETOD", "DSTC9", "DSTC11")
METRICS = ("macro_f1", "pos_f1", "neg_f1", "roc_auc", "average_precision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def metric_values(labels: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "pos_f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "neg_f1": float(f1_score(labels, predictions, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def close(actual: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        raise AssertionError(f"{label}: actual={actual} expected={expected}")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    formal = pd.read_csv(results_dir / "minilm_results_ready.csv")
    if len(formal) != 9:
        raise AssertionError(f"Expected 9 formal rows, got {len(formal)}")
    expected_pairs = {(train, test) for train in DATASET_ORDER for test in DATASET_ORDER}
    actual_pairs = set(zip(formal["train_on"], formal["test_on"]))
    if actual_pairs != expected_pairs:
        raise AssertionError(f"Unexpected train/test pairs: {actual_pairs ^ expected_pairs}")
    if set(formal["model"]) != {"all-MiniLM-L6-v2+LR"}:
        raise AssertionError("Unexpected formal MiniLM model label")
    if set(formal["truncation_side"]) != {"left"}:
        raise AssertionError("Formal results are not uniformly left-truncated")
    if set(formal["max_len"]) != {256}:
        raise AssertionError("Formal results are not uniformly max_len=256")
    if set(formal["threshold"]) != {0.5}:
        raise AssertionError("Formal results are not uniformly threshold=0.5")
    if set(formal["seed"]) != {42}:
        raise AssertionError("Formal results are not uniformly seed=42")

    for row in formal.to_dict(orient="records"):
        train_dataset = row["train_on"]
        test_dataset = row["test_on"]
        prediction_path = results_dir / f"pred_{train_dataset.lower()}_to_{test_dataset.lower()}.csv"
        prediction = pd.read_csv(prediction_path)
        labels = prediction["label"].to_numpy(dtype=int)
        probabilities = prediction["prob_positive"].to_numpy(dtype=float)
        predictions = prediction["prediction"].to_numpy(dtype=int)
        if len(prediction) != int(row["target_rows"]):
            raise AssertionError(f"Row count mismatch: {train_dataset}->{test_dataset}")
        expected_predictions = (probabilities >= 0.5).astype(int)
        if not np.array_equal(predictions, expected_predictions):
            raise AssertionError(f"Threshold mismatch: {train_dataset}->{test_dataset}")
        recomputed = metric_values(labels, probabilities, predictions)
        for metric in METRICS:
            close(row[metric], recomputed[metric], f"{train_dataset}->{test_dataset} {metric}")

        for flag, prefix, count_column in (
            ("raw_exact_overlap_with_source_train", "raw_nonoverlap", "raw_exact_overlap_rows"),
            (
                "tokenized_input_overlap_with_source_train",
                "model_input_nonoverlap",
                "tokenized_input_overlap_rows",
            ),
        ):
            overlap = prediction[flag].astype(bool).to_numpy()
            if int(overlap.sum()) != int(row[count_column]):
                raise AssertionError(f"Overlap count mismatch: {train_dataset}->{test_dataset} {flag}")
            keep = ~overlap
            if int(keep.sum()) != int(row[f"{prefix}_rows"]):
                raise AssertionError(f"Non-overlap row mismatch: {train_dataset}->{test_dataset} {prefix}")
            if np.unique(labels[keep]).size >= 2:
                subset = metric_values(labels[keep], probabilities[keep], predictions[keep])
                for metric in METRICS:
                    close(
                        row[f"{prefix}_{metric}"],
                        subset[metric],
                        f"{train_dataset}->{test_dataset} {prefix}_{metric}",
                    )

    metadata = json.loads((results_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("model_name") != "sentence-transformers/all-MiniLM-L6-v2":
        raise AssertionError("Run metadata MiniLM model mismatch")
    if metadata["tokenizer_truncation_side"] != "left" or metadata["max_len"] != 256:
        raise AssertionError("Run metadata truncation protocol mismatch")
    if metadata["cv_folds"] != 3 or metadata["threshold"] != 0.5:
        raise AssertionError("Run metadata evaluation protocol mismatch")
    if metadata["class_weight"] != "balanced" or metadata["seed"] != 42:
        raise AssertionError("Run metadata LR protocol mismatch")

    audit = pd.read_csv(results_dir / "minilm_input_audit.csv")
    if len(audit) != 6:
        raise AssertionError(f"Expected 6 input-audit rows, got {len(audit)}")
    if not np.allclose(audit["left_current_full_rate"], 1.0):
        raise AssertionError("Left truncation did not fully retain every current turn")
    if not (audit["right_current_full_rate"] < 1.0).all():
        raise AssertionError("Right-vs-left audit did not expose any right-truncation loss")
    if set(audit["default_truncation_side_before_override"]) != {"right"}:
        raise AssertionError("Unexpected default MiniLM tokenizer truncation side")

    summary = json.loads(
        (results_dir / "camera_ready_minilm_summary.json").read_text(encoding="utf-8")
    )
    if summary["protocol_validation"] != "PASS":
        raise AssertionError("Camera-ready MiniLM summary did not pass protocol validation")

    print("MINILM_RESULTS_VERIFICATION=PASS")
    print(f"formal_rows={len(formal)}")
    print(f"prediction_files={len(expected_pairs)}")
    print(f"input_audit_rows={len(audit)}")


if __name__ == "__main__":
    main()
