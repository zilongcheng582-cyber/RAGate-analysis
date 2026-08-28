#!/usr/bin/env python3
"""Render paper Figures 2 and 3 from the compact reference CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
from figure2_spec import PAPER_FIGURE2_LABELS, PAPER_FIGURE2_ORDER


DATASETS = ("KETOD", "DSTC9", "DSTC11")
FIGURE3_SETTINGS = (
    ("DSTC9", "DSTC9 → KETOD"),
    ("DSTC11", "DSTC11 → KETOD"),
    ("KETOD", "KETOD in-domain"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def one_value(
    rows: list[dict[str, str]], source: str, target: str, column: str, path: Path
) -> float:
    matches = [row for row in rows if row["train_on"] == source and row["test_on"] == target]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one {source}->{target} row, found {len(matches)}")
    return float(matches[0][column])


def normalized_feature_importance(reference_root: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    path = reference_root / "lightweight" / "feature_importance.csv"
    rows = read_csv(path)
    if not rows or "feature" not in rows[0]:
        raise ValueError(f"{path}: missing feature rows")
    rows_by_feature = {row["feature"]: row for row in rows}
    if set(rows_by_feature) != set(PAPER_FIGURE2_ORDER):
        raise ValueError(
            f"unexpected Figure 2 feature set: "
            f"{set(rows_by_feature) ^ set(PAPER_FIGURE2_ORDER)}"
        )
    normalized: dict[str, np.ndarray] = {}
    for dataset in DATASETS:
        values = np.asarray(
            [float(rows_by_feature[feature][dataset]) for feature in PAPER_FIGURE2_ORDER],
            dtype=float,
        )
        if (values < 0).any():
            raise ValueError(f"{path}: {dataset} contains a negative absolute coefficient")
        max_value = values.max()
        if max_value > 0:
            values = values / max_value
        if max_value > 0 and not np.isclose(values.max(), 1.0):
            raise AssertionError(f"{dataset}: normalized Figure 2 maximum is not 1")
        normalized[dataset] = values
    return list(PAPER_FIGURE2_ORDER), normalized


def figure3_values(reference_root: Path) -> dict[str, np.ndarray]:
    baseline_path = reference_root / "lightweight" / "baseline_verification.csv"
    minilm_path = reference_root / "minilm" / "minilm_results_ready.csv"
    bert_path = reference_root / "bert" / "bert_results_ready.csv"
    baseline = read_csv(baseline_path)
    minilm = read_csv(minilm_path)
    bert = read_csv(bert_path)
    sources = [source for source, _ in FIGURE3_SETTINGS]
    return {
        "Structural LR": np.asarray(
            [one_value(baseline, source, "KETOD", "rerun_f1_pos", baseline_path) for source in sources]
        ),
        "Embedding + LR": np.asarray(
            [one_value(minilm, source, "KETOD", "pos_f1", minilm_path) for source in sources]
        ),
        "Fine-tuned BERT": np.asarray(
            [one_value(bert, source, "KETOD", "pos_f1", bert_path) for source in sources]
        ),
    }


def make_figure2(reference_root: Path, output_dir: Path) -> Path:
    features, values = normalized_feature_importance(reference_root)
    labels = [PAPER_FIGURE2_LABELS[feature] for feature in features]
    x = np.arange(len(features))
    width = 0.25
    colors = ("steelblue", "darkorange", "seagreen")
    fig, ax = plt.subplots(figsize=(13, 5))
    for index, (dataset, color) in enumerate(zip(DATASETS, colors)):
        ax.bar(x + index * width, values[dataset], width, label=dataset, color=color, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Normalized |coefficient|")
    ax.set_title("Feature Importance — Normalized |Standardized LR Coefficient|")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = output_dir / "figure2_feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_figure3(reference_root: Path, output_dir: Path) -> Path:
    values = figure3_values(reference_root)
    x = np.arange(len(FIGURE3_SETTINGS))
    width = 0.25
    colors = ("steelblue", "darkorange", "seagreen")
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, ((model, model_values), color) in enumerate(zip(values.items(), colors)):
        ax.bar(x + index * width, model_values, width, label=model, color=color, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels([label for _, label in FIGURE3_SETTINGS])
    ax.set_ylabel("Positive-class F1 on KETOD")
    ax.set_title("Representation Checks")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = output_dir / "figure3_representation_checks.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=Path("reference_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure2 = make_figure2(args.reference_root, args.output_dir)
    figure3 = make_figure3(args.reference_root, args.output_dir)
    print(f"FIGURE2={figure2}")
    print(f"FIGURE3={figure3}")


if __name__ == "__main__":
    main()
