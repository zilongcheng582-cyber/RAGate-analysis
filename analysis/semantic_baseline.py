"""
semantic_baseline.py — Semantic Baseline Experiment
Extract sentence embeddings from user turns via sentence-transformers,
train LR probes, and run the same cross-dataset transfer matrix as
the structural LR baseline.

Purpose: Address potential reviewer criticism that structural LR has
insufficient capacity. Results are usable regardless of outcome:
  - Semantic also fails → annotation incompatibility survives even with
    richer representations
  - Semantic slightly better → structural shortcuts are brittle, but
    incompatibility remains annotation-driven

Dependencies:
    pip install sentence-transformers scikit-learn pandas numpy

Usage:
    # Use default paths from config block below:
    python semantic_baseline.py

    # Or override paths via CLI:
    python semantic_baseline.py \\
        --ketod-train /path/to/ketod/train_full.csv \\
        --ketod-test  /path/to/ketod/test_full.csv \\
        --dstc9-train /path/to/dstc9/train.csv \\
        --dstc9-test  /path/to/dstc9/test.csv \\
        --dstc11-train /path/to/dstc11/train.csv \\
        --dstc11-test  /path/to/dstc11/val.csv

    # Skip comparison table if you don't have structural baseline numbers:
    python semantic_baseline.py --no-comparison

Outputs:
    semantic_results.csv    — full transfer matrix (in-domain + cross-dataset)
    semantic_summary.txt    — paper-ready numbers for key directions

Notes:
    - Requires train_full.csv / test_full.csv (files with raw text).
      NOT train_features.csv (structural features, no text column).
    - If label/text column names differ from defaults, edit LABEL_COLS
      and TEXT_COLS below, or use --label-col / --text-col flags.
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, classification_report
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit paths and column names here if not using CLI arguments
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DATASETS = {
    "KETOD": {
        "train": "data/ketod/train_full.csv",
        "test":  "data/ketod/test_full.csv",
    },
    "DSTC9": {
        "train": "data/dstc9/train_dstc9.csv",
        "test":  "data/dstc9/test_dstc9.csv",
    },
    "DSTC11": {
        "train": "data/dstc11/train.csv",
        "test":  "data/dstc11/val.csv",
    },
}

# Column names for label and text in each dataset.
# label column: expects True/False strings or 1/0 integers.
# text column:  raw user turn text.
LABEL_COLS = {
    "KETOD":  "output",
    "DSTC9":  "output",
    "DSTC11": "output",
}
TEXT_COLS = {
    "KETOD":  "input",
    "DSTC9":  "input",
    "DSTC11": "input",
}

# Structural LR baseline numbers for comparison table.
# Set to None to skip the comparison section entirely.
# Format: "TRAIN→TEST": (macro_f1, minority_f1)
STRUCTURAL_BASELINE = {
    "KETOD→KETOD":   (0.5364, 0.3302),
    "KETOD→DSTC9":   (0.4165, 0.1872),
    "KETOD→DSTC11":  (0.4121, 0.3670),
    "DSTC9→KETOD":   (0.4661, 0.0000),
    "DSTC9→DSTC9":   (0.7982, 0.7398),
    "DSTC9→DSTC11":  (0.8359, 0.8564),
    "DSTC11→KETOD":  (0.4661, 0.0000),
    "DSTC11→DSTC9":  (0.8097, 0.7422),
    "DSTC11→DSTC11": (0.8424, 0.8524),
}

# Key cross-dataset directions to highlight in paper-ready summary.
# Format: (train_dataset, test_dataset)
HIGHLIGHT_PAIRS = [
    ("DSTC9",  "KETOD"),
    ("DSTC11", "KETOD"),
]

MODEL_NAME   = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality
BATCH_SIZE   = 256
C_GRID       = [0.01, 0.1, 1.0, 10.0]
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path, label_col, text_col):
    """Load CSV, return (texts: list[str], labels: np.ndarray[int])."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path)
    for col in (label_col, text_col):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {path}. "
                f"Available columns: {list(df.columns)}"
            )
    if df[label_col].dtype == object:
        labels = (df[label_col].str.strip() == "True").astype(int).values
    else:
        labels = df[label_col].astype(int).values
    texts = df[text_col].fillna("").tolist()
    return texts, labels


def encode_texts(texts, model, batch_size=256):
    """Encode texts to embeddings using sentence-transformers."""
    print(f"    Encoding {len(texts)} texts...")
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def train_lr_probe(X_train, y_train):
    """Train LR with StandardScaler and GridSearchCV over C."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )),
    ])
    gs = GridSearchCV(
        pipe,
        {"lr__C": C_GRID},
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_macro",
        n_jobs=1,
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    best.fit(X_train, y_train)
    return best, gs.best_params_["lr__C"]


def evaluate_probe(probe, X_test, y_test):
    """Return macro F1, minority F1, and full classification report."""
    y_pred = probe.predict(X_test)
    return {
        "macro_f1":    round(f1_score(y_test, y_pred, average="macro"), 4),
        "minority_f1": round(f1_score(y_test, y_pred, pos_label=1,
                                      zero_division=0), 4),
        "report":      classification_report(y_test, y_pred, digits=4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ketod-train",  default=None)
    p.add_argument("--ketod-test",   default=None)
    p.add_argument("--dstc9-train",  default=None)
    p.add_argument("--dstc9-test",   default=None)
    p.add_argument("--dstc11-train", default=None)
    p.add_argument("--dstc11-test",  default=None)
    p.add_argument("--model",        default=MODEL_NAME,
                   help="sentence-transformers model name or local path")
    p.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    p.add_argument("--no-comparison", action="store_true",
                   help="Skip structural vs semantic comparison table")
    p.add_argument("--output-dir",   default=".",
                   help="Directory for output CSV and summary files")
    return p.parse_args()


def resolve_datasets(args):
    """Merge CLI path overrides with defaults."""
    datasets = {
        k: dict(v) for k, v in DEFAULT_DATASETS.items()
    }
    overrides = {
        "KETOD":  {"train": args.ketod_train,  "test": args.ketod_test},
        "DSTC9":  {"train": args.dstc9_train,  "test": args.dstc9_test},
        "DSTC11": {"train": args.dstc11_train, "test": args.dstc11_test},
    }
    for ds, splits in overrides.items():
        for split, path in splits.items():
            if path is not None:
                datasets[ds][split] = path
    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    datasets = resolve_datasets(args)
    ds_names = list(datasets.keys())
    os.makedirs(args.output_dir, exist_ok=True)

    # Fix joblib temp dir on Windows if needed
    if sys.platform == "win32":
        tmp = os.path.join(tempfile.gettempdir(), "joblib_ragate")
        os.makedirs(tmp, exist_ok=True)
        os.environ.setdefault("JOBLIB_TEMP_FOLDER", tmp)

    print(f"Loading sentence-transformers model: {args.model}")
    st_model = SentenceTransformer(args.model)

    # ── Step 1: Encode all splits ─────────────────────────────────────────
    embeddings, labels = {}, {}
    for ds in ds_names:
        print(f"\n{'='*54}")
        print(f"Dataset: {ds}")
        for split in ("train", "test"):
            path = datasets[ds][split]
            texts, y = load_dataset(path, LABEL_COLS[ds], TEXT_COLS[ds])
            emb = encode_texts(texts, st_model, args.batch_size)
            embeddings[f"{ds}_{split}"] = emb
            labels[f"{ds}_{split}"]     = y
            print(f"  {split}: {len(texts)} samples | shape {emb.shape} | "
                  f"pos={int(y.sum())} neg={int((y == 0).sum())}")

    # ── Step 2: Train one LR probe per dataset ────────────────────────────
    probes = {}
    print(f"\n{'='*54}")
    print("Training LR probes...")
    for ds in ds_names:
        X, y = embeddings[f"{ds}_train"], labels[f"{ds}_train"]
        probe, best_c = train_lr_probe(X, y)
        probes[ds] = probe
        print(f"  {ds}: best_C={best_c}")

    # ── Step 3: Full transfer matrix ──────────────────────────────────────
    print(f"\n{'='*54}")
    print("SEMANTIC BASELINE — Transfer Matrix (Macro F1)")
    header = f"{'Train→Test':<14}" + "".join(f"{d:>10}" for d in ds_names)
    print(header)
    print("-" * len(header))

    records, f1_matrix = [], {}
    for train_ds in ds_names:
        row = f"{train_ds:<14}"
        for test_ds in ds_names:
            X_test = embeddings[f"{test_ds}_test"]
            y_test = labels[f"{test_ds}_test"]
            m = evaluate_probe(probes[train_ds], X_test, y_test)
            row += f"{m['macro_f1']:>10.4f}"
            key = f"{train_ds}→{test_ds}"
            f1_matrix[key] = (m["macro_f1"], m["minority_f1"])
            records.append({
                "model":       "Semantic-LR",
                "train_on":    train_ds,
                "test_on":     test_ds,
                "in_domain":   train_ds == test_ds,
                "macro_f1":    m["macro_f1"],
                "minority_f1": m["minority_f1"],
            })
        print(row)

    # ── Step 4: Comparison table (optional) ───────────────────────────────
    if not args.no_comparison and STRUCTURAL_BASELINE:
        print(f"\n{'='*70}")
        print("COMPARISON: Structural LR vs Semantic LR")
        print(f"{'Transfer':<20} {'Struct Mac':>10} {'Sem Mac':>8} "
              f"{'Struct Min':>11} {'Sem Min':>8} {'Δ Min':>7}")
        print("-" * 70)
        for key, (s_mac, s_min) in STRUCTURAL_BASELINE.items():
            if key not in f1_matrix:
                continue
            sem_mac, sem_min = f1_matrix[key]
            train_ds, test_ds = key.split("→")
            is_cross = train_ds != test_ds
            flag = " 🔴" if is_cross and test_ds == "KETOD" else ""
            print(f"{key:<20} {s_mac:>10.4f} {sem_mac:>8.4f} "
                  f"{s_min:>11.4f} {sem_min:>8.4f} {sem_min-s_min:>+7.4f}{flag}")

    # ── Step 5: Save outputs ──────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, "semantic_results.csv")
    pd.DataFrame(records).to_csv(results_path, index=False)
    print(f"\nSaved → {results_path}")

    # Paper-ready summary for highlighted pairs
    lines = ["PAPER-READY SUMMARY", "=" * 54]
    for train_ds, test_ds in HIGHLIGHT_PAIRS:
        key = f"{train_ds}→{test_ds}"
        if key not in f1_matrix:
            continue
        sem_mac, sem_min = f1_matrix[key]
        lines.append(f"\n{key}")
        lines.append(f"  Semantic LR:  Macro={sem_mac:.4f}, Minority={sem_min:.4f}")
        if STRUCTURAL_BASELINE and key in STRUCTURAL_BASELINE:
            s_mac, s_min = STRUCTURAL_BASELINE[key]
            lines.append(f"  Structural LR: Macro={s_mac:.4f}, Minority={s_min:.4f}")
            lines.append(f"  Δ Minority:    {sem_min - s_min:+.4f}")

    summary_path = os.path.join(args.output_dir, "semantic_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved → {summary_path}")

    print(f"\n{'='*54}")
    print("Done.")


if __name__ == "__main__":
    main()
