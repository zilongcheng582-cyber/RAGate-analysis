"""
evaluate_mha_transfer.py
评估 DSTC9→KETOD MHA transfer 结果，输出 paper-ready 数字

使用方法：
    python analysis/evaluate_mha_transfer.py \
        --ketod_test data/ketod/test_features.csv \
        --mha_pred   results/mha_dstc9_on_ketod_predictions.csv
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--ketod_test", default="data/ketod/test_features.csv")
parser.add_argument("--mha_pred",   default="results/mha_dstc9_on_ketod_predictions.csv")
args = parser.parse_args()

ketod = pd.read_csv(args.ketod_test)
ketod.columns = [c.strip().lower() for c in ketod.columns]

pred_df = pd.read_csv(args.mha_pred)
pred_df.columns = [c.strip().lower() for c in pred_df.columns]

y_true = ketod["label"].values
y_pred = pred_df["mha_pred"].values
assert len(y_true) == len(y_pred), \
    f"Length mismatch: gold={len(y_true)}, pred={len(y_pred)}"

macro_f1    = f1_score(y_true, y_pred, average="macro")
minority_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
majority_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

print("=" * 60)
print("MHA DSTC9 → KETOD Transfer Evaluation")
print("=" * 60)
print(classification_report(y_true, y_pred, digits=4))
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Minority F1 : {minority_f1:.4f}  ← key number")
print(f"Majority F1 : {majority_f1:.4f}")

# ── 与 LR baseline 对比 ───────────────────────────────────────────────────────
LR_MINORITY_F1 = 0.0000
LR_MACRO_F1    = 0.4661

print("\n" + "=" * 60)
print("Comparison: LR vs MHA Cross-dataset Transfer")
print("=" * 60)
print(f"{'':20s} {'LR':>10s} {'MHA':>10s} {'Delta':>10s}")
print(f"{'Macro F1':20s} {LR_MACRO_F1:>10.4f} {macro_f1:>10.4f} {macro_f1-LR_MACRO_F1:>+10.4f}")
print(f"{'Minority F1':20s} {LR_MINORITY_F1:>10.4f} {minority_f1:>10.4f} {minority_f1-LR_MINORITY_F1:>+10.4f}")

# ── Table 4 LaTeX 行 ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Table 4 LaTeX row (copy-paste)")
print("=" * 60)
print(f"MHA & DSTC9 & KETOD & {majority_f1:.4f} & {minority_f1:.4f} & {macro_f1:.4f} \\\\")

# ── Paper framing ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Suggested Section 4.5 sentence")
print("=" * 60)
if minority_f1 <= 0.02:
    print(
        "RAGate-MHA also fails to generalize across annotation regimes, "
        f"achieving a minority-class F1 of {minority_f1:.2f} on KETOD "
        "when trained on DSTC9, confirming that annotation-induced shortcuts "
        "persist regardless of model complexity."
    )
elif minority_f1 <= 0.20:
    print(
        f"While RAGate-MHA partially improves over LR under cross-dataset transfer "
        f"(minority-class F1: {minority_f1:.2f} vs. {LR_MINORITY_F1:.2f}), "
        f"its performance remains substantially degraded from its in-domain score, "
        f"indicating that annotation-induced shortcuts are not resolved by "
        f"increased model capacity."
    )
else:
    print(
        f"RAGate-MHA achieves a minority-class F1 of {minority_f1:.2f} under "
        "cross-dataset transfer, suggesting partial robustness to annotation-induced "
        "shifts. We leave a full investigation to future work."
    )
