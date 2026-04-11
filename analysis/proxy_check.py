"""
proxy_check.py — MHA cross-dataset 实验前的代理检验

在跑 DSTC9→KETOD MHA transfer 之前，先估计 MHA 在该方向的泛化能力。

逻辑：
1. 用 DSTC9 train_features.csv 训练 LR，在 KETOD test_features.csv 上预测
2. 找出 LR 的 false negatives（label=1 但预测=0）
3. 看这些样本里 MHA（mha_predictions.csv，KETOD in-domain）的预测情况
4. 59% MHA 同样失败 → MHA 部分缓解但问题仍存在（⚠️ 结论）

结果（已跑）：
    MHA also FN: 59.27%  ← MHA 同样失败
    MHA correct: 40.73%  ← MHA 部分缓解
    VERDICT: ⚠️ Uncertain. MHA may partially generalize.

前置条件：
    data/dstc9/train_features.csv
    data/ketod/test_features.csv
    results/mha_predictions.csv（KETOD in-domain MHA 推理结果）

Usage:
    python analysis/proxy_check.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, classification_report

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────
DSTC9_TRAIN  = "data/dstc9/train_features.csv"
KETOD_TEST   = "data/ketod/test_features.csv"
MHA_PRED_CSV = "results/mha_predictions.csv"   # KETOD in-domain MHA 预测

LABEL_COL = "label"
ALL_FEATURES = [
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
]

C_GRID = [0.01, 0.1, 1.0, 10.0]
RANDOM_STATE = 42


def train_lr(train_df):
    X = train_df[ALL_FEATURES].values.astype(float)
    y = train_df[LABEL_COL].values.astype(int)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight="balanced", max_iter=1000,
            random_state=RANDOM_STATE, solver="lbfgs",
        )),
    ])
    gs = GridSearchCV(pipe, {"lr__C": C_GRID},
                      cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
                      scoring="f1_macro", n_jobs=1)
    gs.fit(X, y)
    best = gs.best_estimator_
    best.fit(X, y)
    return best


def main():
    print("Loading data...")
    dstc9_train = pd.read_csv(DSTC9_TRAIN)
    ketod_test  = pd.read_csv(KETOD_TEST)
    mha_df      = pd.read_csv(MHA_PRED_CSV)

    print(f"DSTC9 train: {len(dstc9_train)}")
    print(f"KETOD test:  {len(ketod_test)}")

    print("\nTraining DSTC9 LR...")
    lr_model = train_lr(dstc9_train)

    X_ketod  = ketod_test[ALL_FEATURES].values.astype(float)
    y_ketod  = ketod_test[LABEL_COL].values.astype(int)
    lr_pred  = lr_model.predict(X_ketod)
    mha_pred = mha_df["mha_pred"].values.astype(int)

    print("\n===== DSTC9→KETOD LR Transfer =====")
    print(classification_report(y_ketod, lr_pred, digits=4))
    print(f"LR Macro F1: {f1_score(y_ketod, lr_pred, average='macro'):.4f}")
    print(f"LR Minority-class F1: {f1_score(y_ketod, lr_pred, pos_label=1, zero_division=0):.4f}")

    # LR error analysis
    fn_mask       = (y_ketod == 1) & (lr_pred == 0)
    all_error_mask = (lr_pred != y_ketod)

    print(f"\n===== LR Error Analysis on KETOD =====")
    print(f"Total errors:          {all_error_mask.sum()} / {len(y_ketod)}")
    print(f"False Negatives (1→0): {fn_mask.sum()}  (LR misses positive class)")
    print(f"False Positives (0→1): {((y_ketod==0)&(lr_pred==1)).sum()}  (LR over-predicts positive)")

    # MHA on LR false negatives
    print(f"\n===== MHA on LR's False Negative Samples (label=1, LR pred=0) =====")
    fn_indices = np.where(fn_mask)[0]
    mha_on_fn  = mha_pred[fn_indices]
    mha_also_fn    = (mha_on_fn == 0).sum()
    mha_correct_fn = (mha_on_fn == y_ketod[fn_indices]).sum()
    print(f"  Samples: {len(fn_indices)}")
    print(f"  MHA also predicts 0 (also FN): {mha_also_fn} ({mha_also_fn/len(fn_indices):.2%})")
    print(f"  MHA correctly predicts 1:      {mha_correct_fn} ({mha_correct_fn/len(fn_indices):.2%})")

    print(f"\n===== MHA on ALL LR Error Samples =====")
    err_indices    = np.where(all_error_mask)[0]
    mha_on_err     = mha_pred[err_indices]
    mha_also_wrong = (mha_on_err != y_ketod[err_indices]).sum()
    mha_correct    = (mha_on_err == y_ketod[err_indices]).sum()
    print(f"  Samples: {len(err_indices)}")
    print(f"  MHA also wrong: {mha_also_wrong} ({mha_also_wrong/len(err_indices):.2%})")
    print(f"  MHA correct:    {mha_correct} ({mha_correct/len(err_indices):.2%})")

    print(f"\n===== MHA on Full KETOD Test (reference) =====")
    print(classification_report(y_ketod, mha_pred, digits=4))
    print(f"MHA Macro F1: {f1_score(y_ketod, mha_pred, average='macro'):.4f}")

    # Verdict
    print("\n" + "="*60)
    print("PROXY CHECK VERDICT")
    print("="*60)
    mha_also_fn_rate = mha_also_fn / len(fn_indices) if len(fn_indices) > 0 else 0
    if mha_also_fn_rate > 0.7:
        print(f"✅ MHA also misses {mha_also_fn_rate:.0%} of LR's false negatives.")
        print("   → MHA likely also fails on DSTC→KETOD transfer.")
    elif mha_also_fn_rate > 0.4:
        print(f"⚠️  MHA misses {mha_also_fn_rate:.0%} of LR's false negatives.")
        print("   → Uncertain. MHA may partially generalize.")
        print("   → Consider running experiment but prepare for mixed results.")
    else:
        print(f"❌ MHA only misses {mha_also_fn_rate:.0%} of LR's false negatives.")
        print("   → MHA likely generalizes better than LR on DSTC→KETOD.")


if __name__ == "__main__":
    main()
