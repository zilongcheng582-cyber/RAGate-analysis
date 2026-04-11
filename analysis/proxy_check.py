"""
proxy_check.py — MHA cross-dataset 实验前的代理检验
在跑 DSTC9→KETOD MHA transfer 之前，先估计 MHA 在该方向的泛化能力。

逻辑：
1. 找出 DSTC9→KETOD transfer 中 LR 错误的样本（label=1 但预测=0）
2. 看这些样本里 MHA（已有 mha_predictions.csv）的预测情况
3. 如果 MHA 也大量预测错误 → MHA cross-dataset 大概率也会崩 → 放心跑
4. 如果 MHA 在这些样本上反而预测正确 → 要慎重

前置条件：
    - E:/ketod-main/ketod_release/train_features.csv
    - E:/ketod-main/ketod_release/test_features.csv
    - E:/dstc9-track1/data/train/train_features.csv
    - mha_predictions.csv（已有）
    - transfer_results.csv（已有，或重新生成）

Usage:
    python proxy_check.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.makedirs("C:/Temp", exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = "C:/Temp"

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
DSTC9_TRAIN   = "E:/dstc9-track1/data/train/train_features.csv"
KETOD_TEST    = "E:/ketod-main/ketod_release/test_features.csv"
MHA_PRED_CSV  = "mha_predictions.csv"   # KETOD test set 上的 MHA 预测

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
    # 1. 训练 DSTC9 LR，在 KETOD test 上预测
    print("Loading data...")
    dstc9_train = pd.read_csv(DSTC9_TRAIN)
    ketod_test  = pd.read_csv(KETOD_TEST)
    mha_df      = pd.read_csv(MHA_PRED_CSV)

    print(f"DSTC9 train: {len(dstc9_train)}")
    print(f"KETOD test:  {len(ketod_test)}")

    print("\nTraining DSTC9 LR...")
    lr_model = train_lr(dstc9_train)

    X_ketod = ketod_test[ALL_FEATURES].values.astype(float)
    y_ketod = ketod_test[LABEL_COL].values.astype(int)
    lr_pred = lr_model.predict(X_ketod)
    mha_pred = mha_df["mha_pred"].values.astype(int)

    print("\n===== DSTC9→KETOD LR Transfer =====")
    print(classification_report(y_ketod, lr_pred, digits=4))
    print(f"LR Macro F1: {f1_score(y_ketod, lr_pred, average='macro'):.4f}")
    print(f"LR Minority-class F1: {f1_score(y_ketod, lr_pred, pos_label=1, zero_division=0):.4f}")

    # 2. 找出 LR 在 KETOD 上的错误样本（label=1 但 LR 预测=0）
    fn_mask = (y_ketod == 1) & (lr_pred == 0)   # False Negative：LR 漏报
    fp_mask = (y_ketod == 0) & (lr_pred == 1)   # False Positive：LR 误报
    all_error_mask = (lr_pred != y_ketod)

    print(f"\n===== LR Error Analysis on KETOD =====")
    print(f"Total errors:          {all_error_mask.sum()} / {len(y_ketod)}")
    print(f"False Negatives (1→0): {fn_mask.sum()}  (LR misses positive class)")
    print(f"False Positives (0→1): {fp_mask.sum()}  (LR over-predicts positive)")

    # 3. 在 LR 的错误样本上，MHA 表现如何？
    print(f"\n===== MHA on LR's False Negative Samples (label=1, LR pred=0) =====")
    fn_indices = np.where(fn_mask)[0]
    if len(fn_indices) > 0:
        mha_on_fn = mha_pred[fn_indices]
        mha_correct_on_fn = (mha_on_fn == y_ketod[fn_indices]).sum()
        mha_also_fn = (mha_on_fn == 0).sum()
        print(f"  Samples: {len(fn_indices)}")
        print(f"  MHA also predicts 0 (also FN): {mha_also_fn} ({mha_also_fn/len(fn_indices):.2%})")
        print(f"  MHA correctly predicts 1:      {mha_correct_on_fn} ({mha_correct_on_fn/len(fn_indices):.2%})")

    print(f"\n===== MHA on ALL LR Error Samples =====")
    err_indices = np.where(all_error_mask)[0]
    mha_on_err = mha_pred[err_indices]
    mha_also_wrong = (mha_on_err != y_ketod[err_indices]).sum()
    mha_correct_on_err = (mha_on_err == y_ketod[err_indices]).sum()
    print(f"  Samples: {len(err_indices)}")
    print(f"  MHA also wrong: {mha_also_wrong} ({mha_also_wrong/len(err_indices):.2%})")
    print(f"  MHA correct:    {mha_correct_on_err} ({mha_correct_on_err/len(err_indices):.2%})")

    # 4. MHA 在全部 KETOD test 上的性能（已知，作为参考）
    print(f"\n===== MHA on Full KETOD Test (reference) =====")
    print(classification_report(y_ketod, mha_pred, digits=4))
    print(f"MHA Macro F1: {f1_score(y_ketod, mha_pred, average='macro'):.4f}")

    # 5. 综合判断
    print("\n" + "="*60)
    print("PROXY CHECK VERDICT")
    print("="*60)
    mha_also_fn_rate = mha_also_fn / len(fn_indices) if len(fn_indices) > 0 else 0
    if mha_also_fn_rate > 0.7:
        print(f"✅ MHA also misses {mha_also_fn_rate:.0%} of LR's false negatives.")
        print("   → MHA likely also fails on DSTC→KETOD transfer.")
        print("   → SAFE to run MHA cross-dataset experiment.")
    elif mha_also_fn_rate > 0.4:
        print(f"⚠️  MHA misses {mha_also_fn_rate:.0%} of LR's false negatives.")
        print("   → Uncertain. MHA may partially generalize.")
        print("   → Consider running experiment but prepare for mixed results.")
    else:
        print(f"❌ MHA only misses {mha_also_fn_rate:.0%} of LR's false negatives.")
        print("   → MHA likely generalizes better than LR on DSTC→KETOD.")
        print("   → HIGH RISK: Running MHA transfer may contradict current framing.")
        print("   → Consider Option B (downgrade claim) instead.")


if __name__ == "__main__":
    main()
