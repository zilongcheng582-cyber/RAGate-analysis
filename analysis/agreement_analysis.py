# ── CONFIG ────────────────────────────────────
KETOD_TRAIN  = "data/ketod/train_features.csv"
KETOD_TEST   = "data/ketod/test_features.csv"
MHA_PRED_CSV = "results/mha_predictions.csv"
# ──────────────────────────────────────────────
"""
agreement_analysis.py  —  Step 6
LR Full 模型 vs MHA 的逐样本预测 agreement 分析。
证明 MHA 的决策边界与 LR shortcut 高度一致。

前置条件：
    - mha_predictions.csv（从 AutoDL 下载）
    - E:/ketod-main/ketod_release/test_features.csv

Usage:
    python agreement_analysis.py

输出：
    agreement_results.csv     — 每个样本的 label/lr_pred/mha_pred
    agreement_summary.txt     — 统计摘要（论文用）
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
from sklearn.metrics import (
    f1_score, cohen_kappa_score, classification_report,
    confusion_matrix
)

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────
KETOD_TRAIN    = "E:/ketod-main/ketod_release/train_features.csv"
KETOD_TEST     = "E:/ketod-main/ketod_release/test_features.csv"
MHA_PRED_CSV   = "mha_predictions.csv"   # 从 AutoDL 下载到本地

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


# ─────────────────────────────────────────────
# 训练 LR Full 模型，得到测试集预测
# ─────────────────────────────────────────────

def train_lr_and_predict(train_df, test_df):
    X_train = train_df[ALL_FEATURES].values.astype(float)
    y_train = train_df[LABEL_COL].values.astype(int)
    X_test  = test_df[ALL_FEATURES].values.astype(float)

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
        pipe, {"lr__C": C_GRID},
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_macro",
        n_jobs=1,
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    best.fit(X_train, y_train)

    lr_pred  = best.predict(X_test)
    lr_proba = best.predict_proba(X_test)[:, 1]
    print(f"LR best_C = {gs.best_params_['lr__C']}")
    return lr_pred, lr_proba


# ─────────────────────────────────────────────
# Agreement 分析
# ─────────────────────────────────────────────

def analyze_agreement(label, lr_pred, mha_pred):
    n = len(label)

    # Overall agreement
    agree_mask   = (lr_pred == mha_pred)
    agree_rate   = agree_mask.mean()
    kappa        = cohen_kappa_score(lr_pred, mha_pred)

    # Agreement breakdown by true label
    agree_when_0 = agree_mask[label == 0].mean()
    agree_when_1 = agree_mask[label == 1].mean()

    # 四象限：(lr_pred, mha_pred)
    both_correct   = ((lr_pred == label) & (mha_pred == label)).sum()
    both_wrong     = ((lr_pred != label) & (mha_pred != label)).sum()
    lr_only_right  = ((lr_pred == label) & (mha_pred != label)).sum()
    mha_only_right = ((lr_pred != label) & (mha_pred == label)).sum()

    print("\n" + "="*60)
    print("AGREEMENT ANALYSIS — LR Full vs MHA (KETOD test set)")
    print("="*60)
    print(f"  Total samples:        {n}")
    print(f"  Overall agreement:    {agree_rate:.4f} ({agree_mask.sum()}/{n})")
    print(f"  Cohen's Kappa:        {kappa:.4f}")
    print(f"  Agreement | label=0:  {agree_when_0:.4f}")
    print(f"  Agreement | label=1:  {agree_when_1:.4f}")
    print(f"\n  Both correct:         {both_correct} ({both_correct/n:.3f})")
    print(f"  Both wrong:           {both_wrong} ({both_wrong/n:.3f})")
    print(f"  LR only correct:      {lr_only_right} ({lr_only_right/n:.3f})")
    print(f"  MHA only correct:     {mha_only_right} ({mha_only_right/n:.3f})")

    # LR vs MHA confusion matrix（把 MHA 当"ground truth"看 LR 的预测）
    print(f"\n  LR vs MHA confusion (row=LR, col=MHA):")
    cm = confusion_matrix(mha_pred, lr_pred)
    print(f"  {cm}")

    # 各自的性能
    print(f"\n  LR  Macro F1: {f1_score(label, lr_pred,  average='macro'):.4f}")
    print(f"  MHA Macro F1: {f1_score(label, mha_pred, average='macro'):.4f}")

    return {
        "n":              n,
        "agree_rate":     round(agree_rate, 4),
        "kappa":          round(kappa, 4),
        "agree_label0":   round(agree_when_0, 4),
        "agree_label1":   round(agree_when_1, 4),
        "both_correct":   int(both_correct),
        "both_wrong":     int(both_wrong),
        "lr_only_right":  int(lr_only_right),
        "mha_only_right": int(mha_only_right),
        "lr_macro_f1":    round(f1_score(label, lr_pred,  average='macro'), 4),
        "mha_macro_f1":   round(f1_score(label, mha_pred, average='macro'), 4),
    }


def analyze_disagreement_samples(test_df, label, lr_pred, mha_pred):
    """分析 LR 和 MHA 分歧最大的样本的特征分布"""
    disagree_idx = np.where(lr_pred != mha_pred)[0]
    agree_idx    = np.where(lr_pred == mha_pred)[0]

    print(f"\n{'='*60}")
    print(f"DISAGREEMENT ANALYSIS ({len(disagree_idx)} samples disagree)")
    print(f"{'='*60}")

    # 分歧样本的特征均值 vs 一致样本
    feat_cols = ALL_FEATURES
    disagree_df = test_df.iloc[disagree_idx][feat_cols]
    agree_df    = test_df.iloc[agree_idx][feat_cols]

    print(f"\n  Feature means — Agree vs Disagree:")
    print(f"  {'Feature':<25}  {'Agree':>8}  {'Disagree':>10}  {'Diff':>8}")
    print(f"  {'-'*55}")
    for col in feat_cols:
        a_mean = agree_df[col].mean()
        d_mean = disagree_df[col].mean()
        diff   = d_mean - a_mean
        print(f"  {col:<25}  {a_mean:>8.4f}  {d_mean:>10.4f}  {diff:>+8.4f}")

    # 分歧样本里，LR 对还是 MHA 对？
    dis_label    = label[disagree_idx]
    dis_lr_pred  = lr_pred[disagree_idx]
    dis_mha_pred = mha_pred[disagree_idx]
    lr_right  = (dis_lr_pred  == dis_label).sum()
    mha_right = (dis_mha_pred == dis_label).sum()
    print(f"\n  Among {len(disagree_idx)} disagreement samples:")
    print(f"  LR  correct: {lr_right}  ({lr_right/len(disagree_idx):.3f})")
    print(f"  MHA correct: {mha_right} ({mha_right/len(disagree_idx):.3f})")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    # 加载数据
    train_df  = pd.read_csv(KETOD_TRAIN)
    test_df   = pd.read_csv(KETOD_TEST)
    mha_df    = pd.read_csv(MHA_PRED_CSV)

    assert len(test_df) == len(mha_df), \
        f"行数不匹配: test_features={len(test_df)}, mha_predictions={len(mha_df)}"

    label    = test_df[LABEL_COL].values.astype(int)
    mha_pred = mha_df["mha_pred"].values.astype(int)

    # 训练 LR 并预测
    print("Training LR Full model...")
    lr_pred, lr_proba = train_lr_and_predict(train_df, test_df)

    # Agreement 分析
    summary = analyze_agreement(label, lr_pred, mha_pred)

    # 分歧样本分析
    analyze_disagreement_samples(test_df, label, lr_pred, mha_pred)

    # 保存逐样本结果
    result_df = test_df[ALL_FEATURES].copy()
    result_df["label"]    = label
    result_df["lr_pred"]  = lr_pred
    result_df["lr_prob1"] = np.round(lr_proba, 6)
    result_df["mha_pred"] = mha_pred
    result_df["mha_prob1"] = mha_df["mha_prob_1"].values
    result_df["agree"]    = (lr_pred == mha_pred).astype(int)
    result_df.to_csv("agreement_results.csv", index=False)
    print(f"\nSaved → agreement_results.csv")

    # 保存摘要
    with open("agreement_summary.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    print("Saved → agreement_summary.txt")

    # 最后打印论文用的一句话数字
    print(f"\n{'='*60}")
    print("PAPER-READY NUMBERS:")
    print(f"  LR-MHA agreement rate: {summary['agree_rate']:.4f}")
    print(f"  Cohen's Kappa:         {summary['kappa']:.4f}")
    print(f"  LR  Macro F1:          {summary['lr_macro_f1']:.4f}")
    print(f"  MHA Macro F1:          {summary['mha_macro_f1']:.4f}")
    print(f"  Both correct:          {summary['both_correct']}/{summary['n']} "
          f"({summary['both_correct']/summary['n']:.3f})")
    print(f"  Both wrong:            {summary['both_wrong']}/{summary['n']} "
          f"({summary['both_wrong']/summary['n']:.3f})")


if __name__ == "__main__":
    main()
