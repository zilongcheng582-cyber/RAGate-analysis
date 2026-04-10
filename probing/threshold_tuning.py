# ── CONFIG ────────────────────────────────────
KETOD_TRAIN = "data/ketod/train_features.csv"
KETOD_TEST  = "data/ketod/test_features.csv"
# ──────────────────────────────────────────────
"""
threshold_tuning.py  —  Step 4 补充实验
针对 KETOD 严重类别不平衡（7.4:1）做 threshold 优化分析。
在训练集 hold-out（20%）上搜索最优 threshold，在测试集上验证。

Usage:
    python threshold_tuning.py

输出：
    threshold_tuning_results.csv   — 各 ablation 的 optimal threshold + 对应指标
    threshold_curve_ketod.png      — PR 曲线 + F1-threshold 曲线（Full 模型）
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.makedirs("C:/Temp", exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = "C:/Temp"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    precision_recall_curve, classification_report
)

# ─────────────────────────────────────────────
# 配置（只跑 KETOD，其余数据集不需要）
# ─────────────────────────────────────────────
KETOD_TRAIN = "E:/ketod-main/ketod_release/train_features.csv"
KETOD_TEST  = "E:/ketod-main/ketod_release/test_features.csv"

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

ABLATIONS = {
    "Position only": ["turn_position_ratio", "turn_position_squared"],
    "No position":   ["prev_sys_is_question", "user_has_question",
                      "user_starts_question_word", "user_turn_len_log",
                      "sys_turn_len_log", "dialogue_len_log",
                      "consecutive_sys_turns", "turn_len_ratio"],
    "Question only": ["prev_sys_is_question", "user_has_question",
                      "user_starts_question_word"],
    "Length only":   ["user_turn_len_log", "sys_turn_len_log",
                      "dialogue_len_log", "consecutive_sys_turns",
                      "turn_len_ratio"],
    "Full (10)":     ALL_FEATURES,
}

C_GRID = [0.01, 0.1, 1.0, 10.0]
RANDOM_STATE = 42
HOLDOUT_RATIO = 0.2   # 从训练集切出 20% 用于 threshold 搜索
THRESHOLD_STEP = 0.01


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def build_and_train(X_train, y_train, features):
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
    return best, gs.best_params_["lr__C"]


def find_best_threshold(model, X_val, y_val):
    """在 val 上遍历 threshold，最大化 Macro F1"""
    proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.05, 0.95, THRESHOLD_STEP)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        # 避免全预测同一类时 F1 计算出问题
        if len(np.unique(y_pred)) < 2:
            continue
        f1 = f1_score(y_val, y_pred, average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def evaluate_at_threshold(model, X_test, y_test, threshold):
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    return {
        "macro_f1":  round(f1_score(y_test, y_pred, average="macro"), 4),
        "precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, average="macro"), 4),
        "f1_pos":    round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "f1_neg":    round(f1_score(y_test, y_pred, pos_label=0), 4),
    }


# ─────────────────────────────────────────────
# 画图：Full 模型的 F1-threshold 曲线
# ─────────────────────────────────────────────

def plot_threshold_curve(model, X_val, y_val, X_test, y_test, best_t, out_path):
    proba_val  = model.predict_proba(X_val)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.05, 0.95, THRESHOLD_STEP)

    f1_vals, f1_tests = [], []
    for t in thresholds:
        yv = (proba_val  >= t).astype(int)
        yt = (proba_test >= t).astype(int)
        f1_vals.append(f1_score(y_val,  yv, average="macro") if len(np.unique(yv)) > 1 else 0)
        f1_tests.append(f1_score(y_test, yt, average="macro") if len(np.unique(yt)) > 1 else 0)

    # PR curve（val）
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_val, proba_val)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左：F1 vs threshold
    ax = axes[0]
    ax.plot(thresholds, f1_vals,  label="Val Macro F1",  color="steelblue")
    ax.plot(thresholds, f1_tests, label="Test Macro F1", color="darkorange", linestyle="--")
    ax.axvline(best_t, color="red", linestyle=":", label=f"Best t={best_t:.2f}")
    ax.axvline(0.5,    color="gray", linestyle=":", alpha=0.5, label="Default t=0.50")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Macro F1")
    ax.set_title("KETOD Full Model — Macro F1 vs Threshold")
    ax.legend()
    ax.grid(alpha=0.3)

    # 右：PR curve
    ax2 = axes[1]
    ax2.plot(recall_vals, precision_vals, color="steelblue")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("KETOD Full Model — Precision-Recall Curve (Val)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  图表已保存 → {out_path}")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def run():
    train_df = pd.read_csv(KETOD_TRAIN)
    test_df  = pd.read_csv(KETOD_TEST)

    print(f"KETOD  Train: {len(train_df)}  Test: {len(test_df)}")
    print(f"Label dist (train): {dict(train_df[LABEL_COL].value_counts().sort_index())}")

    # 从训练集切出 hold-out，用于 threshold 搜索
    train_sub, val_df = train_test_split(
        train_df, test_size=HOLDOUT_RATIO,
        stratify=train_df[LABEL_COL], random_state=RANDOM_STATE
    )
    print(f"Train sub: {len(train_sub)}  Val (threshold search): {len(val_df)}\n")

    y_test = test_df[LABEL_COL].values.astype(int)
    y_val  = val_df[LABEL_COL].values.astype(int)

    records = []

    for abl_name, features in ABLATIONS.items():
        X_train = train_sub[features].values.astype(float)
        y_train = train_sub[LABEL_COL].values.astype(int)
        X_val   = val_df[features].values.astype(float)
        X_test  = test_df[features].values.astype(float)

        model, best_c = build_and_train(X_train, y_train, features)

        # default threshold = 0.5
        metrics_default = evaluate_at_threshold(model, X_test, y_test, 0.5)

        # optimal threshold（在 val 上搜索）
        best_t, val_f1 = find_best_threshold(model, X_val, y_val)
        metrics_tuned  = evaluate_at_threshold(model, X_test, y_test, best_t)

        gain = metrics_tuned["macro_f1"] - metrics_default["macro_f1"]

        print(f"[{abl_name:20s}]  best_C={best_c}")
        print(f"  Default (t=0.50):  Macro F1={metrics_default['macro_f1']:.4f}  "
              f"P={metrics_default['precision']:.4f}  R={metrics_default['recall']:.4f}")
        print(f"  Tuned   (t={best_t:.2f}):  Macro F1={metrics_tuned['macro_f1']:.4f}  "
              f"P={metrics_tuned['precision']:.4f}  R={metrics_tuned['recall']:.4f}  "
              f"Gain={gain:+.4f}")
        print()

        records.append({
            "ablation":          abl_name,
            "best_C":            best_c,
            "default_macro_f1":  metrics_default["macro_f1"],
            "default_f1_pos":    metrics_default["f1_pos"],
            "default_f1_neg":    metrics_default["f1_neg"],
            "optimal_threshold": best_t,
            "tuned_macro_f1":    metrics_tuned["macro_f1"],
            "tuned_f1_pos":      metrics_tuned["f1_pos"],
            "tuned_f1_neg":      metrics_tuned["f1_neg"],
            "gain":              round(gain, 4),
        })

        # 画图（仅 Full 模型）
        if abl_name == "Full (10)":
            plot_threshold_curve(
                model, X_val, y_val, X_test, y_test,
                best_t, "threshold_curve_ketod.png"
            )
            # 详细 classification report
            proba = model.predict_proba(X_test)[:, 1]
            y_pred_tuned = (proba >= best_t).astype(int)
            print(f"--- Full model @ t={best_t:.2f} (test) ---")
            print(classification_report(y_test, y_pred_tuned, digits=4))

    df = pd.DataFrame(records)
    df.to_csv("threshold_tuning_results.csv", index=False)
    print("Saved → threshold_tuning_results.csv")

    # 汇总表
    print("\n" + "="*65)
    print("SUMMARY — Default vs Tuned Threshold (KETOD)")
    print("="*65)
    print(f"{'Ablation':<22}  {'Default F1':>10}  {'Tuned F1':>10}  {'Gain':>8}  {'Opt t':>6}")
    print("-"*65)
    for _, r in df.iterrows():
        print(f"{r['ablation']:<22}  {r['default_macro_f1']:>10.4f}  "
              f"{r['tuned_macro_f1']:>10.4f}  {r['gain']:>+8.4f}  {r['optimal_threshold']:>6.2f}")


if __name__ == "__main__":
    run()
    print("\nDone.")
