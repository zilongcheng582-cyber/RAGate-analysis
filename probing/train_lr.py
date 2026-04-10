# ── CONFIG ────────────────────────────────────
DATASETS = {
    "KETOD": {
        "train": "data/ketod/train_features.csv",
        "test":  "data/ketod/test_features.csv",
    },
    "DSTC9": {
        "train": "data/dstc9/train/train_features.csv",
        "test":  "data/dstc9/val/test_features.csv",
    },
    "DSTC11": {
        "train": "data/dstc11/train_features.csv",
        "test":  "data/dstc11/test_features.csv",
    },
}
# ──────────────────────────────────────────────
"""
train_lr.py  —  Step 4: LR Feature Ablation
RAGate · EMNLP 2026

Usage:
    python train_lr.py

输出：
    lr_results.csv        — 完整数字表
    lr_results_table.txt  — 论文用格式
"""

import warnings
warnings.filterwarnings("ignore")

import os
# joblib 在路径含中文时 ASCII 编码出错，强制用纯 ASCII 的 temp 目录
os.makedirs("C:/Temp", exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = "C:/Temp"
import joblib
joblib.parallel.JOBLIB_TEMP_FOLDER = "C:/Temp"
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────
DATASETS = {
    "KETOD": {
        "train": "E:/ketod-main/ketod_release/train_features.csv",
        "test":  "E:/ketod-main/ketod_release/test_features.csv",
    },
    "DSTC9": {
        "train": "E:/dstc9-track1/data/train/train_features.csv",
        "test":  "E:/dstc9-track1/data/val/test_features.csv",
    },
    "DSTC11": {
        "train": "E:/dstc11-track5/train_features.csv",
        "test":  "E:/dstc11-track5/test_features.csv",
    },
}

LABEL_COL = "label"

# 10个特征（按handoff编号）
ALL_FEATURES = [
    "turn_position_ratio",       # 1
    "prev_sys_is_question",      # 2
    "user_has_question",         # 3
    "user_starts_question_word", # 4
    "user_turn_len_log",         # 5
    "sys_turn_len_log",          # 6
    "dialogue_len_log",          # 7
    "consecutive_sys_turns",     # 8
    "turn_len_ratio",            # 9
    "turn_position_squared",     # 10
]

# ─────────────────────────────────────────────
# 6种 ablation 设置
# ─────────────────────────────────────────────
ABLATIONS = {
    "Position only":   ["turn_position_ratio", "turn_position_squared"],
    "No position":     ["prev_sys_is_question", "user_has_question",
                        "user_starts_question_word", "user_turn_len_log",
                        "sys_turn_len_log", "dialogue_len_log",
                        "consecutive_sys_turns", "turn_len_ratio"],
    "Question only":   ["prev_sys_is_question", "user_has_question",
                        "user_starts_question_word"],
    "Length only":     ["user_turn_len_log", "sys_turn_len_log",
                        "dialogue_len_log", "consecutive_sys_turns",
                        "turn_len_ratio"],
    "Full (10)":       ALL_FEATURES,
    "L1 interaction":  None,  # 特殊处理：Full + PolynomialFeatures degree=2
}

C_GRID = [0.01, 0.1, 1.0, 10.0]
CV_FOLDS = 5
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def load_data(paths: dict):
    train = pd.read_csv(paths["train"])
    test  = pd.read_csv(paths["test"])
    return train, test


def get_xy(df: pd.DataFrame, features: list):
    X = df[features].values.astype(float)
    y = df[LABEL_COL].values.astype(int)
    return X, y


def build_pipeline(interaction=False, penalty="l2"):
    steps = [("scaler", StandardScaler())]
    if interaction:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    steps.append(("lr", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        penalty=penalty,
        solver="saga" if penalty == "l1" else "lbfgs",
    )))
    return Pipeline(steps)


def cv_best_c(X_train, y_train, interaction=False, penalty="l2"):
    """5折CV选最优C"""
    pipe = build_pipeline(interaction=interaction, penalty=penalty)
    param_grid = {"lr__C": C_GRID}
    gs = GridSearchCV(
        pipe, param_grid,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_macro",
        n_jobs=1,  # n_jobs=-1 在路径含中文时会触发 joblib UnicodeEncodeError
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_["lr__C"]


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "macro_f1":  round(f1_score(y_test, y_pred, average="macro"), 4),
        "precision": round(precision_score(y_test, y_pred, average="macro"), 4),
        "recall":    round(recall_score(y_test, y_pred, average="macro"), 4),
        "f1_pos":    round(f1_score(y_test, y_pred, pos_label=1), 4),
        "f1_neg":    round(f1_score(y_test, y_pred, pos_label=0), 4),
    }


# ─────────────────────────────────────────────
# 主循环
# ─────────────────────────────────────────────

def run_all():
    records = []

    for dataset_name, paths in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        train_df, test_df = load_data(paths)
        print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
        print(f"  Label dist (train): {dict(train_df[LABEL_COL].value_counts().sort_index())}")

        for ablation_name, features in ABLATIONS.items():
            is_interaction = (ablation_name == "L1 interaction")
            penalty = "l1" if is_interaction else "l2"
            feats = features if features is not None else ALL_FEATURES

            X_train, y_train = get_xy(train_df, feats)
            X_test,  y_test  = get_xy(test_df,  feats)

            best_model, best_c = cv_best_c(
                X_train, y_train,
                interaction=is_interaction,
                penalty=penalty,
            )
            best_model.fit(X_train, y_train)
            metrics = evaluate(best_model, X_test, y_test)

            row = {
                "dataset":    dataset_name,
                "ablation":   ablation_name,
                "n_features": len(feats) if not is_interaction else "10+poly",
                "best_C":     best_c,
                **metrics,
            }
            records.append(row)

            print(f"  [{ablation_name:20s}]  Macro F1={metrics['macro_f1']:.4f}"
                  f"  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}"
                  f"  best_C={best_c}")

    return pd.DataFrame(records)


def print_paper_table(df: pd.DataFrame):
    """输出论文格式的结果表"""
    print("\n" + "="*80)
    print("PAPER TABLE  (Macro F1 per dataset × ablation)")
    print("="*80)

    datasets = df["dataset"].unique()
    ablation_order = list(ABLATIONS.keys())

    header = f"{'Ablation':<22}" + "".join(f"{d:>12}" for d in datasets)
    print(header)
    print("-" * (22 + 12 * len(datasets)))

    for abl in ablation_order:
        row_str = f"{abl:<22}"
        for ds in datasets:
            val = df[(df["dataset"] == ds) & (df["ablation"] == abl)]["macro_f1"]
            row_str += f"{val.values[0]:>12.4f}" if len(val) else f"{'N/A':>12}"
        print(row_str)

    # 额外：No-position vs Full 的差值
    print("\nΔ (Full − No position):")
    for ds in datasets:
        full_f1  = df[(df["dataset"] == ds) & (df["ablation"] == "Full (10)")]["macro_f1"].values
        nopos_f1 = df[(df["dataset"] == ds) & (df["ablation"] == "No position")]["macro_f1"].values
        if len(full_f1) and len(nopos_f1):
            delta = full_f1[0] - nopos_f1[0]
            flag = "✅ >5pt" if delta > 0.05 else ("⚠️ 3-5pt" if delta > 0.03 else "❌ <3pt")
            print(f"  {ds}: Δ = {delta:+.4f}  {flag}")


# ─────────────────────────────────────────────

if __name__ == "__main__":
    results_df = run_all()

    # 保存
    out_csv = "lr_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # 打印论文表
    print_paper_table(results_df)

    # 每个数据集的详细分类报告（Full模型）
    print("\n" + "="*80)
    print("FULL MODEL — per-class breakdown")
    print("="*80)
    for dataset_name, paths in DATASETS.items():
        train_df, test_df = load_data(paths)
        X_train, y_train = get_xy(train_df, ALL_FEATURES)
        X_test,  y_test  = get_xy(test_df,  ALL_FEATURES)
        best_model, _ = cv_best_c(X_train, y_train)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        print(f"\n--- {dataset_name} ---")
        print(classification_report(y_test, y_pred, digits=4))

    print("\nDone.")
