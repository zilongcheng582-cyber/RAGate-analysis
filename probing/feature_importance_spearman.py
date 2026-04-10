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
feature_importance_spearman.py  —  Step 7
Cross-dataset feature importance consistency via Spearman rho.

每个数据集训练 Full LR，提取系数（abs value）作为 feature importance，
计算数据集两两之间的 Spearman rho，判断 shortcut 是否跨数据集一致。

Usage:
    python feature_importance_spearman.py

输出：
    feature_importance.csv         — 各数据集的特征重要性（标准化系数）
    spearman_rho_results.csv       — 两两 Spearman rho + p-value
    feature_importance_plot.png    — 特征重要性对比图
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

from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# ─────────────────────────────────────────────
# 配置
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

ALL_FEATURES = [
    "turn_position_ratio",        # 1
    "prev_sys_is_question",       # 2
    "user_has_question",          # 3
    "user_starts_question_word",  # 4
    "user_turn_len_log",          # 5
    "sys_turn_len_log",           # 6
    "dialogue_len_log",           # 7
    "consecutive_sys_turns",      # 8
    "turn_len_ratio",             # 9
    "turn_position_squared",      # 10
]

# 简短显示名（图用）
FEATURE_LABELS = [
    "pos_ratio",       # 1
    "prev_sys_q",      # 2
    "user_q",          # 3
    "starts_qword",    # 4
    "user_len",        # 5
    "sys_len",         # 6
    "dial_len",        # 7
    "consec_sys",      # 8
    "len_ratio",       # 9
    "pos_squared",     # 10
]

C_GRID = [0.01, 0.1, 1.0, 10.0]
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# 训练 Full LR，提取系数
# ─────────────────────────────────────────────

def train_and_extract_coef(train_df):
    X = train_df[ALL_FEATURES].values.astype(float)
    y = train_df[LABEL_COL].values.astype(int)

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
    gs.fit(X, y)
    best_model = gs.best_estimator_
    best_model.fit(X, y)
    best_c = gs.best_params_["lr__C"]

    # 系数：shape (1, n_features) for binary LR
    # 正系数 → 预测 class 1（需要检索），负系数 → 预测 class 0
    coef = best_model.named_steps["lr"].coef_[0]  # raw signed coef（已在标准化特征上）
    abs_coef = np.abs(coef)

    return coef, abs_coef, best_c


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def run():
    coef_records = {}   # dataset -> signed coef array
    abs_records  = {}   # dataset -> abs coef array

    for ds_name, paths in DATASETS.items():
        print(f"\nDataset: {ds_name}")
        train_df = pd.read_csv(paths["train"])
        coef, abs_coef, best_c = train_and_extract_coef(train_df)
        coef_records[ds_name] = coef
        abs_records[ds_name]  = abs_coef
        print(f"  best_C = {best_c}")
        for fname, c, ac in zip(FEATURE_LABELS, coef, abs_coef):
            bar = "█" * int(ac * 20)
            sign = "+" if c >= 0 else "-"
            print(f"  {fname:<15s}  {sign}{ac:.4f}  {bar}")

    # ─────────────────────────────────────────
    # Feature importance 表
    # ─────────────────────────────────────────
    fi_df = pd.DataFrame(
        {ds: abs_records[ds] for ds in DATASETS},
        index=ALL_FEATURES
    )
    fi_df.index.name = "feature"

    # 加 rank 列（每个数据集内按重要性排名）
    for ds in DATASETS:
        fi_df[f"{ds}_rank"] = fi_df[ds].rank(ascending=False).astype(int)

    fi_df.to_csv("feature_importance.csv")
    print(f"\nSaved → feature_importance.csv")

    # ─────────────────────────────────────────
    # Spearman rho（两两）
    # ─────────────────────────────────────────
    ds_names = list(DATASETS.keys())
    rho_records = []

    print("\n" + "="*55)
    print("Spearman rho — cross-dataset feature importance")
    print("="*55)

    for i in range(len(ds_names)):
        for j in range(i + 1, len(ds_names)):
            ds_a, ds_b = ds_names[i], ds_names[j]
            rho, pval = spearmanr(abs_records[ds_a], abs_records[ds_b])
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
            print(f"  {ds_a} vs {ds_b}:  rho={rho:+.4f}  p={pval:.4f}  {sig}")
            rho_records.append({
                "pair":    f"{ds_a} vs {ds_b}",
                "rho":     round(rho, 4),
                "p_value": round(pval, 4),
                "sig":     sig,
            })

    rho_df = pd.DataFrame(rho_records)
    rho_df.to_csv("spearman_rho_results.csv", index=False)
    print(f"Saved → spearman_rho_results.csv")

    # ─────────────────────────────────────────
    # 打印特征重要性排名对比
    # ─────────────────────────────────────────
    print("\n" + "="*70)
    print("Feature Importance Ranking (by |coef| on standardized features)")
    print("="*70)
    print(f"{'Feature':<25}" + "".join(f"{ds:>12}" for ds in ds_names)
          + "".join(f"  {ds}_rank" for ds in ds_names))
    print("-"*70)
    for feat, label in zip(ALL_FEATURES, FEATURE_LABELS):
        row = f"{label:<25}"
        for ds in ds_names:
            row += f"{fi_df.loc[feat, ds]:>12.4f}"
        for ds in ds_names:
            row += f"  {fi_df.loc[feat, f'{ds}_rank']:>8}"
        print(row)

    # 找最不一致的特征（rank 方差最大）
    rank_cols = [f"{ds}_rank" for ds in ds_names]
    fi_df["rank_std"] = fi_df[rank_cols].std(axis=1)
    print("\nMost inconsistent features (largest rank std across datasets):")
    for feat in fi_df["rank_std"].sort_values(ascending=False).head(3).index:
        label = FEATURE_LABELS[ALL_FEATURES.index(feat)]
        ranks = [fi_df.loc[feat, f"{ds}_rank"] for ds in ds_names]
        print(f"  {label:<20s}  ranks={ranks}  std={fi_df.loc[feat,'rank_std']:.2f}")

    print("\nMost consistent features (smallest rank std):")
    for feat in fi_df["rank_std"].sort_values(ascending=True).head(3).index:
        label = FEATURE_LABELS[ALL_FEATURES.index(feat)]
        ranks = [fi_df.loc[feat, f"{ds}_rank"] for ds in ds_names]
        print(f"  {label:<20s}  ranks={ranks}  std={fi_df.loc[feat,'rank_std']:.2f}")

    # ─────────────────────────────────────────
    # 画图
    # ─────────────────────────────────────────
    plot_feature_importance(fi_df, abs_records, coef_records, ds_names)


def plot_feature_importance(fi_df, abs_records, coef_records, ds_names):
    colors = ["steelblue", "darkorange", "seagreen"]
    x = np.arange(len(ALL_FEATURES))
    width = 0.25

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    # 上图：abs coef（重要性）
    ax = axes[0]
    for i, (ds, color) in enumerate(zip(ds_names, colors)):
        vals = [abs_records[ds][j] for j in range(len(ALL_FEATURES))]
        ax.bar(x + i * width, vals, width, label=ds, color=color, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(FEATURE_LABELS, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("|Coefficient| (standardized)")
    ax.set_title("Feature Importance — |LR Coefficient| on Standardized Features")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 下图：signed coef（方向）
    ax2 = axes[1]
    for i, (ds, color) in enumerate(zip(ds_names, colors)):
        vals = [coef_records[ds][j] for j in range(len(ALL_FEATURES))]
        ax2.bar(x + i * width, vals, width, label=ds, color=color, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(FEATURE_LABELS, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("Coefficient (signed)")
    ax2.set_title("Feature Direction — Signed LR Coefficient (+ = toward retrieval)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("feature_importance_plot.png", dpi=150)
    plt.close()
    print("Saved → feature_importance_plot.png")


if __name__ == "__main__":
    run()
    print("\nDone.")
