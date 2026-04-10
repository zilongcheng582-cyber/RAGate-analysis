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
cross_dataset_transfer.py  —  补充实验
Cross-dataset transfer: train on dataset A, test on dataset B.
证明 shortcut mismatch 导致真实的 generalization failure。

Usage:
    python cross_dataset_transfer.py

输出：
    transfer_results.csv       — 完整 transfer 矩阵
    transfer_matrix.png        — 热力图（论文 Figure 用）
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
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, classification_report

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
# 工具函数
# ─────────────────────────────────────────────

def train_model(train_df):
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
    best = gs.best_estimator_
    best.fit(X, y)
    return best, gs.best_params_["lr__C"]


def evaluate(model, test_df):
    X = test_df[ALL_FEATURES].values.astype(float)
    y = test_df[LABEL_COL].values.astype(int)
    y_pred = model.predict(X)
    return {
        "macro_f1":  round(f1_score(y, y_pred, average="macro"), 4),
        "f1_pos":    round(f1_score(y, y_pred, pos_label=1, zero_division=0), 4),
        "f1_neg":    round(f1_score(y, y_pred, pos_label=0), 4),
        "report":    classification_report(y, y_pred, digits=4),
    }


# ─────────────────────────────────────────────
# 主流程：所有 train→test 组合
# ─────────────────────────────────────────────

def run():
    ds_names = list(DATASETS.keys())

    # 先训练每个数据集的模型
    models = {}
    train_dfs = {}
    test_dfs  = {}

    for ds in ds_names:
        print(f"Training on {ds}...")
        train_df = pd.read_csv(DATASETS[ds]["train"])
        test_df  = pd.read_csv(DATASETS[ds]["test"])
        train_dfs[ds] = train_df
        test_dfs[ds]  = test_df
        model, best_c = train_model(train_df)
        models[ds] = model
        label_dist = dict(train_df[LABEL_COL].value_counts().sort_index())
        print(f"  best_C={best_c}  label_dist={label_dist}")

    # 构建 transfer 矩阵
    records = []
    f1_matrix = pd.DataFrame(index=ds_names, columns=ds_names, dtype=float)

    print("\n" + "="*65)
    print("TRANSFER MATRIX — Macro F1 (row=train, col=test)")
    print("="*65)
    print(f"{'Train \\ Test':<12}" + "".join(f"{ds:>10}" for ds in ds_names))
    print("-"*45)

    for train_ds in ds_names:
        row_str = f"{train_ds:<12}"
        for test_ds in ds_names:
            metrics = evaluate(models[train_ds], test_dfs[test_ds])
            f1 = metrics["macro_f1"]
            f1_matrix.loc[train_ds, test_ds] = f1

            is_in_domain = (train_ds == test_ds)
            marker = " [in]" if is_in_domain else ""
            row_str += f"{f1:>10.4f}"

            records.append({
                "train_on":     train_ds,
                "test_on":      test_ds,
                "in_domain":    is_in_domain,
                "macro_f1":     f1,
                "f1_pos":       metrics["f1_pos"],
                "f1_neg":       metrics["f1_neg"],
            })
        print(row_str)

    # 计算 transfer gap（in-domain - cross-domain）
    print("\nTransfer Gap (in-domain F1 − cross-domain F1):")
    print("-"*55)
    for train_ds in ds_names:
        in_f1 = f1_matrix.loc[train_ds, train_ds]
        for test_ds in ds_names:
            if test_ds == train_ds:
                continue
            cross_f1 = f1_matrix.loc[train_ds, test_ds]
            gap = in_f1 - cross_f1
            severity = "🔴 severe" if gap > 0.2 else ("🟡 moderate" if gap > 0.1 else "🟢 mild")
            print(f"  {train_ds}→{test_ds}: in={in_f1:.4f}  cross={cross_f1:.4f}  "
                  f"gap={gap:+.4f}  {severity}")

    # 详细 classification report（只看跨数据集的）
    print("\n" + "="*65)
    print("Per-class breakdown (cross-domain only)")
    print("="*65)
    for train_ds in ds_names:
        for test_ds in ds_names:
            if train_ds == test_ds:
                continue
            print(f"\n--- Train: {train_ds} → Test: {test_ds} ---")
            metrics = evaluate(models[train_ds], test_dfs[test_ds])
            print(metrics["report"])

    # 保存结果
    df = pd.DataFrame(records)
    df.to_csv("transfer_results.csv", index=False)
    print("Saved → transfer_results.csv")

    # 画热力图
    plot_heatmap(f1_matrix, ds_names)

    return df, f1_matrix


def plot_heatmap(f1_matrix, ds_names):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6, 5))

    # 标注 in-domain 格子
    annot = f1_matrix.copy().astype(str)
    for ds in ds_names:
        val = f1_matrix.loc[ds, ds]
        annot.loc[ds, ds] = f"{val:.4f}\n(in-domain)"

    mask_vals = f1_matrix.values.astype(float)

    sns.heatmap(
        mask_vals,
        annot=f1_matrix.values.astype(float),
        fmt=".4f",
        cmap="RdYlGn",
        vmin=0.3, vmax=0.9,
        xticklabels=ds_names,
        yticklabels=ds_names,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Macro F1"},
    )

    # 加粗对角线边框（in-domain）
    for i in range(len(ds_names)):
        ax.add_patch(plt.Rectangle(
            (i, i), 1, 1,
            fill=False, edgecolor="black", lw=2.5
        ))

    ax.set_xlabel("Test Dataset", fontsize=11)
    ax.set_ylabel("Train Dataset", fontsize=11)
    ax.set_title("Cross-Dataset Transfer — Macro F1\n(bold = in-domain)", fontsize=11)
    plt.tight_layout()
    plt.savefig("transfer_matrix.png", dpi=150)
    plt.close()
    print("Saved → transfer_matrix.png")


if __name__ == "__main__":
    try:
        import seaborn as sns
    except ImportError:
        print("seaborn not found, installing...")
        os.system("pip install seaborn -q")
        import seaborn as sns

    df, matrix = run()
    print("\nDone.")
