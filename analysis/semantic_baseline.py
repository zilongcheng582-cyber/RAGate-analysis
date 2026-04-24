"""
semantic_baseline.py — Semantic Baseline 实验
用 sentence-transformers 提取 user turn embedding，
训练 LR，跑和结构特征完全一样的 cross-dataset transfer。

目的：回应 reviewer "LR capacity 太弱" 的批评。
无论结果如何都能用：
  - semantic 也崩 → annotation incompatibility 连语义都救不了
  - semantic 略好 → structural shortcut 特别脆弱，但问题仍是结构性的

依赖：
    pip install sentence-transformers

Usage:
    python semantic_baseline.py

输出：
    semantic_results.csv     — in-domain + cross-dataset transfer 结果
    semantic_summary.txt     — 论文用数字
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
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────
# 注意：需要 train_full.csv / test_full.csv（含原始文本的文件）
# 不是 train_features.csv（那个是结构特征，没有文本）
DATASETS = {
    "KETOD": {
        "train": "E:/ketod-main/ketod_release/train_full.csv",
        "test":  "E:/ketod-main/ketod_release/test_full.csv",
    },
    "DSTC9": {
        "train": "E:/dstc9-track1/data/train/train_dstc9.csv",
        "test":  "E:/dstc9-track1/data/val/test_dstc9.csv",
    },
    "DSTC11": {
        "train": "E:/dstc11-track5/train.csv",
        "test":  "E:/dstc11-track5/val.csv",
    },
}

# 每个数据集的 label 列名和 text 列名（根据实际文件调整）
# train_full.csv 的列名：output（label）, input（文本）
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

MODEL_NAME = "all-MiniLM-L6-v2"   # 轻量，本地跑，384维
BATCH_SIZE = 256
C_GRID = [0.01, 0.1, 1.0, 10.0]
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def load_and_label(path, label_col, text_col):
    df = pd.read_csv(path)
    # label：True/False 或 1/0，统一转成 int
    if df[label_col].dtype == object:
        df["label_int"] = (df[label_col].str.strip() == "True").astype(int)
    else:
        df["label_int"] = df[label_col].astype(int)
    return df[text_col].fillna("").tolist(), df["label_int"].values


def encode(texts, model, batch_size=256):
    print(f"  Encoding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings


def train_lr(X_train, y_train):
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


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "macro_f1":    round(f1_score(y_test, y_pred, average="macro"), 4),
        "minority_f1": round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "report":      classification_report(y_test, y_pred, digits=4),
    }


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    print(f"Loading sentence-transformers model: {MODEL_NAME}")
    st_model = SentenceTransformer(MODEL_NAME)

    ds_names = list(DATASETS.keys())

    # Step 1: 编码所有数据集的 train + test
    embeddings = {}
    labels = {}

    for ds in ds_names:
        print(f"\n{'='*50}")
        print(f"Dataset: {ds}")
        for split in ["train", "test"]:
            path = DATASETS[ds][split]
            label_col = LABEL_COLS[ds]
            text_col  = TEXT_COLS[ds]
            texts, y = load_and_label(path, label_col, text_col)
            emb = encode(texts, st_model, BATCH_SIZE)
            embeddings[f"{ds}_{split}"] = emb
            labels[f"{ds}_{split}"] = y
            print(f"  {split}: {len(texts)} samples, shape {emb.shape}, "
                  f"pos={y.sum()}, neg={(y==0).sum()}")

    # Step 2: 训练每个数据集的 LR，跑全矩阵 transfer
    records = []
    models = {}

    print(f"\n{'='*50}")
    print("Training LR models...")

    for train_ds in ds_names:
        X_train = embeddings[f"{train_ds}_train"]
        y_train = labels[f"{train_ds}_train"]
        model, best_c = train_lr(X_train, y_train)
        models[train_ds] = model
        print(f"  {train_ds}: best_C={best_c}")

    # Step 3: 全矩阵 transfer 评估
    print(f"\n{'='*50}")
    print("SEMANTIC BASELINE — Transfer Matrix (Macro F1)")
    print(f"{'Train→Test':<12}" + "".join(f"{ds:>10}" for ds in ds_names))
    print("-" * (12 + 10 * len(ds_names)))

    f1_matrix = {}
    for train_ds in ds_names:
        row_str = f"{train_ds:<12}"
        for test_ds in ds_names:
            X_test = embeddings[f"{test_ds}_test"]
            y_test = labels[f"{test_ds}_test"]
            metrics = evaluate(models[train_ds], X_test, y_test)
            f1 = metrics["macro_f1"]
            minority = metrics["minority_f1"]
            row_str += f"{f1:>10.4f}"
            is_in = (train_ds == test_ds)
            records.append({
                "model":       "Semantic-LR",
                "train_on":    train_ds,
                "test_on":     test_ds,
                "in_domain":   is_in,
                "macro_f1":    f1,
                "minority_f1": minority,
            })
            f1_matrix[f"{train_ds}→{test_ds}"] = (f1, minority)
        print(row_str)

    # 保存
    df = pd.DataFrame(records)
    df.to_csv("semantic_results.csv", index=False)
    print(f"\nSaved → semantic_results.csv")

    # 论文用关键数字
    print(f"\n{'='*50}")
    print("PAPER-READY: DSTC→KETOD direction (most important)")
    for key in ["DSTC9→KETOD", "DSTC11→KETOD"]:
        sem_macro, sem_min = f1_matrix[key]
        print(f"  {key}:")
        print(f"    Semantic LR: Macro={sem_macro:.4f}, Minority={sem_min:.4f}")

    with open("semantic_summary.txt", "w") as f:
        f.write(df.to_string())
    print("Saved → semantic_summary.txt")


if __name__ == "__main__":
    main()
