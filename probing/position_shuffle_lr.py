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
position_shuffle_lr.py  —  Step 4 辅助实验
Shuffle turn_position_ratio（以及 turn_position_squared）在测试集上，
测量 Macro F1 下降幅度，证明 position 特征的因果重要性。

Usage:
    python position_shuffle_lr.py

输出：
    position_shuffle_results.csv
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.makedirs("C:/Temp", exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = "C:/Temp"
import joblib
joblib.parallel.JOBLIB_TEMP_FOLDER = "C:/Temp"

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score

# ─────────────────────────────────────────────
# 配置（与 train_lr.py 保持一致）
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

POSITION_FEATURES = ["turn_position_ratio", "turn_position_squared"]

C_GRID = [0.01, 0.1, 1.0, 10.0]
CV_FOLDS = 5
RANDOM_STATE = 42
N_SHUFFLE_REPEATS = 30  # shuffle重复次数，取均值±std


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )),
    ])


def cv_train(X_train, y_train):
    pipe = build_pipeline()
    gs = GridSearchCV(
        pipe, {"lr__C": C_GRID},
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_macro",
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_


def macro_f1(model, X, y):
    return f1_score(y, model.predict(X), average="macro")


def shuffle_features(X: np.ndarray, feature_names: list, cols_to_shuffle: list, rng) -> np.ndarray:
    """对指定列做 permutation，其余列不变"""
    X_shuffled = X.copy()
    for col in cols_to_shuffle:
        idx = feature_names.index(col)
        X_shuffled[:, idx] = rng.permutation(X_shuffled[:, idx])
    return X_shuffled


# ─────────────────────────────────────────────
# 主实验
# ─────────────────────────────────────────────

def run_shuffle_experiment():
    records = []
    rng = np.random.default_rng(RANDOM_STATE)

    for dataset_name, paths in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        train_df = pd.read_csv(paths["train"])
        test_df  = pd.read_csv(paths["test"])

        X_train = train_df[ALL_FEATURES].values.astype(float)
        y_train = train_df[LABEL_COL].values.astype(int)
        X_test  = test_df[ALL_FEATURES].values.astype(float)
        y_test  = test_df[LABEL_COL].values.astype(int)

        # 1. 原始Full模型（baseline）
        model = cv_train(X_train, y_train)
        model.fit(X_train, y_train)
        base_f1 = macro_f1(model, X_test, y_test)
        print(f"  Baseline Full F1: {base_f1:.4f}")

        # 2. Shuffle position features N次
        shuffled_f1s = []
        for i in range(N_SHUFFLE_REPEATS):
            X_test_shuf = shuffle_features(X_test, ALL_FEATURES, POSITION_FEATURES, rng)
            f1 = macro_f1(model, X_test_shuf, y_test)
            shuffled_f1s.append(f1)

        shuffled_mean = np.mean(shuffled_f1s)
        shuffled_std  = np.std(shuffled_f1s)
        delta         = base_f1 - shuffled_mean

        print(f"  Shuffled F1: {shuffled_mean:.4f} ± {shuffled_std:.4f}")
        print(f"  Δ (baseline − shuffled): {delta:+.4f}")

        # 3. 额外：shuffle 每个 position feature 单独
        for feat in POSITION_FEATURES:
            single_f1s = []
            for _ in range(N_SHUFFLE_REPEATS):
                X_shuf = shuffle_features(X_test, ALL_FEATURES, [feat], rng)
                single_f1s.append(macro_f1(model, X_shuf, y_test))
            smean = np.mean(single_f1s)
            print(f"    Shuffle {feat}: F1={smean:.4f}  Δ={base_f1-smean:+.4f}")

            records.append({
                "dataset":        dataset_name,
                "shuffled_cols":  feat,
                "baseline_f1":    round(base_f1, 4),
                "shuffled_mean":  round(smean, 4),
                "shuffled_std":   round(np.std(single_f1s), 4),
                "delta":          round(base_f1 - smean, 4),
            })

        # 4. shuffle 全部 position（主要结果）
        records.append({
            "dataset":        dataset_name,
            "shuffled_cols":  "all_position",
            "baseline_f1":    round(base_f1, 4),
            "shuffled_mean":  round(shuffled_mean, 4),
            "shuffled_std":   round(shuffled_std, 4),
            "delta":          round(delta, 4),
        })

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame):
    print("\n" + "="*70)
    print("SHUFFLE SUMMARY — Δ when position features are permuted")
    print("="*70)
    print(f"{'Dataset':<10}  {'Shuffled':<25}  {'Baseline':>10}  {'Shuffled':>10}  {'Δ':>8}")
    print("-"*70)
    for _, row in df.iterrows():
        flag = "✅" if row["delta"] > 0.05 else ("⚠️" if row["delta"] > 0.02 else "❌")
        print(f"{row['dataset']:<10}  {row['shuffled_cols']:<25}  "
              f"{row['baseline_f1']:>10.4f}  {row['shuffled_mean']:>10.4f}  "
              f"{row['delta']:>+8.4f}  {flag}")


# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = run_shuffle_experiment()

    out = "position_shuffle_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")

    print_summary(df)
    print("\nDone.")
