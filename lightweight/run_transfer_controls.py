from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
C_GRID = [0.01, 0.1, 1.0, 10.0]
LABEL_COL = "label"

FULL_FEATURES = [
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

QUESTION_FEATURES = [
    "prev_sys_is_question",
    "user_has_question",
    "user_starts_question_word",
]

NO_QUESTION_FEATURES = [f for f in FULL_FEATURES if f not in QUESTION_FEATURES]
DATASET_ORDER = ["KETOD", "DSTC9", "DSTC11"]

PUBLISHED_FULL = {
    ("KETOD", "KETOD"): {"macro_f1": 0.5419, "f1_pos": 0.3267, "roc_auc": 0.7194, "average_precision": 0.2700},
    ("KETOD", "DSTC9"): {"macro_f1": 0.4199, "f1_pos": 0.2122, "roc_auc": 0.4628, "average_precision": 0.2356},
    ("KETOD", "DSTC11"): {"macro_f1": 0.4273, "f1_pos": 0.4241, "roc_auc": 0.4119, "average_precision": 0.4399},
    ("DSTC9", "KETOD"): {"macro_f1": 0.5299, "f1_pos": 0.2239, "roc_auc": 0.4812, "average_precision": 0.1347},
    ("DSTC9", "DSTC9"): {"macro_f1": 0.7982, "f1_pos": 0.7398, "roc_auc": 0.9077, "average_precision": 0.7126},
    ("DSTC9", "DSTC11"): {"macro_f1": 0.8359, "f1_pos": 0.8564, "roc_auc": 0.9003, "average_precision": 0.8728},
    ("DSTC11", "KETOD"): {"macro_f1": 0.5264, "f1_pos": 0.2089, "roc_auc": 0.4741, "average_precision": 0.1341},
    ("DSTC11", "DSTC9"): {"macro_f1": 0.8097, "f1_pos": 0.7422, "roc_auc": 0.9064, "average_precision": 0.7039},
    ("DSTC11", "DSTC11"): {"macro_f1": 0.8422, "f1_pos": 0.8523, "roc_auc": 0.9041, "average_precision": 0.8696},
}


@dataclass
class DatasetFrames:
    train: pd.DataFrame
    test: pd.DataFrame
    train_path: Path
    test_path: Path


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.ParserError:
        logging.warning("Standard CSV parser failed for %s; retrying with engine='python'.", path)
        return pd.read_csv(path, engine="python", on_bad_lines="error")


def normalize_and_validate(df: pd.DataFrame, dataset: str, split: str) -> pd.DataFrame:
    required = [LABEL_COL] + FULL_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset} {split} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    bad_counts = out[required].isna().sum()
    bad_counts = bad_counts[bad_counts > 0]
    if not bad_counts.empty:
        raise ValueError(f"{dataset} {split} contains non-numeric/NaN values:\n{bad_counts.to_string()}")

    values = set(out[LABEL_COL].astype(int).unique().tolist())
    if not values.issubset({0, 1}) or len(values) < 2:
        raise ValueError(f"{dataset} {split} label values must contain both 0 and 1; found {sorted(values)}")
    out[LABEL_COL] = out[LABEL_COL].astype(int)

    matrix = out[FULL_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{dataset} {split} contains inf/-inf in feature columns.")
    return out


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    missing = [name for name in DATASET_ORDER if name not in config]
    if missing:
        raise ValueError(f"Config is missing datasets: {missing}")
    return config


def audit_and_load(config: dict[str, Any], output_dir: Path) -> dict[str, DatasetFrames]:
    records: list[dict[str, Any]] = []
    datasets: dict[str, DatasetFrames] = {}

    for name in DATASET_ORDER:
        entry = config[name]
        for key in ["train_features", "test_features", "train_full", "test_full"]:
            path = Path(entry[key])
            records.append({
                "dataset": name,
                "file_role": key,
                "path": str(path),
                "exists": path.exists(),
                "required_for_current_experiments": key in {"train_features", "test_features"},
                "size_bytes": path.stat().st_size if path.exists() else np.nan,
            })

        train_path = Path(entry["train_features"])
        test_path = Path(entry["test_features"])
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Missing required feature CSV for {name}.\n"
                f"train={train_path} (exists={train_path.exists()})\n"
                f"test={test_path} (exists={test_path.exists()})"
            )

        logging.info("Loading %s train features: %s", name, train_path)
        train = normalize_and_validate(read_csv_robust(train_path), name, "train")
        logging.info("Loading %s test features: %s", name, test_path)
        test = normalize_and_validate(read_csv_robust(test_path), name, "test")
        datasets[name] = DatasetFrames(train=train, test=test, train_path=train_path, test_path=test_path)

        for split_name, frame in [("train", train), ("test", test)]:
            counts = frame[LABEL_COL].value_counts().to_dict()
            records.append({
                "dataset": name,
                "file_role": f"{split_name}_loaded_summary",
                "path": str(train_path if split_name == "train" else test_path),
                "exists": True,
                "required_for_current_experiments": True,
                "size_bytes": (train_path if split_name == "train" else test_path).stat().st_size,
                "n_rows": len(frame),
                "n_negative": int(counts.get(0, 0)),
                "n_positive": int(counts.get(1, 0)),
                "positive_rate": float(frame[LABEL_COL].mean()),
            })

    pd.DataFrame(records).to_csv(output_dir / "input_audit.csv", index=False, encoding="utf-8-sig")
    return datasets


def cv_splits(y: np.ndarray, requested: int = 3) -> int:
    min_count = int(np.bincount(y, minlength=2).min())
    if min_count < 2:
        raise ValueError("At least two examples per class are required.")
    return min(requested, min_count)


def make_pipeline(c_value: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )),
    ])


def select_best_c(train_df: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    x = train_df[features].to_numpy(dtype=float)
    y = train_df[LABEL_COL].to_numpy(dtype=int)
    cv = StratifiedKFold(n_splits=cv_splits(y), shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=make_pipeline(1.0),
        param_grid={"lr__C": C_GRID},
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    grid.fit(x, y)
    return float(grid.best_params_["lr__C"]), float(grid.best_score_)


def train_source_model(train_df: pd.DataFrame, features: list[str]) -> tuple[Pipeline, float, float]:
    best_c, cv_score = select_best_c(train_df, features)
    model = make_pipeline(best_c)
    model.fit(train_df[features].to_numpy(dtype=float), train_df[LABEL_COL].to_numpy(dtype=int))
    return model, best_c, cv_score


def safe_auc(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    try:
        roc = float(roc_auc_score(y, prob))
    except ValueError:
        roc = math.nan
    try:
        ap = float(average_precision_score(y, prob))
    except ValueError:
        ap = math.nan
    return roc, ap


def metric_dict(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    roc, ap = safe_auc(y, prob)
    return {
        "threshold": float(threshold),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_pos": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "f1_neg": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "roc_auc": roc,
        "average_precision": ap,
        "predicted_positive_rate": float(pred.mean()),
    }


def best_macro_f1_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, dict[str, float]]:
    """Find the exact score boundary maximizing macro-F1.

    Predictions use prob >= threshold. Ties are resolved by choosing the
    threshold closest to 0.5, which avoids arbitrary extreme thresholds.
    """
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    if len(y) != len(prob) or len(y) == 0:
        raise ValueError("Invalid y/prob arrays for threshold selection.")
    if not np.isfinite(prob).all():
        raise ValueError("Probabilities contain NaN or infinity.")

    order = np.argsort(-prob, kind="mergesort")
    sorted_prob = prob[order]
    sorted_y = y[order]
    total_pos = int(sorted_y.sum())
    total_neg = int(len(sorted_y) - total_pos)

    candidates: list[tuple[float, float, float, float]] = []

    def add_candidate(threshold: float, tp: int, fp: int) -> None:
        fn = total_pos - tp
        tn = total_neg - fp
        denom_pos = 2 * tp + fp + fn
        denom_neg = 2 * tn + fp + fn
        f1_pos = 0.0 if denom_pos == 0 else 2 * tp / denom_pos
        f1_neg = 0.0 if denom_neg == 0 else 2 * tn / denom_neg
        macro = 0.5 * (f1_pos + f1_neg)
        candidates.append((macro, threshold, f1_pos, f1_neg))

    add_candidate(float(np.nextafter(sorted_prob[0], np.inf)), tp=0, fp=0)

    tp = 0
    fp = 0
    i = 0
    n = len(sorted_y)
    while i < n:
        score = sorted_prob[i]
        j = i
        group_pos = 0
        group_total = 0
        while j < n and sorted_prob[j] == score:
            group_pos += int(sorted_y[j])
            group_total += 1
            j += 1
        tp += group_pos
        fp += group_total - group_pos
        add_candidate(float(score), tp=tp, fp=fp)
        i = j

    candidates.sort(key=lambda item: (-item[0], abs(item[1] - 0.5), item[1]))
    best_macro, best_threshold, _, _ = candidates[0]
    metrics = metric_dict(y, prob, best_threshold)
    assert abs(metrics["macro_f1"] - best_macro) < 1e-10
    return best_threshold, metrics


def run_no_question_experiment(datasets: dict[str, DatasetFrames], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("=== Experiment 1: full vs no-question cross-dataset transfer ===")
    feature_sets = {"full_10": FULL_FEATURES, "no_question_7": NO_QUESTION_FEATURES}
    rows: list[dict[str, Any]] = []

    for fs_name, features in feature_sets.items():
        for source in DATASET_ORDER:
            logging.info("Training %s model on %s", fs_name, source)
            model, best_c, cv_score = train_source_model(datasets[source].train, features)
            for target in DATASET_ORDER:
                target_df = datasets[target].test
                x = target_df[features].to_numpy(dtype=float)
                y = target_df[LABEL_COL].to_numpy(dtype=int)
                prob = model.predict_proba(x)[:, 1]
                metrics = metric_dict(y, prob, 0.5)
                rows.append({
                    "feature_set": fs_name,
                    "n_features": len(features),
                    "features": ";".join(features),
                    "train_on": source,
                    "test_on": target,
                    "in_domain": source == target,
                    "best_C": best_c,
                    "source_cv_macro_f1": cv_score,
                    **metrics,
                })
                logging.info(
                    "%s %s -> %s | Macro=%.4f Pos=%.4f ROC=%.4f AP=%.4f",
                    fs_name, source, target, metrics["macro_f1"], metrics["f1_pos"], metrics["roc_auc"], metrics["average_precision"]
                )

    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "no_question_transfer_all.csv", index=False, encoding="utf-8-sig")

    full = raw[raw["feature_set"] == "full_10"].copy()
    noq = raw[raw["feature_set"] == "no_question_7"].copy()
    metric_cols = ["macro_f1", "f1_pos", "f1_neg", "roc_auc", "average_precision", "predicted_positive_rate"]
    full = full[["train_on", "test_on", "best_C"] + metric_cols].rename(
        columns={"best_C": "full_best_C", **{c: f"full_{c}" for c in metric_cols}}
    )
    noq = noq[["train_on", "test_on", "best_C"] + metric_cols].rename(
        columns={"best_C": "noq_best_C", **{c: f"noq_{c}" for c in metric_cols}}
    )
    comparison = full.merge(noq, on=["train_on", "test_on"], how="inner")
    for c in metric_cols:
        comparison[f"delta_noq_minus_full_{c}"] = comparison[f"noq_{c}"] - comparison[f"full_{c}"]
    comparison.to_csv(output_dir / "no_question_comparison.csv", index=False, encoding="utf-8-sig")
    return raw, comparison


def run_baseline_verification(noq_raw: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    full = noq_raw[noq_raw["feature_set"] == "full_10"]
    for _, row in full.iterrows():
        key = (str(row["train_on"]), str(row["test_on"]))
        expected = PUBLISHED_FULL[key]
        record: dict[str, Any] = {"train_on": key[0], "test_on": key[1]}
        max_abs_diff = 0.0
        for metric, expected_value in expected.items():
            observed = float(row[metric])
            diff = observed - expected_value
            record[f"published_{metric}"] = expected_value
            record[f"rerun_{metric}"] = observed
            record[f"diff_{metric}"] = diff
            max_abs_diff = max(max_abs_diff, abs(diff))
        record["max_abs_diff"] = max_abs_diff
        record["status_at_0.005_tolerance"] = "PASS" if max_abs_diff <= 0.005 else "CHECK"
        rows.append(record)
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "baseline_verification.csv", index=False, encoding="utf-8-sig")
    return out


def source_oof_threshold(
    source_df: pd.DataFrame, features: list[str]
) -> tuple[float, dict[str, float], list[float]]:
    """Select a source threshold from genuinely nested out-of-fold scores.

    C is re-selected using only each outer fold's training partition. This
    prevents full-source hyperparameter selection from leaking information
    into the OOF probabilities used for threshold selection.
    """
    x = source_df[features].to_numpy(dtype=float)
    y = source_df[LABEL_COL].to_numpy(dtype=int)
    outer_cv = StratifiedKFold(
        n_splits=cv_splits(y), shuffle=True, random_state=RANDOM_STATE
    )
    oof_prob = np.full(len(y), np.nan, dtype=float)
    fold_cs: list[float] = []
    for train_idx, validation_idx in outer_cv.split(x, y):
        outer_train = source_df.iloc[train_idx]
        fold_c, _ = select_best_c(outer_train, features)
        fold_cs.append(fold_c)
        fold_model = make_pipeline(fold_c)
        fold_model.fit(x[train_idx], y[train_idx])
        oof_prob[validation_idx] = fold_model.predict_proba(x[validation_idx])[:, 1]
    if not np.isfinite(oof_prob).all():
        raise RuntimeError("Nested source OOF prediction did not fill every row.")
    threshold, metrics = best_macro_f1_threshold(y, oof_prob)
    return threshold, metrics, fold_cs


def run_threshold_experiment(datasets: dict[str, DatasetFrames], output_dir: Path) -> pd.DataFrame:
    logging.info("=== Experiment 2: threshold calibration sensitivity ===")
    rows: list[dict[str, Any]] = []

    for source in DATASET_ORDER:
        source_df = datasets[source].train
        model, best_c, cv_score = train_source_model(source_df, FULL_FEATURES)
        src_threshold, src_dev_metrics, src_fold_cs = source_oof_threshold(
            source_df, FULL_FEATURES
        )
        logging.info(
            "%s nested source OOF threshold=%.6f (OOF Macro=%.4f, Pos=%.4f, fold Cs=%s)",
            source,
            src_threshold,
            src_dev_metrics["macro_f1"],
            src_dev_metrics["f1_pos"],
            src_fold_cs,
        )

        for target in DATASET_ORDER:
            if target == source:
                continue
            target_train = datasets[target].train
            target_test = datasets[target].test

            target_dev_x = target_train[FULL_FEATURES].to_numpy(dtype=float)
            target_dev_y = target_train[LABEL_COL].to_numpy(dtype=int)
            target_dev_prob = model.predict_proba(target_dev_x)[:, 1]
            target_threshold, target_dev_metrics = best_macro_f1_threshold(target_dev_y, target_dev_prob)

            test_x = target_test[FULL_FEATURES].to_numpy(dtype=float)
            test_y = target_test[LABEL_COL].to_numpy(dtype=int)
            test_prob = model.predict_proba(test_x)[:, 1]

            settings = [
                ("default_0.5", 0.5, None),
                ("source_oof_macro", src_threshold, src_dev_metrics),
                ("target_dev_macro", target_threshold, target_dev_metrics),
            ]
            for calibration, threshold, dev_metrics in settings:
                metrics = metric_dict(test_y, test_prob, threshold)
                rows.append({
                    "train_on": source,
                    "test_on": target,
                    "calibration": calibration,
                    "threshold_objective": "macro_f1" if calibration != "default_0.5" else "fixed",
                    "threshold": threshold,
                    "best_C": best_c,
                    "source_nested_oof_fold_Cs": ";".join(str(c) for c in src_fold_cs),
                    "source_cv_macro_f1": cv_score,
                    "calibration_dev_macro_f1": np.nan if dev_metrics is None else dev_metrics["macro_f1"],
                    "calibration_dev_f1_pos": np.nan if dev_metrics is None else dev_metrics["f1_pos"],
                    "source_train_positive_rate": float(source_df[LABEL_COL].mean()),
                    "target_dev_positive_rate": float(target_train[LABEL_COL].mean()),
                    "target_test_positive_rate": float(target_test[LABEL_COL].mean()),
                    **metrics,
                })
                logging.info(
                    "%s -> %s [%s] thr=%.5f | Macro=%.4f Pos=%.4f ROC=%.4f",
                    source, target, calibration, threshold, metrics["macro_f1"], metrics["f1_pos"], metrics["roc_auc"]
                )

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "threshold_calibration_all.csv", index=False, encoding="utf-8-sig")

    wide = out.pivot(index=["train_on", "test_on"], columns="calibration", values=["threshold", "macro_f1", "f1_pos", "f1_neg", "roc_auc", "average_precision"])
    wide.columns = [f"{metric}__{cal}" for metric, cal in wide.columns]
    wide = wide.reset_index()
    wide["gain_target_vs_default_macro_f1"] = wide["macro_f1__target_dev_macro"] - wide["macro_f1__default_0.5"]
    wide["gain_target_vs_default_f1_pos"] = wide["f1_pos__target_dev_macro"] - wide["f1_pos__default_0.5"]
    wide["gain_target_vs_source_macro_f1"] = wide["macro_f1__target_dev_macro"] - wide["macro_f1__source_oof_macro"]
    wide["gain_target_vs_source_f1_pos"] = wide["f1_pos__target_dev_macro"] - wide["f1_pos__source_oof_macro"]
    wide.to_csv(output_dir / "threshold_calibration_comparison.csv", index=False, encoding="utf-8-sig")
    return out


def direction_order(df: pd.DataFrame) -> pd.DataFrame:
    order = {
        ("DSTC9", "DSTC11"): 0,
        ("DSTC11", "DSTC9"): 1,
        ("DSTC9", "KETOD"): 2,
        ("DSTC11", "KETOD"): 3,
        ("KETOD", "DSTC9"): 4,
        ("KETOD", "DSTC11"): 5,
    }
    out = df.copy()
    out["_order"] = [order.get((a, b), 999) for a, b in zip(out["train_on"], out["test_on"])]
    return out.sort_values("_order").drop(columns="_order")


def markdown_table(headers: list[str], rows: Iterable[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_summary_outputs(
    no_question_comparison: pd.DataFrame,
    threshold_raw: pd.DataFrame,
    baseline_verification: pd.DataFrame,
    output_dir: Path,
) -> None:
    cross = direction_order(no_question_comparison[no_question_comparison["train_on"] != no_question_comparison["test_on"]])
    noq_rows: list[list[str]] = []
    for _, r in cross.iterrows():
        noq_rows.append([
            f"{r['train_on']} → {r['test_on']}",
            f"{r['full_roc_auc']:.3f}",
            f"{r['noq_roc_auc']:.3f}",
            f"{r['full_f1_pos']:.3f}",
            f"{r['noq_f1_pos']:.3f}",
            f"{r['full_macro_f1']:.3f}",
            f"{r['noq_macro_f1']:.3f}",
        ])
    no_q_table = markdown_table(
        ["Train → Test", "Full AUC", "No-Q AUC", "Full Pos.F1", "No-Q Pos.F1", "Full Macro", "No-Q Macro"],
        noq_rows,
    )

    thr = threshold_raw.pivot(index=["train_on", "test_on"], columns="calibration", values=["threshold", "macro_f1", "f1_pos", "roc_auc", "average_precision"])
    thr.columns = [f"{m}__{c}" for m, c in thr.columns]
    thr = direction_order(thr.reset_index())
    thr_rows_default: list[list[str]] = []
    thr_rows_source: list[list[str]] = []
    for _, r in thr.iterrows():
        direction = f"{r['train_on']} → {r['test_on']}"
        thr_rows_default.append([
            direction,
            f"{r['f1_pos__default_0.5']:.3f}",
            f"{r['f1_pos__target_dev_macro']:.3f}",
            f"{r['macro_f1__default_0.5']:.3f}",
            f"{r['macro_f1__target_dev_macro']:.3f}",
            f"{r['roc_auc__default_0.5']:.3f}",
        ])
        thr_rows_source.append([
            direction,
            f"{r['threshold__source_oof_macro']:.3f}",
            f"{r['threshold__target_dev_macro']:.3f}",
            f"{r['f1_pos__source_oof_macro']:.3f}",
            f"{r['f1_pos__target_dev_macro']:.3f}",
            f"{r['macro_f1__source_oof_macro']:.3f}",
            f"{r['macro_f1__target_dev_macro']:.3f}",
            f"{r['roc_auc__source_oof_macro']:.3f}",
        ])

    default_table = markdown_table(
        ["Train → Test", "Pos.F1 @0.5", "Pos.F1 target-dev", "Macro @0.5", "Macro target-dev", "ROC-AUC"],
        thr_rows_default,
    )
    source_table = markdown_table(
        ["Train → Test", "Source thr.", "Target thr.", "Source Pos.F1", "Target Pos.F1", "Source Macro", "Target Macro", "ROC-AUC"],
        thr_rows_source,
    )

    baseline_passes = int((baseline_verification["status_at_0.005_tolerance"] == "PASS").sum())
    baseline_total = len(baseline_verification)

    same_dirs = cross[
        ((cross["train_on"] == "DSTC9") & (cross["test_on"] == "DSTC11"))
        | ((cross["train_on"] == "DSTC11") & (cross["test_on"] == "DSTC9"))
    ]
    cross_family = cross[~cross.index.isin(same_dirs.index)]

    summary_lines = [
        "# Lightweight transfer-control results",
        "",
        f"Baseline verification: {baseline_passes}/{baseline_total} directions are within 0.005 of the submitted full-feature results.",
        "",
        "## Experiment 1: removing all three question-form features",
        "",
        no_q_table,
        "",
        "Descriptive checks (do not paste without reading the actual values):",
        f"- No-Q same-family AUC range: {same_dirs['noq_roc_auc'].min():.3f}–{same_dirs['noq_roc_auc'].max():.3f}.",
        f"- No-Q cross-family AUC range: {cross_family['noq_roc_auc'].min():.3f}–{cross_family['noq_roc_auc'].max():.3f}.",
        "- The full and No-Q columns use separately tuned C values under the same 3-fold source CV protocol.",
        "",
        "## Experiment 2A: original 0.5 threshold vs target-dev calibration",
        "",
        default_table,
        "",
        "## Experiment 2B: source-OOF threshold vs target-dev calibration",
        "",
        source_table,
        "",
        "Interpretation guardrails:",
        "- Target-dev calibration is an optimistic sensitivity analysis: the classifier is still trained only on the source dataset, but target labels are used to select a threshold.",
        "- ROC-AUC and average precision are unchanged by threshold selection. Calibration can improve thresholded F1 but cannot repair weak or reversed ranking.",
        "- The threshold objective is macro-F1, selected exactly over score boundaries; positive-class F1 is reported at that pre-specified threshold.",
    ]
    (output_dir / "transfer_controls_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )


def write_metadata(config_path: Path, output_dir: Path, elapsed_seconds: float) -> None:
    metadata = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed_seconds,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "C_grid": C_GRID,
        "cv": "StratifiedKFold(n_splits=3, shuffle=True, random_state=42)",
        "class_weight": "balanced",
        "default_threshold": 0.5,
        "threshold_selection_objective": "macro_f1",
        "source_oof_protocol": "nested 3-fold OOF; C re-selected within each outer training fold",
        "config_path": str(config_path.resolve()),
        "full_features": FULL_FEATURES,
        "removed_question_features": QUESTION_FEATURES,
        "no_question_features": NO_QUESTION_FEATURES,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Camera-ready No-Q transfer and threshold-calibration controls."
    )
    parser.add_argument("--config", type=Path, default=Path("config_paths.json"), help="JSON file with local dataset paths.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for result files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    setup_logging(output_dir)
    start = time.time()

    try:
        logging.info("Starting camera-ready lightweight transfer controls")
        logging.info("Working directory: %s", Path.cwd())
        logging.info("Config: %s", args.config)
        config = load_config(args.config)
        datasets = audit_and_load(config, output_dir)
        no_q_raw, no_q_comparison = run_no_question_experiment(datasets, output_dir)
        baseline = run_baseline_verification(no_q_raw, output_dir)
        threshold_raw = run_threshold_experiment(datasets, output_dir)
        write_summary_outputs(no_q_comparison, threshold_raw, baseline, output_dir)
        elapsed = time.time() - start
        write_metadata(args.config, output_dir, elapsed)
        logging.info("All experiments completed in %.1f seconds.", elapsed)
        logging.info("Open: %s", output_dir / "transfer_controls_summary.md")
        return 0
    except Exception as exc:
        logging.exception("Experiment failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
