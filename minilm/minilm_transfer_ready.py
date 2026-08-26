#!/usr/bin/env python3
"""Corrected accumulated-context MiniLM + source-trained LR transfer matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_COL = "input"
LABEL_COL = "output"
DATASET_ORDER = ("KETOD", "DSTC9", "DSTC11")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_paths.json")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--truncation-side", choices=("left", "right"), default="left")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tokenize-batch-size", type=int, default=1024)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path, data_root: Path) -> dict[str, dict[str, Path]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != set(DATASET_ORDER):
        raise ValueError(f"Config datasets must be {DATASET_ORDER}; got {tuple(raw)}")
    resolved: dict[str, dict[str, Path]] = {}
    for dataset in DATASET_ORDER:
        resolved[dataset] = {}
        for key in ("train_full", "test_full"):
            value = Path(raw[dataset][key])
            candidate = value if value.is_absolute() else data_root / value
            candidate = candidate.resolve()
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            resolved[dataset][key] = candidate
    return resolved


def label_to_int(series: pd.Series) -> np.ndarray:
    if series.dtype == object:
        normalized = series.astype(str).str.strip().str.lower()
        valid = normalized.isin(("true", "false", "1", "0"))
        if not valid.all():
            raise ValueError(f"Unexpected labels: {normalized[~valid].unique()[:10]}")
        return normalized.isin(("true", "1")).astype(int).to_numpy()
    values = series.astype(int).to_numpy()
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Labels must be binary")
    return values


def load_split(path: Path) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(path)
    for column in (TEXT_COL, LABEL_COL):
        if column not in frame.columns:
            raise ValueError(f"{path}: missing {column!r}")
    texts = frame[TEXT_COL].fillna("").astype(str).tolist()
    return texts, label_to_int(frame[LABEL_COL])


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def configure_model(args: argparse.Namespace) -> SentenceTransformer:
    device = choose_device(args.device)
    model = SentenceTransformer(
        args.model_name,
        device=device,
        local_files_only=args.local_files_only,
    )
    model.max_seq_length = args.max_len
    model.tokenizer.model_max_length = args.max_len
    model.tokenizer.truncation_side = args.truncation_side
    if model.max_seq_length != args.max_len:
        raise RuntimeError("SentenceTransformer max_seq_length override failed")
    if model.tokenizer.truncation_side != args.truncation_side:
        raise RuntimeError("Tokenizer truncation_side override failed")
    probe = "history " * (args.max_len + 100) + " USER: CURRENT TURN MUST REMAIN VISIBLE"
    expected = model.tokenizer(
        probe,
        add_special_tokens=True,
        truncation=True,
        max_length=args.max_len,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    model_tokens = model.tokenize([probe])
    attention = model_tokens["attention_mask"][0].detach().cpu().numpy().astype(bool)
    actual = model_tokens["input_ids"][0].detach().cpu().numpy()[attention].tolist()
    if actual != expected:
        raise RuntimeError("SentenceTransformer tokenization differs from the audited tokenizer input")
    current_ids = model.tokenizer(
        "CURRENT TURN MUST REMAIN VISIBLE", add_special_tokens=False
    )["input_ids"]
    if args.truncation_side == "left" and actual[-(len(current_ids) + 1) : -1] != current_ids:
        raise RuntimeError("Left-truncation self-test did not retain the synthetic current turn")
    print(
        f"model={args.model_name} device={device} max_len={model.max_seq_length} "
        f"truncation_side={model.tokenizer.truncation_side}",
        flush=True,
    )
    return model


def encode_split(
    model: SentenceTransformer,
    texts: list[str],
    cache_path: Path,
    batch_size: int,
) -> np.ndarray:
    """Encode the exact current inputs.

    Camera-ready runs deliberately do not reuse old embedding caches: a cache
    keyed only by row count can silently become stale after data/model changes.
    The newly produced cache is an output convenience, not an input artifact.
    """
    print(f"Encoding {len(texts)} accumulated-context inputs -> {cache_path.name}", flush=True)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    np.save(cache_path, embeddings)
    return embeddings


def train_lr(embeddings: np.ndarray, labels: np.ndarray, args: argparse.Namespace):
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=args.seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    search = GridSearchCV(
        estimator,
        {"lr__C": (0.01, 0.1, 1.0, 10.0)},
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    search.fit(embeddings, labels)
    return search.best_estimator_, float(search.best_params_["lr__C"]), float(search.best_score_)


def metric_values(labels: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "pos_f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "neg_f1": float(f1_score(labels, predictions, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def add_subset_metrics(
    row: dict[str, Any],
    prefix: str,
    keep: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> None:
    row[f"{prefix}_rows"] = int(keep.sum())
    if keep.sum() == 0 or np.unique(labels[keep]).size < 2:
        for name in ("macro_f1", "pos_f1", "neg_f1", "roc_auc", "average_precision"):
            row[f"{prefix}_{name}"] = float("nan")
        return
    for name, value in metric_values(labels[keep], probabilities[keep], predictions[keep]).items():
        row[f"{prefix}_{name}"] = value


def raw_overlap_mask(source_texts: list[str], target_texts: list[str]) -> np.ndarray:
    source = set(source_texts)
    return np.asarray([text in source for text in target_texts], dtype=bool)


def token_id_digests(
    texts: list[str], tokenizer: Any, max_len: int, batch_size: int
) -> list[bytes]:
    digests: list[bytes] = []
    for offset in range(0, len(texts), batch_size):
        tokenized = tokenizer(
            texts[offset : offset + batch_size],
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        for token_ids in tokenized["input_ids"]:
            packed = np.asarray(token_ids, dtype="<i4").tobytes()
            digests.append(hashlib.sha256(packed).digest())
    return digests


def main() -> None:
    args = parse_args()
    if args.model_name != MODEL_NAME:
        raise ValueError(f"Canonical camera-ready MiniLM model must be {MODEL_NAME!r}; got {args.model_name!r}")
    seed_everything(args.seed)
    started = time.time()
    config_path = Path(args.config).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = output_dir / "embedding_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path, data_root)
    model = configure_model(args)

    texts: dict[tuple[str, str], list[str]] = {}
    labels: dict[tuple[str, str], np.ndarray] = {}
    embeddings: dict[tuple[str, str], np.ndarray] = {}
    token_digests: dict[tuple[str, str], list[bytes]] = {}

    for dataset in DATASET_ORDER:
        for split, key in (("train", "train_full"), ("test", "test_full")):
            split_texts, split_labels = load_split(config[dataset][key])
            texts[(dataset, split)] = split_texts
            labels[(dataset, split)] = split_labels
            embeddings[(dataset, split)] = encode_split(
                model,
                split_texts,
                cache_dir
                / f"{dataset.lower()}_{split}_{args.truncation_side}{args.max_len}.npy",
                args.batch_size,
            )
            token_digests[(dataset, split)] = token_id_digests(
                split_texts, model.tokenizer, args.max_len, args.tokenize_batch_size
            )
            print(
                f"{dataset} {split}: rows={len(split_texts)} "
                f"positive_rate={split_labels.mean():.6f} "
                f"embedding_shape={embeddings[(dataset, split)].shape}",
                flush=True,
            )

    records: list[dict[str, Any]] = []
    for train_dataset in DATASET_ORDER:
        train_embeddings = embeddings[(train_dataset, "train")]
        train_labels = labels[(train_dataset, "train")]
        classifier, best_c, source_cv = train_lr(train_embeddings, train_labels, args)
        print(
            f"{train_dataset}: best_C={best_c} source_cv_macro_f1={source_cv:.6f}",
            flush=True,
        )
        source_digest_set = set(token_digests[(train_dataset, "train")])

        for test_dataset in DATASET_ORDER:
            test_embeddings = embeddings[(test_dataset, "test")]
            test_labels = labels[(test_dataset, "test")]
            probabilities = classifier.predict_proba(test_embeddings)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            metrics = metric_values(test_labels, probabilities, predictions)
            raw_overlap = raw_overlap_mask(
                texts[(train_dataset, "train")], texts[(test_dataset, "test")]
            )
            model_overlap = np.asarray(
                [digest in source_digest_set for digest in token_digests[(test_dataset, "test")]],
                dtype=bool,
            )
            row: dict[str, Any] = {
                "model": "all-MiniLM-L6-v2+LR",
                "train_on": train_dataset,
                "test_on": test_dataset,
                "in_domain": train_dataset == test_dataset,
                "best_C": best_c,
                "source_cv_macro_f1": source_cv,
                "threshold": 0.5,
                **metrics,
                "target_rows": len(test_labels),
                "raw_exact_overlap_rows": int(raw_overlap.sum()),
                "raw_exact_overlap_rate": float(raw_overlap.mean()),
                "tokenized_input_overlap_rows": int(model_overlap.sum()),
                "tokenized_input_overlap_rate": float(model_overlap.mean()),
                "max_len": args.max_len,
                "truncation_side": args.truncation_side,
                "seed": args.seed,
            }
            add_subset_metrics(
                row,
                "raw_nonoverlap",
                ~raw_overlap,
                test_labels,
                probabilities,
                predictions,
            )
            add_subset_metrics(
                row,
                "model_input_nonoverlap",
                ~model_overlap,
                test_labels,
                probabilities,
                predictions,
            )
            records.append(row)

            prediction_frame = pd.DataFrame(
                {
                    "row_index": np.arange(len(test_labels)),
                    "label": test_labels,
                    "prob_positive": probabilities,
                    "prediction": predictions,
                    "raw_exact_overlap_with_source_train": raw_overlap,
                    "tokenized_input_overlap_with_source_train": model_overlap,
                }
            )
            prediction_frame.to_csv(
                output_dir
                / f"pred_{train_dataset.lower()}_to_{test_dataset.lower()}.csv",
                index=False,
            )
            print(
                f"{train_dataset:6s}->{test_dataset:6s} "
                f"Macro={metrics['macro_f1']:.4f} Pos={metrics['pos_f1']:.4f} "
                f"Neg={metrics['neg_f1']:.4f} AUC={metrics['roc_auc']:.4f} "
                f"AP={metrics['average_precision']:.4f} "
                f"raw_overlap={raw_overlap.sum()}/{len(raw_overlap)} "
                f"model_overlap={model_overlap.sum()}/{len(model_overlap)}",
                flush=True,
            )

    result_frame = pd.DataFrame(records)
    result_path = output_dir / "minilm_results_ready.csv"
    result_frame.to_csv(result_path, index=False)
    (output_dir / "minilm_summary.txt").write_text(
        result_frame.to_string(index=False), encoding="utf-8"
    )

    model_commit = getattr(model[0].auto_model.config, "_commit_hash", None)
    metadata = {
        "timestamp_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(model.device),
        "model_name": args.model_name,
        "model_config_commit_hash": model_commit,
        "embedding_dimension": int(embeddings[("KETOD", "train")].shape[1]),
        "tokenizer_truncation_side": model.tokenizer.truncation_side,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "cv_folds": args.cv_folds,
        "C_grid": [0.01, 0.1, 1.0, 10.0],
        "class_weight": "balanced",
        "threshold": 0.5,
        "seed": args.seed,
        "input_policy": "processed accumulated dialogue context ending at evaluated user turn",
        "sentence_transformer_tokenization_self_test": "PASS",
        "average_precision_definition": "sklearn.metrics.average_precision_score",
        "overlap_policy": (
            "raw exact string and exact truncated input_ids; "
            "model_input_nonoverlap is primary"
        ),
        "config": {
            dataset: {key: str(value) for key, value in values.items()}
            for dataset, values in config.items()
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    by_key = {(row["train_on"], row["test_on"]): row for row in records}
    paper_summary = {
        "protocol_validation": "PASS",
        "paper_values_to_update": {
            "KETOD_to_KETOD": by_key[("KETOD", "KETOD")],
            "DSTC9_to_KETOD": by_key[("DSTC9", "KETOD")],
            "DSTC11_to_KETOD": by_key[("DSTC11", "KETOD")],
        },
        "interpretation_rule": (
            "A weak thresholded positive F1 supports a ranking-failure claim only "
            "when ROC-AUC/AP and non-overlap sensitivities are also weak."
        ),
    }
    (output_dir / "camera_ready_minilm_summary.json").write_text(
        json.dumps(paper_summary, indent=2), encoding="utf-8"
    )
    print(f"Saved formal results -> {result_path}", flush=True)


if __name__ == "__main__":
    main()
