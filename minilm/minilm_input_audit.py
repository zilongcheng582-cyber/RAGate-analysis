#!/usr/bin/env python3
"""Audit accumulated-context inputs under MiniLM right vs left truncation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_COL = "input"
LABEL_COL = "output"
USER_MARKER = "USER:"
SYSTEM_MARKER = "SYSTEM:"
DATASET_ORDER = ("KETOD", "DSTC9", "DSTC11")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_paths.json")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--tokenize-batch-size", type=int, default=512)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


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


def retention_stats(texts: list[str], tokenizer: Any, max_len: int, batch_size: int) -> dict[str, np.ndarray | int]:
    content_capacity = max_len - tokenizer.num_special_tokens_to_add(pair=False)
    if content_capacity <= 0:
        raise ValueError("max_len leaves no content-token capacity")

    full_len: list[int] = []
    current_len: list[int] = []
    has_user: list[bool] = []
    right_any: list[bool] = []
    right_full: list[bool] = []
    left_any: list[bool] = []
    left_full: list[bool] = []

    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        tokenized = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        for text, token_ids, offsets in zip(batch, tokenized["input_ids"], tokenized["offset_mapping"]):
            marker_position = text.rfind(USER_MARKER)
            marker_found = marker_position >= 0
            if not marker_found:
                marker_position = 0

            indices = [
                index
                for index, (_start_char, end_char) in enumerate(offsets)
                if end_char > marker_position
            ]
            n_tokens = len(token_ids)
            if indices:
                current_start = indices[0]
                current_end = indices[-1] + 1
            else:
                current_start = current_end = n_tokens

            right_start, right_end = 0, min(content_capacity, n_tokens)
            left_start, left_end = max(0, n_tokens - content_capacity), n_tokens

            full_len.append(n_tokens)
            current_len.append(max(0, current_end - current_start))
            has_user.append(marker_found)
            right_any.append(current_start < right_end and current_end > right_start)
            right_full.append(current_start >= right_start and current_end <= right_end)
            left_any.append(current_start < left_end and current_end > left_start)
            left_full.append(current_start >= left_start and current_end <= left_end)

    return {
        "full_len": np.asarray(full_len),
        "current_len": np.asarray(current_len),
        "has_user": np.asarray(has_user, dtype=bool),
        "right_any": np.asarray(right_any, dtype=bool),
        "right_full": np.asarray(right_full, dtype=bool),
        "left_any": np.asarray(left_any, dtype=bool),
        "left_full": np.asarray(left_full, dtype=bool),
        "content_capacity": content_capacity,
    }


def main() -> None:
    args = parse_args()
    if args.model_name != MODEL_NAME:
        raise ValueError(f"Canonical camera-ready MiniLM model must be {MODEL_NAME!r}; got {args.model_name!r}")
    config = load_config(Path(args.config).resolve(), Path(args.data_root).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(
        args.model_name,
        local_files_only=args.local_files_only,
        device="cpu",
    )
    tokenizer = model.tokenizer
    default_side = tokenizer.truncation_side
    model.max_seq_length = args.max_len

    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        for split, key in (("train", "train_full"), ("test", "test_full")):
            path = config[dataset][key]
            texts, labels = load_split(path)
            stats = retention_stats(texts, tokenizer, args.max_len, args.tokenize_batch_size)
            lengths = stats["full_len"]
            current_lengths = stats["current_len"]
            requires_truncation = lengths > stats["content_capacity"]
            user_counts = np.asarray([text.count(USER_MARKER) for text in texts])
            system_counts = np.asarray([text.count(SYSTEM_MARKER) for text in texts])

            row = {
                "dataset": dataset,
                "split": split,
                "path": str(path),
                "rows": len(texts),
                "positive_rate": float(labels.mean()),
                "system_marker_rate": float(np.mean(system_counts > 0)),
                "multiple_user_marker_rate": float(np.mean(user_counts > 1)),
                "explicit_user_marker_rate": float(np.mean(user_counts > 0)),
                "token_len_p50": float(np.percentile(lengths, 50)),
                "token_len_p90": float(np.percentile(lengths, 90)),
                "token_len_p95": float(np.percentile(lengths, 95)),
                "token_len_p99": float(np.percentile(lengths, 99)),
                "token_len_max": int(lengths.max()),
                "current_turn_token_len_max": int(current_lengths.max()),
                "requires_truncation_rate": float(np.mean(requires_truncation)),
                "right_current_any_rate": float(np.mean(stats["right_any"])),
                "right_current_full_rate": float(np.mean(stats["right_full"])),
                "left_current_any_rate": float(np.mean(stats["left_any"])),
                "left_current_full_rate": float(np.mean(stats["left_full"])),
                "right_minus_left_full_retention": float(
                    np.mean(stats["right_full"]) - np.mean(stats["left_full"])
                ),
                "max_len": args.max_len,
                "content_token_capacity": stats["content_capacity"],
                "model_name": args.model_name,
                "default_truncation_side_before_override": default_side,
            }
            rows.append(row)
            print(
                f"{dataset:6s} {split:5s} n={len(texts):6d} "
                f"truncated={100 * row['requires_truncation_rate']:6.2f}% | "
                f"current fully kept right={100 * row['right_current_full_rate']:6.2f}% "
                f"left={100 * row['left_current_full_rate']:6.2f}%",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    csv_path = output_dir / "minilm_input_audit.csv"
    frame.to_csv(csv_path, index=False)
    summary = {
        "protocol_validation": "PASS",
        "model_name": args.model_name,
        "default_truncation_side_before_override": default_side,
        "formal_truncation_side": "left",
        "max_len": args.max_len,
        "content_token_capacity": int(frame["content_token_capacity"].iloc[0]),
        "all_left_current_turns_fully_retained": bool((frame["left_current_full_rate"] == 1.0).all()),
        "audit_csv": csv_path.name,
    }
    (output_dir / "minilm_input_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
