#!/usr/bin/env python3
"""Audit processed text inputs before the camera-ready BERT rerun.

Checks dataset provenance and BERT tokenization/truncation properties without
training a model. In particular, it measures how right vs left truncation
prioritizes retention of the latest USER turn serialized at the end of the
processed input. A turn longer than the content capacity cannot be fully kept.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from transformers import BertTokenizerFast

MODEL_NAME = "bert-base-uncased"
TEXT_COL = "input"
LABEL_COL = "output"
USER_MARKER = "USER:"
SYSTEM_MARKER = "SYSTEM:"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_paths.json")
    p.add_argument("--output", default="results/bert_input_audit.csv")
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--max-len", type=int, default=256)
    return p.parse_args()


def load_config(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    required = {"KETOD", "DSTC9", "DSTC11"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Missing datasets in config: {sorted(missing)}")
    return cfg


def label_to_int(series: pd.Series) -> np.ndarray:
    if series.dtype == object:
        norm = series.astype(str).str.strip().str.lower()
        valid = norm.isin(["true", "false", "1", "0"])
        if not valid.all():
            bad = sorted(norm[~valid].unique().tolist())[:10]
            raise ValueError(f"Unexpected labels: {bad}")
        return norm.isin(["true", "1"]).astype(int).to_numpy()
    vals = series.astype(int).to_numpy()
    if not np.isin(vals, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1 or True/False")
    return vals


def load_split(path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    for col in (TEXT_COL, LABEL_COL):
        if col not in df.columns:
            raise ValueError(f"{p}: missing column {col!r}")
    if df[TEXT_COL].isna().any():
        df[TEXT_COL] = df[TEXT_COL].fillna("")
    y = label_to_int(df[LABEL_COL])
    return df, y


def token_lengths_and_current_turn(texts, tokenizer, max_len: int):
    """Return arrays describing full token length and turn retention.

    We tokenize without special tokens for counting. For BERT single-sequence
    classification, [CLS] and [SEP] consume two positions, leaving
    ``max_len - 2`` content tokens.
    """
    content_cap = max_len - tokenizer.num_special_tokens_to_add(pair=False)
    if content_cap <= 0:
        raise ValueError("max_len too small after special tokens")

    full_lens = []
    current_lens = []
    has_user = []
    right_any = []
    right_full = []
    left_any = []
    left_full = []

    for text in texts:
        text = str(text)
        marker_pos = text.rfind(USER_MARKER)
        has = marker_pos >= 0
        if has:
            prefix = text[:marker_pos]
            current = text[marker_pos:]
        else:
            # If no explicit USER marker exists, treat the whole input as the
            # current text rather than silently declaring it lost.
            prefix = ""
            current = text

        # Fast-tokenizer offsets let us locate the latest USER turn exactly
        # in the tokenized full sequence, avoiding prefix-tokenization edge
        # cases at punctuation/whitespace boundaries.
        tok = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        full_ids = tok["input_ids"]
        offsets = tok["offset_mapping"]
        n = len(full_ids)

        current_token_idx = [
            i for i, (_start_char, end_char) in enumerate(offsets)
            if end_char > marker_pos
        ]
        if current_token_idx:
            start = current_token_idx[0]
            end = current_token_idx[-1] + 1
        else:
            start = end = n

        right_start, right_end = 0, min(content_cap, n)
        left_start, left_end = max(0, n - content_cap), n

        full_lens.append(n)
        current_lens.append(max(0, end - start))
        has_user.append(has)
        right_any.append(start < right_end and end > right_start)
        right_full.append(start >= right_start and end <= right_end)
        left_any.append(start < left_end and end > left_start)
        left_full.append(start >= left_start and end <= left_end)

    return {
        "full_len": np.asarray(full_lens),
        "current_len": np.asarray(current_lens),
        "has_user": np.asarray(has_user, dtype=bool),
        "right_any": np.asarray(right_any, dtype=bool),
        "right_full": np.asarray(right_full, dtype=bool),
        "left_any": np.asarray(left_any, dtype=bool),
        "left_full": np.asarray(left_full, dtype=bool),
        "content_cap": content_cap,
    }


def pct(x) -> float:
    return 100.0 * float(np.mean(x)) if len(x) else float("nan")


def main() -> None:
    args = parse_args()
    if args.model_name != MODEL_NAME:
        raise ValueError(f"Canonical camera-ready BERT tokenizer must be {MODEL_NAME!r}; got {args.model_name!r}")
    cfg = load_config(args.config)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    rows = []
    for ds in ("KETOD", "DSTC9", "DSTC11"):
        for split, key in (("train", "train_full"), ("test", "test_full")):
            path = cfg[ds][key]
            df, y = load_split(path)
            texts = df[TEXT_COL].astype(str).tolist()
            stats = token_lengths_and_current_turn(texts, tokenizer, args.max_len)
            lens = stats["full_len"]
            user_counts = df[TEXT_COL].astype(str).str.count(USER_MARKER).to_numpy()
            sys_counts = df[TEXT_COL].astype(str).str.count(SYSTEM_MARKER).to_numpy()

            rows.append({
                "dataset": ds,
                "split": split,
                "path": str(path),
                "rows": len(df),
                "positive_rate": float(y.mean()),
                "system_marker_rate": float(np.mean(sys_counts > 0)),
                "multiple_user_marker_rate": float(np.mean(user_counts > 1)),
                "explicit_user_marker_rate": float(np.mean(user_counts > 0)),
                "token_len_p50": float(np.percentile(lens, 50)),
                "token_len_p90": float(np.percentile(lens, 90)),
                "token_len_p95": float(np.percentile(lens, 95)),
                "token_len_p99": float(np.percentile(lens, 99)),
                "token_len_max": int(lens.max()) if len(lens) else 0,
                "requires_truncation_rate": float(np.mean(lens > stats["content_cap"])),
                "right_current_any_rate": float(np.mean(stats["right_any"])),
                "right_current_full_rate": float(np.mean(stats["right_full"])),
                "left_current_any_rate": float(np.mean(stats["left_any"])),
                "left_current_full_rate": float(np.mean(stats["left_full"])),
                "max_len": args.max_len,
                "content_token_capacity": stats["content_cap"],
                "tokenizer": args.model_name,
            })

            print(
                f"{ds:6s} {split:5s} n={len(df):6d} "
                f"requires_truncation={pct(lens > stats['content_cap']):6.2f}% | "
                f"current fully kept: right={pct(stats['right_full']):6.2f}% "
                f"left={pct(stats['left_full']):6.2f}%"
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved audit -> {out}")


if __name__ == "__main__":
    main()
