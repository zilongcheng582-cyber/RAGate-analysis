#!/usr/bin/env python3
"""Camera-ready BERT cross-dataset transfer experiment.

Design decisions:
- Fine-tune bert-base-uncased separately on each source dataset.
- Fixed 3-epoch training; no target-domain early stopping or target calibration.
- Explicit LEFT truncation by default to prioritize retention of the latest
  USER turn at the end of accumulated dialogue context.
- Report thresholded F1, ROC-AUC, and Average Precision (AP).
- Seed Python / NumPy / PyTorch and use a seeded DataLoader generator.
- Save run metadata and optional per-example predictions.

The processed ``input`` field is used as-is. In the audited data, all three
benchmarks predominantly contain accumulated dialogue context rather than a
KETOD/DSTC9 single-turn vs DSTC11-history split.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "bert-base-uncased"
TEXT_COL = "input"
LABEL_COL = "output"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_paths.json")
    p.add_argument("--output-dir", default="results/bert_ready")
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 is the deterministic/Windows-safe default")
    p.add_argument("--truncation-side", choices=["left", "right"], default="left")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--save-models", action="store_true")
    return p.parse_args()


def load_config(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for ds in ("KETOD", "DSTC9", "DSTC11"):
        if ds not in cfg:
            raise ValueError(f"Missing dataset {ds} in {path}")
        for key in ("train_full", "test_full"):
            if key not in cfg[ds]:
                raise ValueError(f"Missing {ds}.{key} in {path}")
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def load_dataset(path: str) -> Tuple[pd.DataFrame, list[str], np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    for col in (TEXT_COL, LABEL_COL):
        if col not in df.columns:
            raise ValueError(f"{p}: missing column {col!r}")
    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    labels = label_to_int(df[LABEL_COL])
    if len(texts) != len(labels):
        raise AssertionError("Text/label length mismatch")
    return df, texts, labels


class GatingDataset(Dataset):
    def __init__(self, texts: Iterable[str], labels: np.ndarray,
                 tokenizer: BertTokenizerFast, max_len: int):
        enc = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.encodings = enc
        # Pandas/NumPy may expose a read-only view.  Copy it so PyTorch never
        # warns about undefined behavior if a tensor consumer writes in-place.
        self.labels = torch.tensor(np.array(labels, copy=True), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"Both classes are required, got counts={counts.tolist()}")
    weights = len(labels) / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool,
                num_workers: int, seed: int) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=gen,
    )


def train_epoch(model, loader, optimizer, scheduler, device, class_weights) -> float:
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        labels = batch.pop("labels").to(device)
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(loader))


@torch.no_grad()
def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, probs, preds = [], [], []
    for batch in loader:
        labels = batch.pop("labels").numpy()
        inputs = {k: v.to(device) for k, v in batch.items()}
        logits = model(**inputs).logits
        p1 = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        pred = (p1 >= 0.5).astype(np.int64)
        ys.append(labels)
        probs.append(p1)
        preds.append(pred)
    return np.concatenate(ys), np.concatenate(probs), np.concatenate(preds)


def metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "pos_f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "neg_f1": float(f1_score(y_true, pred, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "average_precision": float(average_precision_score(y_true, prob)),
    }


def train_model(train_texts, train_labels, tokenizer, device, args, ds_name):
    train_ds = GatingDataset(train_texts, train_labels, tokenizer, args.max_len)
    loader = make_loader(train_ds, args.batch_size, True, args.num_workers, args.seed)

    model = BertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    ).to(device)
    weights = compute_class_weights(train_labels)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\nFine-tuning {ds_name}: {len(train_ds)} rows")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, scheduler, device, weights)
        print(f"  epoch {epoch}/{args.epochs}: loss={loss:.6f}")
    return model


def raw_exact_overlap_mask(source_train_texts: list[str], target_texts: list[str]) -> np.ndarray:
    source_set = set(source_train_texts)
    return np.asarray([t in source_set for t in target_texts], dtype=bool)


def iter_model_input_keys(texts: list[str], tokenizer: BertTokenizerFast,
                          max_len: int, batch_size: int = 2048):
    """Yield the exact non-padding input-id sequence visible to BERT.

    This applies the canonical tokenizer normalization and the tokenizer's
    configured truncation side.  Streaming avoids materializing a second full
    encoded copy of the largest source dataset.
    """
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size],
            truncation=True,
            padding=False,
            max_length=max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        for input_ids in encoded["input_ids"]:
            yield tuple(input_ids)


def model_input_key_set(texts: list[str], tokenizer: BertTokenizerFast,
                        max_len: int) -> set[tuple[int, ...]]:
    return set(iter_model_input_keys(texts, tokenizer, max_len))


def model_input_keys_from_dataset(dataset: GatingDataset) -> list[tuple[int, ...]]:
    """Recover non-padding model-input keys from an already encoded dataset."""
    input_ids = dataset.encodings["input_ids"]
    attention_mask = dataset.encodings["attention_mask"]
    lengths = attention_mask.sum(dim=1).tolist()
    return [
        tuple(row[:int(length)].tolist())
        for row, length in zip(input_ids, lengths)
    ]


def add_subset_metrics(row: dict, prefix: str, keep_mask: np.ndarray,
                       y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> None:
    row[f"{prefix}_rows"] = int(keep_mask.sum())
    if keep_mask.sum() > 0 and len(np.unique(y[keep_mask])) == 2:
        subset = metrics(y[keep_mask], prob[keep_mask], pred[keep_mask])
        row.update({f"{prefix}_{key}": value for key, value in subset.items()})
    else:
        for key in ("macro_f1", "pos_f1", "neg_f1", "roc_auc", "average_precision"):
            row[f"{prefix}_{key}"] = float("nan")


def main() -> None:
    args = parse_args()
    if args.model_name != MODEL_NAME:
        raise ValueError(f"Canonical camera-ready BERT model must be {MODEL_NAME!r}; got {args.model_name!r}")
    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    tokenizer.truncation_side = args.truncation_side
    if args.max_len > tokenizer.model_max_length:
        raise ValueError(
            f"max_len={args.max_len} exceeds tokenizer model_max_length="
            f"{tokenizer.model_max_length}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; truncation_side={tokenizer.truncation_side}; max_len={args.max_len}")
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)}")

    data = {}
    for ds in ("KETOD", "DSTC9", "DSTC11"):
        for split, key in (("train", "train_full"), ("test", "test_full")):
            df, texts, labels = load_dataset(cfg[ds][key])
            data[(ds, split)] = {"df": df, "texts": texts, "labels": labels}
            print(
                f"{ds:6s} {split:5s}: n={len(labels)}, "
                f"pos={int(labels.sum())}, neg={int((labels == 0).sum())}"
            )

    models = {}
    for ds in ("KETOD", "DSTC9", "DSTC11"):
        # Reset RNG state before each source model so its initialization and
        # training order are reproducible independently of preceding sources.
        seed_everything(args.seed)
        models[ds] = train_model(
            data[(ds, "train")]["texts"],
            data[(ds, "train")]["labels"],
            tokenizer,
            device,
            args,
            ds,
        )
        if args.save_models:
            model_dir = out_dir / f"model_{ds.lower()}"
            models[ds].save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

    records = []
    for source in ("KETOD", "DSTC9", "DSTC11"):
        source_train_texts = data[(source, "train")]["texts"]
        # Raw equality can miss collisions introduced by uncased tokenizer
        # normalization and left truncation.  Build the exact model-visible
        # source-input set once per source, then reuse it for all three targets.
        source_model_input_set = model_input_key_set(
            source_train_texts, tokenizer, args.max_len
        )
        for target in ("KETOD", "DSTC9", "DSTC11"):
            target_texts = data[(target, "test")]["texts"]
            target_y = data[(target, "test")]["labels"]
            ds_obj = GatingDataset(target_texts, target_y, tokenizer, args.max_len)
            loader = make_loader(ds_obj, args.eval_batch_size, False, args.num_workers, args.seed)
            y, prob, pred = predict(models[source], loader, device)
            m = metrics(y, prob, pred)
            raw_overlap = raw_exact_overlap_mask(source_train_texts, target_texts)
            target_model_keys = model_input_keys_from_dataset(ds_obj)
            model_input_overlap = np.asarray(
                [key in source_model_input_set for key in target_model_keys],
                dtype=bool,
            )

            row = {
                "model": "BERT",
                "train_on": source,
                "test_on": target,
                "in_domain": source == target,
                "threshold": 0.5,
                **m,
                "target_rows": int(len(y)),
                "raw_exact_overlap_rows": int(raw_overlap.sum()),
                "raw_exact_overlap_rate": float(raw_overlap.mean()),
                "tokenized_input_overlap_rows": int(model_input_overlap.sum()),
                "tokenized_input_overlap_rate": float(model_input_overlap.mean()),
                "max_len": args.max_len,
                "truncation_side": args.truncation_side,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "seed": args.seed,
            }

            # Report both sensitivities.  Model-input non-overlap is primary for
            # BERT because different raw strings can normalize/truncate to the
            # same visible input-id sequence.
            add_subset_metrics(
                row, "raw_nonoverlap", ~raw_overlap, y, prob, pred
            )
            add_subset_metrics(
                row, "model_input_nonoverlap", ~model_input_overlap, y, prob, pred
            )

            records.append(row)
            print(
                f"{source:6s}->{target:6s} "
                f"Macro={m['macro_f1']:.4f} PosF1={m['pos_f1']:.4f} "
                f"AUC={m['roc_auc']:.4f} AP={m['average_precision']:.4f} "
                f"raw_overlap={raw_overlap.sum()}/{len(raw_overlap)} "
                f"model_overlap={model_input_overlap.sum()}/{len(model_input_overlap)}"
            )

            if args.save_predictions:
                pred_df = pd.DataFrame({
                    "row_index": np.arange(len(y)),
                    "label": y,
                    "prob_positive": prob,
                    "prediction": pred,
                    "raw_exact_overlap_with_source_train": raw_overlap,
                    "tokenized_input_overlap_with_source_train": model_input_overlap,
                })
                pred_df.to_csv(
                    out_dir / f"pred_{source.lower()}_to_{target.lower()}.csv",
                    index=False,
                )

    result_df = pd.DataFrame(records)
    result_df.to_csv(out_dir / "bert_results_ready.csv", index=False)

    metadata = {
        "timestamp_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "model_name": args.model_name,
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "tokenizer_commit_hash": tokenizer.init_kwargs.get("_commit_hash"),
        "model_config_commit_hash_by_source": {
            ds: getattr(models[ds].config, "_commit_hash", None)
            for ds in ("KETOD", "DSTC9", "DSTC11")
        },
        "tokenizer_truncation_side": tokenizer.truncation_side,
        "tokenizer_model_max_length": tokenizer.model_max_length,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "config": cfg,
        "evaluation_policy": "final epoch; fixed 0.5 threshold; no target calibration or target early stopping",
        "average_precision_definition": "sklearn.metrics.average_precision_score",
        "overlap_policy": (
            "report raw exact-string overlap and exact non-padding tokenized input_ids overlap; "
            "model_input_nonoverlap is the primary BERT overlap sensitivity"
        ),
    }
    try:
        import transformers
        metadata["transformers"] = transformers.__version__
    except Exception:
        pass
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open(out_dir / "bert_summary.txt", "w", encoding="utf-8") as f:
        f.write(result_df.to_string(index=False))
        f.write("\n")

    print(f"\nSaved results -> {out_dir / 'bert_results_ready.csv'}")
    print(f"Saved metadata -> {out_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
