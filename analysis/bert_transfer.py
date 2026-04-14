"""
bert_transfer.py — BERT Cross-Dataset Transfer Experiment
Fine-tune bert-base-uncased as a knowledge-gating classifier on each dataset,
then run the same 3x3 cross-dataset transfer matrix as the structural LR baseline.

Purpose: Progressive capacity check — if task-specific fine-tuned BERT also
fails to transfer, annotation incompatibility is the only remaining explanation.

Dependencies:
    pip install transformers torch scikit-learn pandas numpy

Usage:
    # AutoDL / local GPU:
    python bert_transfer.py

    # With explicit paths:
    python bert_transfer.py \
        --ketod-train data/ketod/train_full.csv \
        --ketod-test  data/ketod/test_full.csv \
        --dstc9-train data/dstc9/train_dstc9.csv \
        --dstc9-test  data/dstc9/test_dstc9.csv \
        --dstc11-train data/dstc11/train.csv \
        --dstc11-test  data/dstc11/val.csv \
        --output-dir  results/

    # Skip structural comparison table:
    python bert_transfer.py --no-comparison

Outputs:
    bert_results.csv      — full 3x3 transfer matrix
    bert_summary.txt      — paper-ready numbers

Input convention:
    All datasets use 'input' column (raw text) and 'output' column (True/False).
    DSTC11 input already contains multi-turn context; KETOD/DSTC9 contain
    single user turns. We use the raw input field as-is for all datasets,
    which naturally reflects each benchmark's annotation granularity.
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DATASETS = {
    "KETOD": {
        "train": "data/ketod/train_full.csv",
        "test":  "data/ketod/test_full.csv",
    },
    "DSTC9": {
        "train": "data/dstc9/train_dstc9.csv",
        "test":  "data/dstc9/test_dstc9.csv",
    },
    "DSTC11": {
        "train": "data/dstc11/train.csv",
        "test":  "data/dstc11/val.csv",
    },
}

LABEL_COL = "output"
TEXT_COL  = "input"

MODEL_NAME   = "bert-base-uncased"
MAX_LEN      = 256
BATCH_SIZE   = 32
EPOCHS       = 3
LR           = 2e-5
WARMUP_RATIO = 0.1
RANDOM_STATE = 42

# Structural LR baseline for comparison table
STRUCTURAL_BASELINE = {
    "KETOD→KETOD":   (0.5364, 0.3302),
    "KETOD→DSTC9":   (0.4165, 0.1872),
    "KETOD→DSTC11":  (0.4121, 0.3670),
    "DSTC9→KETOD":   (0.4661, 0.0000),
    "DSTC9→DSTC9":   (0.7982, 0.7398),
    "DSTC9→DSTC11":  (0.8359, 0.8564),
    "DSTC11→KETOD":  (0.4661, 0.0000),
    "DSTC11→DSTC9":  (0.8097, 0.7422),
    "DSTC11→DSTC11": (0.8424, 0.8524),
}

SEMANTIC_BASELINE = {
    "DSTC9→KETOD":   (0.4453, 0.1186),
    "DSTC11→KETOD":  (0.3901, 0.1335),
}

HIGHLIGHT_PAIRS = [
    ("DSTC9",  "KETOD"),
    ("DSTC11", "KETOD"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class GatingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings["token_type_ids"][idx],
            "labels":         self.labels[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path, label_col=LABEL_COL, text_col=TEXT_COL):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if df[label_col].dtype == object:
        labels = (df[label_col].str.strip() == "True").astype(int).values
    else:
        labels = df[label_col].astype(int).values
    texts = df[text_col].fillna("").tolist()
    pos = int(labels.sum())
    neg = int((labels == 0).sum())
    print(f"    {len(texts)} samples | pos={pos} neg={neg} "
          f"| ratio={pos/(pos+neg):.2f}")
    return texts, labels


def compute_class_weights(labels):
    """Inverse-frequency weights for imbalanced datasets."""
    counts = np.bincount(labels)
    weights = len(labels) / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(model, loader, optimizer, scheduler, device, class_weights):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["labels"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].numpy())
    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    return {
        "macro_f1":    round(f1_score(y_true, y_pred, average="macro"), 4),
        "minority_f1": round(f1_score(y_true, y_pred, pos_label=1,
                                      zero_division=0), 4),
        "report":      classification_report(y_true, y_pred, digits=4),
    }


def fine_tune(train_texts, train_labels, tokenizer, device, ds_name):
    """Fine-tune BERT on one dataset, return trained model."""
    print(f"\n  Fine-tuning on {ds_name}...")
    dataset = GatingDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=2, pin_memory=True)

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    class_weights = compute_class_weights(train_labels)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, loader, optimizer, scheduler,
                           device, class_weights)
        print(f"    Epoch {epoch}/{EPOCHS} | loss={loss:.4f}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ketod-train",   default=None)
    p.add_argument("--ketod-test",    default=None)
    p.add_argument("--dstc9-train",   default=None)
    p.add_argument("--dstc9-test",    default=None)
    p.add_argument("--dstc11-train",  default=None)
    p.add_argument("--dstc11-test",   default=None)
    p.add_argument("--output-dir",    default="results")
    p.add_argument("--no-comparison", action="store_true")
    p.add_argument("--epochs",        type=int, default=EPOCHS)
    p.add_argument("--batch-size",    type=int, default=BATCH_SIZE)
    return p.parse_args()


def resolve_datasets(args):
    datasets = {k: dict(v) for k, v in DEFAULT_DATASETS.items()}
    overrides = {
        "KETOD":  {"train": args.ketod_train,  "test": args.ketod_test},
        "DSTC9":  {"train": args.dstc9_train,  "test": args.dstc9_test},
        "DSTC11": {"train": args.dstc11_train, "test": args.dstc11_test},
    }
    for ds, splits in overrides.items():
        for split, path in splits.items():
            if path is not None:
                datasets[ds][split] = path
    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    global EPOCHS, BATCH_SIZE
    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch_size

    datasets = resolve_datasets(args)
    ds_names = list(datasets.keys())
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    # ── Step 1: Load all splits ───────────────────────────────────────────
    data = {}
    print("\n" + "="*54)
    print("Loading datasets...")
    for ds in ds_names:
        for split in ("train", "test"):
            path = datasets[ds][split]
            print(f"\n  {ds} {split}: {path}")
            texts, labels = load_dataset(path)
            data[f"{ds}_{split}"] = (texts, labels)

    # ── Step 2: Fine-tune one BERT per dataset ────────────────────────────
    models = {}
    print("\n" + "="*54)
    print("Fine-tuning BERT classifiers...")
    for ds in ds_names:
        texts, labels = data[f"{ds}_train"]
        models[ds] = fine_tune(texts, labels, tokenizer, device, ds)

    # ── Step 3: Transfer matrix ───────────────────────────────────────────
    print("\n" + "="*54)
    print("BERT Transfer Matrix (Macro F1)")
    header = f"{'Train→Test':<14}" + "".join(f"{d:>10}" for d in ds_names)
    print(header)
    print("-" * len(header))

    records, f1_matrix = [], {}
    for train_ds in ds_names:
        row = f"{train_ds:<14}"
        for test_ds in ds_names:
            texts, labels = data[f"{test_ds}_test"]
            test_ds_obj = GatingDataset(texts, labels, tokenizer, MAX_LEN)
            test_loader = DataLoader(test_ds_obj, batch_size=BATCH_SIZE * 2,
                                     shuffle=False, num_workers=2,
                                     pin_memory=True)
            m = evaluate(models[train_ds], test_loader, device)
            key = f"{train_ds}→{test_ds}"
            f1_matrix[key] = (m["macro_f1"], m["minority_f1"])
            row += f"{m['macro_f1']:>10.4f}"
            records.append({
                "model":       "BERT",
                "train_on":    train_ds,
                "test_on":     test_ds,
                "in_domain":   train_ds == test_ds,
                "macro_f1":    m["macro_f1"],
                "minority_f1": m["minority_f1"],
            })
        print(row)

    # ── Step 4: Comparison table ──────────────────────────────────────────
    if not args.no_comparison:
        print("\n" + "="*70)
        print("PROGRESSIVE CAPACITY CHECK — Minority F1 (DSTC→KETOD)")
        print(f"{'Transfer':<20} {'Struct LR':>10} {'Semantic':>9} {'BERT':>7}")
        print("-" * 50)
        for train_ds, test_ds in HIGHLIGHT_PAIRS:
            key = f"{train_ds}→{test_ds}"
            s_mac, s_min = STRUCTURAL_BASELINE.get(key, (None, None))
            sem_mac, sem_min = SEMANTIC_BASELINE.get(key, (None, None))
            bert_mac, bert_min = f1_matrix.get(key, (None, None))
            sem_str  = f"{sem_min:>9.4f}" if sem_min  is not None else f"{'N/A':>9}"
            bert_str = f"{bert_min:>7.4f}" if bert_min is not None else f"{'N/A':>7}"
            print(f"{key:<20} {s_min:>10.4f} {sem_str} {bert_str}")

        print("\n" + "="*70)
        print("FULL COMPARISON: Structural LR vs Semantic LR vs BERT")
        print(f"{'Transfer':<20} {'S-Mac':>7} {'S-Min':>6} "
              f"{'B-Mac':>7} {'B-Min':>6} {'ΔMin':>7}")
        print("-" * 60)
        for key, (s_mac, s_min) in STRUCTURAL_BASELINE.items():
            if key not in f1_matrix:
                continue
            b_mac, b_min = f1_matrix[key]
            train_ds, test_ds = key.split("→")
            flag = " 🔴" if train_ds != test_ds and test_ds == "KETOD" else ""
            print(f"{key:<20} {s_mac:>7.4f} {s_min:>6.4f} "
                  f"{b_mac:>7.4f} {b_min:>6.4f} {b_min-s_min:>+7.4f}{flag}")

    # ── Step 5: Save ──────────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, "bert_results.csv")
    pd.DataFrame(records).to_csv(results_path, index=False)
    print(f"\nSaved → {results_path}")

    lines = ["BERT TRANSFER — PAPER-READY SUMMARY", "=" * 54]
    for train_ds, test_ds in HIGHLIGHT_PAIRS:
        key = f"{train_ds}→{test_ds}"
        if key not in f1_matrix:
            continue
        b_mac, b_min = f1_matrix[key]
        s_mac, s_min = STRUCTURAL_BASELINE.get(key, (None, None))
        lines.append(f"\n{key}")
        lines.append(f"  BERT:          Macro={b_mac:.4f}, Minority={b_min:.4f}")
        if s_min is not None:
            lines.append(f"  Structural LR: Macro={s_mac:.4f}, Minority={s_min:.4f}")
            lines.append(f"  Δ Minority:    {b_min - s_min:+.4f}")
        if key in SEMANTIC_BASELINE:
            sem_mac, sem_min = SEMANTIC_BASELINE[key]
            lines.append(f"  Semantic LR:   Macro={sem_mac:.4f}, Minority={sem_min:.4f}")

    summary_path = os.path.join(args.output_dir, "bert_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved → {summary_path}")

    print("\n" + "="*54)
    print("Done.")


if __name__ == "__main__":
    main()
