"""
train_MHA_dstc9.py
用 DSTC9 数据训练 MHA，用于 cross-dataset transfer 实验（DSTC9 → KETOD）

使用方法（本地）：
    python mha/train_MHA_dstc9.py \
        --train_csv data/dstc9/train_features.csv \
        --val_csv   data/dstc9/test_features.csv

使用方法（AutoDL）：
    python train_MHA_dstc9.py \
        --train_csv /root/autodl-tmp/dstc9/train_features.csv \
        --val_csv   /root/autodl-tmp/dstc9/test_features.csv

产出：
    outputs/MHA-dstc9/MHA_dstc9_e{N}_f1{score}.pt
    outputs/MHA-dstc9/best_ckpt.txt
"""

import os, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from collections import Counter

# ── 参数 ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--train_csv",  default="data/dstc9/train_features.csv")
parser.add_argument("--val_csv",    default="data/dstc9/test_features.csv")
parser.add_argument("--output_dir", default="outputs/MHA-dstc9")
parser.add_argument("--loss",       default="weighted", choices=["ce", "weighted"])
parser.add_argument("--epochs",     type=int, default=50)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--patience",   type=int, default=10)
parser.add_argument("--seed",       type=int, default=42)
args = parser.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 特征列（与 KETOD/DSTC9/DSTC11 test_features.csv 完全一致）────────────────
FEATURE_COLS = [
    "turn_position_ratio", "prev_sys_is_question", "user_has_question",
    "user_starts_question_word", "user_turn_len_log", "sys_turn_len_log",
    "dialogue_len_log", "consecutive_sys_turns", "turn_len_ratio",
    "turn_position_squared",
]

def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    return X, y

print("Loading DSTC9 train …")
X_train, y_train = load_data(args.train_csv)
print(f"  Train: {X_train.shape}, label dist: {Counter(y_train)}")

print("Loading DSTC9 val …")
X_val, y_val = load_data(args.val_csv)
print(f"  Val:   {X_val.shape}, label dist: {Counter(y_val)}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(TabularDataset(X_train, y_train),
                          batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(TabularDataset(X_val, y_val),
                          batch_size=256, shuffle=False)

# ── MHA 模型 ──────────────────────────────────────────────────────────────────
class MHAClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc   = nn.Linear(embed_dim, 2)
    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        x, _ = self.attn(x, x, x)
        return self.fc(self.norm(x).squeeze(1))

model = MHAClassifier(len(FEATURE_COLS)).to(DEVICE)

# ── Loss ──────────────────────────────────────────────────────────────────────
counts = Counter(y_train)
total  = sum(counts.values())
weights = torch.tensor([total / counts[i] for i in range(2)],
                       dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights) if args.loss == "weighted" \
            else nn.CrossEntropyLoss()
print(f"Loss: {args.loss}, weights: {weights.tolist()}")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# ── 训练循环 ──────────────────────────────────────────────────────────────────
best_f1, best_ckpt, no_improve = 0.0, None, 0

for epoch in range(1, args.epochs + 1):
    model.train()
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        criterion(model(X_b.to(DEVICE)), y_b.to(DEVICE)).backward()
        optimizer.step()

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            preds.extend(model(X_b.to(DEVICE)).argmax(1).cpu().tolist())
            labels.extend(y_b.tolist())

    macro_f1    = f1_score(labels, preds, average="macro")
    minority_f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    print(f"Epoch {epoch:3d} | Macro F1={macro_f1:.4f} | Minority F1={minority_f1:.4f}")

    if macro_f1 > best_f1:
        best_f1   = macro_f1
        best_ckpt = os.path.join(args.output_dir,
                                 f"MHA_dstc9_e{epoch}_f1{macro_f1:.4f}.pt")
        torch.save(model.state_dict(), best_ckpt)
        no_improve = 0
        print(f"  ✅ Saved → {best_ckpt}")
    else:
        no_improve += 1
        if no_improve >= args.patience:
            print(f"Early stop at epoch {epoch}")
            break

with open(os.path.join(args.output_dir, "best_ckpt.txt"), "w") as f:
    f.write(best_ckpt)
print(f"\nBest Macro F1={best_f1:.4f} | {best_ckpt}")
