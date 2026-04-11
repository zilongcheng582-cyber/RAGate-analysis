"""
mha_inference_dstc9_on_ketod.py
用 DSTC9 训练的 MHA checkpoint，在 KETOD test 上推理
产出：results/mha_dstc9_on_ketod_predictions.csv

使用方法（本地）：
    python mha/mha_inference_dstc9_on_ketod.py \
        --test_csv  data/ketod/test_features.csv \
        --checkpoint outputs/MHA-dstc9/best_ckpt.txt

使用方法（AutoDL）：
    python mha_inference_dstc9_on_ketod.py \
        --test_csv  /root/autodl-tmp/ketod/test_features.csv \
        --checkpoint /root/autodl-tmp/outputs/MHA-dstc9/best_ckpt.txt
"""

import os, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

# ── 参数 ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--test_csv",    default="data/ketod/test_features.csv")
parser.add_argument("--checkpoint",  default="outputs/MHA-dstc9/best_ckpt.txt",
                    help="checkpoint 路径或 best_ckpt.txt 路径")
parser.add_argument("--output_csv",  default="results/mha_dstc9_on_ketod_predictions.csv")
parser.add_argument("--embed_dim",   type=int, default=64)
parser.add_argument("--num_heads",   type=int, default=4)
parser.add_argument("--batch_size",  type=int, default=256)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 特征列（与训练一致）──────────────────────────────────────────────────────
FEATURE_COLS = [
    "turn_position_ratio", "prev_sys_is_question", "user_has_question",
    "user_starts_question_word", "user_turn_len_log", "sys_turn_len_log",
    "dialogue_len_log", "consecutive_sys_turns", "turn_len_ratio",
    "turn_position_squared",
]

# ── Checkpoint 路径（支持直接传 .pt 或 best_ckpt.txt）───────────────────────
ckpt_path = args.checkpoint
if ckpt_path.endswith(".txt"):
    assert os.path.exists(ckpt_path), f"best_ckpt.txt not found: {ckpt_path}"
    with open(ckpt_path) as f:
        ckpt_path = f.read().strip()
assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
print(f"Checkpoint: {ckpt_path}")

# ── 模型（与训练一致）────────────────────────────────────────────────────────
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

model = MHAClassifier(len(FEATURE_COLS), args.embed_dim, args.num_heads).to(DEVICE)
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
model.eval()

# ── 加载 KETOD test ───────────────────────────────────────────────────────────
df = pd.read_csv(args.test_csv)
df.columns = [c.strip().lower() for c in df.columns]
X = torch.tensor(df[FEATURE_COLS].values.astype("float32"))
print(f"KETOD test: {X.shape}")

# ── 推理 ──────────────────────────────────────────────────────────────────────
class DS(torch.utils.data.Dataset):
    def __init__(self, X): self.X = X
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i]

preds, probs = [], []
with torch.no_grad():
    for X_b in DataLoader(DS(X), batch_size=args.batch_size):
        logits = model(X_b.to(DEVICE))
        preds.extend(logits.argmax(1).cpu().tolist())
        probs.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())

# ── 保存 ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
df["mha_pred"]     = preds
df["mha_prob_pos"] = probs
df.to_csv(args.output_csv, index=False)
print(f"Saved → {args.output_csv}")

# ── 评估（需要 label 列）─────────────────────────────────────────────────────
if "label" in df.columns:
    y = df["label"].values
    p = np.array(preds)
    print(classification_report(y, p, digits=4))
    print(f"Macro F1:    {f1_score(y, p, average='macro'):.4f}")
    print(f"Minority F1: {f1_score(y, p, pos_label=1, zero_division=0):.4f}")
