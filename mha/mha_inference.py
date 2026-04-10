# ── CONFIG ────────────────────────────────────
TRAIN_CSV   = "data/ketod/train_full.csv"
TEST_CSV    = "data/ketod/test_full.csv"
CHECKPOINT  = "outputs/MHA-trained/MHA_weighted_e35_f10.6139.pt"
OUTPUT_CSV  = "results/mha_predictions.csv"
# ──────────────────────────────────────────────
"""
mha_inference.py  —  Step 6 前置
用训练好的 MHA checkpoint 对 KETOD test set 做推理，
输出每个样本的预测结果，用于和 LR 做 agreement 分析。

在 AutoDL 上运行（有 checkpoint 的机器）：
    python mha_inference.py

输出：
    mha_predictions.csv   — 含 label, mha_pred, mha_prob_0, mha_prob_1
"""

import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm

# ─────────────────────────────────────────────
# 路径配置（AutoDL 上的路径）
# ─────────────────────────────────────────────
TRAIN_CSV    = "E:/ketod-main/ketod_release/train_full.csv"
TEST_CSV     = "E:/ketod-main/ketod_release/test_full.csv"
CHECKPOINT   = "E:/MHA_weighted_e35_f10.6139.pt"
OUTPUT_CSV   = "mha_predictions.csv"

# 模型超参（与训练时一致）
NUM_LAYERS = 7       # 你说的 7 层
EMBED_DIM  = 64
NUM_HEADS  = 4
HIDDEN_DIM = 64
NUM_CLASSES = 2
BATCH_SIZE  = 256

# ─────────────────────────────────────────────
# 复用 train_MHA.py 的组件
# ─────────────────────────────────────────────

def get_tokenizer():
    return lambda x: re.findall(r"\b\w+\b", x.lower())

class Vocab:
    def __init__(self, counter, specials=["<unk>"]):
        self.itos = specials + [w for w, _ in counter.most_common()]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.default_index = 0

    def __call__(self, tokens):
        return [self.stoi.get(t, self.default_index) for t in tokens]

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)

    def set_default_index(self, idx):
        self.default_index = idx

def build_vocab_from_iterator(iterator, specials=["<unk>"]):
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    return Vocab(counter, specials)


class TextDataset(Dataset):
    def __init__(self, csv_path):
        self.data = load_dataset('csv', data_files=csv_path, split='train')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx]['output']
        text  = self.data[idx]['input']
        return label, text


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        b = query.shape[0]
        q = self.W_q(query).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(key).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(value).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out  = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, -1, self.embed_dim)
        return self.W_o(out)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :].to(x.device)


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderCell(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward_network = FeedForwardNetwork(embed_dim, hidden_dim, dropout)
        self.norm_attention = nn.LayerNorm(embed_dim)
        self.norm_ffn       = nn.LayerNorm(embed_dim)
        self.dropout        = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm_attention(x + self.dropout(self.multi_head_attention(x, x, x, mask)))
        return self.norm_ffn(x + self.dropout(self.feed_forward_network(x)))


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.encoder_cells = nn.ModuleList([
            TransformerEncoderCell(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        for cell in self.encoder_cells:
            x = cell(x, mask)
        return self.norm(x)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_layers, embed_dim, num_heads,
                 hidden_dim, num_classes, dropout=0.1, pad_token=0):
        super().__init__()
        self.embedding         = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.encoder           = TransformerEncoder(num_layers, embed_dim, num_heads, hidden_dim, dropout)
        self.fc                = nn.Linear(embed_dim, num_classes)
        self.embed_dim         = embed_dim

    def forward(self, text, mask=None):
        x = self.positional_encoding(self.embedding(text) * math.sqrt(self.embed_dim))
        x = self.encoder(x, mask)
        return self.fc(torch.mean(x, dim=1))


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = get_tokenizer()

    # 重建 vocab（必须和训练时完全一致）
    print("Building vocab from train_full.csv ...")
    train_ds = TextDataset(TRAIN_CSV)
    counter  = Counter()
    for _, text in tqdm(train_ds, desc="Tokenizing"):
        counter.update(tokenizer(str(text)))
    vocab = Vocab(counter, specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    print(f"Vocab size: {len(vocab)}")

    PAD_IDX = vocab["<unk>"]

    # collate
    def collate_fn(batch):
        label_list, text_list = [], []
        for label, text in batch:
            label_val = 1 if str(label).strip() == "True" else 0
            label_list.append(label_val)
            tokens = torch.tensor(vocab(tokenizer(str(text))), dtype=torch.int64)
            text_list.append(tokens)
        labels = torch.tensor(label_list, dtype=torch.int64)
        texts  = torch.nn.utils.rnn.pad_sequence(
            text_list, batch_first=True, padding_value=PAD_IDX
        )
        return labels, texts

    # 加载模型
    print(f"Loading checkpoint: {CHECKPOINT}")
    model = TransformerClassifier(
        vocab_size  = len(vocab),
        num_layers  = NUM_LAYERS,
        embed_dim   = EMBED_DIM,
        num_heads   = NUM_HEADS,
        hidden_dim  = HIDDEN_DIM,
        num_classes = NUM_CLASSES,
    ).to(device)

    state_dict = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded.")

    # 推理
    test_ds = TextDataset(TEST_CSV)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn
    )

    all_labels, all_preds, all_prob0, all_prob1 = [], [], [], []

    with torch.no_grad():
        for labels, texts in tqdm(test_loader, desc="Inference"):
            texts  = texts.to(device)
            logits = model(texts, mask=None)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_prob0.extend(probs[:, 0])
            all_prob1.extend(probs[:, 1])

    # 保存
    df = pd.DataFrame({
        "label":    all_labels,
        "mha_pred": all_preds,
        "mha_prob_0": np.round(all_prob0, 6),
        "mha_prob_1": np.round(all_prob1, 6),
    })
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}  ({len(df)} rows)")

    # 简单评估确认
    from sklearn.metrics import classification_report, f1_score
    print("\n===== MHA Test Set Performance =====")
    print(classification_report(all_labels, all_preds, digits=4))
    print(f"Macro F1: {f1_score(all_labels, all_preds, average='macro'):.4f}")


if __name__ == "__main__":
    main()
