# ── CONFIG ────────────────────────────────────
KETOD_TRAIN    = "data/ketod/train_features.csv"
KETOD_TEST     = "data/ketod/test_features.csv"
TRAIN_FULL_CSV = "data/ketod/train_full.csv"
TEST_FULL_CSV  = "data/ketod/test_full.csv"
CHECKPOINT     = "outputs/MHA-trained/MHA_weighted_e35_f10.6139.pt"
MHA_PRED_CSV   = "results/mha_predictions.csv"
# ──────────────────────────────────────────────
"""
counterfactual_analysis.py  —  补强实验3
反事实扰动：把 user_has_question 强制翻转（0→1, 1→0），
观察 LR 和 MHA 的预测是否 flip，量化两个模型对该特征的依赖程度。

前置条件：
    - E:/ketod-main/ketod_release/train_features.csv
    - E:/ketod-main/ketod_release/test_features.csv
    - mha_predictions.csv（agreement 分析时生成的）
    - E:/ketod-main/ketod_release/test_full.csv（用于 MHA 反事实推理）
    - MHA checkpoint（本地）

Usage:
    python counterfactual_analysis.py

输出：
    counterfactual_results.csv   — 逐样本翻转结果
    counterfactual_summary.txt   — 论文用数字
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.makedirs("C:/Temp", exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = "C:/Temp"

import math, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from tqdm import tqdm

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────
KETOD_TRAIN   = "E:/ketod-main/ketod_release/train_features.csv"
KETOD_TEST    = "E:/ketod-main/ketod_release/test_features.csv"
TEST_FULL_CSV = "E:/ketod-main/ketod_release/test_full.csv"
TRAIN_FULL_CSV= "E:/ketod-main/ketod_release/train_full.csv"
CHECKPOINT    = "E:/MHA_weighted_e35_f10.6139.pt"
MHA_PRED_CSV  = "mha_predictions.csv"

LABEL_COL = "label"
ALL_FEATURES = [
    "turn_position_ratio",
    "prev_sys_is_question",
    "user_has_question",        # ← 主要扰动目标
    "user_starts_question_word",
    "user_turn_len_log",
    "sys_turn_len_log",
    "dialogue_len_log",
    "consecutive_sys_turns",
    "turn_len_ratio",
    "turn_position_squared",
]

# 额外也扰动 position 特征做对比
PERTURB_TARGETS = {
    "user_has_question":       "flip",   # 0↔1
    "user_starts_question_word": "flip", # 0↔1
    "turn_position_ratio":     "zero",   # 置0（模拟对话最开始）
}

C_GRID = [0.01, 0.1, 1.0, 10.0]
RANDOM_STATE = 42
NUM_LAYERS = 7
EMBED_DIM  = 64
NUM_HEADS  = 4
HIDDEN_DIM = 64
BATCH_SIZE = 256


# ─────────────────────────────────────────────
# MHA 组件（复用）
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

class TextDataset(Dataset):
    def __init__(self, csv_path):
        self.data = load_dataset('csv', data_files=csv_path, split='train')
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]['output'], self.data[idx]['input']

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
        self.embedding           = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.encoder             = TransformerEncoder(num_layers, embed_dim, num_heads, hidden_dim, dropout)
        self.fc                  = nn.Linear(embed_dim, num_classes)
        self.embed_dim           = embed_dim
    def forward(self, text, mask=None):
        x = self.positional_encoding(self.embedding(text) * math.sqrt(self.embed_dim))
        x = self.encoder(x, mask)
        return self.fc(torch.mean(x, dim=1))


# ─────────────────────────────────────────────
# LR 工具
# ─────────────────────────────────────────────

def train_lr(train_df):
    X = train_df[ALL_FEATURES].values.astype(float)
    y = train_df[LABEL_COL].values.astype(int)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight="balanced", max_iter=1000,
            random_state=RANDOM_STATE, solver="lbfgs",
        )),
    ])
    gs = GridSearchCV(pipe, {"lr__C": C_GRID},
                      cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
                      scoring="f1_macro", n_jobs=1)
    gs.fit(X, y)
    best = gs.best_estimator_
    best.fit(X, y)
    print(f"LR best_C = {gs.best_params_['lr__C']}")
    return best


def lr_predict(model, df):
    X = df[ALL_FEATURES].values.astype(float)
    return model.predict(X)


# ─────────────────────────────────────────────
# MHA 推理（直接用文本，不改特征）
# 注意：MHA 的反事实无法通过改特征实现，
# 只能通过修改原始文本中的问号来模拟
# ─────────────────────────────────────────────

def load_mha_model(vocab_size, device):
    model = TransformerClassifier(
        vocab_size=vocab_size, num_layers=NUM_LAYERS,
        embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM, num_classes=2,
    ).to(device)
    state_dict = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def mha_predict_texts(model, texts, vocab, tokenizer, device):
    """给定文本列表，返回 MHA 预测"""
    PAD_IDX = vocab["<unk>"]
    token_seqs = [torch.tensor(vocab(tokenizer(str(t))), dtype=torch.int64) for t in texts]
    padded = torch.nn.utils.rnn.pad_sequence(token_seqs, batch_first=True, padding_value=PAD_IDX)

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(padded), BATCH_SIZE):
            batch = padded[i:i+BATCH_SIZE].to(device)
            logits = model(batch, mask=None)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)


def add_question_mark(text):
    """在 user turn 最后加问号（模拟 user_has_question=1）"""
    text = str(text)
    # 找最后一个 user 发言，加问号
    if not text.rstrip().endswith("?"):
        return text.rstrip() + "?"
    return text


def remove_question_mark(text):
    """去掉文本中所有问号（模拟 user_has_question=0）"""
    return str(text).replace("?", ".")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载数据
    train_df = pd.read_csv(KETOD_TRAIN)
    test_df  = pd.read_csv(KETOD_TEST)
    mha_df   = pd.read_csv(MHA_PRED_CSV)
    test_full_df = pd.read_csv(TEST_FULL_CSV)

    label       = test_df[LABEL_COL].values.astype(int)
    mha_orig    = mha_df["mha_pred"].values.astype(int)

    # ── LR 训练 ──
    print("\nTraining LR...")
    lr_model = train_lr(train_df)
    lr_orig  = lr_predict(lr_model, test_df)

    # ── MHA vocab 重建 ──
    print("\nBuilding vocab...")
    tokenizer = get_tokenizer()
    train_full_ds = TextDataset(TRAIN_FULL_CSV)
    counter = Counter()
    for _, text in tqdm(train_full_ds, desc="Tokenizing"):
        counter.update(tokenizer(str(text)))
    vocab = Vocab(counter, specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    print(f"Vocab size: {len(vocab)}")

    # ── MHA 模型加载 ──
    print("\nLoading MHA model...")
    mha_model = load_mha_model(len(vocab), device)

    # 原始文本
    orig_texts = test_full_df["input"].tolist()

    records = []

    # ── 对每个扰动目标做实验 ──
    for feat_name, perturb_type in PERTURB_TARGETS.items():
        print(f"\n{'='*60}")
        print(f"Perturbing: {feat_name} ({perturb_type})")
        print(f"{'='*60}")

        # LR 反事实：直接改特征值
        test_cf = test_df.copy()
        if perturb_type == "flip":
            test_cf[feat_name] = 1 - test_cf[feat_name]
        elif perturb_type == "zero":
            test_cf[feat_name] = 0.0

        lr_cf = lr_predict(lr_model, test_cf)

        # LR flip rate
        lr_flip_mask = (lr_orig != lr_cf)
        lr_flip_rate = lr_flip_mask.mean()
        lr_flip_pos  = lr_flip_mask[label == 1].mean()
        lr_flip_neg  = lr_flip_mask[label == 0].mean()

        print(f"  LR flip rate (overall): {lr_flip_rate:.4f}")
        print(f"  LR flip rate | label=1: {lr_flip_pos:.4f}")
        print(f"  LR flip rate | label=0: {lr_flip_neg:.4f}")

        # MHA 反事实：只对 question 特征做文本级扰动
        if feat_name == "user_has_question":
            # 原本没问号的加问号，原本有问号的去问号
            cf_texts = []
            for i, text in enumerate(orig_texts):
                orig_val = test_df["user_has_question"].iloc[i]
                if orig_val == 0:
                    cf_texts.append(add_question_mark(text))
                else:
                    cf_texts.append(remove_question_mark(text))

            print("  Running MHA counterfactual inference...")
            mha_cf = mha_predict_texts(mha_model, cf_texts, vocab, tokenizer, device)

            mha_flip_mask = (mha_orig != mha_cf)
            mha_flip_rate = mha_flip_mask.mean()
            mha_flip_pos  = mha_flip_mask[label == 1].mean()
            mha_flip_neg  = mha_flip_mask[label == 0].mean()

            print(f"  MHA flip rate (overall): {mha_flip_rate:.4f}")
            print(f"  MHA flip rate | label=1: {mha_flip_pos:.4f}")
            print(f"  MHA flip rate | label=0: {mha_flip_neg:.4f}")

            # 两者都 flip 的比例
            both_flip = (lr_flip_mask & mha_flip_mask).mean()
            print(f"  Both LR & MHA flip:      {both_flip:.4f}")

            records.append({
                "feature":         feat_name,
                "perturb_type":    perturb_type,
                "lr_flip_rate":    round(lr_flip_rate, 4),
                "lr_flip_pos":     round(lr_flip_pos, 4),
                "lr_flip_neg":     round(lr_flip_neg, 4),
                "mha_flip_rate":   round(mha_flip_rate, 4),
                "mha_flip_pos":    round(mha_flip_pos, 4),
                "mha_flip_neg":    round(mha_flip_neg, 4),
                "both_flip":       round(both_flip, 4),
            })
        else:
            # position/其他特征：MHA 无法直接做文本级扰动，只报 LR
            records.append({
                "feature":         feat_name,
                "perturb_type":    perturb_type,
                "lr_flip_rate":    round(lr_flip_rate, 4),
                "lr_flip_pos":     round(lr_flip_pos, 4),
                "lr_flip_neg":     round(lr_flip_neg, 4),
                "mha_flip_rate":   "N/A",
                "mha_flip_pos":    "N/A",
                "mha_flip_neg":    "N/A",
                "both_flip":       "N/A",
            })

    # 保存
    df_out = pd.DataFrame(records)
    df_out.to_csv("counterfactual_results.csv", index=False)
    print(f"\nSaved → counterfactual_results.csv")

    # 论文用摘要
    print("\n" + "="*60)
    print("PAPER-READY SUMMARY")
    print("="*60)
    for _, row in df_out.iterrows():
        print(f"\n  Feature: {row['feature']} ({row['perturb_type']})")
        print(f"  LR  flip rate: {row['lr_flip_rate']}")
        print(f"  MHA flip rate: {row['mha_flip_rate']}")
        print(f"  Both flip:     {row['both_flip']}")

    with open("counterfactual_summary.txt", "w") as f:
        f.write(df_out.to_string())
    print("\nSaved → counterfactual_summary.txt")


if __name__ == "__main__":
    main()
