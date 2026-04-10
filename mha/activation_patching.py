# ── CONFIG ────────────────────────────────────
TRAIN_CSV   = "data/ketod/train_full.csv"
TEST_CSV    = "data/ketod/test_full.csv"
CHECKPOINT  = "outputs/MHA-trained/MHA_weighted_e35_f10.6139.pt"
# ──────────────────────────────────────────────
"""
Activation Patching 实验 —— 定位 RAGate 门控因果层
逻辑：
  1. 取正样本（enrich=True）在每一层的激活
  2. 跑负样本（enrich=False）的前向，在指定层把激活替换成正样本的
  3. 看 Class-1 logit 的变化（Δ Logit）
  4. 对照组：用随机配对重跑，验证结论是否依赖等长配对

用法：
    python activation_patching_v2.py \
        --checkpoint "../../outputs/MHA-trained/MHA_focal_e20_f10.4575.pt" \
        --train_csv  ../../data/lm_finetune_data/split_train.csv \
        --test_csv   ../../data/lm_finetune_data/split_test.csv \
        --num_layers 7 --n_pairs 100
"""
import math, argparse, re, random
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# ──────────────────────────────────────────────
# 参数
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--train_csv",  required=True)
parser.add_argument("--test_csv",   required=True)
parser.add_argument("--num_heads",  type=int, default=4)
parser.add_argument("--num_layers", type=int, default=5)
parser.add_argument("--embed_dim",  type=int, default=64)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_pairs",    type=int, default=100)
args = parser.parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ──────────────────────────────────────────────
# Tokenizer & Vocab（与 train_MHA.py 一致）
# ──────────────────────────────────────────────
def get_tokenizer():
    return lambda x: re.findall(r"\b\w+\b", x.lower())

class Vocab:
    def __init__(self, counter, specials=["<unk>"]):
        self.itos = specials + [w for w, _ in counter.most_common()]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.default_index = 0
    def __call__(self, tokens):
        return [self.stoi.get(t, self.default_index) for t in tokens]
    def __len__(self): return len(self.itos)
    def __getitem__(self, token): return self.stoi.get(token, self.default_index)
    def set_default_index(self, idx): self.default_index = idx

def build_vocab(iterator, specials=["<unk>"]):
    counter = Counter()
    for tokens in iterator: counter.update(tokens)
    return Vocab(counter, specials)

class TextClassificationDataset(Dataset):
    def __init__(self, data_dir):
        self.data = load_dataset('csv', data_files=data_dir, split='train')
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]['output'], self.data[idx]['input']

tokenizer = get_tokenizer()
print("Building vocab...")
train_iter = TextClassificationDataset(args.train_csv)
vocab = build_vocab((tokenizer(str(t)) for _, t in train_iter))
vocab.set_default_index(vocab["<unk>"])
PAD_IDX = vocab["<unk>"]
print(f"Vocab size: {len(vocab)}")

def collate_batch(batch):
    labels, texts = [], []
    for l, t in batch:
        labels.append(1 if str(l).strip() == "True" else 0)
        texts.append(torch.tensor(vocab(tokenizer(str(t))), dtype=torch.int64))
    return (torch.tensor(labels, dtype=torch.int64),
            torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=PAD_IDX))

# ──────────────────────────────────────────────
# 模型定义（与 train_MHA.py 一字不差）
# ──────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
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
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, -1, self.embed_dim)
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
    def forward(self, x): return x + self.pe[:, :x.shape[1], :].to(x.device)

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerEncoderCell(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward_network = FeedForwardNetwork(embed_dim, hidden_dim, dropout)
        self.norm_attention = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
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
        for cell in self.encoder_cells: x = cell(x, mask)
        return self.norm(x)

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_layers, embed_dim, num_heads, hidden_dim,
                 num_classes, dropout=0.1, pad_token=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, hidden_dim, dropout)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim
    def forward(self, text, mask=None):
        x = self.positional_encoding(self.embedding(text) * math.sqrt(self.embed_dim))
        x = self.encoder(x, mask)
        return self.fc(torch.mean(x, dim=1))

# ──────────────────────────────────────────────
# 加载模型
# ──────────────────────────────────────────────
print(f"\n加载 checkpoint: {args.checkpoint}")
state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]

CKPT_VOCAB_SIZE = state["embedding.weight"].shape[0]
print(f"Checkpoint vocab size: {CKPT_VOCAB_SIZE}")

model = TransformerClassifier(
    CKPT_VOCAB_SIZE, args.num_layers, args.embed_dim,
    args.num_heads, args.embed_dim, 2
).to(DEVICE)
model.load_state_dict(state)
model.eval()
print("模型加载成功")

# ──────────────────────────────────────────────
# 核心函数
# ──────────────────────────────────────────────
@torch.no_grad()
def get_all_layer_acts(input_ids):
    """
    跑一次完整前向，返回每层输出的激活列表。
    acts[i] shape: [1, seq_len, embed_dim]
    """
    x = model.positional_encoding(model.embedding(input_ids) * math.sqrt(model.embed_dim))
    acts = []
    for cell in model.encoder.encoder_cells:
        x = cell(x, mask=None)
        acts.append(x.clone())  # 存干净的激活，不受后续层影响
    return acts

@torch.no_grad()
def run_with_patch(input_ids, patch_layer, patch_act):
    """
    跑前向，在 patch_layer 层输出后把激活替换成 patch_act，
    然后继续用替换后的激活跑后续层。
    返回 Class-1 logit。
    """
    x = model.positional_encoding(model.embedding(input_ids) * math.sqrt(model.embed_dim))
    for i, cell in enumerate(model.encoder.encoder_cells):
        x = cell(x, mask=None)
        if i == patch_layer:
            # 替换：处理正负样本 seq_len 不同的情况
            neg_len = x.shape[1]
            pos_len = patch_act.shape[1]
            min_len = min(neg_len, pos_len)
            # 只替换有效长度范围内的激活
            x = x.clone()
            x[:, :min_len, :] = patch_act[:, :min_len, :]
    x = model.encoder.norm(x)
    logit = model.fc(torch.mean(x, dim=1))[0, 1].item()
    return logit

@torch.no_grad()
def run_clean(input_ids):
    """跑干净的前向，返回 Class-1 logit。"""
    out = model(input_ids, mask=None)
    return out[0, 1].item()

# ──────────────────────────────────────────────
# 加载测试集，分正负样本
# ──────────────────────────────────────────────
test_loader = DataLoader(
    TextClassificationDataset(args.test_csv),
    batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch
)
all_labels, all_texts = [], []
for l, t in test_loader:
    all_labels.extend(l.tolist())
    all_texts.extend(t.tolist())

# all_texts 里每条是 padded 的列表，去掉 padding 还原原始长度
def strip_padding(t):
    t = list(t)
    while t and t[-1] == PAD_IDX:
        t.pop()
    return t

pos_texts = [strip_padding(t) for t, l in zip(all_texts, all_labels) if l == 1]
neg_texts = [strip_padding(t) for t, l in zip(all_texts, all_labels) if l == 0]
print(f"\n正样本数: {len(pos_texts)}, 负样本数: {len(neg_texts)}")

# ──────────────────────────────────────────────
# 构造配对（只用随机配对，等长配对全量替换导致各层Δ相同，无法区分因果层）
# ──────────────────────────────────────────────
random.seed(42)
neg_shuffled = neg_texts.copy()
random.shuffle(neg_shuffled)
n_rand = min(args.n_pairs, len(pos_texts), len(neg_shuffled))
rand_pairs = list(zip(pos_texts[:n_rand], neg_shuffled[:n_rand]))
print(f"随机配对: {len(rand_pairs)} 对")

# ──────────────────────────────────────────────
# Activation Patching 主函数
# ──────────────────────────────────────────────
def run_patching(pairs, label=""):
    """
    对给定的 pairs 跑 Activation Patching。
    每对 (pos, neg)：
      - 预计算 pos 在每层的激活
      - 对每层，把 pos 的激活注入 neg 的前向，计算 Δ Logit
    返回 {layer: [delta1, delta2, ...]}
    """
    layer_deltas = {i: [] for i in range(args.num_layers)}
    orig_preds = [1 if run_clean(torch.tensor([n], dtype=torch.int64).to(DEVICE)) > 0 else 0 for _, n in pairs]
    print(f"负样本中被预测为 Class-1 的比例: {sum(orig_preds)/len(orig_preds):.1%}")
    for idx, (p, n) in enumerate(pairs):
        p_in = torch.tensor([p], dtype=torch.int64).to(DEVICE)
        n_in = torch.tensor([n], dtype=torch.int64).to(DEVICE)

        # 1. 获取正样本各层激活（干净，不互相影响）
        pos_acts = get_all_layer_acts(p_in)

        # 2. 获取负样本原始 logit
        orig_logit = run_clean(n_in)

        # 3. 逐层 patch
        for layer in range(args.num_layers):
            patched_logit = run_with_patch(n_in, layer, pos_acts[layer])
            layer_deltas[layer].append(patched_logit - orig_logit)

    print(f"\n{'='*60}")
    print(f"Activation Patching {label}")
    print(f"{'='*60}")
    for layer in range(args.num_layers):
        mean_d = np.mean(layer_deltas[layer])
        std_d  = np.std(layer_deltas[layer])
        print(f"  Layer {layer} → Mean Δ Logit: {mean_d:+.4f} ± {std_d:.4f}")

    return layer_deltas

# 只跑随机配对
rand_deltas = run_patching(rand_pairs, label="（随机配对）")

# ──────────────────────────────────────────────
# 对比结论
# ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("层级模式解读")
print(f"{'='*60}")
print("""
  某层 Δ 明显高于其他层 → 该层是因果层，门控决策在此形成
  所有层 Δ 单调递减     → 信号在最早层形成，后续层线性传递
  所有层 Δ 接近 0       → Patching 无效，模型对此类替换不敏感
""")
best_layer = max(range(args.num_layers), key=lambda l: np.mean(rand_deltas[l]))
print(f"最强响应层 = Layer {best_layer}  "
      f"(Δ = {np.mean(rand_deltas[best_layer]):+.4f})")
