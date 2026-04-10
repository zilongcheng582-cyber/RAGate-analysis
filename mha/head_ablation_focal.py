# ── CONFIG ────────────────────────────────────
TRAIN_CSV   = "data/ketod/train_full.csv"
TEST_CSV    = "data/ketod/test_full.csv"
CHECKPOINT  = "outputs/MHA-trained/MHA_weighted_e35_f10.6139.pt"
# ──────────────────────────────────────────────
"""
Head Ablation 实验 —— 适配 Focal Loss Checkpoint (1/10 数据, ~3380 vocab)
用法：python head_ablation_focal.py --checkpoint "路径" --train_csv "路径" --test_csv "路径"
"""
import math, copy, argparse, re
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.metrics import classification_report, recall_score

# ──────────────────────────────────────────────
# 1. 参数与设备
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--train_csv", required=True)
parser.add_argument("--test_csv", required=True)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--num_layers", type=int, default=5)
parser.add_argument("--embed_dim", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# 2. 数据与 Vocab (与 train_MHA.py 完全一致)
# ──────────────────────────────────────────────
def get_tokenizer(mode="basic_english"):
    return lambda x: re.findall(r"\b\w+\b", x.lower())

class Vocab:
    def __init__(self, counter, specials=["<unk>"]):
        self.itos = specials + [w for w, _ in counter.most_common()]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.default_index = 0
    def __call__(self, tokens): return [self.stoi.get(t, self.default_index) for t in tokens]
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
    def __getitem__(self, idx): return self.data[idx]['output'], self.data[idx]['input']

tokenizer = get_tokenizer()
train_iter = TextClassificationDataset(args.train_csv)
vocab = build_vocab((tokenizer(str(t)) for _, t in train_iter))
vocab.set_default_index(vocab["<unk>"])
PAD_IDX = vocab["<unk>"]

def collate_batch(batch):
    labels, texts = [], []
    for l, t in batch:
        labels.append(1 if str(l).strip() == "True" else 0)
        texts.append(torch.tensor(vocab(tokenizer(str(t))), dtype=torch.int64))
    return torch.tensor(labels, dtype=torch.int64), torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)

# ──────────────────────────────────────────────
# 3. 模型定义 (1:1 复制自 train_MHA.py)
# ──────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
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
        self.multi_head_attention  = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward_network  = FeedForwardNetwork(embed_dim, hidden_dim, dropout)
        self.norm_attention        = nn.LayerNorm(embed_dim)
        self.norm_ffn              = nn.LayerNorm(embed_dim)
        self.dropout               = nn.Dropout(dropout)
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
        self.embedding           = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.encoder             = TransformerEncoder(num_layers, embed_dim, num_heads, hidden_dim, dropout)
        self.fc                  = nn.Linear(embed_dim, num_classes)
        self.embed_dim           = embed_dim
    def forward(self, text, mask=None):
        x = self.positional_encoding(self.embedding(text) * math.sqrt(self.embed_dim))
        x = self.encoder(x, mask)
        return self.fc(torch.mean(x, dim=1))

# ──────────────────────────────────────────────
# 4. 加载 Focal Loss Checkpoint (自动识别 vocab)
# ──────────────────────────────────────────────
print(f"📥 加载 checkpoint: {args.checkpoint}")
state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]

# ✅ 自动提取词表大小，兼容 1/10 数据训练
CKPT_VOCAB_SIZE = state["embedding.weight"].shape[0]
print(f"✅ 自动识别词表大小: {CKPT_VOCAB_SIZE}")

model = TransformerClassifier(CKPT_VOCAB_SIZE, args.num_layers, args.embed_dim, 
                              args.num_heads, args.embed_dim, 2).to(DEVICE)
model.load_state_dict(state)
model.eval()
print("✅ 模型完整加载成功 (Focal Loss 权重已恢复)")

# ──────────────────────────────────────────────
# 5. 测试集 DataLoader
# ──────────────────────────────────────────────
test_iter = TextClassificationDataset(args.test_csv)
test_loader = DataLoader(test_iter, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

def run_inference(m):
    m.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for labels, texts in test_loader:
            labels, texts = labels.to(DEVICE), texts.to(DEVICE)
            preds = m(texts, mask=None).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return all_labels, all_preds

# ──────────────────────────────────────────────
# 6. Baseline
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("BASELINE (Focal Loss 完整模型)")
print("="*60)
gt_labels, base_preds = run_inference(model)
base_recall = recall_score(gt_labels, base_preds, pos_label=1)
print(classification_report(gt_labels, base_preds, target_names=["Class-0", "Class-1"]))
print(f"Class-1 Recall = {base_recall:.4f}")

# ──────────────────────────────────────────────
# 7. Head Ablation (W_o pre-hook 零化指定 Head)
# ──────────────────────────────────────────────
head_dim = args.embed_dim // args.num_heads
results = {}

print("\n" + "="*60)
print("HEAD ABLATION (Focal Loss 模型)")
print("="*60)

for ablate_head in range(args.num_heads):
    ablated = copy.deepcopy(model).to(DEVICE)
    hooks = []
    start = ablate_head * head_dim
    end = (ablate_head + 1) * head_dim

    def make_pre_hook(s, e):
        def hook_fn(module, inputs):
            x = inputs[0].clone()
            x[..., s:e] = 0.0
            return (x,)
        return hook_fn

    for cell in ablated.encoder.encoder_cells:
        h = cell.multi_head_attention.W_o.register_forward_pre_hook(make_pre_hook(start, end))
        hooks.append(h)

    gt, preds = run_inference(ablated)
    recall = recall_score(gt, preds, pos_label=1)
    delta = recall - base_recall
    results[ablate_head] = {"recall": recall, "delta": delta}

    for h in hooks: h.remove()

    print(f"\n▶ Ablate Head {ablate_head}")
    print(f"  Class-1 Recall = {recall:.4f}   Δ = {delta:+.4f}")

# ──────────────────────────────────────────────
# 8. 汇总
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("汇总")
print("="*60)
print(f"Baseline Class-1 Recall = {base_recall:.4f}\n")
print(f"  {'Head':<8} {'Recall':<10} {'Δ Recall':<12} 重要性")
print(f"  {'-'*45}")

for h, v in sorted(results.items(), key=lambda x: x[1]["delta"]):
    if   v["delta"] <= -0.10: importance = "★★★  关键 head"
    elif v["delta"] <= -0.05: importance = "★★   有贡献"
    elif v["delta"] >= +0.03: importance = "⚠    关掉反而变好"
    else:                     importance = "—    影响不大"
    print(f"  {h:<8} {v['recall']:<10.4f} {v['delta']:<+12.4f} {importance}")

critical = min(results, key=lambda h: results[h]["delta"])
print(f"\n最关键 Head: Head {critical} (Δ = {results[critical]['delta']:+.4f})")
