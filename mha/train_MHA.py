# ── CONFIG ────────────────────────────────────
TRAIN_CSV   = "data/ketod/train_full.csv"
TEST_CSV    = "data/ketod/test_full.csv"
CKPT_DIR    = "outputs/MHA-trained"
# ──────────────────────────────────────────────
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from typing import Iterable
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import re
import os
import argparse

# ===== 参数 =====
parser = argparse.ArgumentParser()
parser.add_argument("--loss", type=str, default="cross_entropy",
                    choices=["cross_entropy", "weighted", "focal"],
                    help="损失函数类型: cross_entropy / weighted / focal")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.0005)      # 原作者 lr
parser.add_argument("--batch_size", type=int, default=256)   # 原作者 batch_size
parser.add_argument("--focal_gamma", type=float, default=2.0,
                    help="Focal Loss 的 gamma 参数，越大越关注难样本")
args = parser.parse_args()

print(f"Loss type: {args.loss}, lr: {args.lr}, batch_size: {args.batch_size}")

# ===== 替代 torchtext 的简单实现 =====
def get_tokenizer(mode):
    if mode == "basic_english":
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


# ===== 模型定义 =====
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
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
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
        for cell in self.encoder_cells:
            x = cell(x, mask)
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


# ===== Focal Loss =====
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ===== 数据集 =====
class TextClassificationDataset(Dataset):
    def __init__(self, data_dir):
        self.data = load_dataset('csv', data_files=data_dir, split='train')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data[idx]['output'], self.data[idx]['input']
        return label, text


# ===== 主流程 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = get_tokenizer("basic_english")

print("Building vocab...")
train_iter = TextClassificationDataset("/root/train_full.csv")

def yield_tokens(data_iter: Iterable, tokenizer):
    for _, text in data_iter:
        yield tokenizer(str(text))

vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
print(f"Vocab size: {len(vocab)}")

PAD_IDX = vocab["<unk>"]

def collate_batch(batch):
    label_list, text_list = [], []
    for label, text in batch:
        label_val = 1 if str(label).strip() == "True" else 0
        label_list.append(label_val)
        processed = torch.tensor(vocab(tokenizer(str(text))), dtype=torch.int64)
        text_list.append(processed)
    labels = torch.tensor(label_list, dtype=torch.int64)
    texts = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=PAD_IDX)
    return labels, texts

# 计算 class weight
all_labels = [1 if str(train_iter.data[i]['output']).strip() == "True" else 0
              for i in range(len(train_iter))]
n_class0 = all_labels.count(0)
n_class1 = all_labels.count(1)
total = len(all_labels)
w0 = total / (2 * n_class0)
w1 = total / (2 * n_class1)
class_weights = torch.tensor([w0, w1], dtype=torch.float).to(device)
print(f"Class weights: class0={w0:.3f}, class1={w1:.3f}")

# 选择 loss
if args.loss == "cross_entropy":
    criterion = nn.CrossEntropyLoss()
    print("Using: standard CrossEntropyLoss (no reweighting)")
elif args.loss == "weighted":
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("Using: weighted CrossEntropyLoss")
elif args.loss == "focal":
    criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    print(f"Using: FocalLoss (gamma={args.focal_gamma})")

# DataLoader
train_dataloader = DataLoader(train_iter, batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate_batch)

test_iter = TextClassificationDataset("/root/test_full.csv")
test_dataloader = DataLoader(test_iter, batch_size=256,
                              shuffle=False, collate_fn=collate_batch)

# 模型
model = TransformerClassifier(
    vocab_size=len(vocab),
    num_layers=5,
    embed_dim=64,
    num_heads=4,
    hidden_dim=64,
    num_classes=2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# ===== 新增：CosineAnnealingLR scheduler（和原作者一致）=====
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

# 训练
print("\nStart training...")
best_f1 = 0.0
best_epoch = 0

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0
    for labels, texts in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
        labels, texts = labels.to(device), texts.to(device)
        optimizer.zero_grad()
        logits = model(texts, mask=None)
        loss = criterion(logits, labels)
        loss.backward()
        # gradient clipping（原作者也有）
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # scheduler 每个 epoch 更新一次
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    # 每 5 个 epoch 评估一次
    if epoch % 5 == 0:
        model.eval()
        all_preds, all_labels_eval = [], []
        with torch.no_grad():
            for labels, texts in test_dataloader:
                labels, texts = labels.to(device), texts.to(device)
                preds = model(texts, mask=None).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels_eval.extend(labels.cpu().numpy())

        macro_f1 = f1_score(all_labels_eval, all_preds, average='macro')
        recall_1 = f1_score(all_labels_eval, all_preds, average=None)[1]
        print(f"\nEpoch {epoch} | Loss: {total_loss/len(train_dataloader):.4f} "
              f"| Macro F1: {macro_f1:.4f} | Class-1 Recall: {recall_1:.4f} | LR: {current_lr:.2e}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            os.makedirs("../../outputs/MHA-trained", exist_ok=True)
            ckpt_name = f"../../outputs/MHA-trained/MHA_{args.loss}_e{epoch}_f1{macro_f1:.4f}.pt"
            torch.save(model.state_dict(), ckpt_name)
            print(f"  -> Saved best checkpoint: {ckpt_name}")

print(f"\nTraining done. Best Macro F1: {best_f1:.4f} at epoch {best_epoch}")

# 最终评估
print("\n===== Final Evaluation =====")
model.eval()
all_preds, all_labels_eval = [], []
with torch.no_grad():
    for labels, texts in test_dataloader:
        labels, texts = labels.to(device), texts.to(device)
        preds = model(texts, mask=None).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels_eval.extend(labels.cpu().numpy())

print(classification_report(all_labels_eval, all_preds))
print(f"Macro F1: {f1_score(all_labels_eval, all_preds, average='macro'):.4f}")
