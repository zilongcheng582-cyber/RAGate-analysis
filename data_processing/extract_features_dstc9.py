# ── CONFIG ── 修改为你的本地路径 ──────────────
KETOD_TRAIN = "data/ketod/train_features.csv"
KETOD_TEST  = "data/ketod/test_features.csv"
# ──────────────────────────────────────────────
"""
extract_features_dstc9.py
从 DSTC9 logs.json + labels.json 提取 10 个特征，输出 CSV 供 LR 训练。
用法：python extract_features_dstc9.py
"""
import json
import csv
import math
import re

# ===== 路径 =====
TRAIN_LOGS   = "E:/dstc9-track1/data/train/logs.json"
TRAIN_LABELS = "E:/dstc9-track1/data/train/labels.json"
TRAIN_OUT    = "E:/dstc9-track1/data/train/train_features.csv"

VAL_LOGS     = "E:/dstc9-track1/data/val/logs.json"
VAL_LABELS   = "E:/dstc9-track1/data/val/labels.json"
VAL_OUT      = "E:/dstc9-track1/data/val/test_features.csv"

QUESTION_WORDS = {'what','how','where','when','why','is','does','can','do'}

def count_tokens(text):
    return len(re.findall(r"\b\w+\b", text.lower())) if text else 0

def extract_features(log, label, idx):
    """
    每条 log 是一个完整对话，最后一个 U turn 是当前 turn。
    label 对应整条对话是否需要 knowledge。
    """
    turns = log

    # 统计 USER turns 总数（用于 position ratio）
    user_turns = [t for t in turns if t['speaker'] == 'U']
    dialogue_len = len(user_turns)

    # 当前 turn = 最后一个 USER turn
    user_turn_idx = dialogue_len  # 1-indexed

    # 当前 USER turn 文本
    user_text = user_turns[-1]['text'] if user_turns else ""

    # 找上一个 SYSTEM turn（当前 USER turn 之前最近的 S turn）
    prev_sys_text = ""
    consecutive_sys = 0
    found_current_user = False
    for t in reversed(turns):
        if t['speaker'] == 'U' and not found_current_user:
            found_current_user = True
            continue
        if found_current_user:
            if t['speaker'] == 'S':
                if not prev_sys_text:
                    prev_sys_text = t['text']
                consecutive_sys += 1
            else:
                break

    # ===== 10 个特征 =====
    turn_position_ratio = user_turn_idx / dialogue_len if dialogue_len > 0 else 0.0
    prev_sys_is_question = 1 if prev_sys_text.strip().endswith('?') else 0
    user_has_question = 1 if '?' in user_text else 0
    first_word = re.findall(r"\b\w+\b", user_text.lower())
    user_starts_question_word = 1 if (first_word and first_word[0] in QUESTION_WORDS) else 0
    user_turn_len_log = math.log(1 + count_tokens(user_text))
    sys_turn_len_log = math.log(1 + count_tokens(prev_sys_text))
    dialogue_len_log = math.log(dialogue_len) if dialogue_len > 1 else 0.0
    consecutive_sys_turns = consecutive_sys
    user_len = count_tokens(user_text)
    sys_len = count_tokens(prev_sys_text) if prev_sys_text else 1
    turn_len_ratio = min(user_len / sys_len, 5.0)
    turn_position_squared = turn_position_ratio ** 2

    return {
        'dialogue_id': idx,
        'turn_idx': user_turn_idx,
        'turn_position_ratio': turn_position_ratio,
        'prev_sys_is_question': prev_sys_is_question,
        'user_has_question': user_has_question,
        'user_starts_question_word': user_starts_question_word,
        'user_turn_len_log': user_turn_len_log,
        'sys_turn_len_log': sys_turn_len_log,
        'dialogue_len_log': dialogue_len_log,
        'consecutive_sys_turns': consecutive_sys_turns,
        'turn_len_ratio': turn_len_ratio,
        'turn_position_squared': turn_position_squared,
        'label': 1 if label['target'] else 0
    }

def process_file(logs_path, labels_path, out_path):
    with open(logs_path, encoding='utf-8') as f:
        logs = json.load(f)
    with open(labels_path, encoding='utf-8') as f:
        labels = json.load(f)

    fieldnames = [
        'dialogue_id', 'turn_idx',
        'turn_position_ratio', 'prev_sys_is_question', 'user_has_question',
        'user_starts_question_word', 'user_turn_len_log', 'sys_turn_len_log',
        'dialogue_len_log', 'consecutive_sys_turns', 'turn_len_ratio',
        'turn_position_squared', 'label'
    ]

    all_rows = []
    for idx, (log, label) in enumerate(zip(logs, labels)):
        row = extract_features(log, label, idx)
        all_rows.append(row)

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    pos = sum(1 for r in all_rows if r['label'] == 1)
    neg = sum(1 for r in all_rows if r['label'] == 0)
    print(f"完成: {out_path}")
    print(f"  总样本: {len(all_rows)}, 正样本: {pos}, 负样本: {neg}, 比例: {neg/pos:.1f}:1")

if __name__ == '__main__':
    print("处理 DSTC9 train...")
    process_file(TRAIN_LOGS, TRAIN_LABELS, TRAIN_OUT)
    print("\n处理 DSTC9 val（测试集）...")
    process_file(VAL_LOGS, VAL_LABELS, VAL_OUT)
