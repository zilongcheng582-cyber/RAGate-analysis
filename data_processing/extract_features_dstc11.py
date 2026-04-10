# ── CONFIG ────────────────────────────────────
TRAIN_CSV   = "data/dstc11/train.csv"
TEST_CSV    = "data/dstc11/val.csv"
TRAIN_OUT   = "data/dstc11/train_features.csv"
TEST_OUT    = "data/dstc11/test_features.csv"
# ──────────────────────────────────────────────
"""
extract_features_dstc11.py
从 DSTC11 train.csv / val.csv 提取 10 个特征，输出 CSV 供 LR 训练。
input 列是累积对话历史，格式：USER: ... SYSTEM: ... USER: ...
用法：python extract_features_dstc11.py
"""
import csv
import math
import re
import pandas as pd

# ===== 路径 =====
TRAIN_CSV = "E:/dstc11-track5/train.csv"
VAL_CSV   = "E:/dstc11-track5/val.csv"
TRAIN_OUT = "E:/dstc11-track5/train_features.csv"
VAL_OUT   = "E:/dstc11-track5/test_features.csv"

QUESTION_WORDS = {'what','how','where','when','why','is','does','can','do'}

def count_tokens(text):
    return len(re.findall(r"\b\w+\b", text.lower())) if text else 0

def parse_turns(input_text):
    """
    把累积对话历史解析成 turn 列表。
    返回 [{'speaker': 'USER'/'SYSTEM', 'text': '...'}, ...]
    """
    # 用 USER: 和 SYSTEM: 作为分隔符
    pattern = r'(USER:|SYSTEM:)'
    parts = re.split(pattern, input_text.strip())
    turns = []
    i = 1
    while i < len(parts) - 1:
        speaker = parts[i].strip().rstrip(':')
        text = parts[i+1].strip()
        turns.append({'speaker': speaker, 'text': text})
        i += 2
    return turns

def extract_features_from_input(input_text, label, idx):
    turns = parse_turns(input_text)

    user_turns = [t for t in turns if t['speaker'] == 'USER']
    dialogue_len = len(user_turns)
    user_turn_idx = dialogue_len  # 当前是第几个 USER turn（最后一个）

    user_text = user_turns[-1]['text'] if user_turns else ""

    # 找上一个 SYSTEM turn
    prev_sys_text = ""
    consecutive_sys = 0
    found_current_user = False
    for t in reversed(turns):
        if t['speaker'] == 'USER' and not found_current_user:
            found_current_user = True
            continue
        if found_current_user:
            if t['speaker'] == 'SYSTEM':
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
        'label': 1 if str(label).strip() == 'True' else 0
    }

def process_file(csv_path, out_path):
    df = pd.read_csv(csv_path)

    fieldnames = [
        'dialogue_id', 'turn_idx',
        'turn_position_ratio', 'prev_sys_is_question', 'user_has_question',
        'user_starts_question_word', 'user_turn_len_log', 'sys_turn_len_log',
        'dialogue_len_log', 'consecutive_sys_turns', 'turn_len_ratio',
        'turn_position_squared', 'label'
    ]

    all_rows = []
    for idx, row in df.iterrows():
        r = extract_features_from_input(str(row['input']), row['output'], idx)
        all_rows.append(r)

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    pos = sum(1 for r in all_rows if r['label'] == 1)
    neg = sum(1 for r in all_rows if r['label'] == 0)
    ratio = neg/pos if pos > 0 else float('inf')
    print(f"完成: {out_path}")
    print(f"  总样本: {len(all_rows)}, 正样本: {pos}, 负样本: {neg}, 比例: {ratio:.1f}:1")

if __name__ == '__main__':
    print("处理 DSTC11 train...")
    process_file(TRAIN_CSV, TRAIN_OUT)
    print("\n处理 DSTC11 val（测试集）...")
    process_file(VAL_CSV, VAL_OUT)
