# ── CONFIG ────────────────────────────────────
TRAIN_JSON  = "data/ketod/train.json"
TEST_JSON   = "data/ketod/test.json"
TRAIN_OUT   = "data/ketod/train_features.csv"
TEST_OUT    = "data/ketod/test_features.csv"
# ──────────────────────────────────────────────
"""
extract_features_ketod.py
从 KETOD 原始 JSON 提取 10 个特征，输出 CSV 供 LR 训练。
用法：python extract_features_ketod.py
"""
import json
import csv
import math
import re

# ===== 路径 =====
TRAIN_JSON = "E:/ketod-main/ketod_release/train.json"
TEST_JSON  = "E:/ketod-main/ketod_release/test.json"
TRAIN_OUT  = "E:/ketod-main/ketod_release/train_features.csv"
TEST_OUT   = "E:/ketod-main/ketod_release/test_features.csv"

QUESTION_WORDS = {'what','how','where','when','why','is','does','can','do'}

def count_tokens(text):
    return len(re.findall(r"\b\w+\b", text.lower())) if text else 0

def extract_features_from_dialogue(dialogue):
    """
    从单个对话提取所有 SYSTEM turn 的特征。
    每个 SYSTEM turn 对应一条训练样本。
    """
    turns = dialogue['turns']
    total_turns = len(turns)
    rows = []

    # 统计对话中 USER turn 总数（用于 dialogue_len）
    user_turns = [t for t in turns if t['speaker'] == 'USER']
    dialogue_len = len(user_turns)

    prev_sys_text = ""
    consecutive_sys = 0
    user_turn_idx = 0  # 当前是第几个 USER turn（0-indexed）

    for i, turn in enumerate(turns):
        if turn['speaker'] == 'USER':
            user_turn_idx += 1
            consecutive_sys = 0
            user_text = turn['utterance']

            # 下一个是 SYSTEM turn，预取
            if i + 1 < total_turns and turns[i+1]['speaker'] == 'SYSTEM':
                sys_turn = turns[i+1]
                sys_text = sys_turn['utterance']
                label = sys_turn.get('enrich', False)

                # ===== 10 个特征 =====
                # 1. turn_position_ratio
                turn_position_ratio = user_turn_idx / dialogue_len if dialogue_len > 0 else 0.0

                # 2. prev_sys_is_question
                prev_sys_is_question = 1 if prev_sys_text.strip().endswith('?') else 0

                # 3. user_has_question
                user_has_question = 1 if '?' in user_text else 0

                # 4. user_starts_question_word
                first_word = re.findall(r"\b\w+\b", user_text.lower())
                user_starts_question_word = 1 if (first_word and first_word[0] in QUESTION_WORDS) else 0

                # 5. user_turn_len_log
                user_turn_len_log = math.log(1 + count_tokens(user_text))

                # 6. sys_turn_len_log
                sys_turn_len_log = math.log(1 + count_tokens(prev_sys_text))

                # 7. dialogue_len_log
                dialogue_len_log = math.log(dialogue_len) if dialogue_len > 1 else 0.0

                # 8. consecutive_sys_turns
                consecutive_sys_turns = consecutive_sys

                # 9. turn_len_ratio
                user_len = count_tokens(user_text)
                sys_len = count_tokens(prev_sys_text) if prev_sys_text else 1
                turn_len_ratio = min(user_len / sys_len, 5.0)

                # 10. turn_position_squared
                turn_position_squared = turn_position_ratio ** 2

                rows.append({
                    'dialogue_id': dialogue['dialogue_id'],
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
                    'label': 1 if label else 0
                })

                prev_sys_text = sys_text
                consecutive_sys = 0

        elif turn['speaker'] == 'SYSTEM':
            consecutive_sys += 1
            prev_sys_text = turn['utterance']

    return rows

def process_file(json_path, out_path):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    all_rows = []
    for dialogue in data:
        all_rows.extend(extract_features_from_dialogue(dialogue))

    fieldnames = [
        'dialogue_id', 'turn_idx',
        'turn_position_ratio', 'prev_sys_is_question', 'user_has_question',
        'user_starts_question_word', 'user_turn_len_log', 'sys_turn_len_log',
        'dialogue_len_log', 'consecutive_sys_turns', 'turn_len_ratio',
        'turn_position_squared', 'label'
    ]

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    pos = sum(1 for r in all_rows if r['label'] == 1)
    neg = sum(1 for r in all_rows if r['label'] == 0)
    print(f"完成: {out_path}")
    print(f"  总样本: {len(all_rows)}, 正样本(enrich=True): {pos}, 负样本: {neg}, 比例: {neg/pos:.1f}:1")

if __name__ == '__main__':
    print("处理 KETOD train...")
    process_file(TRAIN_JSON, TRAIN_OUT)
    print("\n处理 KETOD test...")
    process_file(TEST_JSON, TEST_OUT)
