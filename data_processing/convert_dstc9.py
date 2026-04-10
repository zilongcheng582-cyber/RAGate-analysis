# ── CONFIG ────────────────────────────────────
LOGS_JSON   = "data/dstc9/train/logs.json"
LABELS_JSON = "data/dstc9/train/labels.json"
OUTPUT_CSV  = "data/dstc9/train/train_dstc9.csv"
# ──────────────────────────────────────────────
import json
import csv

def convert_dstc9(logs_path, labels_path, output_path):
    with open(logs_path, encoding='utf-8') as f:
        logs = json.load(f)
    with open(labels_path, encoding='utf-8') as f:
        labels = json.load(f)

    rows = []
    for log, label in zip(logs, labels):
        # 拼接对话历史
        turns = []
        for turn in log:
            speaker = 'USER' if turn['speaker'] == 'U' else 'SYSTEM'
            turns.append(f"{speaker}: {turn['text']}")
        context = ' '.join(turns)
        output = label['target']
        rows.append({'input': context, 'output': output})

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['input', 'output'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"完成，共 {len(rows)} 条，保存到 {output_path}")

convert_dstc9(
    'E:/dstc9-track1/data/train/logs.json',
    'E:/dstc9-track1/data/train/labels.json',
    'E:/dstc9-track1/data/train/train_dstc9.csv'
)
# 转换 val（作为测试集）
convert_dstc9(
    'E:/dstc9-track1/data/val/logs.json',
    'E:/dstc9-track1/data/val/labels.json',
    'E:/dstc9-track1/data/val/test_dstc9.csv'
)
