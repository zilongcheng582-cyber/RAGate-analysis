"""
Computes class-conditional user_has_question rates for §4.6.

Outputs:
    results/class_conditional_qrate.csv

Usage:
    python analysis/class_conditional_qrate.py
"""
import pandas as pd

DATASETS = {
    "KETOD": {
        "path": "E:/ketod-main/ketod_release/train_full.csv",
        "engine": "c",
    },
    "DSTC9": {
        "path": "E:/dstc9-track1/data/train/train_dstc9.csv",
        "engine": "python",
    },
    "DSTC11": {
        "path": "E:/dstc11-track5/train.csv",
        "engine": "c",
    },
}

rows = []
for name, cfg in DATASETS.items():
    df = pd.read_csv(cfg["path"], engine=cfg["engine"], on_bad_lines="skip")

    # output column is True/False string in raw CSVs; normalize to 0/1.
    if df["output"].dtype == object:
        df["label"] = (df["output"].str.strip() == "True").astype(int)
    else:
        df["label"] = df["output"].astype(int)

    # Re-derive user_has_question from the input column for transparency.
    df["user_has_question"] = df["input"].str.contains(r"\?", na=False).astype(int)

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    pos_q = pos["user_has_question"].mean()
    neg_q = neg["user_has_question"].mean()

    rows.append({
        "dataset": name,
        "N_pos": len(pos),
        "N_neg": len(neg),
        "pos_q_rate": round(pos_q, 4),
        "neg_q_rate": round(neg_q, 4),
        "delta": round(pos_q - neg_q, 4),
    })

    print(f"{name}: N_pos={len(pos)}, N_neg={len(neg)}, "
          f"pos_q={pos_q:.3f}, neg_q={neg_q:.3f}, "
          f"\u0394={pos_q - neg_q:+.3f}")

out = pd.DataFrame(rows)
out.to_csv("results/class_conditional_qrate.csv", index=False)
print(f"\nSaved to results/class_conditional_qrate.csv")
