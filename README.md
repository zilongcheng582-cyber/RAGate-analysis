# Annotation Shortcuts: Lightweight Probing Reveals Incompatibility in Knowledge-Gating Benchmarks

This repository contains code for our EMNLP 2026 submission analyzing shortcut learning in knowledge-gating benchmarks (KETOD, DSTC9, DSTC11).

We show that knowledge-gating models rely on **dataset-specific surface shortcuts** rather than genuine semantic reasoning, and that these shortcuts are fundamentally incompatible across benchmarks due to differences in annotation protocols.

---

## Datasets

Download and place in the following structure:

| Dataset | Source | Path |
|---|---|---|
| KETOD | [facebookresearch/ketod](https://github.com/facebookresearch/ketod) | `data/ketod/` |
| DSTC9 Track 1 | [alexa/alexa-with-dstc9-track1-dataset](https://github.com/alexa/alexa-with-dstc9-track1-dataset) | `data/dstc9/` |
| DSTC11 Track 5 | [alexa/dstc11-track5](https://github.com/alexa/dstc11-track5) | `data/dstc11/` |

---

## Environment

```bash
pip install torch scikit-learn pandas numpy scipy matplotlib seaborn datasets tqdm
```

For the semantic baseline only:
```bash
pip install sentence-transformers
```

For the BERT transfer experiment (requires GPU):
```bash
pip install transformers accelerate
```

Tested on Python 3.12, PyTorch 2.5.1 (LR/MHA experiments) and PyTorch 2.7.0 + CUDA 12.8 (BERT experiment, RTX 5090).

---

## Configuration

Before running, update the data paths at the top of each script to match your local setup. All paths are defined in a `# CONFIG` block at the top of each file.

Default structure assumed:
```
data/
├── ketod/
│   ├── train_full.csv
│   ├── test_full.csv
│   ├── train_features.csv
│   └── test_features.csv
├── dstc9/
│   ├── train/
│   └── val/
└── dstc11/
    ├── train_features.csv
    └── test_features.csv
```

## Reproduction

Run in order:

### 1. Data Processing

```bash
python data_processing/convert_dstc9.py
python data_processing/extract_features_ketod.py
python data_processing/extract_features_dstc9.py
python data_processing/extract_features_dstc11.py
```

### 2. MHA Baseline (KETOD)

```bash
# Train (requires GPU)
python mha/train_MHA.py --loss weighted --epochs 50

# Inference with checkpoint
python mha/mha_inference.py
```

MHA is adapted from the original RAGate implementation.
Modifications: replaced `torchtext` with a custom `Vocab` class, added `CosineAnnealingLR` scheduler, switched to weighted cross-entropy loss.

### 3. LR Probing (all datasets)

```bash
# Main ablation experiment (Table 1)
python probing/train_lr.py

# Position shuffle test
python probing/position_shuffle_lr.py

# Threshold tuning for KETOD (imbalanced)
python probing/threshold_tuning.py

# Feature importance + Spearman rho (Table 2 / Figure 2)
python probing/feature_importance_spearman.py
```

### 4. Analysis

```bash
# Cross-dataset transfer — structural LR (Table 3)
python analysis/cross_dataset_transfer.py

# LR vs MHA agreement (requires mha_predictions.csv)
python analysis/agreement_analysis.py

# Counterfactual perturbation
python analysis/counterfactual_analysis.py

# MHA proxy check on LR false negatives
python analysis/proxy_check.py

# Semantic baseline — capacity robustness check (Section 4.3)
# Requires train_full.csv (raw text), not train_features.csv
pip install sentence-transformers
python analysis/semantic_baseline.py

# BERT cross-dataset transfer — progressive capacity check (Section 4.3)
# Requires GPU; downloads bert-base-uncased from HuggingFace
# For mainland China: export HF_ENDPOINT=https://hf-mirror.com
pip install transformers accelerate
python analysis/bert_transfer.py \
    --ketod-train  data/ketod/train_full.csv \
    --ketod-test   data/ketod/test_full.csv \
    --dstc9-train  data/dstc9/train_dstc9.csv \
    --dstc9-test   data/dstc9/test_dstc9.csv \
    --dstc11-train data/dstc11/train.csv \
    --dstc11-test  data/dstc11/val.csv \
    --output-dir   results/
```

> **Note on DSTC9 training file:** `train_dstc9.csv` contains malformed rows due to unescaped quotes in dialogue text. If you encounter a CSV parse error, run the following to produce a cleaned version before proceeding:
> ```bash
> python -c "
> import pandas as pd
> df = pd.read_csv('data/dstc9/train_dstc9.csv', engine='python', on_bad_lines='skip')
> df.to_csv('data/dstc9/train_dstc9_fixed.csv', index=False)
> print(f'Saved {len(df)} rows')
> "
> ```
> Then replace `--dstc9-train data/dstc9/train_dstc9.csv` with `--dstc9-train data/dstc9/train_dstc9_fixed.csv`.

---

## Results

Pre-computed results are in `results/`. Key numbers:

| Experiment | Key Finding |
|---|---|
| Feature ablation | Position only F1 ≈ random on DSTC9/11; No position = Full |
| Spearman ρ | DSTC9 vs DSTC11: ρ=+0.94 (p<0.001); KETOD vs DSTC: ρ≈−0.43 |
| Cross-dataset transfer (structural LR) | DSTC→KETOD minority F1=0.00; gap=+0.33 (severe) |
| Cross-dataset transfer (semantic LR) | DSTC→KETOD minority F1=0.12–0.13; transfer still fails vs in-domain 0.36 |
| Cross-dataset transfer (BERT) | DSTC→KETOD minority F1=0.07–0.10; lower than semantic baseline; within-DSTC intact |
| LR-MHA agreement | κ=0.25; MHA counterfactual flip rate=2.4% vs LR 16.6% |

Results files:

```
results/
├── lr_results.csv
├── transfer_results.csv
├── spearman_rho_results.csv
├── feature_importance.csv
├── position_shuffle_results.csv
├── threshold_tuning_results.csv
├── agreement_results.csv
├── counterfactual_results.csv
├── mha_predictions.csv
├── semantic_results.csv        ← semantic baseline transfer matrix
├── semantic_summary.txt        ← paper-ready numbers (semantic)
├── bert_results.csv            ← BERT transfer matrix
└── bert_summary.txt            ← paper-ready numbers (BERT)
```

---

## Features

10 hand-crafted structural features used for probing:

| # | Feature | Description |
|---|---|---|
| 1 | `turn_position_ratio` | turn index / total turns |
| 2 | `prev_sys_is_question` | previous system turn ends with ? |
| 3 | `user_has_question` | user turn contains ? |
| 4 | `user_starts_question_word` | user turn starts with what/how/where/... |
| 5 | `user_turn_len_log` | log(1 + user turn tokens) |
| 6 | `sys_turn_len_log` | log(1 + previous system turn tokens) |
| 7 | `dialogue_len_log` | log(total turns) |
| 8 | `consecutive_sys_turns` | consecutive system turns before current user turn |
| 9 | `turn_len_ratio` | user\_len / sys\_len, clipped to [0, 5] |
| 10 | `turn_position_squared` | (turn index / total)², captures nonlinear position effect |

---

## Citation

```bibtex
@article{
  title={Annotation Shortcuts: Lightweight Probing Reveals Incompatibility in Knowledge-Gating Benchmarks},
  year={2026}
}
```

---

## Acknowledgements

RAGate-MHA architecture is based on [RAGate](https://github.com/...)(original paper citation).
KETOD, DSTC9, and DSTC11 datasets are used under their respective licenses.
