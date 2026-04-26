# Shortcuts or Semantics? Probing Knowledge-Gating Benchmarks via Lightweight Feature Analysis

Code for an ARR 2026 short-paper submission analyzing shortcut learning in knowledge-gating benchmarks (KETOD, DSTC9, DSTC11).

We show that knowledge-gating models rely on **annotation-specific structural shortcuts** rather than transferable gating ability, and that these shortcuts are largely incompatible across annotation protocols. The collapse persists at higher capacity (sentence embeddings, fine-tuned BERT), and fine-tuning even deepens annotation-specific reliance.

---

## Datasets

Download and place under the following structure:

| Dataset | Source | Path |
|---|---|---|
| KETOD | [facebookresearch/ketod](https://github.com/facebookresearch/ketod) | `data/ketod/` |
| DSTC9 Track 1 | [alexa/alexa-with-dstc9-track1-dataset](https://github.com/alexa/alexa-with-dstc9-track1-dataset) | `data/dstc9/` |
| DSTC11 Track 5 | [alexa/dstc11-track5](https://github.com/alexa/dstc11-track5) | `data/dstc11/` |

Default layout assumed by the scripts:

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

Before running, update the data paths in the `# CONFIG` block at the top of each script if your local layout differs.

---

## Environment

```bash
pip install -r requirements.txt
```

Tested with Python 3.12 on:
- PyTorch 2.5.1 (LR / MHA experiments)
- PyTorch 2.7.0 + CUDA 12.8 on RTX 5090 (BERT experiment)

---

## Reproduction

Run in order. All result files are written to `results/`.

### 1. Data processing

```bash
python data_processing/convert_dstc9.py
python data_processing/extract_features_ketod.py
python data_processing/extract_features_dstc9.py
python data_processing/extract_features_dstc11.py
```

### 2. RAGate-MHA baseline (KETOD)

```bash
# Train (requires GPU)
python mha/train_MHA.py --loss weighted --epochs 50

# Run inference with the saved checkpoint
python mha/mha_inference.py
```

The MHA implementation is reimplemented from the architectural description in the original RAGate paper (NAACL 2025 Findings). Modifications: replaced the deprecated `torchtext` API with a custom `Vocab` class, added a `CosineAnnealingLR` scheduler, and switched to weighted cross-entropy loss for class imbalance. See Appendix B of the paper for details.

### 3. LR probing (all three datasets)

```bash
# Feature-subset ablation (Table 2)
python probing/train_lr.py

# Position-feature permutation test (§4.1)
python probing/position_shuffle_lr.py

# Threshold tuning for KETOD (class imbalance)
python probing/threshold_tuning.py

# Feature importance + Spearman rho (Figure 2, §4.2)
python probing/feature_importance_spearman.py
```

### 4. Analysis

```bash
# Cross-dataset transfer — structural LR (Table 3, Appendix Table 5)
python analysis/cross_dataset_transfer.py

# LR vs MHA agreement on KETOD (§4.5)
python analysis/agreement_analysis.py

# Counterfactual perturbation of user_has_question (§4.5)
python analysis/counterfactual_analysis.py

# Class-conditional question-mark rates for KETOD/DSTC9/DSTC11 (Table 4, §4.6)
python analysis/class_conditional_qrate.py

# Sentence-embedding capacity check (Figure 3, §4.4)
python analysis/semantic_baseline.py

# BERT cross-dataset transfer (Figure 3, §4.4)
# Requires GPU; downloads bert-base-uncased from HuggingFace
# For mainland China: export HF_ENDPOINT=https://hf-mirror.com
python analysis/bert_transfer.py \
    --ketod-train  data/ketod/train_full.csv \
    --ketod-test   data/ketod/test_full.csv \
    --dstc9-train  data/dstc9/train_dstc9.csv \
    --dstc9-test   data/dstc9/test_dstc9.csv \
    --dstc11-train data/dstc11/train.csv \
    --dstc11-test  data/dstc11/val.csv \
    --output-dir   results/
```

> **Note on DSTC9 training file:** `train_dstc9.csv` contains malformed rows due to unescaped quotes in dialogue text. If `pandas.read_csv` errors out, produce a cleaned copy first:
>
> ```bash
> python -c "
> import pandas as pd
> df = pd.read_csv('data/dstc9/train_dstc9.csv', engine='python', on_bad_lines='skip')
> df.to_csv('data/dstc9/train_dstc9_fixed.csv', index=False)
> print(f'Saved {len(df)} rows')
> "
> ```
>
> Then pass `--dstc9-train data/dstc9/train_dstc9_fixed.csv` to `bert_transfer.py`.

---

## Results

Pre-computed result files are in `results/`. Key numbers reproduced in the paper:

| Experiment | Key finding |
|---|---|
| Feature ablation (Table 2) | Position-only F1 ≈ random on DSTC9/11; No-position matches Full on DSTC; KETOD shows no dominant feature group |
| Spearman ρ (§4.2) | DSTC9 vs DSTC11: ρ = +0.94 (*p* < 0.001); KETOD vs DSTC: ρ ≈ −0.25 (mean, *p* > 0.1) |
| Cross-dataset transfer — structural LR (Table 3) | DSTC → KETOD minority F1 ≤ 0.22, ROC-AUC ≈ 0.48 (near chance); macro F1 gap +0.27 / +0.32 |
| Cross-dataset transfer — sentence embeddings (§4.4) | DSTC → KETOD minority F1 = 0.12 / 0.13; transfer still fails vs in-domain 0.36 |
| Cross-dataset transfer — fine-tuned BERT (§4.4) | DSTC → KETOD minority F1 = 0.07 / 0.10; *worse* than embeddings despite higher in-domain F1 (0.50) |
| LR–MHA agreement (§4.5) | Agreement 67.2%, κ = 0.25; LR-FN ∩ MHA-fail 79.6% (160/201) vs chance 59.3% (binomial *p* < 0.001) |
| Counterfactual flip (§4.5) | LR 18.1% vs MHA 2.4% on `user_has_question` |
| Class-conditional question rate (Table 4, §4.6) | DSTC9 / DSTC11 P(q∣pos) = 1.000 / 0.999; KETOD P(q∣pos) = 0.907 vs P(q∣neg) = 0.925 |

Result file layout:

```
results/
├── lr_results.csv                  # Table 2 (feature subset ablation)
├── transfer_results.csv            # Table 3 + Appendix Table 5
├── spearman_rho_results.csv        # §4.2
├── feature_importance.csv          # Figure 2
├── position_shuffle_results.csv    # §4.1 permutation test
├── threshold_tuning_results.csv    # KETOD threshold search
├── agreement_results.csv           # §4.5 LR vs MHA
├── counterfactual_results.csv      # §4.5 counterfactual flip
├── mha_predictions.csv             # MHA test-set predictions on KETOD
├── class_conditional_qrate.csv     # Table 4
├── semantic_results.csv            # §4.4 semantic baseline transfer
├── semantic_summary.txt
├── bert_results.csv                # §4.4 BERT transfer
└── bert_summary.txt
```

---

## Features

Ten hand-crafted structural features used for probing (Appendix A):

| # | Feature | Description |
|---|---|---|
| 1 | `turn_position_ratio` | turn index / total turns |
| 2 | `turn_position_squared` | (turn index / total)², captures nonlinear position effect |
| 3 | `user_turn_len_log` | log(1 + user turn tokens) |
| 4 | `sys_turn_len_log` | log(1 + previous system turn tokens) |
| 5 | `dialogue_len_log` | log(total turns) |
| 6 | `consecutive_sys_turns` | consecutive system turns before current user turn |
| 7 | `turn_len_ratio` | user_len / sys_len, clipped to [0, 5] |
| 8 | `user_has_question` | user turn contains `?` |
| 9 | `prev_sys_is_question` | previous system turn ends with `?` |
| 10 | `user_starts_question_word` | user turn starts with what / how / where / when / why / is / does / can / do |

No utterance text or semantic content is accessed — only metadata.

---

## Citation

This work is currently under double-blind review at ARR 2026. Citation information will be released after the review process.

---

## License

Code is released under the MIT License. Datasets are used under their respective licenses; please refer to the source repositories for details.
