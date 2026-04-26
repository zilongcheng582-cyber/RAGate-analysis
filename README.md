# Shortcuts or Semantics? Probing Knowledge-Gating Benchmarks via Lightweight Feature Analysis

Code for an ARR 2026 short-paper submission analyzing shortcut learning in knowledge-gating benchmarks: KETOD, DSTC9, and DSTC11.

The experiments test whether lightweight structural signals can expose benchmark-specific shortcut patterns in knowledge-gating evaluation. The results suggest that same-protocol DSTC datasets share highly similar structural signals, while transfer from DSTC to KETOD is weak on minority-class detection and ROC-AUC. Higher-capacity representations, including sentence embeddings and fine-tuned BERT, do not remove the DSTC→KETOD transfer failure in these experiments. Because KETOD and DSTC also differ in corpus family, the paper treats this as evidence consistent with protocol-linked shortcut mismatch rather than as a fully controlled causal decomposition.

---

## Datasets

Download the datasets and place them under the following structure:

| Dataset | Source | Path |
|---|---|---|
| KETOD | `facebookresearch/ketod` | `data/ketod/` |
| DSTC9 Track 1 | `alexa/alexa-with-dstc9-track1-dataset` | `data/dstc9/` |
| DSTC11 Track 5 | `alexa/dstc11-track5` | `data/dstc11/` |

Default layout assumed by the scripts:

```text
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

- PyTorch 2.5.1 for LR and MHA experiments
- PyTorch 2.7.0 + CUDA 12.8 on RTX 5090 for the BERT experiment

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

### 2. RAGate-MHA baseline on KETOD

```bash
# Train; requires GPU
python mha/train_MHA.py --loss weighted --epochs 50

# Run inference with the saved checkpoint
python mha/mha_inference.py
```

The MHA implementation is reimplemented from the architectural description in the original RAGate paper. It is not an exact reproduction of the released training setup. The implementation replaces the deprecated `torchtext` API with a custom `Vocab` class, adds a `CosineAnnealingLR` scheduler, and uses weighted cross-entropy for class imbalance. See Appendix B of the paper for details.

### 3. LR probing

```bash
# Feature-subset ablation, Table 2; uses 5-fold CV
python probing/train_lr.py

# Position-feature permutation test, Section 4.1
python probing/position_shuffle_lr.py

# KETOD threshold-tuning diagnostic; not used for the transfer table
python probing/threshold_tuning.py

# Feature importance and Spearman correlation, Figure 1 and Section 4.2
python probing/feature_importance_spearman.py
```

### 4. Transfer and model-comparison analyses

```bash
# Cross-dataset transfer with structural LR, Table 3 and Appendix Table 6; uses 3-fold CV
python analysis/cross_dataset_transfer.py

# LR vs MHA agreement on KETOD, Section 4.5
python analysis/agreement_analysis.py

# Counterfactual perturbation of user_has_question, Section 4.5
python analysis/counterfactual_analysis.py

# Class-conditional question-marker rates, Table 4 and Section 4.6
python analysis/class_conditional_qrate.py

# Sentence-embedding capacity check, Figure 2 and Section 4.4
python analysis/semantic_baseline.py

# BERT cross-dataset transfer, Figure 2 and Section 4.4
python analysis/bert_transfer.py \
    --ketod-train  data/ketod/train_full.csv \
    --ketod-test   data/ketod/test_full.csv \
    --dstc9-train  data/dstc9/train_dstc9.csv \
    --dstc9-test   data/dstc9/test_dstc9.csv \
    --dstc11-train data/dstc11/train.csv \
    --dstc11-test  data/dstc11/val.csv \
    --output-dir   results/
```

Note on DSTC9 training file: `train_dstc9.csv` may contain malformed rows due to unescaped quotes in dialogue text. If `pandas.read_csv` errors out, create a cleaned copy first:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/dstc9/train_dstc9.csv', engine='python', on_bad_lines='skip')
df.to_csv('data/dstc9/train_dstc9_fixed.csv', index=False)
print(f'Saved {len(df)} rows')
"
```

Then pass `--dstc9-train data/dstc9/train_dstc9_fixed.csv` to `bert_transfer.py`.

---

## Results

Pre-computed result files are in `results/`. Key numbers reproduced in the paper:

| Experiment | Key result |
|---|---|
| Feature ablation, Table 2 | Question-type features are strong on DSTC9/11; KETOD shows no single dominant feature group. |
| Spearman correlation, Section 4.2 | DSTC9 vs DSTC11: ρ = +0.94, p < 0.001. KETOD vs DSTC: mean ρ ≈ −0.25, p > 0.1. |
| Structural LR transfer, Table 3 | DSTC→KETOD minority F1 ≤ 0.22 and ROC-AUC ≈ 0.48. Macro F1 alone is less informative because of KETOD class imbalance. |
| Sentence-embedding transfer, Section 4.4 | DSTC→KETOD minority F1 = 0.12 / 0.13, compared with KETOD in-domain 0.36. |
| Fine-tuned BERT transfer, Section 4.4 | DSTC→KETOD minority F1 = 0.10 / 0.07, while KETOD in-domain minority F1 reaches 0.50. |
| LR–MHA agreement, Section 4.5 | Agreement 67.2%, κ = 0.25; LR false-negative cases overlap strongly with MHA failures. |
| Counterfactual flip, Section 4.5 | Perturbing `user_has_question` flips 18.1% of LR predictions and 2.4% of MHA predictions. |
| Class-conditional question rate, Table 4 | DSTC9/DSTC11 positives are almost always questions; KETOD positives and negatives have similar question-marker rates. |

Result file layout:

```text
results/
├── lr_results.csv                  # Table 2 feature-subset ablation
├── transfer_results.csv            # Table 3 and Appendix Table 6
├── spearman_rho_results.csv        # Section 4.2
├── feature_importance.csv          # Figure 1
├── position_shuffle_results.csv    # Section 4.1 permutation test
├── threshold_tuning_results.csv    # KETOD threshold-tuning diagnostic
├── agreement_results.csv           # Section 4.5 LR vs MHA
├── counterfactual_results.csv      # Section 4.5 counterfactual flip
├── mha_predictions.csv             # MHA test-set predictions on KETOD
├── class_conditional_qrate.csv     # Table 4
├── semantic_results.csv            # Section 4.4 semantic baseline transfer
├── semantic_summary.txt
├── bert_results.csv                # Section 4.4 BERT transfer
└── bert_summary.txt
```

---

## Features

Ten hand-crafted structural features used for probing:

| # | Feature | Description |
|---:|---|---|
| 1 | `turn_position_ratio` | turn index / total turns |
| 2 | `turn_position_squared` | squared turn-position ratio |
| 3 | `user_turn_len_log` | log(1 + user-turn tokens) |
| 4 | `sys_turn_len_log` | log(1 + previous-system-turn tokens) |
| 5 | `dialogue_len_log` | log(total turns) |
| 6 | `consecutive_sys_turns` | number of consecutive system turns before the current user turn |
| 7 | `turn_len_ratio` | user length / system length, clipped to [0, 5] |
| 8 | `user_has_question` | user turn contains `?` |
| 9 | `prev_sys_is_question` | previous system turn ends with `?` |
| 10 | `user_starts_question_word` | user turn starts with what / how / where / when / why / is / does / can / do |

No utterance text or semantic content is accessed by the structural LR probe.

---

## Review-stage note

This repository is for a double-blind ARR 2026 submission. If used during review, the repository should be anonymized or submitted as supplementary material without author-identifying metadata.

---

## Citation

This work is currently under double-blind review at ARR 2026. Citation information will be released after the review process.

---

## License

Code is released under the MIT License. Datasets are used under their respective licenses; please refer to the source repositories for details.

