# Paper-to-code map

This map follows the tables, figures, and appendices in the accompanying
paper. `reference_results/` contains compact reported-result tables for
comparison; it does not contain benchmark data, predictions, embedding caches,
model weights, or checkpoints.

| Paper item | Reproduction entry point | Reference result |
|---|---|---|
| Table 1: datasets, held-out splits, and class ratios | `preprocessing/prepare_all.py`; [data/README.md](../data/README.md); `docs/PROTOCOL.md` | Data are obtained from their upstream releases and are not committed. |
| Table 2: feature-subset LR ablation | `lightweight/train_lr_ablation.py` | `reference_results/lightweight/lr_results.csv` |
| Figure 2: coefficient ranks and Spearman comparison | `lightweight/feature_importance_spearman.py` | `reference_results/lightweight/feature_importance.csv`, `spearman_rho_results.csv` |
| Table 3: 3 × 3 structural transfer | `lightweight/run_transfer_controls.py` | `reference_results/lightweight/baseline_verification.csv` |
| Table 4: current-turn question-form rates | `lightweight/run_feature_controls.py` | `reference_results/lightweight/class_conditional_qrate_fixed.csv` |
| Figure 3: MiniLM and BERT representation checks | `minilm/minilm_transfer_ready.py`; `bert/run_autodl.sh` | `reference_results/minilm/minilm_results_ready.csv`, `reference_results/bert/bert_results_ready.csv` |
| Appendix B: tokenization/truncation and input-overlap audits | `minilm/minilm_input_audit.py`; `bert/bert_input_audit.py` | `reference_results/minilm/minilm_input_audit*`, `reference_results/bert/bert_input_audit.csv` |
| Appendix D: no-question and position-permutation controls | `lightweight/run_transfer_controls.py`; `lightweight/run_feature_controls.py` | `reference_results/lightweight/no_question_*.csv`, `position_shuffle_group_fixed.csv` |
| Appendix E: source and target threshold-calibration analysis | `lightweight/run_transfer_controls.py` | `reference_results/lightweight/threshold_calibration_*.csv` |
