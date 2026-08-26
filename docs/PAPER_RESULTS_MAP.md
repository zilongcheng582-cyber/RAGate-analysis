# Paper-to-artifact map

| Paper evidence | Reproduction code | Archived reference output |
|---|---|---|
| Dataset statistics and ten feature definitions | `data_processing/` and hashed feature CSV inputs | `data/data_manifest.csv` |
| LR feature ablation table | `lightweight/train_lr_ablation.py` | `reference_results/lightweight/lr_results.csv` |
| Feature rankings and Spearman comparison | `lightweight/feature_importance_spearman.py` | `feature_importance.csv`, `spearman_rho_results.csv` |
| Full 3x3 structural transfer | `lightweight/run_transfer_controls.py` | `baseline_verification.csv` |
| No-Q transfer control | `lightweight/run_transfer_controls.py` | `no_question_*.csv` |
| Threshold-calibration sensitivity | `lightweight/run_transfer_controls.py` | `threshold_calibration_*.csv` |
| Current-turn question-rate control | `lightweight/run_feature_controls.py` | `class_conditional_qrate_fixed.csv` |
| Joint position-permutation control | `lightweight/run_feature_controls.py` | `position_shuffle_group_fixed.csv` |
| Corrected MiniLM transfer and truncation audit | `minilm/` | `reference_results/minilm/` |
| Corrected BERT transfer and input audit | `bert/` | `reference_results/bert/` |

The reference outputs are compact verification artifacts. Prediction-level
CSVs, embedding caches, model checkpoints, pretrained weights and benchmark
payloads are intentionally excluded from this public code package.
