# Paper-to-code and results map

| Paper evidence | Reproduction code | Compact reference output |
|---|---|---|
| Dataset statistics and ten feature definitions | `preprocessing/`, `data_processing/` | `data/data_manifest.csv` |
| LR feature ablation table | `lightweight/train_lr_ablation.py` | `reference_results/lightweight/lr_results.csv` |
| Figure 2: within-dataset normalized absolute standardized LR coefficients | `lightweight/feature_importance_spearman.py`; training-free entry `scripts/make_paper_figures.py` | `reference_results/lightweight/feature_importance.csv`; output `figure2_feature_importance.png` |
| Feature rankings and Spearman comparison | `lightweight/feature_importance_spearman.py` | `reference_results/lightweight/feature_importance.csv`, `reference_results/lightweight/spearman_rho_results.csv` |
| Full 3x3 structural transfer | `lightweight/run_transfer_controls.py` | `reference_results/lightweight/baseline_verification.csv` |
| No-Q transfer control | `lightweight/run_transfer_controls.py` | `reference_results/lightweight/no_question_*.csv` |
| Threshold-calibration sensitivity | `lightweight/run_transfer_controls.py` | `reference_results/lightweight/threshold_calibration_*.csv` |
| Current-turn question-rate control | `lightweight/run_feature_controls.py` | `reference_results/lightweight/class_conditional_qrate_fixed.csv` |
| Joint position-permutation control | `lightweight/run_feature_controls.py` | `reference_results/lightweight/position_shuffle_group_fixed.csv` |
| Corrected MiniLM transfer and truncation audit | `minilm/` | `reference_results/minilm/` |
| Corrected BERT transfer and input audit | `bert/` | `reference_results/bert/` |
| Figure 3: positive-class F1 on KETOD for structural LR, embedding + LR, and fine-tuned BERT | `scripts/make_paper_figures.py` | reads `baseline_verification.csv`, `minilm_results_ready.csv`, and `bert_results_ready.csv`; output `figure3_representation_checks.png` |

Regenerate both paper figures from the compact reference CSVs without training:

```bash
python scripts/make_paper_figures.py \
  --reference-root reference_results \
  --output-dir outputs/paper_figures
```

For Figure 2, raw absolute coefficients remain unchanged in the CSV and in the
Spearman calculation. Only the plotting layer divides each dataset's values by
that dataset's maximum, so every nonzero dataset maximum is 1. Rank display
uses the explicit `method="min"` tie policy; the Spearman statistic continues
to operate directly on raw coefficient importance. Both plotting entry points
use the single paper order and abbreviation specification in `figure2_spec.py`.

`reference_results/minilm/` is a compact paper-reference directory containing
formal tables and small audits. It intentionally omits the nine prediction
CSVs and full `run_metadata.json`, and therefore is not a valid full-verifier
bundle. A real rerun writes the complete bundle to `outputs/minilm/`; validate
that directory with:

```bash
python minilm/verify_minilm_results.py --results-dir outputs/minilm
```

The reference outputs are compact verification files. Prediction-level
CSVs, embedding caches, model checkpoints, pretrained weights, and benchmark
payloads are intentionally excluded from this public repository.
