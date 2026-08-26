# Knowledge-gating benchmark audit: camera-ready code

This package reproduces the experiments supporting the camera-ready paper. It
contains the lightweight structural probe and controls, the corrected MiniLM
representation check, and the corrected fine-tuned BERT representation check.
The removed MHA experiment is not part of this artifact.

## Package scope

- `lightweight/`: LR ablation, feature ranking, 3x3 transfer, No-Q,
  threshold-calibration, current-turn question rate and grouped position
  permutation.
- `minilm/`: accumulated-context MiniLM transfer, explicit left-256 truncation,
  truncation audit and overlap sensitivity.
- `bert/`: accumulated-context BERT transfer, explicit left-256 truncation,
  input audit, 3x3 evaluation and fail-closed result validation.
- `data_processing/`: historical feature-extraction interfaces.
- `preprocessing/`: deterministic raw-benchmark to processed-text and structural-feature conversion, inventory, and fail-closed verification.
- `reference_results/`: compact outputs used to check the paper values.
- `docs/`: exact protocol and paper-to-artifact mapping.
- `scripts/verify_package.py`: dependency-free package/protocol audit.

Benchmark files, weights, checkpoints, predictions, caches and logs are not
included. See `data/README.md` for data placement and split provenance.

## 1. Verify the downloaded package

From the package root:

```bash
python scripts/verify_package.py
```

This checks file completeness, Python syntax, protocol constants, formal 3x3
result matrices, absence of removed experiment code, absence of model/data
payloads, relative path configuration and every entry in `SHA256SUMS.txt`.

## 2. Lightweight experiments

Create a CPU environment and place the six feature CSVs under `data/`:

```bash
python -m venv .venv-lightweight
python -m pip install -r requirements-lightweight.txt

python lightweight/train_lr_ablation.py \
  --data-root . --output-dir outputs/lightweight/core

python lightweight/feature_importance_spearman.py \
  --data-root . --output-dir outputs/lightweight/core

python lightweight/run_transfer_controls.py \
  --config config_paths.json --output-dir outputs/lightweight/controls

python lightweight/run_feature_controls.py \
  --config config_paths.json --output-dir outputs/lightweight/controls
```

These experiments use NumPy, pandas and scikit-learn and do not require a GPU.

## 3. Corrected MiniLM experiment

Use a separate environment, place the six processed-text CSVs under `data/`,
and run:

```bash
python -m venv .venv-minilm
python -m pip install -r requirements-minilm.txt

python minilm/minilm_input_audit.py \
  --config config_paths.json --data-root . --output-dir outputs/minilm

python minilm/minilm_transfer_ready.py \
  --config config_paths.json --data-root . --output-dir outputs/minilm \
  --truncation-side left --max-len 256
```

Omit network restrictions on the first run so that
`sentence-transformers/all-MiniLM-L6-v2` can be downloaded. Add
`--local-files-only` only when that exact model is already cached. The formal run used `sentence-transformers/all-MiniLM-L6-v2`, sentence-transformers 5.2.2, and seed 42. The formal scripts fail closed if a different MiniLM model identifier is supplied.

## 4. Corrected BERT experiment

Use a Linux CUDA environment with at least 24 GB GPU memory. Keep the
environment's CUDA-enabled PyTorch and install only:

```bash
python -m pip install -r requirements-bert.txt
bash bert/run_autodl.sh
```

The runner verifies the separately supplied CSV hashes before training and
fails closed unless the complete protocol and 3x3 output are present. If the
Hugging Face endpoint is inaccessible in mainland China, the public model can
be fetched through a configured mirror, for example:

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash bert/run_autodl.sh
```

No checkpoint is saved unless the underlying Python command is explicitly
invoked with `--save-models`.

## 5. Reproducibility notes

- KETOD uses its released test split. DSTC9/DSTC11 use released validation
  splits as held-out evaluation sets.
- The processed `input` field is accumulated dialogue context for all three
  datasets and ends at the evaluated user turn.
- MiniLM and BERT both use explicit left truncation at 256 tokens.
- Source selection never uses held-out target labels. The target-development
  threshold is reported only as an optimistic sensitivity analysis.
- Formal reference values are under `reference_results/`; the authoritative
  paper-to-file mapping is `docs/PAPER_RESULTS_MAP.md`.

For the exact definitions and interpretation boundary, see
`docs/PROTOCOL.md`.


## 6. Raw-to-processed preprocessing

The package does not redistribute benchmark payloads. Place the upstream raw
releases in a local directory, preserving their original JSON files. KETOD
requires both its released annotation archive and the matching Google SGD
dialogue release; DSTC9 and DSTC11 require their official train/validation
`logs.json` and `labels.json` files.

Run the deterministic conversion into a directory outside the package:

```bash
python preprocessing/prepare_all.py \
  --raw-root "<path-to-upstream-raw-data>" \
  --output-root reproduced_data
```

Then verify the generated files against the package manifest. If private
historical canonical files are available locally, pass their dataset roots to
the optional canonical-root arguments for byte and semantic comparison:

```bash
python preprocessing/verify_preprocessing.py \
  --generated-root reproduced_data \
  --manifest data/data_manifest.csv \
  --report-dir preprocessing
```

The converter fails closed on malformed required fields, split length
mismatches, unknown speakers, unexpected labels, row-count violations, and
feature invariant violations. It never writes into the raw directory or
overwrites historical files. See `preprocessing/PREPROCESSING_AUDIT.md` and
`preprocessing/preprocessing_manifest.json` for the verified staging record.
