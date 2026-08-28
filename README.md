# RAGate-analysis

Official reproducibility repository for:

**Probing Benchmark-Specific Regularities in Knowledge Gating via Lightweight Feature Analysis**<br>
EMNLP 2026

This repository provides the preprocessing pipeline, lightweight structural
probe, MiniLM and BERT representation checks, and compact reference results
used in the paper.

## A. License

Original software in this repository is released under the [MIT License](LICENSE).
KETOD, SGD, DSTC9, and DSTC11 remain subject to their respective upstream
licenses. Their benchmark data are not redistributed here, and this repository's
MIT License grants no rights to third-party data or code.

## B. Environment

Use separate environments for the three experiment families. The lightweight
experiments require NumPy, pandas, scikit-learn, SciPy, and Matplotlib and run
on CPU:

```bash
python -m venv .venv-lightweight
source .venv-lightweight/bin/activate
# Windows PowerShell:
# .\.venv-lightweight\Scripts\Activate.ps1

python -m pip install -r requirements-lightweight.txt
```

MiniLM additionally uses sentence-transformers and PyTorch:

```bash
python -m venv .venv-minilm
source .venv-minilm/bin/activate
# Windows PowerShell:
# .\.venv-minilm\Scripts\Activate.ps1

python -m pip install -r requirements-minilm.txt
```

BERT requires a Linux CUDA environment with at least 24 GB GPU memory. Keep the
environment's CUDA-enabled PyTorch, then install:

```bash
python -m pip install -r requirements-bert.txt
```

Repository and locator tests use:

```bash
python -m pip install -r requirements-test.txt
```

## C. Data acquisition

Obtain, license, and unpack data yourself from the official upstream repositories:

- KETOD: <https://github.com/facebookresearch/ketod>
- Schema-Guided Dialogue (SGD): <https://github.com/google-research-datasets/dstc8-schema-guided-dialogue>
- DSTC9 Track 1: <https://github.com/alexa/alexa-with-dstc9-track1-dataset>
- DSTC11 Track 5: <https://github.com/alexa/dstc11-track5>

The KETOD release ZIP must be extracted. `prepare_ketod.py` recursively locates
the extracted `train_ketod.json` and `test_ketod.json`, plus SGD's
`train/dialogues_*.json` and `test/dialogues_*.json`. Upstream repositories may
keep their default clone names. One valid layout is:

```text
raw_data/
├── ketod/
│   ├── ... upstream repository files ...
│   └── ... extracted release .../
│       ├── train_ketod.json
│       └── test_ketod.json
├── dstc8-schema-guided-dialogue/
│   ├── train/
│   │   └── dialogues_*.json
│   ├── dev/
│   │   └── dialogues_*.json
│   └── test/
│       └── dialogues_*.json
├── alexa-with-dstc9-track1-dataset/
│   └── data/
│       ├── train/
│       │   ├── logs.json
│       │   └── labels.json
│       └── val/
│           ├── logs.json
│           └── labels.json
└── dstc11-track5/
    └── data/
        ├── train/
        │   ├── logs.json
        │   └── labels.json
        └── val/
            ├── logs.json
            └── labels.json
```

See [docs/DATA_ACQUISITION.md](docs/DATA_ACQUISITION.md) for provenance,
split semantics, and the versioning limitation. No upstream commit/tag was
recorded in the original experiment provenance, so none is claimed here.

## D. Preprocessing

From the repository root, convert all three benchmarks with:

```bash
python preprocessing/prepare_all.py \
  --raw-root raw_data \
  --output-root data
```

The converter validates these exact released example counts:

| Dataset | Training | Held-out | Released held-out split |
|---|---:|---:|---|
| KETOD | 41,939 | 4,964 | test |
| DSTC9 | 71,348 | 9,663 | validation |
| DSTC11 | 28,431 | 4,173 | validation |

In processed filenames and code, `test` means the held-out split used by this
paper; for DSTC9/DSTC11 it does not mean a hidden leaderboard test. The
converter never downloads, deletes, or modifies upstream data. Generated files
can be checked with `preprocessing/verify_preprocessing.py`; see
[data/README.md](data/README.md).

## E. Lightweight reproduction

After preprocessing, run the four CPU commands:

```bash
python lightweight/train_lr_ablation.py \
  --data-root . --output-dir outputs/lightweight/core

python lightweight/feature_importance_spearman.py \
  --data-root . --output-dir outputs/lightweight/core

python lightweight/run_transfer_controls.py \
  --config config_paths.json --output-dir outputs/lightweight/controls

python lightweight/run_feature_controls.py \
  --config config_paths.json --output-dir outputs/lightweight/controls
```

The feature-importance command writes the paper-ready normalized
`figure2_feature_importance.png` separately from the unnormalized signed
diagnostic plot.

## F. MiniLM reproduction

The formal model is `sentence-transformers/all-MiniLM-L6-v2`, with accumulated
context, explicit left truncation at 256 tokens, and seed 42:

```bash
python minilm/minilm_input_audit.py \
  --config config_paths.json --data-root . --output-dir outputs/minilm

python minilm/minilm_transfer_ready.py \
  --config config_paths.json --data-root . --output-dir outputs/minilm \
  --truncation-side left --max-len 256

python minilm/verify_minilm_results.py \
  --results-dir outputs/minilm
```

Allow the official model to download on the first run. Use
`--local-files-only` only when that exact model is cached. The verifier requires
a complete rerun bundle: nine `pred_*.csv` files, `run_metadata.json`, result
tables, audits, and hashes. By contrast, `reference_results/minilm/` is a
compact paper-reference directory and is intentionally **not** valid input to the
full-run verifier.

## G. BERT reproduction

The formal BERT run uses full `bert-base-uncased` fine-tuning (no freezing),
three epochs, AdamW at `2e-5`, weight decay `0.01`, 10% linear warmup/decay,
batch size 32, class-frequency-weighted cross entropy, threshold 0.5, seed 42,
and explicit left truncation at 256 tokens:

```bash
bash bert/run_autodl.sh
```

No target calibration or dev early stopping is used. No checkpoint is saved
unless the underlying Python command is explicitly invoked with `--save-models`.

## H. Paper-to-code map

The authoritative mapping is [docs/PAPER_RESULTS_MAP.md](docs/PAPER_RESULTS_MAP.md).
Figures 2 and 3 can be regenerated without model training from compact CSVs:

```bash
python scripts/make_paper_figures.py \
  --reference-root reference_results \
  --output-dir outputs/paper_figures
```

## I. Reference results and repository verification

`reference_results/` contains compact reported-result tables and small audit
summaries. It is not raw benchmark data, a prediction-level verifier bundle,
an embedding cache, or a set of model artifacts. Prediction CSVs, weights,
checkpoints, benchmark payloads, caches, and logs are intentionally excluded.

Run the dependency-free repository/protocol sanity checker and the tests with:

```bash
python scripts/verify_package.py
pytest -q
```

For the exact experimental definitions and interpretation boundary, see
[docs/PROTOCOL.md](docs/PROTOCOL.md).
