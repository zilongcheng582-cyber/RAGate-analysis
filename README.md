# Probing Benchmark-Specific Regularities in Knowledge Gating via Lightweight Feature Analysis

Code for the experiments in the accompanying paper, *Probing Benchmark-Specific
Regularities in Knowledge Gating via Lightweight Feature Analysis*. The package
converts the released raw data into the paper's processed inputs, runs the
lightweight structural analyses, and includes MiniLM and fine-tuned BERT
representation checks.

The repository does **not** redistribute KETOD, SGD, DSTC9, or DSTC11 data;
model weights; checkpoints; prediction files; or embedding caches. Follow the
original data licences and terms of use.

## Paper scope and reported findings

The paper uses logistic regression over ten dialogue-metadata cues (position,
length, and question form), without utterance semantics, as a **dataset-audit
probe** for knowledge-gating labels. It does not present this probe as an
online gating model or as a causal account of how the benchmarks were built.

- Transfer within the related DSTC9/DSTC11 pair is strong (macro F1
  0.836 and 0.810; ROC-AUC 0.900 and 0.906).
- Across the KETOD--DSTC boundary, structural-transfer ordering is weak
  (ROC-AUC 0.412--0.481), and DSTC-to-KETOD positive-class F1 is
  0.224/0.209.
- The qualitative DSTC-to-KETOD weakness also appears with accumulated-context
  MiniLM embeddings and fine-tuned BERT. The paper separately audits exact
  source-training/target-held-out input overlap after tokenization and left
  truncation.

The paper treats KETOD versus DSTC9/11 as a benchmark-family comparison:
the datasets differ jointly in corpus, annotation procedure, and label
distribution, so the reported gap is not a causal attribution to one factor.

## Datasets and evaluation splits

| Dataset | Train | Held-out | Training ratio (negative:positive) | Paper description |
|---|---:|---:|---:|---|
| KETOD | 41,939 | 4,964 | 7.4:1 | Entity-linked enrichment of SGD dialogues |
| DSTC9 | 71,348 | 9,663 | 2.7:1 | Knowledge-seeking turns |
| DSTC11 | 28,431 | 4,173 | 1.0:1 | Subjective knowledge seeking |

KETOD uses its released test split. DSTC9 and DSTC11 use their released
validation splits as held-out evaluation; a code variable named `test` refers
to this held-out split, not a hidden leaderboard test.

## Repository layout

- `preprocessing/`: deterministic conversion from the released raw data to the
  processed-text and structural-feature CSVs used by this paper.
- `lightweight/`: LR ablation, feature ranking, transfer, No-Q,
  threshold-calibration, question-rate, and position-permutation analyses.
- `minilm/`: accumulated-context MiniLM + LR transfer experiment and input
  truncation audit.
- `bert/`: fine-tuned BERT transfer experiment and input audit.
- `reference_results/`: compact result tables reported in the paper.
- `docs/`: protocol and paper-to-file mapping.

## Environment

The lightweight experiments run on CPU:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

For MiniLM, install `requirements-minilm.txt`. The formal MiniLM run used
`sentence-transformers==5.2.2`,
`sentence-transformers/all-MiniLM-L6-v2`, and seed 42.

The BERT experiment requires Linux and a CUDA GPU with at least 24 GB VRAM.
The reported BERT run used Python 3.12.3, PyTorch 2.5.1+cu124, CUDA 12.4, and
`transformers==5.3.0`; install a CUDA-enabled PyTorch build appropriate for
your system, then install `requirements-bert.txt`.

## Data acquisition and preprocessing

Download the upstream releases without modifying their contents:

- KETOD: <https://github.com/facebookresearch/ketod>
- Schema-Guided Dialogue (SGD), required to recover the KETOD dialogue text:
  <https://github.com/google-research-datasets/dstc8-schema-guided-dialogue>
- DSTC9 Track 1: <https://github.com/alexa/alexa-with-dstc9-track1-dataset>
- DSTC11 Track 5: <https://github.com/alexa/dstc11-track5>

Place these releases under one local `raw_data/` directory. The converter
locates KETOD annotations, SGD dialogue shards, and the DSTC train/validation
`logs.json`/`labels.json` files recursively. It writes the paper inputs to the
ignored `data/{ketod,dstc9,dstc11}/` directories:

```bash
python preprocessing/prepare_all.py \
  --raw-root raw_data \
  --output-root data
```

The command checks the expected split sizes and fails on malformed speakers,
labels, or incompatible raw layouts. KETOD uses its released test split;
DSTC9 and DSTC11 use their released validation splits as held-out evaluation.
KETOD processing recovers the dialogue text from SGD shards, while DSTC9/11
processing reads the released dialogue logs and labels. See
[data/README.md](data/README.md) for expected files and licences.

## Reproduce the lightweight analyses

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

These commands reproduce the structural results: Table 2, Figure 2, Table 3,
Table 4, and the Appendix D--E controls. The exact code/output mapping is in
[docs/PAPER_RESULTS_MAP.md](docs/PAPER_RESULTS_MAP.md).

## Reproduce the MiniLM check

```bash
python -m pip install -r requirements-minilm.txt

python minilm/minilm_input_audit.py \
  --config config_paths.json --data-root . --output-dir outputs/minilm

python minilm/minilm_transfer_ready.py \
  --config config_paths.json --data-root . --output-dir outputs/minilm \
  --truncation-side left --max-len 256

python minilm/verify_minilm_results.py --results-dir outputs/minilm
```

On the first run, allow the public `all-MiniLM-L6-v2` model to download. Use
`--local-files-only` only after that exact model is present in the local cache.

## Reproduce the BERT check

```bash
python -m pip install -r requirements-bert.txt
bash bert/run_autodl.sh
```

The runner trains the full `bert-base-uncased` model for three epochs with
explicit left truncation at 256 tokens and saves outputs under `outputs/bert/`.
It does not save checkpoints unless `--save-models` is explicitly passed to
`bert_transfer_ready.py`.

## Interpreting reruns

- All formal experiments use seed 42. The full model, hyperparameter, split,
  feature, and evaluation definitions are in
  [docs/PROTOCOL.md](docs/PROTOCOL.md).
- `reference_results/` is provided for comparison with the paper; rerunning
  GPU experiments can show small platform-dependent numerical differences.
- The structural probe is a dataset audit. KETOD position and dialogue-length
  variables are retrospective metadata, not features available at inference
  time.

## Licence

The original datasets retain their own licences and must not be redistributed
through this repository. The authors should add the intended outbound software
licence for this repository before public archival.
