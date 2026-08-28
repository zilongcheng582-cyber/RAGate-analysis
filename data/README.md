# Data placement and provenance

Benchmark data are not redistributed. Obtain KETOD, SGD, DSTC9, and DSTC11
from the official URLs in [`docs/DATA_ACQUISITION.md`](../docs/DATA_ACQUISITION.md),
accept their upstream terms, and place the extracted releases below a local
`raw_data/` directory. The default upstream clone names are supported.

KETOD depends on two upstream sources: its extracted release must expose
`train_ketod.json` and `test_ketod.json` somewhere below `raw_data/`, while the
SGD clone must expose `train/dialogues_*.json` and `test/dialogues_*.json`.
The official SGD `dev/` directory may remain present but is not consumed by
this converter. DSTC9 and DSTC11 each require `data/train/{logs,labels}.json`
and `data/val/{logs,labels}.json` under a repository directory whose component
contains the `dstc9` or `dstc11` token.

Run:

```bash
python preprocessing/prepare_all.py \
  --raw-root raw_data \
  --output-root data
```

The resulting layout is:

```text
data/
├── ketod/
│   ├── train_full.csv
│   ├── test_full.csv
│   ├── train_features.csv
│   └── test_features.csv
├── dstc9/
│   ├── train_dstc9.csv
│   ├── test_dstc9.csv
│   ├── train_features.csv
│   └── test_features.csv
└── dstc11/
    ├── train.csv
    ├── val.csv
    ├── train_features.csv
    └── test_features.csv
```

KETOD `test_full.csv`/`test_features.csv` use the released test split. DSTC9
`test_dstc9.csv`/`test_features.csv` and DSTC11 `val.csv`/`test_features.csv`
use released validation, not hidden leaderboard test data. The expected
train/held-out row counts are KETOD 41,939/4,964, DSTC9 71,348/9,663, and
DSTC11 28,431/4,173; preprocessing enforces these counts.

Processed text files contain accumulated dialogue context ending at the
evaluated USER turn and an `output` label. Feature files contain `label` plus
the exact ten columns in [`docs/PROTOCOL.md`](../docs/PROTOCOL.md).

`data_manifest.csv` records hashes and schemas of the private canonical inputs
used for the formal experiments. To check separately supplied canonical files:

```bash
python scripts/verify_package.py --verify-data --data-root .
```

To verify newly generated files structurally (and optionally compare private
canonical files when available):

```bash
python preprocessing/verify_preprocessing.py \
  --generated-root data \
  --manifest data/data_manifest.csv \
  --report-dir preprocessing
```

No raw or generated benchmark payload is committed. The repository MIT License
does not relicense or grant rights to any upstream benchmark data.
