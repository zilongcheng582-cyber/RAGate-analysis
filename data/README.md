# Data placement

Benchmark data are not redistributed. Obtain KETOD, DSTC9 and DSTC11 from
their original sources and place the six processed-text CSVs and six
structural-feature CSVs as follows:

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

`test_dstc9.csv` and both DSTC9 `test_features.csv` refer to the released
validation evaluation split. For DSTC11, `val.csv` and `test_features.csv`
refer to the released validation evaluation split, not an official leaderboard
test set.

The processed-text files must contain `input` and `output`. The `input` field
is accumulated dialogue context ending at the evaluated user turn. The feature
files must contain `label` plus the ten columns documented in
`docs/PROTOCOL.md`.

`data_manifest.csv` records the hashes and schemas of the exact private files
used for the camera-ready experiments. Run:

```bash
python scripts/verify_package.py --verify-data --data-root .
```

to check separately supplied files against that manifest.

The scripts in `data_processing/` preserve the historical feature-extraction
interfaces. The executable raw-to-processed chain is in `preprocessing/`:

```bash
python preprocessing/prepare_all.py \
  --raw-root "<path-to-upstream-raw-data>" \
  --output-root reproduced_data

python preprocessing/verify_preprocessing.py \
  --generated-root reproduced_data \
  --manifest data/data_manifest.csv \
  --report-dir preprocessing
```

The KETOD stage joins the released KETOD annotations to Google SGD by
`dialogue_id`; DSTC9 and DSTC11 serialize the official raw logs/labels into
the canonical accumulated-context CSVs. No raw or generated benchmark data
is committed. The hashed CSVs in `data/data_manifest.csv` remain the
authoritative inputs for reproducing the reported lightweight results.
