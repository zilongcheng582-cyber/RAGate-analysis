# Data

This repository does not include benchmark data. Download the original
releases and comply with their terms of use:

- [KETOD](https://github.com/facebookresearch/ketod)
- [Schema-Guided Dialogue (SGD)](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue), used with KETOD
- [DSTC9 Track 1](https://github.com/alexa/alexa-with-dstc9-track1-dataset)
- [DSTC11 Track 5](https://github.com/alexa/dstc11-track5)

The SGD release is CC BY-SA 4.0. DSTC9 and DSTC11 publish their licence terms
in their respective repositories; DSTC11 Track 5 data are released under
CDLA-Sharing 1.0. This repository only contains code derived for the paper and
does not grant rights to those datasets.

Run the converter from the repository root:

```bash
python preprocessing/prepare_all.py --raw-root raw_data --output-root data
```

It creates the ignored files below:

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

The processed-text files contain `input` and `output`; `input` is accumulated
dialogue context ending at the evaluated user turn. The feature files contain
`label` and the ten structural features defined in
[docs/PROTOCOL.md](../docs/PROTOCOL.md). KETOD uses the released test split;
DSTC9 and DSTC11 use their released validation splits as held-out evaluation.
