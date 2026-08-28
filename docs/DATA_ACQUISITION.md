# Data acquisition and split provenance

This repository does not redistribute raw benchmark data. Obtain each release
from its official upstream repository and comply with its own license:

| Resource | Official source | Files used here |
|---|---|---|
| KETOD | <https://github.com/facebookresearch/ketod> | extracted `train_ketod.json`, `test_ketod.json` |
| Schema-Guided Dialogue (SGD) | <https://github.com/google-research-datasets/dstc8-schema-guided-dialogue> | `train/dialogues_*.json`, `test/dialogues_*.json` |
| DSTC9 Track 1 | <https://github.com/alexa/alexa-with-dstc9-track1-dataset> | `data/train/` and `data/val/` logs/labels |
| DSTC11 Track 5 | <https://github.com/alexa/dstc11-track5> | `data/train/` and `data/val/` logs/labels |

## Placement

Clone or download the four sources below one `raw_data/` directory. Do not
rename official clone directories: `alexa-with-dstc9-track1-dataset`,
`dstc11-track5`, and `dstc8-schema-guided-dialogue` are recognized directly.
An alias directory named exactly `dstc9` or `dstc11` is also supported.

The KETOD release archive must be extracted first. The converter recursively
searches below `--raw-root`, so the two annotation JSON files may remain in the
release's extracted subdirectory. If the JSON files are absent while a KETOD
ZIP is present, preprocessing reports that the release still appears compressed.
It never downloads, deletes, or modifies upstream files.

SGD `dev/` may remain in the official clone. The formal converter reads SGD
`train/` and `test/`: training KETOD annotation IDs are joined to SGD train;
released KETOD test IDs are looked up in SGD test with the released generator's
train fallback behavior preserved and audited.

See the complete example tree in the repository [README](../README.md#c-data-acquisition).

## Formal split semantics

| Dataset | Formal training split | Formal held-out split | Expected examples |
|---|---|---|---:|
| KETOD | released train | released test | 41,939 train / 4,964 held-out |
| DSTC9 | released train | released validation | 71,348 train / 9,663 held-out |
| DSTC11 | released train | released validation | 28,431 train / 4,173 held-out |

The variable or filename `test` in shared experiment code means the held-out
evaluation split used by this paper. For DSTC9 and DSTC11 it does not designate
the hidden leaderboard test set. Preprocessing validates all six expected split
sizes and fails closed if they differ.

## Versioning boundary

The original experiment provenance did not record upstream commit hashes or tags.
Accordingly, this document does not claim that a current upstream HEAD is the
exact formal version and does not invent a revision identifier. The repository
instead validates the expected released split sizes during preprocessing and
provides hashes of the exact private processed inputs in `data/data_manifest.csv`.

All benchmark data retain their upstream licenses. The repository's MIT License
applies only to original repository software and grants no third-party data rights.
