# PREPROCESSING_AUDIT

- Verification timestamp (UTC): `2026-08-25T09:41:41.724632+00:00`
- Status: `BYTE_EXACT = PASS` and `SEMANTIC_EXACT = PASS` for all 12 target files (6 processed-text CSVs and 6 structural-feature CSVs), each independently compared with the supplied canonical oracle and `data/data_manifest.csv`.
- KETOD held-out is the released test split.
- DSTC9 and DSTC11 held-out are released validation splits; neither is a hidden leaderboard test.
- Raw files are recorded relative to the staged `RAW_ROOT`; no raw or generated benchmark payload is part of the public code package.
- Exact E-drive oracle paths are kept only in the local provenance manifest, not this public-facing audit note.
- KETOD upstream turn-length mismatches are recorded in `alignment_audit.json` and reproduce the released generator's explicit `zip()` truncation behavior.
- Git commit is unavailable because the audited source artifact was a ZIP without `.git` metadata.
