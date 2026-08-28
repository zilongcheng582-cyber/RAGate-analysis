# Preprocessing provenance

- Verification timestamp (UTC): `2026-08-25T08:22:21.613124+00:00`
- Status: `BYTE_EXACT = PASS` and `SEMANTIC_EXACT = PASS` for all 12 target files.
- KETOD held-out is the released test split.
- DSTC9 and DSTC11 held-out are released validation splits; neither is a hidden leaderboard test.
- Raw files are recorded relative to the staged `RAW_ROOT`; no raw or generated benchmark payload is part of the public repository.
- Canonical comparison paths were local-only and are not included in the public repository.
- KETOD upstream turn-length mismatches are recorded in `alignment_audit.json` and reproduce the released generator's explicit `zip()` truncation behavior.
- Upstream Git revisions are unavailable because they were not recorded in the original experiment provenance.
