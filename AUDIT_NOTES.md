# Code audit notes

Audit date: 2026-08-24.

## Result-level status

No new result-changing bug was found in the archived LR/Full-transfer, No-Q,
nested threshold-calibration, grouped position-permutation, corrected MiniLM,
or corrected BERT protocols. The archived numerical reference values were not
changed by this audit.

## Release fixes applied

- Removed a personal absolute path from the archived MiniLM input-audit summary
  and strengthened package scanning to detect JSON-escaped Windows paths.
- Removed the unsafe MiniLM `--reuse-embeddings` input path; formal runs always
  recompute embeddings from the exact current text/model configuration.
- Made the formal MiniLM and BERT scripts fail early if a different model
  identifier is supplied.
- Made MiniLM result verification enforce the canonical model identity.
- Added a DSTC9 logs/labels length check to prevent silent `zip()` truncation.
- Made DSTC11 processed-label parsing fail closed on unexpected values.
- Renamed the lightweight `pr_auc` result field to `average_precision`, matching
  the actual `sklearn.metrics.average_precision_score` computation. Numerical
  values are unchanged.
- Clarified that KETOD position/dialogue-length features use completed-dialogue
  USER-turn counts and are retrospective audit metadata.

## Remaining artifact boundary

The package hashes the exact processed-text CSVs used for MiniLM/BERT and the
feature CSVs used for the structural experiments, but it does not contain the
complete raw-to-processed text serialization pipeline for all three
benchmarks. This is a reproducibility/packaging gap rather than a discrepancy
in the archived reported results. Before public archival, release or
executable-document that conversion step if licensing permits.
