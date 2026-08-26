# Artifact manifest

This is the canonical no-MHA camera-ready experiment code package.

Included:

- feature-extraction provenance scripts;
- deterministic raw-to-processed preprocessing and fail-closed verification scripts;
- preprocessing schemas, manifests, and small audit summaries without benchmark payloads;
- LR ablation, feature-importance, transfer, No-Q, calibration, question-rate
  and grouped-position-permutation code;
- corrected MiniLM code with explicit left-256 truncation;
- corrected BERT code with explicit left-256 truncation;
- protocol and paper-to-artifact documentation;
- compact formal result tables and input audits;
- private-data hashes and a complete package SHA-256 manifest.

Excluded:

- the removed MHA experiment and all associated source/checkpoints/results;
- KETOD, DSTC9 and DSTC11 payloads;
- pretrained and fine-tuned weights;
- prediction-level CSVs, logs, embedding caches and Python caches;
- obsolete review-stage MiniLM/BERT results;
- raw benchmark archives and generated benchmark CSVs;
- personal absolute paths and credentials.

The source package can therefore be uploaded for code review without
redistributing benchmark data or model weights.
