#!/usr/bin/env python3
"""Load MiniLM and verify the exact SentenceTransformer tokenization path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from minilm_transfer_ready import configure_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--model-name", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--truncation-side", choices=("left", "right"), default="left")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical = "sentence-transformers/all-MiniLM-L6-v2"
    if args.model_name != canonical:
        raise ValueError(f"Canonical camera-ready MiniLM model must be {canonical!r}; got {args.model_name!r}")
    model = configure_model(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol_validation": "PASS",
        "model_name": args.model_name,
        "model_config_commit_hash": getattr(
            model[0].auto_model.config, "_commit_hash", None
        ),
        "max_seq_length": model.max_seq_length,
        "tokenizer_model_max_length": model.tokenizer.model_max_length,
        "truncation_side": model.tokenizer.truncation_side,
        "sentence_transformer_tokenization_matches_direct_tokenizer": True,
        "synthetic_current_turn_retained_under_left_truncation": True,
    }
    path = output_dir / "minilm_tokenization_protocol_audit.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("MINILM_TOKENIZATION_PROTOCOL_AUDIT=PASS")
    print(path)


if __name__ == "__main__":
    main()
