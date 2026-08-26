"""Create preprocessing manifests and an audit summary from verified outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


OUTPUTS = [
    ("KETOD", "train", "ketod/train_full.csv"),
    ("KETOD", "test", "ketod/test_full.csv"),
    ("DSTC9", "train", "dstc9/train_dstc9.csv"),
    ("DSTC9", "held-out", "dstc9/test_dstc9.csv"),
    ("DSTC11", "train", "dstc11/train.csv"),
    ("DSTC11", "held-out", "dstc11/val.csv"),
    ("KETOD", "train", "ketod/train_features.csv"),
    ("KETOD", "test", "ketod/test_features.csv"),
    ("DSTC9", "train", "dstc9/train_features.csv"),
    ("DSTC9", "held-out", "dstc9/test_features.csv"),
    ("DSTC11", "train", "dstc11/train_features.csv"),
    ("DSTC11", "held-out", "dstc11/test_features.csv"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--preprocessing-dir", type=Path, required=True)
    parser.add_argument("--local-provenance-root", type=Path, required=True)
    parser.add_argument("--canonical-ketod-root", type=Path, required=True)
    parser.add_argument("--canonical-dstc9-root", type=Path, required=True)
    parser.add_argument("--canonical-dstc11-root", type=Path, required=True)
    args = parser.parse_args()
    now = datetime.now(timezone.utc).isoformat()
    inventory_path = args.preprocessing_dir / "raw_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.exists() else []
    verification = json.loads((args.preprocessing_dir / "verification_summary.json").read_text(encoding="utf-8"))
    manifest_path = args.preprocessing_dir.parent / "data" / "data_manifest.csv"
    if not verification.get("passed"):
        raise SystemExit("refusing to write provenance: verification_summary.json is not passed")
    verification_rows = {row["relative_path"]: row for row in verification.get("results", [])}
    expected_relatives = {relative for _, _, relative in OUTPUTS}
    if len(verification_rows) != len(OUTPUTS) or set(verification_rows) != expected_relatives:
        raise SystemExit("refusing to write provenance: verification summary must contain exactly 12 target results")
    generated_rows = []
    for dataset, split, relative in OUTPUTS:
        generated = args.generated_root / Path(relative).parts[0] / Path(relative).parts[1]
        if not generated.exists():
            raise FileNotFoundError(generated)
        result = verification_rows[relative]
        actual_generated_hash = sha256(generated)
        if result.get("generated_sha256", "").lower() != actual_generated_hash.lower():
            raise SystemExit(f"refusing to write provenance: verifier hash mismatch for {relative}")
        if result.get("byte_exact") is not True or result.get("semantic_exact") is not True:
            raise SystemExit(f"refusing to write provenance: canonical comparison is not exact for {relative}")
        canonical_hash = result.get("canonical_sha256")
        if not canonical_hash:
            raise SystemExit(f"refusing to write provenance: missing canonical hash for {relative}")
        with generated.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            row_count = sum(1 for _ in reader)
        generated_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "relative_path": relative,
                "raw_source_paths_relative_to_RAW_ROOT": "see raw_inventory.json",
                "generated_sha256": actual_generated_hash,
                "canonical_historical_path": "LOCAL-ONLY oracle; see local_provenance manifest",
                "canonical_historical_hash": canonical_hash,
                "row_count": row_count,
                "schema": ",".join(header),
                "byte_exact": result["byte_exact"],
                "semantic_exact": result["semantic_exact"],
                "known_split_semantics": "KETOD held-out=released test; DSTC9/DSTC11 held-out=released validation",
            }
        )
    converter_hashes = {
        path.name: sha256(path)
        for path in sorted(args.preprocessing_dir.glob("*.py"))
    }
    manifest = {
        "timestamp_utc": now,
        "git_commit": "unavailable: audited source was a ZIP without .git metadata",
        "raw_root_label": "WORK_ROOT/raw_extracted",
        "raw_sources": inventory,
        "converter_script_sha256": converter_hashes,
        "generated_files": generated_rows,
        "verification": verification,
        "split_semantics": {
            "KETOD held-out": "released test split",
            "DSTC9 held-out": "released validation split",
            "DSTC11 held-out": "released validation split",
        },
    }
    args.preprocessing_dir.mkdir(parents=True, exist_ok=True)
    (args.preprocessing_dir / "preprocessing_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    fields = list(generated_rows[0])
    with (args.preprocessing_dir / "preprocessing_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(generated_rows)
    audit_lines = [
        "# PREPROCESSING_AUDIT",
        "",
        f"- Verification timestamp (UTC): `{now}`",
        "- Status: `BYTE_EXACT = PASS` and `SEMANTIC_EXACT = PASS` for all 12 target files (6 processed-text CSVs and 6 structural-feature CSVs), each independently compared with the supplied canonical oracle and `data/data_manifest.csv`.",
        "- KETOD held-out is the released test split.",
        "- DSTC9 and DSTC11 held-out are released validation splits; neither is a hidden leaderboard test.",
        "- Raw files are recorded relative to the staged `RAW_ROOT`; no raw or generated benchmark payload is part of the public code package.",
        "- Exact E-drive oracle paths are kept only in the local provenance manifest, not this public-facing audit note.",
        "- KETOD upstream turn-length mismatches are recorded in `alignment_audit.json` and reproduce the released generator's explicit `zip()` truncation behavior.",
        "- Git commit is unavailable because the audited source artifact was a ZIP without `.git` metadata.",
        "",
    ]
    (args.preprocessing_dir / "PREPROCESSING_AUDIT.md").write_text("\n".join(audit_lines), encoding="utf-8", newline="\n")
    local = {
        "timestamp_utc": now,
        "canonical_historical_paths": {
            relative: str(
                (args.canonical_ketod_root / Path(relative).name)
                if dataset == "KETOD"
                else (args.canonical_dstc9_root / ("train" if split == "train" else "val") / Path(relative).name)
                if dataset == "DSTC9"
                else (args.canonical_dstc11_root / Path(relative).name)
            )
            for dataset, split, relative in OUTPUTS
        },
        "canonical_hashes": {relative: verification_rows[relative]["canonical_sha256"] for _, _, relative in OUTPUTS},
        "verification_summary": verification,
    }
    args.local_provenance_root.mkdir(parents=True, exist_ok=True)
    (args.local_provenance_root / "preprocessing_manifest.local.json").write_text(json.dumps(local, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote manifests for {len(generated_rows)} generated files")


if __name__ == "__main__":
    main()
