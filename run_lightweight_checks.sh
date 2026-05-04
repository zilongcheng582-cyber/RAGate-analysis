#!/usr/bin/env bash
set -euo pipefail

python -m py_compile analysis/*.py probing/*.py data_processing/*.py mha/*.py

grep -R "E:/" -n . --exclude-dir=.git --exclude-dir=__pycache__ --exclude=run_lightweight_checks.sh && exit 1 || true
grep -R "/root/" -n . --exclude-dir=.git --exclude-dir=__pycache__ --exclude=run_lightweight_checks.sh && exit 1 || true
grep -R "C:/" -n . --exclude-dir=.git --exclude-dir=__pycache__ --exclude=run_lightweight_checks.sh && exit 1 || true

echo "Lightweight checks passed."
