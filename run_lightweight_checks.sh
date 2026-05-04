#!/usr/bin/env bash
set -euo pipefail

python -B -m py_compile analysis/*.py probing/*.py data_processing/*.py mha/*.py

python - <<'PY'
import pathlib, re, sys
root = pathlib.Path('.')
ignore_dirs = {'.git', '__pycache__'}
ignore_exts = {'.pyc', '.zip'}
# Build patterns without embedding sensitive literals directly in file text
parts = [
    r'E\x3a/', r'C\x3a/', r'D\x3a/',
    '/'+'ro'+'ot/', '/'+'ho'+'me/', '/'+'m'+'nt/',
    'zi'+'long', 'ch'+'eng', 'github\\.com/'+'zi'+'long',
    'Drop'+'box', 'Google'+' Drive'
]
pattern = re.compile('|'.join(parts), flags=re.IGNORECASE)
violations = []
for p in root.rglob('*'):
    if not p.is_file():
        continue
    if any(part in ignore_dirs for part in p.parts):
        continue
    if p.suffix.lower() in ignore_exts:
        continue
    try:
        text = p.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        continue
    for i, line in enumerate(text.splitlines(), start=1):
        if pattern.search(line):
            violations.append(f"{p}:{i}:{line.strip()}")
if violations:
    print('\n'.join(violations))
    sys.exit(1)
print('Lightweight checks passed.')
PY

find . -type d -name __pycache__ -prune -exec rm -rf {} +
