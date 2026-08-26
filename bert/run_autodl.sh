#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PACKAGE_ROOT"
mkdir -p outputs/bert/bert_ready

export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false

MODEL_NAME="bert-base-uncased"

echo "[0/4] Checking CUDA and package versions"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU is required'; print('torch=', torch.__version__); print('cuda=', torch.version.cuda); print('gpu=', torch.cuda.get_device_name(0)); print('vram_gb=', round(torch.cuda.get_device_properties(0).total_memory/1e9, 2))"
python -c "import numpy, pandas, sklearn, transformers, safetensors; print('numpy=', numpy.__version__); print('pandas=', pandas.__version__); print('sklearn=', sklearn.__version__); print('transformers=', transformers.__version__); print('safetensors=', safetensors.__version__)"

echo "[1/4] Verifying separately supplied benchmark CSVs"
sha256sum --check bert/DATA_SHA256.txt

echo "[2/4] Auditing processed inputs and truncation"
python -u bert/bert_input_audit.py \
  --config config_paths.json \
  --model-name "$MODEL_NAME" \
  --max-len 256 \
  --output outputs/bert/bert_input_audit.csv \
  2>&1 | tee outputs/bert/bert_input_audit.log

echo "[3/4] Running corrected camera-ready BERT transfer"
python -u bert/bert_transfer_ready.py \
  --config config_paths.json \
  --model-name "$MODEL_NAME" \
  --output-dir outputs/bert/bert_ready \
  --epochs 3 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --max-len 256 \
  --truncation-side left \
  --seed 42 \
  --num-workers 0 \
  --save-predictions \
  2>&1 | tee outputs/bert/bert_ready/train.log

echo "[4/4] Validating result completeness and protocol metadata"
python -u bert/verify_bert_results.py \
  --result-dir outputs/bert/bert_ready \
  --output outputs/bert/bert_ready/camera_ready_bert_summary.json

echo "AutoDL BERT rerun completed successfully."
