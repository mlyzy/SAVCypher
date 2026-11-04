#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${ROOT}:${ROOT}/convert:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${ROOT}"
deepspeed --include="${CUDA_INCLUDE:-localhost:0,1}" \
  --master_port="${MASTER_PORT:-25000}" \
  train/train.py \
  --base-model "${BASE_MODEL:-${ROOT}/../models/Qwen__Qwen3.6-35B-A3B}" \
  --train-file "${TRAIN_FILE:-${ROOT}/sirc_data/mtl/train.jsonl}" \
  --dev-file "${DEV_FILE:-${ROOT}/sirc_data/mtl/dev.jsonl}" \
  --output-dir "${OUTPUT_DIR:-${ROOT}/outputs/qwen36_sirc_mlora}" \
  --deepspeed "${ROOT}/train/config/ds3.json" \
  --max-length "${MAX_LENGTH:-1024}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-16}" \
  --epochs "${EPOCHS:-3}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  "$@"
