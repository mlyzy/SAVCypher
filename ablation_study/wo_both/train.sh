#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}/MTL-LoRA:${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${ROOT}"

deepspeed --include="${CUDA_INCLUDE:-localhost:0,1}" --master_port="${MASTER_PORT:-25000}" \
  Ablation_Study/wo_both/train.py \
  --base-model "${BASE_MODEL:-${ROOT}/models/Qwen__Qwen3.6-35B-A3B}" \
  --train-file "${TRAIN_FILE:-${ROOT}/natcypher_data/compact_v03_text2cypher/train.jsonl}" \
  --dev-file "${DEV_FILE:-${ROOT}/natcypher_data/compact_v03_text2cypher/dev.jsonl}" \
  --output-dir "${OUTPUT_DIR:-${ROOT}/outputs/ablation/wo_both}" \
  --deepspeed "${ROOT}/MTL-LoRA/config/ds3_natcypher.json" \
  --max-length "${MAX_LENGTH:-1024}" \
  --epochs "${EPOCHS:-3}" \
  --max-steps "${MAX_STEPS:--1}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --weight-decay 0.0 \
  --batch-size "${BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-16}" \
  --warmup-ratio 0.03 \
  --logging-steps 10 \
  --eval-strategy no \
  --adapter-save-steps 50 \
  --rank 8 \
  --alpha 16 \
  --dropout 0.05 \
  --target-modules q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_a,in_proj_b,in_proj_z,out_proj \
  --seed 42 \
  "$@"
