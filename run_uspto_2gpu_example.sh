#!/usr/bin/env bash
set -euo pipefail

: "${CHEFNMR_DIR:?Please set CHEFNMR_DIR}"
: "${CHEFNMR_CKPT_DIR:?Please set CHEFNMR_CKPT_DIR}"

run_shard () {
  local gpu="$1"
  local start_idx="$2"
  local max_samples="$3"
  local out="$4"

  CHEFNMR_CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 conda run -n nmr3d python -u generate_offline_cache.py \
    --datasets uspto \
    --split test \
    --start-idx "${start_idx}" \
    --max-samples "${max_samples}" \
    --chunk-samples 1024 \
    --top-k 10 \
    --test-batch-size 64 \
    --batch-size 64 \
    --output "${out}" \
    --overwrite
}

run_shard 0 0 37784 "${CHEFNMR_CKPT_DIR}/uspto_shard0.pkl" &
run_shard 1 37784 37783 "${CHEFNMR_CKPT_DIR}/uspto_shard1.pkl" &

wait

python merge_shards.py \
  --inputs "${CHEFNMR_CKPT_DIR}/uspto_shard0.pkl" \
           "${CHEFNMR_CKPT_DIR}/uspto_shard1.pkl" \
  --output "${CHEFNMR_CKPT_DIR}/chefnmr_offline_cache_uspto.pkl"
