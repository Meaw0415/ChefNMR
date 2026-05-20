#!/usr/bin/env bash
set -euo pipefail

: "${CHEFNMR_DIR:?Please set CHEFNMR_DIR}"
: "${CHEFNMR_CKPT_DIR:?Please set CHEFNMR_CKPT_DIR}"

PYTHONUNBUFFERED=1 conda run -n nmr3d python -u generate_offline_cache.py \
  --datasets uspto \
  --split test \
  --start-idx 0 \
  --max-samples -1 \
  --chunk-samples 1024 \
  --top-k 10 \
  --test-batch-size 64 \
  --batch-size 64 \
  --output "${CHEFNMR_CKPT_DIR}/chefnmr_offline_cache_uspto.pkl" \
  --overwrite

