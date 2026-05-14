#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python train_mvp.py \
  --set exp_name=debug_smoke \
  --set max_steps=5 \
  --set eval_every=5 \
  --set max_eval_batches=2 \
  --set max_train_eval_batches=2 \
  --set num_workers=0 \
  --set persistent_workers=False \
  --set batch_size=1 \
  --set grad_accum=1
