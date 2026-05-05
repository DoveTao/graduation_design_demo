#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_CKPT="$("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
policy = json.loads(Path("checkpoints/S2b_clean_fine_rot_policy.json").read_text(encoding="utf-8"))
print(policy["base_checkpoint_path"])
PY
)"

"$PYTHON_BIN" train_mvp.py \
  --set exp_name=S3a0_coupled_pose_residual_head_train \
  --set ckpt_dir=checkpoints \
  --set dt_bucket_scale_anchor_policy_json="checkpoints/S2b_clean_fine_rot_policy.json" \
  --set init_checkpoint="$BASE_CKPT" \
  --set strict_load_checkpoint=False \
  --set use_fine_stage=True \
  --set use_coupled_pose_residual_head=True \
  --set coupled_pose_residual_trainable=True \
  --set train_coupled_pose_residual_only=True \
  --set freeze_backbone_for_coupled_pose=True \
  --set freeze_tmag_for_coupled_pose=True \
  --set freeze_dt_anchor_for_coupled_pose=True \
  --set lr=5e-5 \
  --set batch_size=1 \
  --set grad_accum=1 \
  --set max_steps=300 \
  --set eval_every=50 \
  --set max_eval_batches=32 \
  --set max_train_eval_batches=32
