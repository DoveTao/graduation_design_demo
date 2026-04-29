#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python train_mvp.py \
  --set exp_name=W0_wide_baseline \
  --set 'k_choices=(5,10,20,40)' \
  --set 'k_probs=(0.25,0.30,0.30,0.15)' \
  --set 'eval_k_list=(5,10,20,40)' \
  --set min_dt=0.1 \
  --set max_dt=5.0 \
  --set eval_min_dt=0.1 \
  --set eval_max_dt=5.0 \
  --set stable_eval_min_dt=0.5 \
  --set use_translation_magnitude_head=True \
  --set use_fine_stage=False \
  --set use_epipolar_loss=True \
  --set epi_loss_type=gt_band_nll \
  --set w_epi=0.01
