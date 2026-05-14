#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:-debug}"

case "$mode" in
  debug)
    python train_mvp.py \
      --set exp_name=O0_odom_small_k_debug \
      --set max_steps=200 \
      --set 'k_choices=(1,2,3,5,10,20)' \
      --set 'k_probs=(0.15,0.20,0.20,0.25,0.15,0.05)' \
      --set 'eval_k_list=(1,2,3,5,10,20)' \
      --set min_dt=0.02 \
      --set eval_min_dt=0.02 \
      --set max_dt=5.0 \
      --set eval_max_dt=5.0 \
      --set 'eval_dt_bucket_edges=(0.05,0.1,0.2,0.5,1.0,2.0,5.0)' \
      --set stable_eval_min_dt=0.2 \
      --set use_translation_magnitude_head=True \
      --set use_fine_stage=False \
      --set use_epipolar_loss=True \
      --set epi_loss_type=gt_band_nll \
      --set epi_angle_thresh_deg=10.0 \
      --set w_epi=0.01
    ;;
  main)
    python train_mvp.py \
      --set exp_name=O1_odom_small_k_main \
      --set max_steps=1200 \
      --set 'k_choices=(1,2,3,5,10,20)' \
      --set 'k_probs=(0.15,0.20,0.20,0.25,0.15,0.05)' \
      --set 'eval_k_list=(1,2,3,5,10,20)' \
      --set min_dt=0.02 \
      --set eval_min_dt=0.02 \
      --set max_dt=5.0 \
      --set eval_max_dt=5.0 \
      --set 'eval_dt_bucket_edges=(0.05,0.1,0.2,0.5,1.0,2.0,5.0)' \
      --set stable_eval_min_dt=0.2 \
      --set use_translation_magnitude_head=True \
      --set use_fine_stage=False \
      --set use_epipolar_loss=True \
      --set epi_loss_type=gt_band_nll \
      --set epi_angle_thresh_deg=10.0 \
      --set w_epi=0.01
    ;;
  *)
    echo "Usage: $0 [debug|main]" >&2
    exit 2
    ;;
esac
