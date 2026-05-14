#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:-f0}"

case "$mode" in
  f0|F0_fine_no_bias_debug)
    python train_mvp.py \
      --set exp_name=F0_fine_no_bias_debug \
      --set use_fine_stage=True \
      --set use_epipolar_bias=False \
      --set fine_pose_fuse_strength=0.0 \
      --set topk_coarse=64 \
      --set use_epipolar_loss=True \
      --set epi_loss_type=gt_band_nll \
      --set w_epi=0.01 \
      --set max_steps=200
    ;;
  f1|F1_fine_weak_bias)
    python train_mvp.py \
      --set exp_name=F1_fine_weak_bias \
      --set use_fine_stage=True \
      --set use_epipolar_bias=True \
      --set epi_mode=bias \
      --set epi_angle_thresh_deg=15.0 \
      --set epi_bias_strength=0.5 \
      --set fine_pose_fuse_strength=0.25 \
      --set topk_coarse=64 \
      --set use_epipolar_loss=True \
      --set epi_loss_type=gt_band_nll \
      --set w_epi=0.01
    ;;
  f2|F2_fine_pose_full)
    python train_mvp.py \
      --set exp_name=F2_fine_pose_full \
      --set use_fine_stage=True \
      --set use_epipolar_bias=True \
      --set epi_mode=bias \
      --set epi_angle_thresh_deg=10.0 \
      --set epi_bias_strength=0.5 \
      --set fine_pose_fuse_strength=0.5 \
      --set topk_coarse=64 \
      --set use_epipolar_loss=True \
      --set epi_loss_type=gt_band_nll \
      --set w_epi=0.01
    ;;
  *)
    echo "Usage: $0 [f0|f1|f2]" >&2
    exit 2
    ;;
esac
