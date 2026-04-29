#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:-help}"

common_debug=(
  --set max_steps=200
  --set eval_every=100
  --set max_eval_batches=16
  --set max_train_eval_batches=8
)

common_main=(
  --set max_steps=1200
  --set eval_every=100
)

common_odom_small_k=(
  --set 'k_choices=(1,2,3,5,10,20)'
  --set 'k_probs=(0.15,0.20,0.20,0.25,0.15,0.05)'
  --set 'eval_k_list=(1,2,3,5,10,20)'
  --set min_dt=0.02
  --set eval_min_dt=0.02
  --set max_dt=5.0
  --set eval_max_dt=5.0
  --set 'eval_dt_bucket_edges=(0.05,0.1,0.2,0.5,1.0,2.0,5.0)'
  --set stable_eval_min_dt=0.2
  --set use_translation_magnitude_head=True
)

run_epi_debug() {
  python train_mvp.py \
    --set exp_name=A10_epi_w000_a10_debug \
    --set use_fine_stage=False \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_band_nll \
    --set epi_angle_thresh_deg=10.0 \
    --set w_epi=0.0 \
    "${common_debug[@]}"

  python train_mvp.py \
    --set exp_name=A11_epi_w005_a15_debug \
    --set use_fine_stage=False \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_band_nll \
    --set epi_angle_thresh_deg=15.0 \
    --set w_epi=0.005 \
    "${common_debug[@]}"

  python train_mvp.py \
    --set exp_name=A12_epi_w010_a10_debug \
    --set use_fine_stage=False \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_band_nll \
    --set epi_angle_thresh_deg=10.0 \
    --set w_epi=0.01 \
    "${common_debug[@]}"

  python train_mvp.py \
    --set exp_name=A13_epi_w020_a05_debug \
    --set use_fine_stage=False \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_band_nll \
    --set epi_angle_thresh_deg=5.0 \
    --set w_epi=0.02 \
    "${common_debug[@]}"
}

run_epi_main() {
  python train_mvp.py \
    --set exp_name=A12_epi_w010_a10_main \
    --set use_fine_stage=False \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_band_nll \
    --set epi_angle_thresh_deg=10.0 \
    --set w_epi=0.01 \
    "${common_main[@]}"

  python train_mvp.py \
    --set exp_name=A13_epi_w020_a05_main \
    --set use_fine_stage=False \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_band_nll \
    --set epi_angle_thresh_deg=5.0 \
    --set w_epi=0.02 \
    "${common_main[@]}"
}

run_fine_debug() {
  python train_mvp.py \
    --set exp_name=F10_topk32_nobias_fuse0_debug \
    --set use_fine_stage=True \
    --set use_epipolar_bias=False \
    --set topk_coarse=32 \
    --set fine_pose_fuse_strength=0.0 \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_band_nll \
    --set epi_angle_thresh_deg=10.0 \
    --set w_epi=0.01 \
    "${common_debug[@]}"

  python train_mvp.py \
    --set exp_name=F11_topk64_bias03_fuse025_debug \
    --set use_fine_stage=True \
    --set use_epipolar_bias=True \
    --set epi_mode=bias \
    --set epi_bias_strength=0.3 \
    --set topk_coarse=64 \
    --set fine_pose_fuse_strength=0.25 \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_band_nll \
    --set epi_angle_thresh_deg=15.0 \
    --set w_epi=0.01 \
    "${common_debug[@]}"

  python train_mvp.py \
    --set exp_name=F12_topk96_bias05_fuse025_debug \
    --set use_fine_stage=True \
    --set use_epipolar_bias=True \
    --set epi_mode=bias \
    --set epi_bias_strength=0.5 \
    --set topk_coarse=96 \
    --set fine_pose_fuse_strength=0.25 \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_band_nll \
    --set epi_angle_thresh_deg=15.0 \
    --set w_epi=0.01 \
    "${common_debug[@]}"

  python train_mvp.py \
    --set exp_name=F13_topk96_bias05_fuse05_debug \
    --set use_fine_stage=True \
    --set use_epipolar_bias=True \
    --set epi_mode=bias \
    --set epi_bias_strength=0.5 \
    --set topk_coarse=96 \
    --set fine_pose_fuse_strength=0.5 \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_band_nll \
    --set epi_angle_thresh_deg=10.0 \
    --set w_epi=0.01 \
    "${common_debug[@]}"
}

run_fine_main() {
  python train_mvp.py \
    --set exp_name=F11_topk64_bias03_fuse025_main \
    --set use_fine_stage=True \
    --set use_epipolar_bias=True \
    --set epi_mode=bias \
    --set epi_bias_strength=0.3 \
    --set topk_coarse=64 \
    --set fine_pose_fuse_strength=0.25 \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_band_nll \
    --set epi_angle_thresh_deg=15.0 \
    --set w_epi=0.01 \
    "${common_main[@]}"

  python train_mvp.py \
    --set exp_name=F12_topk96_bias05_fuse025_main \
    --set use_fine_stage=True \
    --set use_epipolar_bias=True \
    --set epi_mode=bias \
    --set epi_bias_strength=0.5 \
    --set topk_coarse=96 \
    --set fine_pose_fuse_strength=0.25 \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_band_nll \
    --set epi_angle_thresh_deg=15.0 \
    --set w_epi=0.01 \
    "${common_main[@]}"
}

run_odom_debug() {
  python train_mvp.py \
    --set exp_name=O10_odom_small_k_tmag_debug \
    "${common_odom_small_k[@]}" \
    --set use_fine_stage=False \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_band_nll \
    --set epi_angle_thresh_deg=10.0 \
    --set w_epi=0.01 \
    --set w_tmag=0.1 \
    "${common_debug[@]}"
}

run_odom_main() {
  python train_mvp.py \
    --set exp_name=O10_odom_small_k_tmag_main \
    "${common_odom_small_k[@]}" \
    --set use_fine_stage=False \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_band_nll \
    --set epi_angle_thresh_deg=10.0 \
    --set w_epi=0.01 \
    --set w_tmag=0.1 \
    "${common_main[@]}"
}

run_coarse_tmag_schedule_main() {
  python train_mvp.py \
    --set exp_name=C26_coarse_gtmatch_notmag_workers6_800 \
    --set use_fine_stage=False \
    --set use_epipolar_bias=False \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_match_ce \
    --set epi_angle_thresh_deg=10.0 \
    --set w_epi=0.01 \
    --set use_translation_magnitude_head=False \
    --set w_tmag=0.0 \
    --set max_steps=800 \
    --set eval_every=100 \
    --set num_workers=6 \
    --set persistent_workers=True

  python train_mvp.py \
    --set exp_name=C28_coarse_gtmatch_tmag_delay_workers6_800 \
    --set use_fine_stage=False \
    --set use_epipolar_bias=False \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_match_ce \
    --set epi_angle_thresh_deg=10.0 \
    --set w_epi=0.01 \
    --set use_translation_magnitude_head=True \
    --set w_tmag=0.1 \
    --set tmag_start_updates=200 \
    --set tmag_ramp_updates=100 \
    --set max_steps=800 \
    --set eval_every=100 \
    --set num_workers=6 \
    --set persistent_workers=True
}

case "$mode" in
  epi_debug)
    run_epi_debug
    ;;
  epi_main)
    run_epi_main
    ;;
  fine_debug)
    run_fine_debug
    ;;
  fine_main)
    run_fine_main
    ;;
  odom_debug)
    run_odom_debug
    ;;
  odom_main)
    run_odom_main
    ;;
  coarse_tmag_main)
    run_coarse_tmag_schedule_main
    ;;
  all_debug)
    run_epi_debug
    run_fine_debug
    run_odom_debug
    ;;
  all_main)
    run_epi_main
    run_fine_main
    run_odom_main
    ;;
  help|-h|--help)
    cat <<'USAGE'
Usage: scripts/run_effect_ablation.sh MODE

Modes:
  epi_debug   4x max_steps=200: w_epi / epi_angle_thresh_deg
  fine_debug  4x max_steps=200: topk_coarse / epi_bias_strength / fine_pose_fuse_strength
  odom_debug  1x max_steps=200: odom-small-k t_mag probe
  all_debug   Run all debug experiments above

  epi_main    2x max_steps=1200: best epipolar candidates
  fine_main   2x max_steps=1200: best fine routing candidates
  odom_main   1x max_steps=1200: odom-small-k t_mag candidate
  coarse_tmag_main
              2x max_steps=800: gt_match_ce no-tmag baseline and delayed t_mag
  all_main    Run all main experiments above

Run debug modes first. Main modes are intentionally compact and should only be
started after reviewing debug matching, t_mag, and odometry metrics.
USAGE
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    echo "Use: scripts/run_effect_ablation.sh --help" >&2
    exit 2
    ;;
esac
