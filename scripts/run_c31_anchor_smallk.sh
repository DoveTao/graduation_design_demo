#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

C31_CKPT="${C31_CKPT:-checkpoints/C31_coarse_gtmatch_tmag_detach_workers6_800/best_joint_local_A_abs.pt}"

common_args=(
  --set "init_checkpoint=${C31_CKPT}"
  --set use_tdir_anchor_loss=True
  --set "tdir_anchor_checkpoint=${C31_CKPT}"
  --set tdir_anchor_min_dt=0.0
  --set strict_load_checkpoint=False
  --set 'k_choices=(1,2,3,5,10)'
  --set 'k_probs=(0.15,0.20,0.25,0.25,0.15)'
  --set 'eval_k_list=(1,2,3,5,10,20)'
  --set min_dt=0.02
  --set eval_min_dt=0.02
  --set max_dt=5.0
  --set eval_max_dt=5.0
  --set 'eval_dt_bucket_edges=(0.05,0.1,0.2,0.5,1.0,2.0,5.0)'
  --set stable_eval_min_dt=0.2
  --set use_translation_magnitude_head=True
  --set tmag_detach_features=True
  --set use_fine_stage=False
  --set use_epipolar_loss=True
  --set epi_loss_type=gt_match_ce
  --set epi_angle_thresh_deg=10.0
  --set w_epi=0.01
  --set w_tmag=0.1
  --set small_dt_thresh=0.3
  --set small_dt_t_weight=0.15
  --set lr=1e-5
  --set eval_every=100
)

run_800() {
  test -f "$C31_CKPT" || { echo "Missing C31_CKPT=$C31_CKPT"; exit 1; }
  python train_mvp.py \
    --set exp_name=O25_c31_smallk_anchor_all_800 \
    "${common_args[@]}" \
    --set w_tdir_anchor=1.0 \
    --set max_steps=800
}

run_800_safe() {
  test -f "$C31_CKPT" || { echo "Missing C31_CKPT=$C31_CKPT"; exit 1; }
  python train_mvp.py \
    --set exp_name=O27_c31_smallk_anchor2_all_800 \
    "${common_args[@]}" \
    --set w_tdir_anchor=2.0 \
    --set max_steps=800
}

run_1200_probe() {
  test -f "$C31_CKPT" || { echo "Missing C31_CKPT=$C31_CKPT"; exit 1; }
  python train_mvp.py \
    --set exp_name=O26_c31_smallk_anchor_all_1200 \
    "${common_args[@]}" \
    --set w_tdir_anchor=1.0 \
    --set max_steps=1200
}

case "${1:-help}" in
  800) run_800 ;;
  800_safe) run_800_safe ;;
  1200) run_1200_probe ;;
  *)
    echo "Usage: $0 {800|800_safe|1200}"
    echo "O27/800_safe uses stronger tdir anchor and is the current preferred small-k finetune."
    echo "O25/800 is the original conservative run; prefer its best_joint.pt if final tdir_abs crosses 25 deg."
    echo "Override C31_CKPT to test another teacher/init checkpoint."
    ;;
esac
