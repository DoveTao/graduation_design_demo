#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PHASE1_CKPT="${PHASE1_CKPT:-checkpoints/O20_curriculum_stable_dt/best_joint_local_A_abs.pt}"
PHASE2_CKPT="${PHASE2_CKPT:-checkpoints/O21_curriculum_mixed_dt/best_joint_local_A_abs.pt}"

run_phase1() {
  python train_mvp.py \
    --set exp_name=O20_curriculum_stable_dt \
    --set 'k_choices=(5,10,20)' \
    --set 'k_probs=(0.35,0.40,0.25)' \
    --set 'eval_k_list=(1,2,3,5,10,20)' \
    --set min_dt=0.5 \
    --set eval_min_dt=0.02 \
    --set max_dt=5.0 \
    --set eval_max_dt=5.0 \
    --set 'eval_dt_bucket_edges=(0.05,0.1,0.2,0.5,1.0,2.0,5.0)' \
    --set stable_eval_min_dt=0.5 \
    --set use_translation_magnitude_head=True \
    --set tmag_detach_features=True \
    --set use_fine_stage=False \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_match_ce \
    --set epi_angle_thresh_deg=10.0 \
    --set w_epi=0.01 \
    --set w_tmag=0.1 \
    --set small_dt_thresh=0.3 \
    --set small_dt_t_weight=0.2 \
    --set max_steps=800 \
    --set eval_every=100
}

run_phase2() {
  test -f "$PHASE1_CKPT" || { echo "Missing PHASE1_CKPT=$PHASE1_CKPT"; exit 1; }
  python train_mvp.py \
    --set exp_name=O21_curriculum_mixed_dt \
    --set init_checkpoint="$PHASE1_CKPT" \
    --set strict_load_checkpoint=False \
    --set 'k_choices=(2,3,5,10,20)' \
    --set 'k_probs=(0.20,0.25,0.25,0.20,0.10)' \
    --set 'eval_k_list=(1,2,3,5,10,20)' \
    --set min_dt=0.1 \
    --set eval_min_dt=0.02 \
    --set max_dt=5.0 \
    --set eval_max_dt=5.0 \
    --set 'eval_dt_bucket_edges=(0.05,0.1,0.2,0.5,1.0,2.0,5.0)' \
    --set stable_eval_min_dt=0.2 \
    --set use_translation_magnitude_head=True \
    --set tmag_detach_features=True \
    --set use_fine_stage=False \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_match_ce \
    --set epi_angle_thresh_deg=10.0 \
    --set w_epi=0.01 \
    --set w_tmag=0.1 \
    --set small_dt_thresh=0.3 \
    --set small_dt_t_weight=0.2 \
    --set max_steps=800 \
    --set eval_every=100
}

run_phase3() {
  test -f "$PHASE2_CKPT" || { echo "Missing PHASE2_CKPT=$PHASE2_CKPT"; exit 1; }
  python train_mvp.py \
    --set exp_name=O22_curriculum_small_k_finetune \
    --set init_checkpoint="$PHASE2_CKPT" \
    --set strict_load_checkpoint=False \
    --set 'k_choices=(1,2,3,5,10)' \
    --set 'k_probs=(0.18,0.24,0.24,0.24,0.10)' \
    --set 'eval_k_list=(1,2,3,5,10,20)' \
    --set min_dt=0.02 \
    --set eval_min_dt=0.02 \
    --set max_dt=5.0 \
    --set eval_max_dt=5.0 \
    --set 'eval_dt_bucket_edges=(0.05,0.1,0.2,0.5,1.0,2.0,5.0)' \
    --set stable_eval_min_dt=0.2 \
    --set use_translation_magnitude_head=True \
    --set tmag_detach_features=True \
    --set use_fine_stage=False \
    --set use_epipolar_loss=True \
    --set epi_loss_type=gt_match_ce \
    --set epi_angle_thresh_deg=10.0 \
    --set w_epi=0.01 \
    --set w_tmag=0.1 \
    --set small_dt_thresh=0.3 \
    --set small_dt_t_weight=0.05 \
    --set max_steps=800 \
    --set eval_every=100
}

case "${1:-help}" in
  phase1) run_phase1 ;;
  phase2) run_phase2 ;;
  phase3) run_phase3 ;;
  all)
    run_phase1
    PHASE1_CKPT="$PHASE1_CKPT" run_phase2
    PHASE2_CKPT="$PHASE2_CKPT" run_phase3
    ;;
  *)
    echo "Usage: $0 {phase1|phase2|phase3|all}"
    echo "Override PHASE1_CKPT/PHASE2_CKPT when using custom checkpoint paths."
    ;;
esac
