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

run_1200_safe() {
  test -f "$C31_CKPT" || { echo "Missing C31_CKPT=$C31_CKPT"; exit 1; }
  python train_mvp.py \
    --set exp_name=O28_c31_smallk_anchor2_all_1200 \
    "${common_args[@]}" \
    --set w_tdir_anchor=2.0 \
    --set max_steps=1200
}

run_1200_odomselect() {
  test -f "$C31_CKPT" || { echo "Missing C31_CKPT=$C31_CKPT"; exit 1; }
  python train_mvp.py \
    --set exp_name=O29_c31_smallk_anchor2_odomselect_1200 \
    "${common_args[@]}" \
    --set w_tdir_anchor=2.0 \
    --set save_best_odom_checkpoint=True \
    --set odom_select_max_tdir_abs=25.0 \
    --set odom_select_max_tmag_rel=0.9 \
    --set max_steps=1200
}

run_1200_smallkselect() {
  test -f "$C31_CKPT" || { echo "Missing C31_CKPT=$C31_CKPT"; exit 1; }
  python train_mvp.py \
    --set exp_name=O30_c31_smallk_anchor2_smallkselect_1200 \
    "${common_args[@]}" \
    --set w_tdir_anchor=2.0 \
    --set save_best_odom_checkpoint=True \
    --set odom_select_max_tdir_abs=25.0 \
    --set odom_select_max_tmag_rel=0.9 \
    --set save_best_smallk_odom_checkpoint=True \
    --set smallk_select_metric=odom_metric_drift \
    --set 'smallk_select_k_list=(1,2,3)' \
    --set smallk_select_max_tdir_abs=28.0 \
    --set smallk_select_max_tmag_rel=0.95 \
    --set max_steps=1200
}

run_1200_ignore_tiny_dt() {
  test -f "$C31_CKPT" || { echo "Missing C31_CKPT=$C31_CKPT"; exit 1; }
  python train_mvp.py \
    --set exp_name=O31_c31_smallk_anchor2_ignore_tiny_dt_1200 \
    "${common_args[@]}" \
    --set w_tdir_anchor=2.0 \
    --set save_best_odom_checkpoint=True \
    --set odom_select_max_tdir_abs=25.0 \
    --set odom_select_max_tmag_rel=0.9 \
    --set save_best_smallk_odom_checkpoint=True \
    --set smallk_select_metric=odom_metric_drift \
    --set 'smallk_select_k_list=(1,2,3)' \
    --set smallk_select_max_tdir_abs=28.0 \
    --set smallk_select_max_tmag_rel=0.95 \
    --set tdir_loss_ignore_dt_below=0.05 \
    --set tdir_loss_ignore_weight=0.0 \
    --set max_steps=1200
}

run_1200_soft_tiny_dt() {
  test -f "$C31_CKPT" || { echo "Missing C31_CKPT=$C31_CKPT"; exit 1; }
  python train_mvp.py \
    --set exp_name=O32_c31_smallk_anchor2_soft_tiny_dt_1200 \
    "${common_args[@]}" \
    --set w_tdir_anchor=2.0 \
    --set save_best_odom_checkpoint=True \
    --set odom_select_max_tdir_abs=25.0 \
    --set odom_select_max_tmag_rel=0.9 \
    --set save_best_smallk_odom_checkpoint=True \
    --set smallk_select_metric=odom_metric_drift \
    --set 'smallk_select_k_list=(1,2,3)' \
    --set smallk_select_max_tdir_abs=28.0 \
    --set smallk_select_max_tmag_rel=0.95 \
    --set tdir_loss_ignore_dt_below=0.05 \
    --set tdir_loss_ignore_weight=0.05 \
    --set max_steps=1200
}

run_1200_dt_ramp() {
  test -f "$C31_CKPT" || { echo "Missing C31_CKPT=$C31_CKPT"; exit 1; }
  python train_mvp.py \
    --set exp_name=O33_c31_smallk_anchor2_dt_ramp_1200 \
    "${common_args[@]}" \
    --set w_tdir_anchor=2.0 \
    --set save_best_odom_checkpoint=True \
    --set odom_select_max_tdir_abs=25.0 \
    --set odom_select_max_tmag_rel=0.9 \
    --set save_best_smallk_odom_checkpoint=True \
    --set smallk_select_metric=odom_metric_drift \
    --set 'smallk_select_k_list=(1,2,3)' \
    --set smallk_select_max_tdir_abs=28.0 \
    --set smallk_select_max_tmag_rel=0.95 \
    --set tdir_loss_dt_ramp_enable=True \
    --set tdir_loss_dt_ramp_start=0.02 \
    --set tdir_loss_dt_ramp_end=0.10 \
    --set tdir_loss_dt_ramp_start_weight=0.05 \
    --set tdir_loss_dt_ramp_end_weight=-1.0 \
    --set max_steps=1200
}

run_1200_smoothselect() {
  test -f "$C31_CKPT" || { echo "Missing C31_CKPT=$C31_CKPT"; exit 1; }
  python train_mvp.py \
    --set exp_name=O34_c31_smallk_anchor2_smoothselect_1200 \
    "${common_args[@]}" \
    --set w_tdir_anchor=2.0 \
    --set save_best_odom_checkpoint=True \
    --set odom_select_metric=odom_metric_smooth_tmag_drift \
    --set odom_select_max_tdir_abs=25.0 \
    --set odom_select_max_tmag_rel=0.9 \
    --set save_best_smallk_odom_checkpoint=True \
    --set smallk_select_metric=odom_metric_smooth_tmag_drift \
    --set 'smallk_select_k_list=(1,2,3)' \
    --set smallk_select_max_tdir_abs=28.0 \
    --set smallk_select_max_tmag_rel=0.95 \
    --set odom_eval_smooth_tmag_window=5 \
    --set save_odom_trajectory_debug=True \
    --set max_steps=1200
}

run_1200_tmag_bias() {
  test -f "$C31_CKPT" || { echo "Missing C31_CKPT=$C31_CKPT"; exit 1; }
  python train_mvp.py \
    --set exp_name=O36_c31_smallk_anchor2_tmag_bias_1200 \
    "${common_args[@]}" \
    --set w_tdir_anchor=2.0 \
    --set use_tmag_global_bias=True \
    --set tmag_global_bias_init=0.0 \
    --set save_best_odom_checkpoint=True \
    --set odom_select_metric=odom_metric_drift \
    --set odom_select_max_tdir_abs=25.0 \
    --set odom_select_max_tmag_rel=0.9 \
    --set save_best_smallk_odom_checkpoint=True \
    --set smallk_select_metric=odom_metric_drift \
    --set 'smallk_select_k_list=(1,2,3)' \
    --set smallk_select_max_tdir_abs=28.0 \
    --set smallk_select_max_tmag_rel=0.95 \
    --set odom_eval_scale_fit=True \
    --set max_steps=1200
}

case "${1:-help}" in
  800) run_800 ;;
  800_safe) run_800_safe ;;
  1200) run_1200_probe ;;
  1200_safe) run_1200_safe ;;
  1200_odomselect) run_1200_odomselect ;;
  1200_smallkselect) run_1200_smallkselect ;;
  1200_ignore_tiny_dt) run_1200_ignore_tiny_dt ;;
  1200_soft_tiny_dt) run_1200_soft_tiny_dt ;;
  1200_dt_ramp) run_1200_dt_ramp ;;
  1200_smoothselect) run_1200_smoothselect ;;
  1200_tmag_bias) run_1200_tmag_bias ;;
  *)
    echo "Usage: $0 {800|800_safe|1200|1200_safe|1200_odomselect|1200_smallkselect|1200_ignore_tiny_dt|1200_soft_tiny_dt|1200_dt_ramp|1200_smoothselect|1200_tmag_bias}"
    echo "O27/800_safe uses stronger tdir anchor and is the current preferred small-k finetune."
    echo "O28/1200_safe extends that recipe to 1200 steps; adopt only if drift improves and tdir_abs stays <=25 deg."
    echo "O29/1200_odomselect keeps the O28 recipe and also saves best_odom_drift.pt under odom gates."
    echo "O30/1200_smallkselect keeps O29 and additionally saves best_smallk_odom.pt under k=1/2/3 gates."
    echo "O31/1200_ignore_tiny_dt additionally ignores dt<0.05 samples for tdir loss only."
    echo "O32/1200_soft_tiny_dt keeps weak tdir supervision for dt<0.05 with weight 0.05."
    echo "O33/1200_dt_ramp ramps tdir weight from 0.05 at dt=0.02 to small_dt_t_weight by dt=0.10."
    echo "O34/1200_smoothselect keeps O30 training but selects checkpoints by smoothed t_mag odom drift."
    echo "O36/1200_tmag_bias keeps O30 training and learns a global log_tmag_bias for scale calibration."
    echo "O25/800 is the original conservative run; prefer its best_joint.pt if final tdir_abs crosses 25 deg."
    echo "Override C31_CKPT to test another teacher/init checkpoint."
    ;;
esac
