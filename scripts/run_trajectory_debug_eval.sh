#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

read -r -a PYTHON_CMD <<< "${PYTHON_BIN:-python}"

O37_CKPT="${O37_CKPT:-checkpoints/O37_c31_smallk_anchor2_tmag_nodetach_1200/best_smallk_odom.pt}"
O39_CKPT="${O39_CKPT:-checkpoints/O39_c31_smallk_anchor3_tmag_nodetach_1200/best_smallk_odom.pt}"
ODOM_DEBUG_MAX_PAIRS="${ODOM_DEBUG_MAX_PAIRS:-0}"
ODOM_DEBUG_MAX_CHAINS="${ODOM_DEBUG_MAX_CHAINS:-2}"

common_eval=(
  --set eval_only=True
  --set strict_load_checkpoint=False
  --set max_eval_batches=128
  --set max_train_eval_batches=0
  --set num_workers=0
  --set persistent_workers=False
  --set batch_size=1
  --set use_translation_magnitude_head=True
  --set tmag_detach_features=False
  --set use_tmag_global_bias=True
  --set use_fine_stage=False
  --set use_epipolar_loss=True
  --set epi_loss_type=gt_match_ce
  --set epi_angle_thresh_deg=10.0
  --set w_epi=0.01
  --set w_tmag=0.1
  --set 'eval_k_list=(1,2,3,5,10,20)'
  --set eval_min_dt=0.02
  --set eval_max_dt=5.0
  --set stable_eval_min_dt=0.2
  --set 'eval_dt_bucket_edges=(0.05,0.1,0.2,0.5,1.0,2.0,5.0)'
  --set save_odom_trajectory_debug=True
  --set "odom_eval_max_pairs=${ODOM_DEBUG_MAX_PAIRS}"
  --set "odom_trajectory_debug_max_chains=${ODOM_DEBUG_MAX_CHAINS}"
  --set odom_trajectory_debug_segment_count=4
  --set odom_trajectory_debug_topk_steps=10
  --set odom_eval_smooth_tmag_window=5
  --set odom_eval_scale_fit=True
  --set odom_eval_dtcalib=True
  --set odom_eval_dtcalib_min_count=1
)

run_eval() {
  local exp_name="$1"
  local ckpt="$2"
  test -f "$ckpt" || { echo "Missing checkpoint: $ckpt"; exit 1; }
  "${PYTHON_CMD[@]}" train_mvp.py \
    --set "exp_name=${exp_name}" \
    --set "init_checkpoint=${ckpt}" \
    "${common_eval[@]}"
}

case "${1:-help}" in
  o37)
    run_eval E_O37_trajectory_debug "$O37_CKPT"
    ;;
  o39)
    run_eval E_O39_trajectory_debug "$O39_CKPT"
    ;;
  both)
    run_eval E_O37_trajectory_debug "$O37_CKPT"
    run_eval E_O39_trajectory_debug "$O39_CKPT"
    ;;
  help|-h|--help)
    cat <<'USAGE'
Usage: scripts/run_trajectory_debug_eval.sh {o37|o39|both}

Environment overrides:
  PYTHON_BIN              Python command, e.g. "conda run --no-capture-output -n pytorch python"
  O37_CKPT                O37 checkpoint path
  O39_CKPT                O39 checkpoint path
  ODOM_DEBUG_MAX_PAIRS    Limit odom pairs for quick debugging; 0 means full eval
  ODOM_DEBUG_MAX_CHAINS   Number of chains exported to debug CSV/JSON

Outputs are saved under checkpoints/<exp_name>/:
  odom_trajectory_debug_latest.npz
  odom_trajectory_debug_latest.json
  odom_trajectory_steps_latest.csv
USAGE
    ;;
  *)
    echo "Unknown mode: $1" >&2
    echo "Use: scripts/run_trajectory_debug_eval.sh --help" >&2
    exit 2
    ;;
esac
