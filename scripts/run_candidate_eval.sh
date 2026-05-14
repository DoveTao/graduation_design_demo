#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

read -r -a PYTHON_CMD <<< "${PYTHON_BIN:-python}"

C31_CKPT="${C31_CKPT:-checkpoints/C31_coarse_gtmatch_tmag_detach_workers6_800/best_joint_local_A_abs.pt}"
O27_CKPT="${O27_CKPT:-checkpoints/O27_c31_smallk_anchor2_all_800/best_joint_local_A_abs.pt}"

common_eval=(
  --set eval_only=True
  --set strict_load_checkpoint=False
  --set max_eval_batches=128
  --set max_train_eval_batches=0
  --set num_workers=0
  --set persistent_workers=False
  --set batch_size=1
  --set use_translation_magnitude_head=True
  --set tmag_detach_features=True
  --set use_fine_stage=False
  --set use_epipolar_loss=True
  --set epi_loss_type=gt_match_ce
  --set epi_angle_thresh_deg=10.0
  --set w_epi=0.01
  --set w_tmag=0.1
)

smallk_eval=(
  --set 'eval_k_list=(1,2,3,5,10,20)'
  --set eval_min_dt=0.02
  --set eval_max_dt=5.0
  --set stable_eval_min_dt=0.2
  --set 'eval_dt_bucket_edges=(0.05,0.1,0.2,0.5,1.0,2.0,5.0)'
)

wide_eval=(
  --set 'eval_k_list=(5,10,20,40)'
  --set eval_min_dt=0.1
  --set eval_max_dt=5.0
  --set stable_eval_min_dt=0.5
  --set 'eval_dt_bucket_edges=(0.1,0.3,0.5,1.0,2.0,3.5,5.0)'
)

run_eval() {
  local exp_name="$1"
  local ckpt="$2"
  shift 2
  test -f "$ckpt" || { echo "Missing checkpoint: $ckpt"; exit 1; }
  "${PYTHON_CMD[@]}" train_mvp.py \
    --set "exp_name=${exp_name}" \
    --set "init_checkpoint=${ckpt}" \
    "${common_eval[@]}" \
    "$@"
}

run_smallk() {
  run_eval E_C31_smallk_eval "$C31_CKPT" "${smallk_eval[@]}"
  run_eval E_O27_smallk_eval "$O27_CKPT" "${smallk_eval[@]}"
}

run_wide() {
  run_eval E_C31_wide_eval "$C31_CKPT" "${wide_eval[@]}"
  run_eval E_O27_wide_eval "$O27_CKPT" "${wide_eval[@]}"
}

case "${1:-help}" in
  smallk)
    run_smallk
    ;;
  wide)
    run_wide
    ;;
  all)
    run_smallk
    run_wide
    ;;
  help|-h|--help)
    cat <<'USAGE'
Usage: scripts/run_candidate_eval.sh {smallk|wide|all}

Environment overrides:
  PYTHON_BIN  Python command, e.g. "conda run --no-capture-output -n pytorch python"
  C31_CKPT    C31 baseline checkpoint path
  O27_CKPT    O27 small-k candidate checkpoint path
USAGE
    ;;
  *)
    echo "Unknown mode: $1" >&2
    echo "Use: scripts/run_candidate_eval.sh --help" >&2
    exit 2
    ;;
esac
