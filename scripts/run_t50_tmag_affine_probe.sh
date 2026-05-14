#!/usr/bin/env bash
# scripts/run_t50_tmag_affine_probe.sh
#
# T50: Short tmag affine calibration probe starting from O49m3 checkpoint.
# Goal: let the model learn a log-scale bias+scale on the translation
# magnitude head while keeping rotation/direction frozen (or fine-tuned).
#
# Usage:
#   bash scripts/run_t50_tmag_affine_probe.sh dryrun
#   bash scripts/run_t50_tmag_affine_probe.sh short
#
# Notes:
#   - dryrun prints the command without executing.
#   - short runs the full probe (expects CUDA GPU).
#   - Does NOT commit, push, or launch 8-hour scripts.
#   - Must be run from repo root: cd /home/dovetao/graduation_design_demo

set -euo pipefail

cd "$(dirname "$0")/.."

read -r -a PYTHON_CMD <<< "${PYTHON_BIN:-conda run --no-capture-output -n pytorch python}"

# ---- checkpoints ----
O49M3_CKPT="${O49M3_CKPT:-checkpoints/O49m3_o39_seqturn_pair_w007_acosclip_rerun3_260/best_odom_drift.pt}"
C31_CKPT="${C31_CKPT:-checkpoints/C31_coarse_gtmatch_tmag_detach_workers6_800/best_joint_local_A_abs.pt}"

require_file() {
  local path="$1"
  test -f "$path" || { echo "Missing required file: $path" >&2; exit 1; }
}

# ---- CUDA check ----
check_cuda() {
  "${PYTHON_CMD[@]}" -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'CUDA OK: {torch.cuda.get_device_name(0)}')
"
}

# ---- common args inherited from O49m ----
common_args=(
  --set "init_checkpoint=${O49M3_CKPT}"
  --set "strict_load_checkpoint=False"
  --set "use_tdir_anchor_loss=True"
  --set "tdir_anchor_checkpoint=${C31_CKPT}"
  --set "w_tdir_anchor=3.0"
  --set "tdir_anchor_min_dt=0.0"
  --set "k_choices=(1,2,3,5,10)"
  --set "k_probs=(0.85,0.10,0.03,0.01,0.01)"
  --set "eval_k_list=(1,2,3,5,10,20)"
  --set "min_dt=0.02"
  --set "eval_min_dt=0.02"
  --set "max_dt=5.0"
  --set "eval_max_dt=5.0"
  --set "stable_eval_min_dt=0.2"
  --set "use_translation_magnitude_head=True"
  --set "tmag_detach_features=False"
  --set "use_tmag_global_bias=True"
  --set "use_tmag_affine_calib=True"
  --set "tmag_affine_init_scale=1.0"
  --set "tmag_affine_init_bias=1.6"
  --set "use_fine_stage=False"
  --set "use_epipolar_loss=True"
  --set "epi_loss_type=gt_match_ce"
  --set "epi_angle_thresh_deg=10.0"
  --set "w_epi=0.01"
  --set "w_tmag=0.1"
  --set "small_dt_thresh=0.3"
  --set "small_dt_t_weight=0.15"
  --set "lr=1e-5"
  --set "eval_every=50"
  --set "max_steps=400"
  --set "max_eval_batches=128"
  --set "max_train_eval_batches=64"
  --set "num_workers=0"
  --set "persistent_workers=False"
  --set "batch_size=4"
  --set "grad_accum=1"
  --set "save_best_odom_checkpoint=True"
  --set "odom_select_metric=odom_metric_drift"
  --set "odom_select_max_tdir_abs=25.0"
  --set "odom_select_max_tmag_rel=0.95"
  --set "save_best_smallk_odom_checkpoint=True"
  --set "smallk_select_metric=odom_metric_drift"
  --set "smallk_select_k_list=(1,2,3)"
  --set "smallk_select_max_tdir_abs=28.0"
  --set "smallk_select_max_tmag_rel=0.95"
  --set "use_seq_turn_loss=True"
  --set "seq_turn_loss_w=0.007"
  --set "seq_turn_only_k=1"
  --set "seq_turn_min_dt=0.05"
  --set "seq_turn_max_dt=0.25"
  --set "seq_turn_start_updates=100"
  --set "seq_turn_ramp_updates=200"
  --set "seq_turn_acos_eps=0.0001"
  --set "use_seq_turn_chain_loss=False"
  --set "save_odom_trajectory_debug=True"
)

run_case() {
  local exp_name="$1"
  shift
  require_file "$O49M3_CKPT"
  require_file "$C31_CKPT"
  "${PYTHON_CMD[@]}" train_mvp.py \
    --set "exp_name=${exp_name}" \
    "${common_args[@]}" \
    "$@"
}

# ---- short: T50a probe ----
run_short() {
  echo "=== T50a: tmag affine probe (bias=1.6, max_steps=400) ==="
  run_case "T50a_o49m_tmag_affine_bias16_400" \
    --set "use_tmag_affine_calib=True" \
    --set "tmag_affine_init_scale=1.0" \
    --set "tmag_affine_init_bias=1.6" \
    --set "max_steps=400"
}

# ---- dryrun: print command ----
dryrun_case() {
  local exp_name="$1"
  shift
  echo "  exp_name=${exp_name}"
  echo "  init_checkpoint=${O49M3_CKPT}"
  echo "  tmag_affine: scale=1.0 bias=1.6"
  echo "  max_steps=400  eval_every=50  lr=1e-5"
  echo "  use_fine_stage=False  use_seq_turn_loss=True(w=0.007)"
  echo "  save_odom_trajectory_debug=True"
  local full_cmd="${PYTHON_CMD[*]} train_mvp.py"
  for arg in "${common_args[@]}" "$@"; do
    full_cmd="$full_cmd $arg"
  done
  echo "  full_cmd: ${full_cmd:0:200}..."
  echo ""
}

run_dryrun() {
  echo "=== T50 dryrun ==="
  echo "O49M3_CKPT=${O49M3_CKPT}"
  echo "C31_CKPT=${C31_CKPT}"
  echo ""
  dryrun_case "T50a_o49m_tmag_affine_bias16_400" \
    --set "use_tmag_affine_calib=True" \
    --set "tmag_affine_init_scale=1.0" \
    --set "tmag_affine_init_bias=1.6" \
    --set "max_steps=400"
  echo "=== dryrun done (no execution) ==="
}

# ---- main ----
usage() {
  echo "Usage: $0 {dryrun|short}"
  echo ""
  echo "  dryrun   Print the launch command, do not execute"
  echo "  short    Run T50a tmag affine probe (400 steps, CUDA required)"
  exit 1
}

MODE="${1:-}"
case "$MODE" in
  dryrun)
    check_cuda
    run_dryrun
    ;;
  short)
    check_cuda
    run_short
    ;;
  *)
    usage
    ;;
esac
