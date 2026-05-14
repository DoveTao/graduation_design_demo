#!/usr/bin/env bash
# scripts/run_fine_matching_ablation.sh
#
# Fine-stage matching-only ablation.  Opens fine stage to evaluate
# fine token routing/match quality, but keeps fine_pose_fuse_strength=0
# so that the final pose/output is not affected by fine-stage predictions.
#
# Goal: answer whether fine matching improves routing recall / top-k
# hitrate compared to coarse-only, before committing to fine pose fusion.
#
# Usage:
#   bash scripts/run_fine_matching_ablation.sh dryrun
#   bash scripts/run_fine_matching_ablation.sh f0_topk64
#   bash scripts/run_fine_matching_ablation.sh f0_topk96
#   bash scripts/run_fine_matching_ablation.sh f0_topk128
#
# Notes:
#   - dryrun checks CUDA + checkpoint and prints the command.
#   - f0_topk* runs a short experiment (300 steps) with the given topk.
#   - fine_pose_fuse_strength=0 ensures fine pose does NOT affect final output.
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

# ---- batch size: lower for fine stage (768 tokens × 2 = more VRAM) ----
FINE_BS="${FINE_BS:-1}"

# ---- common args ----
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
  --set "use_tmag_affine_calib=False"
  --set "use_fine_stage=True"
  --set "fine_pose_fuse_strength=0.0"
  --set "use_depth_branch=False"
  --set "use_epipolar_loss=True"
  --set "epi_loss_type=gt_match_ce"
  --set "epi_angle_thresh_deg=10.0"
  --set "w_epi=0.01"
  --set "w_tmag=0.1"
  --set "small_dt_thresh=0.3"
  --set "small_dt_t_weight=0.15"
  --set "lr=1e-5"
  --set "eval_every=50"
  --set "max_steps=300"
  --set "max_eval_batches=128"
  --set "max_train_eval_batches=64"
  --set "num_workers=0"
  --set "persistent_workers=False"
  --set "batch_size=${FINE_BS}"
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
  local topk="$2"
  shift 2
  require_file "$O49M3_CKPT"
  require_file "$C31_CKPT"
  "${PYTHON_CMD[@]}" train_mvp.py \
    --set "exp_name=${exp_name}" \
    --set "topk_coarse=${topk}" \
    "${common_args[@]}" \
    "$@"
}

# ---- f0: matching-only with varying topk ----
run_f0_topk64() {
  echo "=== F0-topk64: fine matching-only (topk_coarse=64) ==="
  run_case "F0_fine_match_topk64_fromO49m3_300" 64
}

run_f0_topk96() {
  echo "=== F0-topk96: fine matching-only (topk_coarse=96) ==="
  run_case "F0_fine_match_topk96_fromO49m3_300" 96
}

run_f0_topk128() {
  echo "=== F0-topk128: fine matching-only (topk_coarse=128) ==="
  run_case "F0_fine_match_topk128_fromO49m3_300" 128
}

# ---- dryrun ----
dryrun_case() {
  local label="$1"
  local exp_name="$2"
  local topk="$3"
  echo "  [$label]"
  echo "    exp_name=${exp_name}"
  echo "    topk_coarse=${topk}"
  echo "    init_checkpoint=${O49M3_CKPT}"
  echo "    use_fine_stage=True  fine_pose_fuse_strength=0.0"
  echo "    max_steps=300  eval_every=50  lr=1e-5"
  echo "    batch_size=${FINE_BS}  save_odom_trajectory_debug=True"
  echo ""
}

run_dryrun() {
  echo "=== Fine Matching Ablation dryrun ==="
  echo "O49M3_CKPT=${O49M3_CKPT}"
  echo "C31_CKPT=${C31_CKPT}"
  echo ""
  dryrun_case "f0_topk64"  "F0_fine_match_topk64_fromO49m3_300"  64
  dryrun_case "f0_topk96"  "F0_fine_match_topk96_fromO49m3_300"  96
  dryrun_case "f0_topk128" "F0_fine_match_topk128_fromO49m3_300" 128
  echo "=== dryrun done (no execution) ==="
}

# ---- main ----
usage() {
  echo "Usage: $0 {dryrun|f0_topk64|f0_topk96|f0_topk128}"
  echo ""
  echo "  dryrun       Print all launch commands, do not execute"
  echo "  f0_topk64    Fine matching probe, topk_coarse=64"
  echo "  f0_topk96    Fine matching probe, topk_coarse=96"
  echo "  f0_topk128   Fine matching probe, topk_coarse=128"
  exit 1
}

MODE="${1:-}"
case "$MODE" in
  dryrun)
    check_cuda
    run_dryrun
    ;;
  f0_topk64)
    check_cuda
    run_f0_topk64
    ;;
  f0_topk96)
    check_cuda
    run_f0_topk96
    ;;
  f0_topk128)
    check_cuda
    run_f0_topk128
    ;;
  *)
    usage
    ;;
esac
