#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

read -r -a PYTHON_CMD <<< "${PYTHON_BIN:-conda run --no-capture-output -n pytorch python}"

O39_CKPT="${O39_CKPT:-checkpoints/O39_c31_smallk_anchor3_tmag_nodetach_1200/best_smallk_odom.pt}"
TDIR_ANCHOR_CKPT="${TDIR_ANCHOR_CKPT:-checkpoints/C31_coarse_gtmatch_tmag_detach_workers6_800/best_joint_local_A_abs.pt}"
RESULTS_MD="${RESULTS_MD:-checkpoints/O45_six_hour_cycle_results.md}"
STATE_JSON="${STATE_JSON:-checkpoints/O45_six_hour_cycle_state.json}"
SELECTION_TXT="${SELECTION_TXT:-checkpoints/O45_six_hour_cycle_selection.txt}"
AUTO_COMMIT="${AUTO_COMMIT:-true}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-exp: record improved O45 odom probe}"

log() {
  printf '[O45-cycle] %s\n' "$*"
}

require_file() {
  local path="$1"
  test -f "$path" || { echo "Missing required file: $path" >&2; exit 1; }
}

best_ckpt_for() {
  local exp_name="$1"
  local candidate
  for candidate in \
    "checkpoints/${exp_name}/best_smallk_odom.pt" \
    "checkpoints/${exp_name}/best_joint_local_A_abs.pt" \
    "checkpoints/${exp_name}/best_joint.pt"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

run_compile_check() {
  log "Running py_compile gate."
  "${PYTHON_CMD[@]}" -m py_compile \
    config.py dataset_pano_only.py losses.py model.py train_mvp.py \
    interaction.py transformer_encoder.py pose_head.py geometry_refine.py \
    scripts/summarize_odom_cycle.py
}

run_debug_eval() {
  local exp_name="$1"
  local ckpt="$2"
  require_file "$ckpt"
  log "Trajectory debug for ${exp_name}: ${ckpt}"
  PYTHON_BIN="${PYTHON_BIN:-conda run --no-capture-output -n pytorch python}" \
  O40C_CKPT="$ckpt" \
  ODOM_DEBUG_MAX_PAIRS="${ODOM_DEBUG_MAX_PAIRS:-0}" \
  ODOM_DEBUG_MAX_CHAINS="${ODOM_DEBUG_MAX_CHAINS:-2}" \
    scripts/run_trajectory_debug_eval.sh o40c

  local out_dir="checkpoints/E_${exp_name}_trajectory_debug"
  mkdir -p "$out_dir"
  cp -f checkpoints/E_O40C_trajectory_debug/final_summary.json "$out_dir/final_summary.json"
  cp -f checkpoints/E_O40C_trajectory_debug/odom_trajectory_debug_latest.json "$out_dir/odom_trajectory_debug_latest.json"
  cp -f checkpoints/E_O40C_trajectory_debug/odom_trajectory_steps_latest.csv "$out_dir/odom_trajectory_steps_latest.csv"
  if [[ -f checkpoints/E_O40C_trajectory_debug/odom_trajectory_debug_latest.npz ]]; then
    cp -f checkpoints/E_O40C_trajectory_debug/odom_trajectory_debug_latest.npz "$out_dir/odom_trajectory_debug_latest.npz"
  fi
}

summarize() {
  log "Refreshing result table."
  "${PYTHON_CMD[@]}" scripts/summarize_odom_cycle.py \
    --output-md "$RESULTS_MD" \
    --state-json "$STATE_JSON" \
    --selection-txt "$SELECTION_TXT"
}

maybe_commit() {
  if [[ "$AUTO_COMMIT" != "true" ]]; then
    log "AUTO_COMMIT=false; skipping git commit."
    return 0
  fi
  if [[ ! -f "$STATE_JSON" ]]; then
    return 0
  fi
  local recommended
  recommended="$("${PYTHON_CMD[@]}" -c 'import json,sys; p=sys.argv[1]; print("true" if json.load(open(p)).get("commit_recommended") else "false")' "$STATE_JSON")"
  if [[ "$recommended" != "true" ]]; then
    log "No gated improvement yet; not committing."
    return 0
  fi

  log "Gated improvement found; committing scripts and lightweight summary."
  git add scripts/run_six_hour_odom_cycle.sh scripts/summarize_odom_cycle.py
  git add -f "$RESULTS_MD" "$STATE_JSON"
  if git diff --cached --quiet; then
    log "Nothing staged for commit."
    return 0
  fi
  if ! git commit -m "$COMMIT_MESSAGE"; then
    log "git commit failed; continuing so the experiment cycle does not stop."
  fi
}

common_train_args=(
  --set "strict_load_checkpoint=False"
  --set "use_tdir_anchor_loss=True"
  --set "tdir_anchor_checkpoint=${TDIR_ANCHOR_CKPT}"
  --set "w_tdir_anchor=3.0"
  --set "tdir_anchor_min_dt=0.0"
  --set "k_choices=(1,2,3,5,10)"
  --set "eval_k_list=(1,2,3,5,10,20)"
  --set "min_dt=0.02"
  --set "eval_min_dt=0.02"
  --set "stable_eval_min_dt=0.2"
  --set "use_translation_magnitude_head=True"
  --set "tmag_detach_features=False"
  --set "use_tmag_global_bias=True"
  --set "use_fine_stage=False"
  --set "use_epipolar_loss=True"
  --set "epi_loss_type=gt_match_ce"
  --set "epi_angle_thresh_deg=10.0"
  --set "w_epi=0.01"
  --set "w_tmag=0.1"
  --set "small_dt_thresh=0.3"
  --set "small_dt_t_weight=0.15"
  --set "lr=1e-5"
  --set "eval_every=100"
  --set "max_eval_batches=16"
  --set "max_train_eval_batches=8"
  --set "batch_size=4"
  --set "grad_accum=1"
  --set "num_workers=0"
  --set "persistent_workers=False"
  --set "save_best_smallk_odom_checkpoint=True"
)

run_arm() {
  local exp_name="$1"
  local init_ckpt="$2"
  local steps="$3"
  local arm="$4"
  shift 4

  require_file "$init_ckpt"
  log "Training ${exp_name} (${arm}) for ${steps} steps from ${init_ckpt}."
  local arm_args=()
  case "$arm" in
    scale_affine)
      arm_args=(
        --set "k_probs=(0.15,0.20,0.25,0.25,0.15)"
        --set "use_tmag_affine_calib=True"
        --set "tmag_affine_init_scale=1.08"
        --set "tmag_affine_init_bias=0.03"
        --set "use_seq_turn_loss=False"
        --set "use_seq_turn_chain_loss=False"
      )
      ;;
    seqturn_pair)
      arm_args=(
        --set "k_probs=(0.90,0.06,0.03,0.01,0.0)"
        --set "use_seq_turn_loss=True"
        --set "seq_turn_loss_w=0.012"
        --set "seq_turn_only_k=1"
        --set "seq_turn_min_dt=0.02"
        --set "seq_turn_max_dt=0.25"
        --set "use_seq_turn_chain_loss=False"
      )
      ;;
    seqturn_chain)
      arm_args=(
        --set "k_probs=(0.95,0.03,0.02,0.0,0.0)"
        --set "use_seq_turn_loss=False"
        --set "use_seq_turn_chain_loss=True"
        --set "seq_turn_chain_loss_w=0.006"
        --set "seq_turn_chain_min_pairs=1"
        --set "seq_turn_only_k=1"
        --set "seq_turn_min_dt=0.02"
        --set "seq_turn_max_dt=0.18"
      )
      ;;
    k1_curriculum)
      arm_args=(
        --set "k_probs=(0.55,0.22,0.13,0.07,0.03)"
        --set "small_dt_t_weight=0.10"
        --set "use_seq_turn_loss=False"
        --set "use_seq_turn_chain_loss=False"
      )
      ;;
    *)
      echo "Unknown arm: $arm" >&2
      exit 2
      ;;
  esac

  "${PYTHON_CMD[@]}" train_mvp.py \
    --set "exp_name=${exp_name}" \
    --set "init_checkpoint=${init_ckpt}" \
    --set "max_steps=${steps}" \
    "${common_train_args[@]}" \
    "${arm_args[@]}" \
    "$@"

  local ckpt
  ckpt="$(best_ckpt_for "$exp_name")"
  run_debug_eval "$exp_name" "$ckpt"
  summarize
  maybe_commit
}

run_stage1() {
  run_arm "O45a_scale_affine_probe_150" "$O39_CKPT" 150 "scale_affine"
  run_arm "O45b_seqturn_pair_probe_150" "$O39_CKPT" 150 "seqturn_pair"
  run_arm "O45c_seqturn_chain_probe_150" "$O39_CKPT" 150 "seqturn_chain"
  run_arm "O45d_curriculum_k1_probe_150" "$O39_CKPT" 150 "k1_curriculum"
}

run_stage2() {
  summarize
  if [[ ! -s "$SELECTION_TXT" ]]; then
    log "No stage-1 branch passed gates; skipping Top-2 continuation."
    return 0
  fi
  local line exp_name arm ckpt cont_name count
  local selected_lines=()
  mapfile -t selected_lines < "$SELECTION_TXT"
  count=0
  for line in "${selected_lines[@]}"; do
    IFS=$'\t' read -r exp_name arm ckpt <<< "$line"
    [[ -n "${exp_name:-}" ]] || continue
    count=$((count + 1))
    [[ "$count" -le 2 ]] || break
    cont_name="${exp_name}_cont250"
    run_arm "$cont_name" "$ckpt" 250 "$arm"
  done
}

run_stage3() {
  summarize
  if [[ ! -s "$SELECTION_TXT" ]]; then
    log "No branch passed gates after continuation; O39 remains the mainline."
    return 0
  fi
  local exp_name arm ckpt
  IFS=$'\t' read -r exp_name arm ckpt < "$SELECTION_TXT"
  if [[ -z "${exp_name:-}" || -z "${ckpt:-}" ]]; then
    log "Selection file is empty; stopping."
    return 0
  fi
  run_arm "O45_winner_cont300_${arm}" "$ckpt" 300 "$arm"
}

main() {
  require_file "$O39_CKPT"
  require_file "$TDIR_ANCHOR_CKPT"
  run_compile_check

  log "Rebuilding O39 trajectory-debug baseline."
  PYTHON_BIN="${PYTHON_BIN:-conda run --no-capture-output -n pytorch python}" \
  ODOM_DEBUG_MAX_PAIRS="${ODOM_DEBUG_MAX_PAIRS:-0}" \
  ODOM_DEBUG_MAX_CHAINS="${ODOM_DEBUG_MAX_CHAINS:-2}" \
    scripts/run_trajectory_debug_eval.sh o39
  summarize

  run_stage1
  run_stage2
  run_stage3
  summarize
  maybe_commit
  log "Done. Results: ${RESULTS_MD}"
}

main "$@"
