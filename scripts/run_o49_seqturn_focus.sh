#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

read -r -a PYTHON_CMD <<< "${PYTHON_BIN:-conda run --no-capture-output -n pytorch python}"

O39_CKPT="${O39_CKPT:-checkpoints/O39_c31_smallk_anchor3_tmag_nodetach_1200/best_smallk_odom.pt}"
C31_CKPT="${C31_CKPT:-checkpoints/C31_coarse_gtmatch_tmag_detach_workers6_800/best_joint_local_A_abs.pt}"

require_file() {
  local path="$1"
  test -f "$path" || { echo "Missing required file: $path" >&2; exit 1; }
}

run_compile_check() {
  "${PYTHON_CMD[@]}" -m py_compile \
    config.py dataset_pano_only.py losses.py model.py train_mvp.py \
    interaction.py transformer_encoder.py pose_head.py geometry_refine.py
}

common_args=(
  --set "init_checkpoint=${O39_CKPT}"
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
  --set "use_fine_stage=False"
  --set "use_epipolar_loss=True"
  --set "epi_loss_type=gt_match_ce"
  --set "epi_angle_thresh_deg=10.0"
  --set "w_epi=0.01"
  --set "w_tmag=0.1"
  --set "small_dt_thresh=0.3"
  --set "small_dt_t_weight=0.15"
  --set "lr=1e-5"
  # Evaluate multiple times after seq-turn becomes active.
  --set "eval_every=50"
  --set "max_steps=260"
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
  --set "use_seq_turn_loss=False"
  --set "use_seq_turn_chain_loss=False"
)

run_case() {
  local exp_name="$1"
  shift
  require_file "$O39_CKPT"
  require_file "$C31_CKPT"
  "${PYTHON_CMD[@]}" train_mvp.py \
    --set "exp_name=${exp_name}" \
    "${common_args[@]}" \
    "$@"
}

run_control() {
  run_case "O49a_o39_k1_curriculum_control_260"
}

run_pair_light() {
  run_case "O49b_o39_seqturn_pair_light_260" \
    --set "use_seq_turn_loss=True" \
    --set "seq_turn_loss_w=0.006" \
    --set "seq_turn_only_k=1" \
    --set "seq_turn_min_dt=0.05" \
    --set "seq_turn_max_dt=0.20" \
    --set "seq_turn_start_updates=100" \
    --set "seq_turn_ramp_updates=200"
}

run_pair_midband() {
  run_case "O49c_o39_seqturn_pair_midband_260" \
    --set "use_seq_turn_loss=True" \
    --set "seq_turn_loss_w=0.008" \
    --set "seq_turn_only_k=1" \
    --set "seq_turn_min_dt=0.08" \
    --set "seq_turn_max_dt=0.30" \
    --set "seq_turn_start_updates=120" \
    --set "seq_turn_ramp_updates=220"
}

run_pair_late() {
  run_case "O49d_o39_seqturn_pair_late_260" \
    --set "use_seq_turn_loss=True" \
    --set "seq_turn_loss_w=0.004" \
    --set "seq_turn_only_k=1" \
    --set "seq_turn_min_dt=0.05" \
    --set "seq_turn_max_dt=0.20" \
    --set "seq_turn_start_updates=150" \
    --set "seq_turn_ramp_updates=250"
}

run_pair_hitrate() {
  run_case "O49e_o39_seqturn_pair_hitrate_260" \
    --set "use_seq_turn_loss=True" \
    --set "seq_turn_loss_w=0.006" \
    --set "seq_turn_only_k=1" \
    --set "seq_turn_min_dt=0.04" \
    --set "seq_turn_max_dt=0.30" \
    --set "seq_turn_start_updates=100" \
    --set "seq_turn_ramp_updates=200"
}

run_pair_soft_hitrate() {
  run_case "O49f_o39_seqturn_pair_soft_hitrate_260" \
    --set "use_seq_turn_loss=True" \
    --set "seq_turn_loss_w=0.006" \
    --set "seq_turn_only_k=1" \
    --set "seq_turn_min_dt=0.05" \
    --set "seq_turn_max_dt=0.25" \
    --set "seq_turn_start_updates=100" \
    --set "seq_turn_ramp_updates=200"
}

run_pair_soft_hitrate_wlow() {
  run_case "O49g_o39_seqturn_pair_soft_hitrate_w005_260" \
    --set "use_seq_turn_loss=True" \
    --set "seq_turn_loss_w=0.005" \
    --set "seq_turn_only_k=1" \
    --set "seq_turn_min_dt=0.05" \
    --set "seq_turn_max_dt=0.25" \
    --set "seq_turn_start_updates=100" \
    --set "seq_turn_ramp_updates=200"
}

run_pair_soft_hitrate_whigh() {
  run_case "O49h_o39_seqturn_pair_soft_hitrate_w007_260" \
    --set "use_seq_turn_loss=True" \
    --set "seq_turn_loss_w=0.007" \
    --set "seq_turn_only_k=1" \
    --set "seq_turn_min_dt=0.05" \
    --set "seq_turn_max_dt=0.25" \
    --set "seq_turn_start_updates=100" \
    --set "seq_turn_ramp_updates=200"
}

run_pair_soft_hitrate_whigh_rerun() {
  run_case "O49h2_o39_seqturn_pair_soft_hitrate_w007_rerun_260" \
    --set "use_seq_turn_loss=True" \
    --set "seq_turn_loss_w=0.007" \
    --set "seq_turn_only_k=1" \
    --set "seq_turn_min_dt=0.05" \
    --set "seq_turn_max_dt=0.25" \
    --set "seq_turn_start_updates=100" \
    --set "seq_turn_ramp_updates=200"
}

run_pair_soft_hitrate_whigh_slowramp() {
  run_case "O49j_o39_seqturn_pair_soft_hitrate_w007_ramp300_260" \
    --set "use_seq_turn_loss=True" \
    --set "seq_turn_loss_w=0.007" \
    --set "seq_turn_only_k=1" \
    --set "seq_turn_min_dt=0.05" \
    --set "seq_turn_max_dt=0.25" \
    --set "seq_turn_start_updates=100" \
    --set "seq_turn_ramp_updates=300"
}

run_pair_soft_hitrate_wmid() {
  run_case "O49l_o39_seqturn_pair_soft_hitrate_w0065_260" \
    --set "use_seq_turn_loss=True" \
    --set "seq_turn_loss_w=0.0065" \
    --set "seq_turn_only_k=1" \
    --set "seq_turn_min_dt=0.05" \
    --set "seq_turn_max_dt=0.25" \
    --set "seq_turn_start_updates=100" \
    --set "seq_turn_ramp_updates=200"
}

run_pair_soft_hitrate_late() {
  run_case "O49i_o39_seqturn_pair_soft_hitrate_start120_260" \
    --set "use_seq_turn_loss=True" \
    --set "seq_turn_loss_w=0.006" \
    --set "seq_turn_only_k=1" \
    --set "seq_turn_min_dt=0.05" \
    --set "seq_turn_max_dt=0.25" \
    --set "seq_turn_start_updates=120" \
    --set "seq_turn_ramp_updates=200"
}

usage() {
  cat <<'EOF'
Usage: scripts/run_o49_seqturn_focus.sh {compile|control|pair_light|pair_midband|pair_late|pair_hitrate|pair_soft_hitrate|pair_soft_hitrate_wlow|pair_soft_hitrate_whigh|pair_soft_hitrate_whigh_rerun|pair_soft_hitrate_whigh_slowramp|pair_soft_hitrate_wmid|pair_soft_hitrate_late|o49_next|all}

  compile       Run the py_compile gate only.
  control       O49a, k1-heavy curriculum only; no seq-turn.
  pair_light    O49b, light pair-turn on the most observable dt band.
  pair_midband  O49c, slightly wider dt band and slightly higher pair-turn weight.
  pair_late     O49d, later and weaker pair-turn schedule for a safer probe.
  pair_hitrate  O49e, higher seq-turn hit-rate via a wider dt band with the same weight.
  pair_soft_hitrate
                O49f, O49b plus a gentler hit-rate bump via max_dt=0.25 only.
  pair_soft_hitrate_wlow
                O49g, O49f with seq_turn_loss_w=0.005.
  pair_soft_hitrate_whigh
                O49h, O49f with seq_turn_loss_w=0.007.
  pair_soft_hitrate_whigh_rerun
                O49h2, rerun O49h under a fresh experiment name.
  pair_soft_hitrate_whigh_slowramp
                O49j, O49h with a slower seq-turn ramp of 300 updates.
  pair_soft_hitrate_wmid
                O49l, O49f with seq_turn_loss_w=0.0065.
  pair_soft_hitrate_late
                O49i, O49f with seq_turn_start_updates=120.
  o49_next      Run O49h2, O49j, and O49l.
  all           Run control and all eight seq-turn variants.
EOF
}

main() {
  local mode="${1:-all}"
  case "$mode" in
    compile)
      run_compile_check
      ;;
    control)
      run_compile_check
      run_control
      ;;
    pair_light)
      run_compile_check
      run_pair_light
      ;;
    pair_midband)
      run_compile_check
      run_pair_midband
      ;;
    pair_late)
      run_compile_check
      run_pair_late
      ;;
    pair_hitrate)
      run_compile_check
      run_pair_hitrate
      ;;
    pair_soft_hitrate)
      run_compile_check
      run_pair_soft_hitrate
      ;;
    pair_soft_hitrate_wlow)
      run_compile_check
      run_pair_soft_hitrate_wlow
      ;;
    pair_soft_hitrate_whigh)
      run_compile_check
      run_pair_soft_hitrate_whigh
      ;;
    pair_soft_hitrate_whigh_rerun)
      run_compile_check
      run_pair_soft_hitrate_whigh_rerun
      ;;
    pair_soft_hitrate_whigh_slowramp)
      run_compile_check
      run_pair_soft_hitrate_whigh_slowramp
      ;;
    pair_soft_hitrate_wmid)
      run_compile_check
      run_pair_soft_hitrate_wmid
      ;;
    pair_soft_hitrate_late)
      run_compile_check
      run_pair_soft_hitrate_late
      ;;
    o49_next)
      run_compile_check
      run_pair_soft_hitrate_whigh_rerun
      run_pair_soft_hitrate_whigh_slowramp
      run_pair_soft_hitrate_wmid
      ;;
    all)
      run_compile_check
      run_control
      run_pair_light
      run_pair_midband
      run_pair_late
      run_pair_hitrate
      run_pair_soft_hitrate
      run_pair_soft_hitrate_wlow
      run_pair_soft_hitrate_whigh
      run_pair_soft_hitrate_whigh_rerun
      run_pair_soft_hitrate_whigh_slowramp
      run_pair_soft_hitrate_wmid
      run_pair_soft_hitrate_late
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
