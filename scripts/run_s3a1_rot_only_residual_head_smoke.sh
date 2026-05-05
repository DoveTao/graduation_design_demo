#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTDIR="checkpoints/S3a1_rot_only_residual_head_smoke"
REPORT="checkpoints/S3a1_rot_only_residual_head_smoke_report.md"
POLICY_JSON="checkpoints/S2b_clean_fine_rot_policy.json"
BASE_CKPT="$("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
policy = json.loads(Path("checkpoints/S2b_clean_fine_rot_policy.json").read_text(encoding="utf-8"))
print(policy["base_checkpoint_path"])
PY
)"

mkdir -p "$OUTDIR"

"$PYTHON_BIN" -m py_compile config.py interaction.py model.py train_mvp.py losses.py tools/s3a1_rot_only_residual_audit.py
"$PYTHON_BIN" tools/s3a1_rot_only_residual_audit.py

"$PYTHON_BIN" train_mvp.py \
  --set exp_name=S3a1_rot_only_residual_head_smoke \
  --set ckpt_dir=checkpoints \
  --set dt_bucket_scale_anchor_policy_json="$POLICY_JSON" \
  --set init_checkpoint="$BASE_CKPT" \
  --set strict_load_checkpoint=False \
  --set use_fine_stage=True \
  --set use_coupled_pose_residual_head=True \
  --set coupled_pose_residual_trainable=True \
  --set coupled_pose_residual_enable_rot=True \
  --set coupled_pose_residual_enable_tdir=False \
  --set coupled_pose_residual_force_tdir_zero=True \
  --set coupled_pose_residual_rot_scale=0.02 \
  --set coupled_pose_residual_tdir_scale=0.0 \
  --set coupled_pose_residual_gate_max=0.05 \
  --set train_coupled_pose_residual_only=True \
  --set freeze_backbone_for_coupled_pose=True \
  --set freeze_tmag_for_coupled_pose=True \
  --set freeze_dt_anchor_for_coupled_pose=True \
  --set coupled_pose_tdir_loss_w=0.0 \
  --set coupled_pose_joint_loss_w=0.0 \
  --set coupled_pose_chain_loss_w=0.0 \
  --set use_tdir_anchor_loss=False \
  --set w_tdir_anchor=0.0 \
  --set use_seq_turn_loss=False \
  --set seq_turn_loss_w=0.0 \
  --set use_seq_turn_chain_loss=False \
  --set seq_turn_chain_loss_w=0.0 \
  --set use_odom_chain_len_loss=False \
  --set odom_chain_len_loss_w=0.0 \
  --set use_odom_chain_vec_loss=False \
  --set odom_chain_vec_loss_w=0.0 \
  --set lr=5e-5 \
  --set batch_size=1 \
  --set grad_accum=1 \
  --set max_steps=30 \
  --set eval_every=10 \
  --set max_eval_batches=8 \
  --set max_train_eval_batches=8 \
  --set num_workers=0 \
  --set pin_memory=False \
  --set persistent_workers=False \
  --set prefetch_factor=2 \
  --set log_every=5 \
  --set save_last_train_state=False \
  --set save_last_eval_checkpoint=False \
  --set save_metric_checkpoints=False \
  --set save_best_joint_checkpoint=False \
  --set save_best_local_joint_checkpoint=False \
  --set save_best_odom_checkpoint=False \
  --set save_best_smallk_odom_checkpoint=False \
  --set save_odom_trajectory_debug=True \
  --set save_vis_examples=False \
  --set save_vis_payload_npz=False \
  --set save_vis_diag_json=False \
  --set amp=False

"$PYTHON_BIN" - <<'PY'
import json
import shutil
from pathlib import Path

audit_path = Path("checkpoints/S3a1_rot_only_residual_audit_report.md")
summary_path = Path("checkpoints/S3a1_rot_only_residual_head_smoke/final_summary.json")
report_path = Path("checkpoints/S3a1_rot_only_residual_head_smoke_report.md")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
last_eval = dict(summary.get("last_eval", {}))
first_train = dict(summary.get("first_train", {}))
last_train = dict(summary.get("last_train", {}))
path_ratio = last_eval.get("odom_shape_metric_mean_path_length_ratio", float("nan"))
ate = last_eval.get("odom_metric_ATE", float("nan"))
drift = last_eval.get("odom_metric_drift", float("nan"))
trainable_names = summary.get("trainable_names", [])
optimizer_only_rot_gate = all(
    name.startswith(("coupled_pose_head.backbone.", "coupled_pose_head.rot_head.", "coupled_pose_head.gate_head."))
    for name in trainable_names
)
no_tdir_trainable = all("tdir_head" not in name for name in trainable_names)
bad_forward = int(summary.get("bad_forward", 0))
skip_updates = int(summary.get("skip_updates", 0))
smoke_no_nan = bad_forward == 0 and skip_updates == 0
path_ratio_ok = (path_ratio == path_ratio) and (path_ratio >= 0.90)
ate_ok = (ate == ate) and (ate <= 7.652371)
drift_ok = (drift == drift) and (drift <= 1.45)
tdir_diff = float(last_train.get("tdir_before_after_max_diff", float("nan")))
tmag_diff = float(last_train.get("tmag_before_after_max_diff", float("nan")))

lines = [
    "# S3a1 Rot-Only Residual Head Smoke Report",
    "",
    "## Inputs",
    "",
    "- base policy: `checkpoints/S2b_clean_fine_rot_policy.json`",
    "- base checkpoint: `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`",
    "- updates attempted: `30`",
    "",
    "## Audit Summary",
    "",
    f"- audit markdown: `{audit_path}`",
    f"- trainable_param_count: `{summary.get('trainable_param_count', 0)}`",
    f"- frozen_param_count: `{summary.get('frozen_param_count', 0)}`",
    f"- optimizer_param_count: `{summary.get('optimizer_param_count', 0)}`",
    f"- optimizer only rot residual/gate: `{optimizer_only_rot_gate}`",
    f"- no tdir residual params trainable: `{no_tdir_trainable}`",
    f"- forbidden trainable params: `{summary.get('forbidden_trainable_params', [])}`",
    "",
    "## Smoke Train Result",
    "",
    f"- bad_forward: `{bad_forward}`",
    f"- skip_updates: `{skip_updates}`",
    f"- total_updates: `{summary.get('total_updates', 0)}`",
    f"- loss_total_last: `{last_train.get('loss_total', float('nan'))}`",
    f"- loss_total_first: `{first_train.get('loss_total', float('nan'))}`",
    f"- loss_coupled_rot_last: `{last_train.get('loss_coupled_rot', float('nan'))}`",
    f"- loss_coupled_rot_first: `{first_train.get('loss_coupled_rot', float('nan'))}`",
    f"- loss_coupled_reg_last: `{last_train.get('loss_coupled_reg', float('nan'))}`",
    f"- loss_coupled_reg_first: `{first_train.get('loss_coupled_reg', float('nan'))}`",
    f"- delta_rot_norm_mean_last: `{last_train.get('delta_rot_norm_mean', float('nan'))}`",
    f"- delta_tdir_norm_mean_last: `{last_train.get('delta_tdir_norm_mean', float('nan'))}`",
    f"- tdir_before_after_max_diff_last: `{tdir_diff}`",
    f"- tmag_before_after_max_diff_last: `{tmag_diff}`",
    f"- latest eval drift: `{drift}`",
    f"- latest eval ATE: `{ate}`",
    f"- latest eval path_ratio: `{path_ratio}`",
    "",
    "## Verdict",
    "",
    f"- smoke passed without NaN-like events: `{smoke_no_nan}`",
    f"- optimizer only rot residual/gate: `{optimizer_only_rot_gate and no_tdir_trainable}`",
    f"- tdir unchanged: `{tdir_diff == tdir_diff and tdir_diff <= 1.0e-9}`",
    f"- tmag unchanged: `{tmag_diff == tmag_diff and tmag_diff <= 1.0e-9}`",
    f"- path_ratio >= 0.90: `{path_ratio_ok}`",
    f"- ATE within +0.3 of S2b: `{ate_ok}`",
    f"- drift <= 1.45: `{drift_ok}`",
    f"- recommend train-CV small run next: `{smoke_no_nan and optimizer_only_rot_gate and no_tdir_trainable and path_ratio_ok and ate_ok and drift_ok}`",
]
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(report_path)

shutil.rmtree(Path("checkpoints/S3a1_rot_only_residual_head_smoke"), ignore_errors=True)
PY
