#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTDIR="checkpoints/S3a0_coupled_pose_residual_head_smoke"
REPORT="checkpoints/S3a0_coupled_pose_residual_head_smoke_report.md"
POLICY_JSON="checkpoints/S2b_clean_fine_rot_policy.json"
BASE_CKPT="$("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
policy = json.loads(Path("checkpoints/S2b_clean_fine_rot_policy.json").read_text(encoding="utf-8"))
print(policy["base_checkpoint_path"])
PY
)"

mkdir -p "$OUTDIR"

"$PYTHON_BIN" -m py_compile config.py interaction.py model.py train_mvp.py losses.py tools/s3a0_audit_coupled_pose_head.py
"$PYTHON_BIN" tools/s3a0_audit_coupled_pose_head.py

"$PYTHON_BIN" train_mvp.py \
  --set exp_name=S3a0_coupled_pose_residual_head_smoke \
  --set ckpt_dir=checkpoints \
  --set init_checkpoint="$BASE_CKPT" \
  --set strict_load_checkpoint=False \
  --set use_fine_stage=True \
  --set fine_rot_fuse_strength=0.45 \
  --set fine_tdir_fuse_strength=0.0 \
  --set fine_tmag_fuse_strength=0.0 \
  --set use_geometry_refine=False \
  --set tmag_condition_on_dt=False \
  --set use_coupled_pose_residual_head=True \
  --set coupled_pose_residual_trainable=True \
  --set train_coupled_pose_residual_only=True \
  --set freeze_backbone_for_coupled_pose=True \
  --set freeze_tmag_for_coupled_pose=True \
  --set freeze_dt_anchor_for_coupled_pose=True \
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
from pathlib import Path

audit_path = Path("checkpoints/S3a0_coupled_pose_head_smoke_audit.md")
summary_path = Path("checkpoints/S3a0_coupled_pose_residual_head_smoke/final_summary.json")
report_path = Path("checkpoints/S3a0_coupled_pose_residual_head_smoke_report.md")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
last_eval = dict(summary.get("last_eval", {}))
path_ratio = last_eval.get("odom_shape_metric_mean_path_length_ratio", float("nan"))
ate = last_eval.get("odom_metric_ATE", float("nan"))
drift = last_eval.get("odom_metric_drift", float("nan"))
trainable_names = summary.get("trainable_names", [])
optimizer_only_coupled = all(name.startswith("coupled_pose_head.") for name in trainable_names)
forbidden = summary.get("forbidden_trainable_params", [])
path_ratio_ok = (path_ratio == path_ratio) and (path_ratio >= 0.90)
smoke_no_nan = summary.get("bad_forward", 0) == 0 and summary.get("skip_updates", 0) == 0

lines = [
    "# S3a0 Coupled Pose Residual Head Smoke Report",
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
    f"- optimizer only coupled head: `{optimizer_only_coupled}`",
    f"- forbidden trainable params: `{forbidden}`",
    "",
    "## Smoke Train Result",
    "",
    f"- bad_forward: `{summary.get('bad_forward', 0)}`",
    f"- skip_updates: `{summary.get('skip_updates', 0)}`",
    f"- total_updates: `{summary.get('total_updates', 0)}`",
    f"- latest eval drift: `{drift}`",
    f"- latest eval ATE: `{ate}`",
    f"- latest eval path_ratio: `{path_ratio}`",
    "",
    "## Verdict",
    "",
    f"- smoke passed without NaN-like events: `{smoke_no_nan}`",
    f"- path_ratio not obviously broken: `{path_ratio_ok}`",
    f"- recommend long train now: `False`",
    f"- recommend train-CV small run next: `{smoke_no_nan and optimizer_only_coupled and path_ratio_ok}`",
    "",
    "## Notes",
    "",
    "- This smoke run only checks wiring, optimizer isolation, and numerical stability.",
    "- It does not claim improvement over the S2b clean candidate.",
]
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(report_path)
PY
