# S3a1 Rot-Only Residual Head Implementation Report

## 1. 修改文件

- `config.py`
- `interaction.py`
- `model.py`
- `train_mvp.py`
- `losses.py`
- `tools/s3a1_rot_only_residual_audit.py`
- `scripts/run_s3a1_rot_only_residual_head_smoke.sh`
- `checkpoints/S3a1_rot_only_residual_audit_report.md`
- `checkpoints/S3a1_rot_only_residual_head_smoke_report.md`

## 2. Rot-Only Mode 如何实现

- 新增 rot-only 配置:
  - `coupled_pose_residual_enable_rot=True`
  - `coupled_pose_residual_enable_tdir=False`
  - `coupled_pose_residual_force_tdir_zero=True`
  - `coupled_pose_residual_rot_scale=0.02`
  - `coupled_pose_residual_tdir_scale=0.0`
  - `coupled_pose_residual_gate_max=0.05`
- `CoupledPoseResidualHead` 现在支持按配置禁用 `tdir` 分支。
- 当 `enable_tdir=False` 或 `force_tdir_zero=True` 时:
  - `delta_tdir_vec` 强制为零
  - `tdir_after_coupled == tdir_before_coupled`
  - `tmag_after_coupled == tmag_before_coupled`
  - 最终只允许 `delta_R` 改变输出 pose
- `gate` 现在带硬上界 `gate_max`，避免 residual 在 smoke 阶段过快放大。

## 3. Tdir / Tmag Invariance

来源: [checkpoints/S3a1_rot_only_residual_audit_report.md](/home/dovetao/graduation_design_demo/checkpoints/S3a1_rot_only_residual_audit_report.md)

- rot-only gate-zero 复现 `S2b`:
  - `drift=1.327402`
  - `ATE=7.352371`
  - `path_ratio=0.934984`
- `tdir_before_after_max_diff = 0.000000000000`
- `tmag_before_after_max_diff = 0.000000000000`
- `delta_tdir_norm_mean = 0.000000000000`

来源: [checkpoints/S3a1_rot_only_residual_head_smoke_report.md](/home/dovetao/graduation_design_demo/checkpoints/S3a1_rot_only_residual_head_smoke_report.md)

- smoke 最后一步:
  - `delta_tdir_norm_mean_last = 0.0`
  - `tdir_before_after_max_diff_last = 0.0`
  - `tmag_before_after_max_diff_last = 0.0`

## 4. Optimizer Audit

- trainable 参数只包含:
  - `coupled_pose_head.backbone.*`
  - `coupled_pose_head.rot_head.*`
  - `coupled_pose_head.gate_head.*`
- `tdir_head` 不在 trainable 名单中。
- audit 结果:
  - `trainable_param_count = 119076`
  - `optimizer_param_count = 119076`
  - `optimizer only rot residual/gate = True`
  - `no tdir residual params trainable = True`

## 5. Smoke Train Result

- 数值稳定性:
  - `bad_forward = 0`
  - `skip_updates = 0`
  - `total_updates = 30`
- loss:
  - `loss_total_first = 1.614173412322998`
  - `loss_total_last = 1.223811388015747`
  - `loss_coupled_rot_first = 0.5942341089248657`
  - `loss_coupled_rot_last = 0.5624608993530273`
- residual 量级:
  - `delta_rot_norm_mean_last = 7.061625365167856e-05`
  - `delta_tdir_norm_mean_last = 0.0`
- smoke eval:
  - `drift = 1.3284205512539493`
  - `ATE = 7.356587799281789`
  - `path_ratio = 0.9350349269488233`

结论:

- `path_ratio >= 0.90`: `PASS`
- `ATE` 未比 `S2b` 恶化 `0.3` 以上: `PASS`
- `drift <= 1.45`: `PASS`

## 6. 是否允许进入 S3a1 Train-CV Small Run

- 是。
- 当前 rot-only smoke 满足安全条件，已经证明:
  - policy-aligned path 正常
  - optimizer 只训练 rot residual / gate
  - `tdir` 与 `tmag` 未被破坏
  - odom `path_ratio` 没有崩坏
- 因此允许进入 `S3a1` 的 train-CV small run。

## 7. 当前 S2b 是否仍保持 Clean Candidate

- 是。
- 虽然 `S3a1` smoke 通过，但这还不是 train-CV 选优结果。
- 在出现新的 clean 选择结果之前，`S2b` 仍然保持当前 clean candidate:
  - `drift=1.327402`
  - `ATE=7.352371`
  - `path_ratio=0.934984`
