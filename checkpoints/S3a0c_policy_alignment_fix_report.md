# S3a0c Policy Alignment Fix Report

## 1. S3a0b Failure Reinterpretation

- `S3a0b` 的核心结论保持成立：之前没有足够 artifact 证明 trained coupled residual 本身导致了 `path_ratio=0.286127`。
- 本轮对齐审计进一步说明，旧 smoke/eval 路径首先漏掉了 `S2b` clean magnitude policy。
- 因此，旧 smoke 的失败不能直接归因到 `delta_R` 或 `delta_tdir`。

## 2. 原路径为什么没走 S2b Policy

- `scripts/run_s3a0_coupled_pose_residual_head_smoke.sh` 和 `scripts/run_s3a0_coupled_pose_residual_head_train.sh` 之前只从 `checkpoints/S2b_clean_fine_rot_policy.json` 读取 `base_checkpoint_path`。
- 这两条路径手动设置了 `fine_rot=0.45 / fine_tdir=0.0 / fine_tmag=0.0`，但没有把 `effective_bucket_factors` 接入 `train_mvp.py` / `model.py`。
- 所以旧 smoke 的 `tmag_before_coupled == tmag_after_coupled` 仅表示 coupled head 没改 raw checkpoint 的 `tmag`，不表示它复用了 `S2b` clean candidate 的 bucket-scaled dt-anchor magnitude policy。
- 这也是为什么旧版 `gate=0` 只能回到 `raw_no_policy path_ratio=0.496344`，而不是 `S2b path_ratio=0.934984`。

## 3. 修复了哪些文件

- `config.py`
- `model.py`
- `train_mvp.py`
- `tools/s3a0_audit_coupled_pose_head.py`
- `tools/s3a0c_policy_alignment_audit.py`
- `scripts/run_s3a0_coupled_pose_residual_head_smoke.sh`
- `scripts/run_s3a0_coupled_pose_residual_head_train.sh`

## 4. Policy-Alignment Audit Table

来源: [checkpoints/S3a0c_policy_alignment_audit_report.md](/home/dovetao/graduation_design_demo/checkpoints/S3a0c_policy_alignment_audit_report.md)

| variant | drift | ATE | path_ratio | note |
|---|---|---|---|---|
| raw_no_policy | 1.326834 | 7.522779 | 0.496344 | raw base checkpoint without S2b bucket scaling |
| S2b_policy_baseline | 1.327402 | 7.352371 | 0.934984 | official wrapper-style S2b clean path |
| coupled_head_off_with_policy | 1.327402 | 7.352371 | 0.934984 | integrated policy path, coupled head off |
| coupled_gate_zero_with_policy | 1.327402 | 7.352371 | 0.934984 | integrated policy path, gate forced zero |
| coupled_residual_scale_zero_with_policy | 1.327402 | 7.352371 | 0.934984 | integrated policy path, residual scales zero |

## 5. PASS 条件

- `S2b_policy_baseline` 复现参考指标: `PASS`
- `coupled_head_off_with_policy` 复现 `S2b`: `PASS`
- `coupled_gate_zero_with_policy` 复现 `S2b`: `PASS`
- `coupled_residual_scale_zero_with_policy` 复现 `S2b`: `PASS`
- `tmag before/after` 保持不变: `PASS`
- `unexpected == 0`: `PASS`
- `fine_rot == 0.45`: `PASS`
- 未启动训练: `PASS`

结论: `S3a0c policy-alignment audit PASS`

## 6. 是否允许进入下一步 S3a1

- 允许进入下一步 `S3a1`，但前提是后续所有 smoke / eval / 小训都必须继续显式加载 `checkpoints/S2b_clean_fine_rot_policy.json`。
- 建议的下一个实验仍然是 `S3a1_rot_only_residual_head`，因为现在 baseline path 已经与 `S2b` clean candidate 对齐，后续才有资格讨论 residual 本身是否安全。
- 本轮仍然没有跑正式训练，也没有进入 train-CV。

## 7. 当前 S2b 是否仍为 Clean Candidate

- 是。`S2b` 仍然是当前 clean candidate。
- `S3a0c` 只修复了 smoke/eval 对齐问题，没有产生新的训练结果，也没有替代 `S2b`。
