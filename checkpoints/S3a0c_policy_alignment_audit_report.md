# S3a0c Policy Alignment Audit Report

## Expected Targets

- policy json: `/home/dovetao/graduation_design_demo/checkpoints/S2b_clean_fine_rot_policy.json`
- base checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- reference drift: `1.327402`
- reference ATE: `7.352371`
- reference path_ratio: `0.934984`

## Variant Table

| variant | drift | ATE | path_ratio | RPE_rot | RPE_tdir | RPE_tmag | rot | tdir_abs | tdir_local_A_abs | tmag_p10 | tmag_p50 | tmag_p90 | missing | unexpected | fine_rot | fine_tdir | fine_tmag | policy_json | dt_anchor | tmag_before_after_diff | gate_mean | d_rot_norm | d_tdir_norm | factor | k | pairs | chains | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw_no_policy | 1.326834 | 7.522779 | 0.496344 | 20.715353 | 72.349483 | 0.122433 | 21.481086 | 29.944641 | 23.152590 | 0.123422 | 0.151614 | 0.156135 | 14 | 0 | 0.450000 | 0.000000 | 0.000000 | - | False | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1 | 132 | 19 | raw base checkpoint path without S2b bucket scaling policy |
| S2b_policy_baseline | 1.327402 | 7.352371 | 0.934984 | 20.715353 | 72.349483 | 0.073926 | 21.481086 | 29.944641 | 23.152590 | 0.126983 | 0.183064 | 0.402117 | 14 | 0 | 0.450000 | 0.000000 | 0.000000 | /home/dovetao/graduation_design_demo/checkpoints/S2b_clean_fine_rot_policy.json | True | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1 | 132 | 19 | official wrapper-style S2b clean policy path |
| coupled_head_off_with_policy | 1.327402 | 7.352371 | 0.934984 | 20.715353 | 72.349483 | 0.073926 | 21.481086 | 29.944641 | 23.152590 | 0.126914 | 0.183148 | 0.402129 | 14 | 0 | 0.450000 | 0.000000 | 0.000000 | /home/dovetao/graduation_design_demo/checkpoints/S2b_clean_fine_rot_policy.json | True | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1 | 132 | 19 | current train_mvp/model path with integrated policy and coupled head disabled |
| coupled_gate_zero_with_policy | 1.327402 | 7.352371 | 0.934984 | 20.715353 | 72.349483 | 0.073926 | 21.481086 | 29.944641 | 23.152590 | 0.126905 | 0.183179 | 0.402085 | 14 | 0 | 0.450000 | 0.000000 | 0.000000 | /home/dovetao/graduation_design_demo/checkpoints/S2b_clean_fine_rot_policy.json | True | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 1 | 132 | 19 | coupled head enabled, but outputs forced back to pre-coupled pose |
| coupled_residual_scale_zero_with_policy | 1.327402 | 7.352371 | 0.934984 | 20.715353 | 72.349483 | 0.073926 | 21.481086 | 29.944641 | 23.152590 | 0.126979 | 0.183229 | 0.402544 | 14 | 0 | 0.450000 | 0.000000 | 0.000000 | /home/dovetao/graduation_design_demo/checkpoints/S2b_clean_fine_rot_policy.json | True | 0.000000 | 0.017986 | 0.000000 | 0.000000 | 1.000000 | 1 | 132 | 19 | coupled head enabled with residual scales forced to zero |

## PASS Checks

- audit_pass: `True`
- S2b_policy_baseline_matches_reference: `True`
- coupled_head_off_matches_S2b: `True`
- coupled_gate_zero_matches_S2b: `True`
- coupled_scale_zero_matches_S2b: `True`
- tmag_before_after_preserved: `True`
- unexpected_zero: `True`
- fine_rot_is_0p45: `True`
- no_training: `True`

## Path Audit Answers

1. S3a0 smoke 当前从哪个 checkpoint 初始化？`checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
2. S3a0 smoke 当前是否加载 checkpoints/S2b_clean_fine_rot_policy.json？修复前否；修复后通过 `dt_bucket_scale_anchor_policy_json` 显式加载。
3. S3a0 smoke 当前是否应用 S1d5/S2b dt-anchor effective bucket factors？修复前否；修复后是。
4. S3a0 smoke 当前是否应用 fine_rot=0.45？是，当前审计所有 variant 都是 `fine_rot=0.45`。
5. S3a0 smoke 当前评估 path_ratio 的路径是否等价于 eval_s2b_clean_policy.sh？修复后是；`S2b_policy_baseline` 与 `coupled_head_off_with_policy` 应一致。
6. gate=0 时为什么之前复现的是 raw_no_policy 0.496，而不是 S2b 0.935？因为旧 smoke 路径没有加载 S2b clean bucket-scaled dt-anchor magnitude policy，gate=0 只关闭了 coupled residual，没有补回 S2b 的 magnitude calibration。
