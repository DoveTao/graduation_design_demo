# S3a1 Rot-Only Residual Audit Report

- policy: `/home/dovetao/graduation_design_demo/checkpoints/S2b_clean_fine_rot_policy.json`
- base checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- audit_pass: `True`

## Baseline

- drift: `1.327402`
- ATE: `7.352371`
- path_ratio: `0.934984`

## Rot-Only Gate-Zero

- drift: `1.327402`
- ATE: `7.352371`
- path_ratio: `0.934984`
- delta_rot_norm_mean: `0.000000000000`
- delta_tdir_norm_mean: `0.000000000000`
- gate_mean: `0.000000000000`
- tdir_before_after_max_diff: `0.000000000000`
- tmag_before_after_max_diff: `0.000000000000`

## Optimizer Audit

- trainable_param_count: `119076`
- frozen_param_count: `10904782`
- optimizer_param_count: `119076`
- optimizer only rot residual/gate: `True`
- no tdir residual params trainable: `True`
- trainable_names: `['coupled_pose_head.backbone.0.weight', 'coupled_pose_head.backbone.0.bias', 'coupled_pose_head.backbone.1.weight', 'coupled_pose_head.backbone.1.bias', 'coupled_pose_head.backbone.4.weight', 'coupled_pose_head.backbone.4.bias', 'coupled_pose_head.rot_head.weight', 'coupled_pose_head.rot_head.bias', 'coupled_pose_head.gate_head.weight', 'coupled_pose_head.gate_head.bias']`

## Load Summary

- baseline missing/unexpected: `14/0`
- rot-only missing/unexpected: `14/0`
