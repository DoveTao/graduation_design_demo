# S3a0 Coupled Pose Head Smoke Audit

- policy: `/home/dovetao/graduation_design_demo/checkpoints/S2b_clean_fine_rot_policy.json`
- base checkpoint: `/home/dovetao/graduation_design_demo/checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- device: `cuda`
- audit_pass: `True`

## Load Summary

- plain missing=14 unexpected=0 skipped=0
- coupled missing=14 unexpected=0 skipped=0
- coupled missing preview: `['coarse.mag_head.ridge_calib_raw_center', 'fine.mag_head.ridge_calib_raw_center', 'coupled_pose_head.backbone.0.weight', 'coupled_pose_head.backbone.0.bias', 'coupled_pose_head.backbone.1.weight', 'coupled_pose_head.backbone.1.bias', 'coupled_pose_head.backbone.4.weight', 'coupled_pose_head.backbone.4.bias']`
- allowed base missing: `['coarse.mag_head.ridge_calib_raw_center', 'fine.mag_head.ridge_calib_raw_center']`
- coupled unexpected preview: `[]`

## Coupled Head Presence

- coupled param count: `12`
- coupled param preview: `['coupled_pose_head.backbone.0.weight', 'coupled_pose_head.backbone.0.bias', 'coupled_pose_head.backbone.1.weight', 'coupled_pose_head.backbone.1.bias', 'coupled_pose_head.backbone.4.weight', 'coupled_pose_head.backbone.4.bias', 'coupled_pose_head.rot_head.weight', 'coupled_pose_head.rot_head.bias']`

## Forward Sanity

- baseline vs coupled `R` max abs diff: `0.000000000000`
- baseline vs coupled `t_dir_out` max abs diff: `0.000000000000`
- baseline vs coupled `t_mag` max abs diff: `0.000000000000`
- `tmag_before_coupled` vs `tmag_after_coupled` max abs diff: `0.000000000000`
- residual rot norm mean: `0.000000000000`
- residual tdir norm mean: `0.000000000000`
- gate mean: `0.017986210063`

## Freeze / Optimizer Audit

- trainable_param_count: `119463`
- frozen_param_count: `10904395`
- optimizer_param_count: `119463`
- optimizer only coupled head: `True`
- forbidden trainable params: `[]`
- trainable names: `['coupled_pose_head.backbone.0.weight', 'coupled_pose_head.backbone.0.bias', 'coupled_pose_head.backbone.1.weight', 'coupled_pose_head.backbone.1.bias', 'coupled_pose_head.backbone.4.weight', 'coupled_pose_head.backbone.4.bias', 'coupled_pose_head.rot_head.weight', 'coupled_pose_head.rot_head.bias', 'coupled_pose_head.tdir_head.weight', 'coupled_pose_head.tdir_head.bias', 'coupled_pose_head.gate_head.weight', 'coupled_pose_head.gate_head.bias']`

## Verdict

- missing keys limited to new coupled head params after allowed-base filtering: `True`
- unexpected keys empty: `True`
- baseline output unchanged when coupled head is enabled at zero residual init: `True`
- `tmag` preserved before/after coupled residual: `True`
- optimizer isolated to coupled head params: `True`
