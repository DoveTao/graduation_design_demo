# S15c Clean Policy Wrapped Training Harness Fix Report

## Executive summary
- final classification: `HARNESS-FIX-PARTIAL`
- S15 can continue: `False`
- S5 remains final clean candidate: `yes`

## Policy wrapper audit
- policy path: `/home/dovetao/graduation_design_demo/checkpoints/S5_clean_tmag_calibration_policy.json`
- base policy path: `/home/dovetao/graduation_design_demo/checkpoints/S2b_clean_fine_rot_policy.json`
- base checkpoint path: `/home/dovetao/graduation_design_demo/checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- fine rot/tdir/tmag: `0.45` / `0.0` / `0.0`
- S5 thresholds: `{'q90_value': 0.49714615345001223, 'q95_upper_tail_value': 0.5355591773986816, 'quantile_fit_protocol': 'deterministic subsample of 512 train pairs from S2b-wrapped model predictions'}`
- S5 scales: `{'mid_scale': 0.95, 'high_scale': 0.8, 'identity_scale': 1.0}`
- dt-anchor apply after fix: `True`
- policy wrapping enabled after fix: `True`
- restore enabled after fix: `True`

## Dropout train/eval drift after fix
- dropout modules: `17`
- dropout frozen eval: `17`
- dropout trainable train: `0`
- eval_vs_train_R_diff: `0.533686`
- eval_vs_train_tvec_diff: `0.045334`
- eval_vs_train_tmag_diff: `0.000180`

## Zero-update wrapped audit
- direct missing/unexpected: `14 / 0`
- training missing/unexpected: `14 / 0`
- wrapped R diff: `0.000000`
- wrapped tvec diff: `0.000000`
- wrapped tmag diff: `0.000000`

## One-update tiny audit
- lr=`5e-06`: R_delta=`0.773798`, tvec_delta=`0.034043`, tmag_delta=`0.001210`, param_delta_l2=`0.005612`
- lr=`5e-07`: R_delta=`0.773673`, tvec_delta=`0.034476`, tmag_delta=`0.000157`, param_delta_l2=`0.000560`

## Tiny smoke after fix
- tiny smoke failed to run: `train_mvp failed, see /tmp/s15_trajectory_level_training_runs/S15c_B_pair_plus_ate_path_wrapped_smoke_u10/train_stdout.log`

## Decision
- final classification: `HARNESS-FIX-PARTIAL`
- whether S15 can continue: `False`
- next step: `S15d tiny trajectory retest` only if this harness-fix pass is accepted; otherwise stop S15.
- S5 remains final clean candidate: `yes`
