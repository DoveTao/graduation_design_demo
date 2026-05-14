# S3a1 Rot-Only Residual Head Smoke Report

## Inputs

- base policy: `checkpoints/S2b_clean_fine_rot_policy.json`
- base checkpoint: `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- updates attempted: `30`

## Audit Summary

- audit markdown: `checkpoints/S3a1_rot_only_residual_audit_report.md`
- trainable_param_count: `119076`
- frozen_param_count: `10904782`
- optimizer_param_count: `119076`
- optimizer only rot residual/gate: `True`
- no tdir residual params trainable: `True`
- forbidden trainable params: `[]`

## Smoke Train Result

- bad_forward: `0`
- skip_updates: `0`
- total_updates: `30`
- loss_total_last: `1.223811388015747`
- loss_total_first: `1.614173412322998`
- loss_coupled_rot_last: `0.5624608993530273`
- loss_coupled_rot_first: `0.5942341089248657`
- loss_coupled_reg_last: `8.142813499034673e-07`
- loss_coupled_reg_first: `8.087594096650719e-07`
- delta_rot_norm_mean_last: `7.061625365167856e-05`
- delta_tdir_norm_mean_last: `0.0`
- tdir_before_after_max_diff_last: `0.0`
- tmag_before_after_max_diff_last: `0.0`
- latest eval drift: `1.3284205512539493`
- latest eval ATE: `7.356587799281789`
- latest eval path_ratio: `0.9350349269488233`

## Verdict

- smoke passed without NaN-like events: `True`
- optimizer only rot residual/gate: `True`
- tdir unchanged: `True`
- tmag unchanged: `True`
- path_ratio >= 0.90: `True`
- ATE within +0.3 of S2b: `True`
- drift <= 1.45: `True`
- recommend train-CV small run next: `True`
