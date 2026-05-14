# S3a0 Coupled Pose Residual Head Smoke Report

## Inputs

- base policy: `checkpoints/S2b_clean_fine_rot_policy.json`
- base checkpoint: `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- updates attempted: `30`

## Audit Summary

- audit markdown: `checkpoints/S3a0_coupled_pose_head_smoke_audit.md`
- trainable_param_count: `119463`
- frozen_param_count: `10572485`
- optimizer_param_count: `119463`
- optimizer only coupled head: `True`
- forbidden trainable params: `[]`

## Smoke Train Result

- bad_forward: `0`
- skip_updates: `0`
- total_updates: `30`
- latest eval drift: `7.3170828428151955`
- latest eval ATE: `8.738240317016485`
- latest eval path_ratio: `0.28612668773627314`

## Verdict

- smoke passed without NaN-like events: `True`
- path_ratio not obviously broken: `False`
- recommend long train now: `False`
- recommend train-CV small run next: `False`

## Notes

- This smoke run validates wiring, optimizer isolation, and numerical stability.
- It does not preserve the clean S2b path_ratio yet, so S2b remains the current clean candidate.
