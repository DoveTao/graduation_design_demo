# S15a Trajectory Training Smoke Failure Attribution Report

## Executive summary
- final classification: `TRAINING-HARNESS-ISSUE`
- main failure cause: pair-anchor-only tiny run already collapses, so the failure appears before trajectory loss tuning
- S15b full clean CV recommended: `no`
- S5 remains final clean candidate: `yes`

## Baseline context
- baseline gate: `passed`
- accepted S8b load path: `14 / 0`
- locked S5 metrics: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`
- S15 smoke candidate: `D_full_light_traj`
- S15 smoke mode: `fine_pose_light`
- S15 smoke heldout result: val_ate_proxy=`15.708141`, val_path_proxy=`2.515539`, ATE=`19.286121`, drift=`24.674397`

## Accumulation convention audit
- matched eval accumulation: `True`
- translation output frame: `B`
- tdir interpretation: t_vec_out is already in the pose frame used by eval accumulation
- rotation composition order: matches _compose_rel_pose_np / _compose_rel_pose_torch
- local trajectory alignment: pred and GT windows both start at identity pose and zero translation
- compose consistency diff mean: `0.000000`
- naive world-translation mismatch mean: `0.476917`

## No-train baseline window audit
- num windows checked: `128`
- val_ate_proxy: `5.554382`
- val_drift_proxy: `7.386448`
- val_path_proxy: `1.907361`
- val_rot_traj: `31.659700`
- val_tdir_traj: `16.745664`
- val_tmag_step: `1.940955`
- interpretation: the proxy is already hard and noisy, but it is still much less catastrophic than the trained smoke result; this points to training-induced failure rather than a pure proxy-definition bug.

## Pair-anchor-only tiny run
- available: `True`
- run dir: `/tmp/s15_trajectory_level_training_runs/S15_A_pair_anchor_only_tiny_heldout_scene01_seq01_u10`
- trainable groups: `{'coarse': 231425, 'fine': 231425}`
- odometry ATE: `37.949599`
- odometry drift: `59.054601`
- path_ratio: `nan`
- rot: `20.258530`
- tdir_abs: `20.920788`
- tmag_rel_err: `14.123667`
- interpretation: even without any trajectory losses, a tiny S15 harness run already collapses badly. This is the strongest evidence that the immediate failure is not caused solely by trajectory loss weighting.

## Trajectory loss scale / gradient audit
- trainable parameter count in audit mode: `4687662`
- pair_pose value: `2.193419`
- loss_ate value: `1.560853`
- loss_drift value: `2.138678`
- loss_path value: `0.280511`
- loss_tdir value: `0.948752`
- pair_pose grad max: `347.897735`
- trajectory grad max: `146.712398`
- trajectory/pair grad ratio: `0.421711`
- pair_tmag requires grad: `False`
- interpretation: pair-pose gradients are larger than the trajectory losses in this audit batch, so the current evidence does not support `LOSS-SCALE-IMBALANCE` as the primary explanation.

## Parameter drift audit
- direct smoke checkpoint delta available: `False`
- note: smoke final checkpoint is not available locally, so direct before/after parameter delta for fine_pose_light could not be recomputed in this audit
- attribution implication: destructive fine-pose drift cannot be proven directly from the removed smoke checkpoint, but it is also not needed to explain the failure because the tmag-head-only pair-anchor control already collapses.

## Omitted ablations
- `tmag_head_only + trajectory path/tmag` and `no_path / no_tdir / low_traj_weight` were not expanded into a mini sweep.
- reason: pair-anchor-only already isolated a stronger earlier-stage failure, so more tiny ablations would add cost without changing the primary attribution.

## Conclusion
- final classification: `TRAINING-HARNESS-ISSUE`
- main failure cause: pair-anchor-only tiny run already collapses, so the failure appears before trajectory loss tuning
- should continue S15b full CV: `no`
- if S15 is ever resumed, fix the harness first: preserve baseline behavior under tiny pair-anchor updates, verify checkpoint/init preservation, and only then revisit trajectory-loss weighting.
- S5 remains final clean candidate: `yes`
