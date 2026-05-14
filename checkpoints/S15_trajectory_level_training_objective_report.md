# S15 Trajectory Level Training Objective Report
## Executive summary
- final classification: `INCONCLUSIVE`
- smoke passed: `True`
- lightweight CV run: `False`
- best diagnostic candidate: `D_full_light_traj`
- S15 replaces S5: `False`
- S15b full clean CV recommended: `False`

## Motivation from S13/S14
- S13 showed that the practical gap is dominated by coupled R/tdir error and chain accumulation.
- S14 showed that lightweight inference-time local-window optimization did not stably solve that gap.
- S15 therefore moves the experiment back into training-time trajectory-aware supervision.

## Baseline gate
- contract path: `/home/dovetao/graduation_design_demo/checkpoints/S8b_reproduction_contract.json`
- load_missing / load_unexpected: `14 / 0`
- locked S5 metrics: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`

## Window dataset audit
- proxy evaluation window cap per fold/candidate: `128`
- `heldout_scene01_seq01`: train_windows=`322`, val_windows=`462`, train_seq=`['scene01/seq01']`, val_seq=`['scene01/seq02']`, avg_edges_per_window=`2.0`
  dt_ab_mean=`0.080529`, dt_bc_mean=`0.080340`, gt_tmag_ab_mean=`0.080529`, gt_tmag_bc_mean=`0.080340`, k_dist=`{'1': 644}`

## Accumulation convention
- S15 uses the same relative-pose composition convention already used by evaluation: for a window `(A,B,C)`, the predicted short trajectory composes `T_BA` then `T_CB` to obtain `T_CA`, and camera centers are recovered from the composed extrinsics.
- This keeps S15 aligned with the existing odometry accumulation path and avoids convention drift.

## Loss definitions
- Pair-level pose and tmag losses remain enabled as anchors.
- Trajectory-level diagnostic losses are defined on triplet windows (`W=3`): local ATE proxy, drift proxy, path-ratio proxy, composed rotation consistency, step-direction consistency, step-magnitude log loss, and speed log loss.
- This first pass does not expand to `W=5`; it stays on the smallest stable trajectory window.

## Candidate definitions

## Trainable parameter audit

## Smoke result
- smoke candidate=`D_full_light_traj`, mode=`fine_pose_light`, max_steps=`20`, val_ate_proxy=`15.708141`, val_path_proxy=`2.515539`, ATE=`19.286121`, drift=`24.674397`, path_ratio=`nan`

## Lightweight two-fold CV result
| candidate | mode | cv_ate_proxy | cv_drift_proxy | cv_path_proxy | cv_rot_traj | cv_tdir_traj | cv_tmag_step | cv_speed_log | cv_ATE | cv_drift | cv_path_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Odometry eval result
- odometry eval was run: `True`

## Component diagnostics

## Hard-gate audit
- no reproduction mismatch: `True`
- missing/unexpected stayed on accepted path: `True`
- coupled residual head remained disabled/frozen: `True`
- path_ratio evidence available: `True`

## Leakage audit
- S15 uses gt pose / gt tdir / gt tmag only as training supervision.
- Candidate selection is train-split two-fold only; test-set tuning is not used.
- Inference-time policy is unchanged and does not depend on gt features.

## Whether S15b full clean CV is recommended
- `False`

## Whether S15 replaces S5
- `False`

## Final classification
- `INCONCLUSIVE`
