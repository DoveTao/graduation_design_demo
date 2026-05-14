# S15e Tiny Trajectory Retest After Harness Fix Report

## Executive summary
- final classification: `TRAINING-STILL-UNSTABLE`
- main finding: `pair-only baseline still collapses after harness parity fix`
- candidates run: `A_pair_only_baseline_fixed_harness, D_full_light_traj_fixed_harness`
- training budget: `W=3, updates=20, batch_size=1`
- trainable mode used: `tmag_head_only`
- S15f lightweight CV recommended: `False`
- S5 remains final clean candidate: `yes`

## Baseline context
- branch: `optimize/s15-trajectory-level-training-objective`
- S15d commit: `8aeeb84`
- S15d classification: `FORWARD-PARITY-FIX-PASS`
- locked S5 metrics: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`

## Tiny retest setup
- fold: `heldout_scene01_seq01`
- split_seed: `0`
- manifest root: `/tmp/s15_trajectory_level_training_manifests`
- policy wrapper expected: `True`
- dt anchor apply: `True`

## Candidate results
| candidate | val_ate_proxy | val_drift_proxy | val_path_proxy | val_rot_traj | val_tdir_traj | val_tmag_step | odom_ATE | odom_drift | odom_path_ratio | R_delta | tvec_delta | tmag_delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_pair_only_baseline_fixed_harness | 5.457961 | 7.262767 | 1.734796 | 31.659700 | 16.745664 | 1.774747 | 21.710394 | 35.126991 | 2.647609 | 0.000000 | 0.001327 | 0.001327 |
| D_full_light_traj_fixed_harness | 5.528377 | 7.356461 | 1.742522 | 31.659700 | 16.745665 | 1.782470 | 21.995400 | 35.549201 | 2.680616 | 0.000000 | 0.000089 | 0.000089 |

## Trainable parameter audit
- `A_pair_only_baseline_fixed_harness`: trainable_param_count=`794760`, optimizer_param_count=`794760`, trainable_groups=`{'log_tmag_bias': 1, 'coarse': 563334, 'fine': 231425}`
- `D_full_light_traj_fixed_harness`: trainable_param_count=`794760`, optimizer_param_count=`794760`, trainable_groups=`{'log_tmag_bias': 1, 'coarse': 563334, 'fine': 231425}`

## Train loss snapshot
- `A_pair_only_baseline_fixed_harness` first_train=`{'loss_total': 0.6612477898597717, 'loss_tmag': 2.1404099464416504, 'loss_tmag_speed': 0.0, 'loss_tmag_ratio': 0.0, 'loss_tmag_chain_sum': 0.0, 'loss_traj_ate': 0.0, 'loss_traj_drift': 0.0, 'loss_traj_path': 0.0, 'loss_traj_rot': 0.0, 'loss_traj_tdir': 0.0, 'loss_traj_tmag_step': 0.0, 'loss_traj_speed': 0.0, 'traj_triplet_n': 0.0, 'traj_path_ratio_mean': nan, 'traj_endpoint_err_mean': nan, 'loss_coupled_rot': 0.0, 'loss_coupled_tdir': 0.0, 'loss_coupled_joint': 0.0, 'loss_coupled_reg': 0.0, 'coupled_gate_mean': 0.0, 'delta_rot_norm_mean': 0.0, 'delta_tdir_norm_mean': 0.0, 'tdir_before_after_max_diff': 0.0, 'tmag_before_after_max_diff': 0.0}`
- `A_pair_only_baseline_fixed_harness` last_train=`{'loss_total': 1.712873935699463, 'loss_tmag': 2.9022672176361084, 'loss_tmag_speed': 0.0, 'loss_tmag_ratio': 0.0, 'loss_tmag_chain_sum': 0.0, 'loss_traj_ate': 0.0, 'loss_traj_drift': 0.0, 'loss_traj_path': 0.0, 'loss_traj_rot': 0.0, 'loss_traj_tdir': 0.0, 'loss_traj_tmag_step': 0.0, 'loss_traj_speed': 0.0, 'traj_triplet_n': 0.0, 'traj_path_ratio_mean': nan, 'traj_endpoint_err_mean': nan, 'loss_coupled_rot': 0.0, 'loss_coupled_tdir': 0.0, 'loss_coupled_joint': 0.0, 'loss_coupled_reg': 0.0, 'coupled_gate_mean': 0.0, 'delta_rot_norm_mean': 0.0, 'delta_tdir_norm_mean': 0.0, 'tdir_before_after_max_diff': 0.0, 'tmag_before_after_max_diff': 0.0}`
- `D_full_light_traj_fixed_harness` first_train=`{'loss_total': 0.8529090285301208, 'loss_tmag': 2.1404099464416504, 'loss_tmag_speed': 0.0, 'loss_tmag_ratio': 0.0, 'loss_tmag_chain_sum': 0.0, 'loss_traj_ate': 0.32990163564682007, 'loss_traj_drift': 0.28086599707603455, 'loss_traj_path': 0.8784886598587036, 'loss_traj_rot': 0.1684839427471161, 'loss_traj_tdir': 9.19857484404929e-05, 'loss_traj_tmag_step': 2.1753923892974854, 'loss_traj_speed': 2.1753926277160645, 'traj_triplet_n': 1.0, 'traj_path_ratio_mean': 0.25195908546447754, 'traj_endpoint_err_mean': 0.7530387043952942, 'loss_coupled_rot': 0.0, 'loss_coupled_tdir': 0.0, 'loss_coupled_joint': 0.0, 'loss_coupled_reg': 0.0, 'coupled_gate_mean': 0.0, 'delta_rot_norm_mean': 0.0, 'delta_tdir_norm_mean': 0.0, 'tdir_before_after_max_diff': 0.0, 'tmag_before_after_max_diff': 0.0}`
- `D_full_light_traj_fixed_harness` last_train=`{'loss_total': 1.9467381238937378, 'loss_tmag': 2.913541555404663, 'loss_tmag_speed': 0.0, 'loss_tmag_ratio': 0.0, 'loss_tmag_chain_sum': 0.0, 'loss_traj_ate': 0.5074971318244934, 'loss_traj_drift': 0.5140776634216309, 'loss_traj_path': 0.85500168800354, 'loss_traj_rot': 0.16287776827812195, 'loss_traj_tdir': 0.4722944498062134, 'loss_traj_tmag_step': 2.1405670642852783, 'loss_traj_speed': 2.1405673027038574, 'traj_triplet_n': 1.0, 'traj_path_ratio_mean': 0.2579468786716461, 'traj_endpoint_err_mean': 1.0192164182662964, 'loss_coupled_rot': 0.0, 'loss_coupled_tdir': 0.0, 'loss_coupled_joint': 0.0, 'loss_coupled_reg': 0.0, 'coupled_gate_mean': 0.0, 'delta_rot_norm_mean': 0.0, 'delta_tdir_norm_mean': 0.0, 'tdir_before_after_max_diff': 0.0, 'tmag_before_after_max_diff': 0.0}`

## Load / wrapper audit
- `A_pair_only_baseline_fixed_harness` missing/unexpected init=`14 / 0`, final=`0 / 0`, wrapper=`True`
- `D_full_light_traj_fixed_harness` missing/unexpected init=`14 / 0`, final=`0 / 0`, wrapper=`True`

## Pre-fix smoke comparison
- pre-fix D smoke: val_ate_proxy=`15.708141`, val_path_proxy=`2.515539`, odom_ATE=`19.286121`, drift=`24.674397`
- post-fix D retest: val_ate_proxy=`5.528377`, val_path_proxy=`1.742522`, odom_ATE=`21.995400`, drift=`35.549201`

## Decision
- final classification: `TRAINING-STILL-UNSTABLE`
- whether S15f lightweight CV is recommended: `False`
- S5 remains final clean candidate: `yes`
