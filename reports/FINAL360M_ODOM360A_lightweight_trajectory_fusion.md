# ODOM360A lightweight trajectory fusion

## 1. Executive summary
- fusion executed: `true`
- training executed: `false`
- pair model modified: `false`
- selected fusion method: `local_window_pose_fusion`
- selected params: `{'use_scale_pre_smoothing': 'ema', 'scale_ema_alpha': 0.4, 'use_rotation_pre_smoothing': 'ema', 'rotation_strength': 0.2, 'use_tdir_pre_suppression': True, 'tdir_window': 3, 'tdir_threshold_deg': 120.0, 'tdir_replacement_strength': 0.25, 'k_max': 2, 'k_decay': 0.7}`
- classification: `partial`
- final recommendation: `proceed_to_ODOM360B_local_pose_graph_with_kstep_constraints`

## 2. Motivation
- FINAL360I pair-level behavior is stable but direct trajectory composition still drifts.
- SEQ360B improves scale/path and SE3 ATE, but Sim3 shape barely changes.
- FINAL360J/K/L show that pair-level small fixes are not the right next lever.

## 3. Data sources
- checkpoint: `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded/best_full_val.pt`
- manifests: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl`, `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl`
- pair prediction cache: `{'cache_dir': '/home/dovetao/graduation_design_demo/external_baselines/results/final360m_odom360a_lightweight_trajectory_fusion/cache', 'reused': {}, 'generated': {'val': '/home/dovetao/graduation_design_demo/external_baselines/results/final360m_odom360a_lightweight_trajectory_fusion/cache/val_all_pair_predictions.jsonl', 'test': '/home/dovetao/graduation_design_demo/external_baselines/results/final360m_odom360a_lightweight_trajectory_fusion/cache/test_all_pair_predictions.jsonl'}}`
- reports used: `reports/TRAIN360E_sequence_trajectory_export_and_ATE_eval.md`, `reports/SEQ360B_train_lightweight_scale_smoothing_head.md`, `reports/RESULTS360_main_results_table.md`, `reports/RESULTS360_experiment_narrative.md`

## 4. Baseline reproduction
- direct val ATE none / SE3 / Sim3: `312.70197966233707` / `29.308235888381514` / `20.618804473381058`
- direct test ATE none / SE3 / Sim3: `251.21451329406275` / `119.93412838535708` / `29.423058946191894`
- direct test path ratio: `1.8675113295746928`
- TRAIN360E reference test ATE none / SE3 / Sim3: `222.56856382785097` / `118.6037794846689` / `27.564661865900444`

## 5. Fusion methods
- scale-only smoothing: adjacent log-tmag EMA or median smoothing.
- rotation smoothing: SO(3) EMA/window smoothing on adjacent relative rotations.
- tdir outlier suppression: local angular-threshold replacement toward neighborhood mean direction.
- local window pose fusion: adjacent-smoothed rotations plus k-step translation constraints solved as a lightweight anchored least-squares position graph.

## 6. Val selection
- score = ATE_SE3 + 2 * ATE_Sim3 + 30 * |log(path_ratio)|
- test not used for selection.
- selected method on val: `local_window_pose_fusion` with `{'use_scale_pre_smoothing': 'ema', 'scale_ema_alpha': 0.4, 'use_rotation_pre_smoothing': 'ema', 'rotation_strength': 0.2, 'use_tdir_pre_suppression': True, 'tdir_window': 3, 'tdir_threshold_deg': 120.0, 'tdir_replacement_strength': 0.25, 'k_max': 2, 'k_decay': 0.7}`

## 7. Test results
- TRAIN360E direct: none=`222.56856382785097`, se3=`118.6037794846689`, sim3=`27.564661865900444`, path_ratio=`1.756343083453392`
- SEQ360B: none=`139.22717463963536`, se3=`75.94691348103409`, sim3=`27.567211313835486`, path_ratio=`1.3502369615185652`
- ODOM360A selected: none=`132.0750603964614`, se3=`53.83113131619235`, sim3=`29.394188009602544`, path_ratio=`1.186821200940653`

## 8. Per-sequence results
- best improved sequences: `[{'sequence': 'snowmobile', 'selected_ate_se3': 57.47870265102845, 'baseline_ate_se3': 141.28089698735872, 'delta_ate_se3': -83.80219433633027, 'selected_ate_sim3': 14.531690580688919, 'baseline_ate_sim3': 14.636688842845073, 'delta_ate_sim3': -0.10499826215615471, 'selected_path_ratio': 1.7177165710312197, 'baseline_path_ratio': 2.6991421431636073}, {'sequence': 'ridge_to_lake', 'selected_ate_se3': 48.671832041654255, 'baseline_ate_se3': 84.38677126608508, 'delta_ate_se3': -35.71493922443082, 'selected_ate_sim3': 41.40503348875037, 'baseline_ate_sim3': 41.40407673511702, 'delta_ate_sim3': 0.0009567536333463522, 'selected_path_ratio': 0.7751965827697378, 'baseline_path_ratio': 1.2227143782577}]`
- worst degraded sequences: `[]`

## 9. Shape vs scale interpretation
- Sim3 shape improved: `False`
- path ratio moved closer to 1.0 than SEQ360B: `True`
- If SE3/path improve without Sim3 improvement, the gain should be interpreted as mostly scale/path correction rather than true shape repair.

## 10. Limitations
- post-processing only
- no image-level geometric verification
- no mature VO claim
- no real-time guarantee measured here
- no loop closure

## 11. Next recommendation
- `proceed_to_ODOM360B_local_pose_graph_with_kstep_constraints`

## 12. Compliance checklist
- `training_executed = false`
- `pair_model_modified = false`
- `test_used_for_selection = false`
- `explicit_matching_used = false`
- `image_ransac_used = false`
- `pnp_used = false`
- `ba_used = false`
- `hkust_teacher_used = false`
- `orbslam_teacher_used = false`
- `checkpoints_committed = false`
- `raw_data_committed = false`
- `large_prediction_dump_committed = false`
- `metrics_modified_existing = false`
