# ODOM360B local pose graph with k-step constraints

## 1. Executive summary
- pose graph executed: `true`
- training executed: `false`
- pair model modified: `false`
- selected method: `pg_k3`
- classification: `regression`
- final recommendation: `keep_ODOM360A_as_best_sequence_backend`

## 2. Motivation
- FINAL360I pair-level behavior is stable, but direct trajectory composition still drifts.
- SEQ360B improved scale/path and SE3 but not Sim3.
- ODOM360A improved path/SE3 strongly, but still did not improve Sim3 shape.
- ODOM360B upgrades the backend from local fusion to an explicit local pose graph with k-step constraints.

## 3. Data sources
- checkpoint: `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded/best_full_val.pt`
- manifests: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl`, `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl`
- pair predictions: `{'source': 'odom360a_cache', 'val_cache': '/home/dovetao/graduation_design_demo/external_baselines/results/final360m_odom360a_lightweight_trajectory_fusion/cache/val_all_pair_predictions.jsonl', 'test_cache': '/home/dovetao/graduation_design_demo/external_baselines/results/final360m_odom360a_lightweight_trajectory_fusion/cache/test_all_pair_predictions.jsonl'}`
- k-step predictions available: `[1, 2, 3]`
- ODOM360A cache reused: `True`

## 4. Baseline reproduction
- TRAIN360E direct reproduced test SE3 / Sim3 / path: `119.93412838535708` / `29.423058946191894` / `1.8675113295746928`
- ODOM360A reproduced test SE3 / Sim3 / path: `53.83113131619235` / `29.394188009602544` / `1.186821200940653`
- reproduction was accepted before ODOM360B search continued.

## 5. Pose graph design
- variables: absolute frame rotations and positions, first pose fixed as gauge anchor.
- constraints: adjacent and k-step relative rotation / translation direction / log-magnitude constraints.
- objective: weighted rotation + tdir + tmag residuals with optional smoothness and initialization priors.
- robust loss: Huber-style robustification for PG-k2-robust and later variants.
- k-step weights: geometric decay `k_decay^(k-1)`.
- initialization: direct composition or ODOM360A selected local-window fusion.

## 6. Val selection
- score = ATE_SE3 + 2 * ATE_Sim3 + 30 * |log(path_ratio)| + 5 * degradation_penalty
- test not used for selection.
- selected method on val: `pg_k3` with `{'method_name': 'pg_k3', 'init_method': 'odom360a_selected', 'k_max': 3, 'k_decay': 0.5, 'w_rot': 1.0, 'w_tdir': 1.0, 'w_tmag': 0.5, 'w_smooth_rot': 0.1, 'w_smooth_t': 0.05, 'w_path': 0.05, 'w_rot_prior': 0.2, 'w_pos_prior': 0.1, 'robust_delta': 0.5, 'use_robust': True, 'use_outlier_downweight': True, 'outlier_angle_deg': 120.0, 'outlier_weight': 0.35, 'use_scale_pre_smoothing': 'ema', 'scale_ema_alpha': 0.4, 'use_rotation_pre_smoothing': 'ema', 'rotation_strength': 0.2, 'use_tdir_pre_suppression': True, 'tdir_window': 3, 'tdir_threshold_deg': 120.0, 'tdir_replacement_strength': 0.25, 'iterations': 40, 'lr': 0.005, 'optimizer': 'adam'}`

## 7. Test results
- TRAIN360E: none=`222.56856382785097`, se3=`118.6037794846689`, sim3=`27.564661865900444`, path_ratio=`1.756343083453392`
- SEQ360B: none=`139.22717463963536`, se3=`75.94691348103409`, sim3=`27.567211313835486`, path_ratio=`1.3502369615185652`
- ODOM360A: none=`132.0750603964614`, se3=`53.83113131619235`, sim3=`29.394188009602544`, path_ratio=`1.186821200940653`
- ODOM360B: none=`132.07632134132956`, se3=`53.83390038341053`, sim3=`29.395213807223342`, path_ratio=`1.221642376834591`

## 8. Per-sequence results
- best improved sequences: `[{'sequence': 'ridge_to_lake', 'selected_ate_se3': 48.674197932798165, 'odom360a_ate_se3': 48.671832041654255, 'delta_ate_se3': 0.0023658911439099484, 'selected_ate_sim3': 41.4060862097912, 'odom360a_ate_sim3': 41.40503348875037, 'delta_ate_sim3': 0.0010527210408284304, 'selected_path_ratio': 0.8117583914907458, 'odom360a_path_ratio': 0.7751965827697378}, {'sequence': 'snowmobile', 'selected_ate_se3': 57.48174892113805, 'odom360a_ate_se3': 57.47870265102845, 'delta_ate_se3': 0.0030462701095999023, 'selected_ate_sim3': 14.533055680916359, 'odom360a_ate_sim3': 14.531690580688919, 'delta_ate_sim3': 0.0013651002274404078, 'selected_path_ratio': 1.750292755085201, 'odom360a_path_ratio': 1.7177165710312197}]`
- worst degraded sequences: `[{'sequence': 'ridge_to_lake', 'selected_ate_se3': 48.674197932798165, 'odom360a_ate_se3': 48.671832041654255, 'delta_ate_se3': 0.0023658911439099484, 'selected_ate_sim3': 41.4060862097912, 'odom360a_ate_sim3': 41.40503348875037, 'delta_ate_sim3': 0.0010527210408284304, 'selected_path_ratio': 0.8117583914907458, 'odom360a_path_ratio': 0.7751965827697378}, {'sequence': 'snowmobile', 'selected_ate_se3': 57.48174892113805, 'odom360a_ate_se3': 57.47870265102845, 'delta_ate_se3': 0.0030462701095999023, 'selected_ate_sim3': 14.533055680916359, 'odom360a_ate_sim3': 14.531690580688919, 'delta_ate_sim3': 0.0013651002274404078, 'selected_path_ratio': 1.750292755085201, 'odom360a_path_ratio': 1.7177165710312197}]`

## 9. Shape vs scale interpretation
- Sim3 improved vs ODOM360A: `False`
- path ratio improved vs ODOM360A: `False`
- If Sim3 stays flat while SE3 and path improve, the gain should still be read as mostly scale/path repair rather than true shape recovery.

## 10. k-step constraints interpretation
- k=2 was explicitly searched and selected if beneficial.
- k=3 was evaluated as an optional ablation and kept only if val remained stable.
- Robust loss and outlier downweight were evaluated separately from plain PG-k2.

## 11. Limitations
- post-processing only
- no image-level geometric verification
- no mature VO claim
- no online real-time guarantee unless measured
- no loop closure
- no BA / reprojection constraints

## 12. Next recommendation
- `keep_ODOM360A_as_best_sequence_backend`

## 13. Compliance checklist
- `training_executed = false`
- `pair_model_modified = false`
- `test_used_for_selection = false`
- `explicit_matching_used = false`
- `image_ransac_used = false`
- `pnp_used = false`
- `ba_used = false`
- `reprojection_error_used = false`
- `hkust_teacher_used = false`
- `orbslam_teacher_used = false`
- `checkpoints_committed = false`
- `raw_data_committed = false`
- `large_prediction_dump_committed = false`
- `metrics_modified_existing = false`
