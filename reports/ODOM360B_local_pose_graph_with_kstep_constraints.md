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
- checkpoint: `checkpoints/FINAL360I_struct360b_final/seed0/best_val.pt`
- manifests: `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_val.jsonl`, `external_baselines/results/dset2c_360dvo_canonical/pair_manifest_test.jsonl`
- pair predictions: `{'source': 'odom360a_cache', 'val_cache': '/home/dovetao/graduation_design_demo/external_baselines/results/odom360a_lightweight_trajectory_fusion/cache/val_all_pair_predictions.jsonl', 'test_cache': '/home/dovetao/graduation_design_demo/external_baselines/results/odom360a_lightweight_trajectory_fusion/cache/test_all_pair_predictions.jsonl'}`
- k-step predictions available: `[1, 2, 3]`
- ODOM360A cache reused: `True`

## 4. Baseline reproduction
- TRAIN360E direct reproduced test SE3 / Sim3 / path: `118.60378021625287` / `27.56466329811743` / `1.7563430093095245`
- ODOM360A reproduced test SE3 / Sim3 / path: `52.1944638233902` / `27.580518809274864` / `1.1104915601875927`
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
- ODOM360A: none=`94.59871076755088`, se3=`52.1944638233902`, sim3=`27.580518809274864`, path_ratio=`1.1104915601875927`
- ODOM360B: none=`94.60021909820709`, se3=`52.19664744072544`, sim3=`27.581094116622864`, path_ratio=`1.1444784143411664`

## 8. Per-sequence results
- best improved sequences: `[{'sequence': 'snowmobile', 'selected_ate_se3': 48.54677556289495, 'odom360a_ate_se3': 48.54476908792696, 'delta_ate_se3': 0.002006474967991778, 'selected_ate_sim3': 10.024318170307346, 'odom360a_ate_sim3': 10.02301789973129, 'delta_ate_sim3': 0.001300270576056306, 'selected_path_ratio': 1.5281259271725396, 'odom360a_path_ratio': 1.4915080009990203}, {'sequence': 'ridge_to_lake', 'selected_ate_se3': 56.598881088314236, 'odom360a_ate_se3': 56.59648598747024, 'delta_ate_se3': 0.002395100843997966, 'selected_ate_sim3': 40.25779409992323, 'odom360a_ate_sim3': 40.257308258742036, 'delta_ate_sim3': 0.00048584118119521236, 'selected_path_ratio': 0.8470210058229958, 'odom360a_path_ratio': 0.8150741280500888}]`
- worst degraded sequences: `[{'sequence': 'snowmobile', 'selected_ate_se3': 48.54677556289495, 'odom360a_ate_se3': 48.54476908792696, 'delta_ate_se3': 0.002006474967991778, 'selected_ate_sim3': 10.024318170307346, 'odom360a_ate_sim3': 10.02301789973129, 'delta_ate_sim3': 0.001300270576056306, 'selected_path_ratio': 1.5281259271725396, 'odom360a_path_ratio': 1.4915080009990203}, {'sequence': 'ridge_to_lake', 'selected_ate_se3': 56.598881088314236, 'odom360a_ate_se3': 56.59648598747024, 'delta_ate_se3': 0.002395100843997966, 'selected_ate_sim3': 40.25779409992323, 'odom360a_ate_sim3': 40.257308258742036, 'delta_ate_sim3': 0.00048584118119521236, 'selected_path_ratio': 0.8470210058229958, 'odom360a_path_ratio': 0.8150741280500888}]`

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
