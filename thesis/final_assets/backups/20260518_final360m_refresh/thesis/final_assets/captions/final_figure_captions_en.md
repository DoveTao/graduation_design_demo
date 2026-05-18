# Final Figure Captions (EN)

## thesis/final_assets/figures/main_results/pair_level_main_results_bar.png
- Title: Pair-level main result bar chart
- Caption: This figure is prepared from `reports/RESULTS360_main_results_table.json, reports/TRAIN360E_metrics_test.json, reports/RESULTS360_main_results_table.json, reports/TRAIN360E_metrics_test.json, reports/FINAL360I_metrics_test.json, reports/BASE360D_metrics_test.json, reports/RESULTS360_main_results_table.json` and visualizes Pair-level main result bar chart. T57b rot_mean bar is omitted because the cleaned source set does not retain that value.

## thesis/final_assets/figures/ablations/structural_ablation_multimetric.png
- Title: Structural ablation multi-metric chart
- Caption: This figure is prepared from `reports/ABLDVO2_metrics_test.json, reports/ABLDVO3_metrics_test.json` and visualizes Structural ablation multi-metric chart. Covers rotation, translation direction, anti-parallel rate, scale ratio, and path ratio together.

## thesis/final_assets/figures/trajectories/trajectory_metric_comparison.png
- Title: Trajectory ATE and path-ratio comparison
- Caption: This figure is prepared from `reports/TRAIN360E_metrics_test.json, reports/SEQ360B_trajectory_metrics_test.json, external_baselines/results/odom360a_lightweight_trajectory_fusion/test, external_baselines/results/odom360b_local_pose_graph_kstep/test, reports/BASE360D_metrics_test.json` and visualizes Trajectory ATE and path-ratio comparison. ODOM360A/B values are computed from existing TUM exports only.

## thesis/final_assets/figures/trajectories/trajectory_overlay_ridge_to_lake.png
- Title: Representative trajectory overlay
- Caption: This figure is prepared from `external_baselines/results/odom360a_lightweight_trajectory_fusion, external_baselines/results/odom360b_local_pose_graph_kstep, external_baselines/results/base360_hkust_360dvo_official` and visualizes Representative trajectory overlay. Uses `ridge_to_lake` and only retained TUM exports that still exist in the cleaned worktree; direct TRAIN360E/SEQ360B TUM files are unavailable, so the overlay emphasizes export-available sequence backends plus a direct-composition reference.

## thesis/final_assets/figures/training_curves/final360i_validation_curves.png
- Title: FINAL360I validation curves
- Caption: This figure is prepared from `checkpoints/FINAL360I_struct360b_final/seed0/train_log.jsonl` and visualizes FINAL360I validation curves. Validation curves extracted from retained train_log.jsonl.

## thesis/final_assets/figures/training_curves/no_cross_interaction_validation_curves.png
- Title: NoCrossInteraction validation curves
- Caption: This figure is prepared from `checkpoints/FINAL360I_struct360b_final/seed0/train_log.jsonl` and visualizes NoCrossInteraction validation curves. Representative ablation curve from retained minival logs.

## thesis/final_assets/figures/training_curves/single_stage_validation_curves.png
- Title: SingleStage validation curves
- Caption: This figure is prepared from `checkpoints/FINAL360I_struct360b_final/seed0/train_log.jsonl` and visualizes SingleStage validation curves. Representative ablation curve from retained minival logs.

## thesis/final_assets/figures/training_curves/final360i_joint_vs_single_best.png
- Title: Joint-score vs single-metric best
- Caption: This figure is prepared from `checkpoints/FINAL360I_struct360b_final/seed0/train_log.jsonl` and visualizes Joint-score vs single-metric best. Compares checkpoint selection by joint score against minimum signed_tdir epoch.

## thesis/final_assets/figures/diagnostics/data360a_tmag_bucket_signed_tdir.png
- Title: DATA360A tmag-bucket signed_tdir plot
- Caption: This figure is prepared from `external_baselines/results/data360a_tdir_regime_diagnostic` and visualizes DATA360A tmag-bucket signed_tdir plot. Derived from retained final360i test pair rows.

## thesis/final_assets/figures/diagnostics/data360a_rotation_bucket_signed_tdir.png
- Title: DATA360A rotation-bucket signed_tdir plot
- Caption: This figure is prepared from `external_baselines/results/data360a_tdir_regime_diagnostic` and visualizes DATA360A rotation-bucket signed_tdir plot. Derived from retained final360i test pair rows.

## thesis/final_assets/figures/diagnostics/anti_parallel_regime_distribution.png
- Title: Anti-parallel regime distribution
- Caption: This figure is prepared from `external_baselines/results/data360a_tdir_regime_diagnostic` and visualizes Anti-parallel regime distribution. Shows anti-parallel remains a broad failure regime rather than a single-bucket issue.

## thesis/final_assets/figures/diagnostics/final360i_j_k_l_validation_compare.png
- Title: FINAL360I / J / K / L comparison
- Caption: This figure is prepared from `external_baselines/results/data360a_tdir_regime_diagnostic` and visualizes FINAL360I / J / K / L comparison. Validation-only comparison because standalone J/K/L test json files are unavailable.

## thesis/final_assets/figures/diagnostics/final360i_vs_seq360b_bucket_compare.png
- Title: FINAL360I vs SEQ360B bucket comparison
- Caption: This figure is prepared from `external_baselines/results/data360a_tdir_regime_diagnostic` and visualizes FINAL360I vs SEQ360B bucket comparison. Shows SEQ360B mainly changes scale/path behavior rather than direction error.

## thesis/final_assets/figures/matching_geometry/thesis360_cross_image_interaction_sample01_normal_mountains.png
- Title: Cross-image interaction visualization
- Caption: The heatmap shows token-level relation responses between two panoramic frames. Brighter values indicate stronger cross-image interactions. The figure illustrates that the model estimates relative pose through inter-frame feature relations rather than simple global feature concatenation. This is a cross-image interaction view, not explicit keypoint matching.

## thesis/final_assets/figures/matching_geometry/thesis360_residual_gate_diagnostic.png
- Title: Residual-gate diagnostic
- Caption: The figure shows fine-gate and residual-gate responses together with the magnitude of the applied residual pose updates. The results suggest that the fine residual branch performs conservative refinements while preserving the stability of the coarse pose estimate.

## thesis/final_assets/figures/matching_geometry/thesis360_cross_image_interaction_sample02_normal_downhill.png
- Title: Cross-image interaction visualization
- Caption: This figure shows a second validation sample and visualizes the token-level cross-image relation heatmap. It indicates that the model consistently performs inter-frame relation modeling across scenes. This is a cross-image interaction view rather than explicit keypoint correspondence.

## thesis/final_assets/figures/matching_geometry/thesis360_cross_image_interaction_sample03_scale_path_k5.png
- Title: Cross-image interaction visualization on a k-step sample
- Caption: This figure visualizes token-level cross-image relations on a larger temporal-gap sample, which is useful for inspecting scale/path-related behavior. The plot is a relation-level visualization, not an explicit matching figure.

## thesis/final_assets/figures/matching_geometry/thesis360_coarse_final_pose_diagnostic.png
- Title: Coarse-to-final pose diagnostic
- Caption: This figure compares coarse-pose and final-pose errors against GT on fixed eval-only samples, showing how the residual refinement branch changes rotation, translation direction, and scale. It illustrates that the model mainly performs controlled corrections on top of the coarse estimate.

## thesis/final_assets/figures/matching_geometry/thesis360_token_layout_visualization.png
- Title: Token layout visualization
- Caption: This figure shows the ERP-coordinate layout of the coarse and fine spherical tokens, illustrating the panoramic tokenization structure used by the model instead of a conventional keypoint detection-and-matching pipeline.
