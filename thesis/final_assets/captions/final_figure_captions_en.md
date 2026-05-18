# Final Figure Captions (EN)

## thesis/final_assets/figures/main_results/pair_level_main_results_bar.png
- Title: Pair-level direction error comparison for FINAL360M, old FINAL360I, and BASE360D
- Caption: FINAL360M replaces FINAL360I as the thesis main model, while the old FINAL360I result is retained only as a subset-trained candidate reference. This promotion is protocol-based rather than evidence of uniform metric superiority.

## thesis/final_assets/figures/trajectories/trajectory_metric_comparison.png
- Title: FINAL360M trajectory backend comparison
- Caption: Compares FINAL360M-direct, FINAL360M-ODOM360A, FINAL360M-ODOM360B, TRAIN360E, SEQ360B, and BASE360D on ATE and path ratio. FINAL360M-ODOM360A improves FINAL360M-direct, but it still does not surpass the old subset-model-based ODOM360A reference.

## thesis/final_assets/figures/trajectories/trajectory_overlay_ridge_to_lake.png
- Title: Representative FINAL360M trajectory overlay
- Caption: Uses ridge_to_lake to visualize the shape difference between FINAL360M direct/fusion/pose-graph trajectories, GT, and BASE360D. The figure highlights that backend fusion helps, but does not fully resolve trajectory-shape error.
