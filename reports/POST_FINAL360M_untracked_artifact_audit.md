# POST FINAL360M untracked artifact audit

## Workspace
- current branch: `research/abldvo3-single-stage-pose-regression-ablation`
- latest commit: `85f12ed FINAL360M: promote full-train thesis mainline and refresh final assets`
- untracked file count: `91`
- untracked total size: `28856300` bytes (`27.52` MiB)

## Classification
- `commit_now` count: `26`
- rationale: small thesis plot-data csv/json files plus a tiny final-asset manifest are reproducibility-friendly and cheap to track
- `keep_local_only` count: `3`
- rationale: eval-only trajectory exports, pair prediction dumps, jsonl row dumps, and TUM files are bulky intermediate artifacts that can be regenerated
- `safe_to_delete` count: `5`
- rationale: cache directories only
- `risky_files` count: `1`

## commit_now
- `thesis/final_assets/manifests/matching_geometry_eval_only_manifest.json` (565 bytes)
- `thesis/final_assets/plot_data/anti_parallel_regime_distribution.csv` (11961 bytes)
- `thesis/final_assets/plot_data/data360a_rotation_bucket_signed_tdir.csv` (1763 bytes)
- `thesis/final_assets/plot_data/data360a_tmag_bucket_signed_tdir.csv` (1669 bytes)
- `thesis/final_assets/plot_data/final360i_j_k_l_validation_compare.csv` (2413 bytes)
- `thesis/final_assets/plot_data/final360i_joint_vs_single_best.csv` (1159 bytes)
- `thesis/final_assets/plot_data/final360i_validation_curves.csv` (2842 bytes)
- `thesis/final_assets/plot_data/final360i_vs_seq360b_bucket_compare.csv` (3376 bytes)
- `thesis/final_assets/plot_data/no_cross_interaction_validation_curves.csv` (2033 bytes)
- `thesis/final_assets/plot_data/pair_level_main_results_bar.csv` (5310 bytes)
- `thesis/final_assets/plot_data/single_stage_validation_curves.csv` (2096 bytes)
- `thesis/final_assets/plot_data/structural_ablation_multimetric.csv` (5245 bytes)
- `thesis/final_assets/plot_data/thesis360_coarse_final_pose_diagnostic.json` (6509 bytes)
- `thesis/final_assets/plot_data/thesis360_cross_image_interaction_sample01_normal_mountains.json` (6530 bytes)
- `thesis/final_assets/plot_data/thesis360_cross_image_interaction_sample02_normal_downhill.json` (6529 bytes)
- `thesis/final_assets/plot_data/thesis360_cross_image_interaction_sample03_scale_path_k5.json` (6527 bytes)
- `thesis/final_assets/plot_data/thesis360_residual_gate_diagnostic.json` (6505 bytes)
- `thesis/final_assets/plot_data/thesis360_token_layout_visualization.json` (6507 bytes)
- `thesis/final_assets/plot_data/trajectory_metric_comparison.csv` (4482 bytes)
- `thesis/final_assets/plot_data/trajectory_overlay_ridge_to_lake.json` (621 bytes)
- `thesis/final_assets/plot_data/trajectory_overlay_ridge_to_lake__BASE360D.csv` (89031 bytes)
- `thesis/final_assets/plot_data/trajectory_overlay_ridge_to_lake__GT.csv` (94622 bytes)
- `thesis/final_assets/plot_data/trajectory_overlay_ridge_to_lake__ODOM360A-direct.csv` (99398 bytes)
- `thesis/final_assets/plot_data/trajectory_overlay_ridge_to_lake__ODOM360A.csv` (98477 bytes)
- `thesis/final_assets/plot_data/trajectory_overlay_ridge_to_lake__ODOM360B.csv` (84153 bytes)
- `thesis/final_assets/plot_data/unavailable_plot_data.json` (40808 bytes)

## keep_local_only
- `external_baselines/results/final360m_odom360a_lightweight_trajectory_fusion/` (8868209 bytes)
- `external_baselines/results/odom360a_lightweight_trajectory_fusion/` (8873982 bytes)
- `external_baselines/results/data360a_tdir_regime_diagnostic/` (10521399 bytes)

## safe_to_delete
- `models/__pycache__/`
- `tools/__pycache__/`
- `tools/current/__pycache__/`
- `datasets/__pycache__/`
- `train360/core/__pycache__/`

## risky_files
- `reports/MAIN362_blocker_worktree_not_clean.md`: small blocker/debug artifact with unclear long-term value

## Findings
- checkpoint / raw data under the audited untracked set: `false / false`
- large dump untracked: `true`
- recommended commit for `thesis/final_assets/plot_data/`: `true`
- recommended commit for `thesis/final_assets/manifests/`: `true`
- recommended keep `external_baselines/results/final360m_odom360a_lightweight_trajectory_fusion/` local only: `true`
- recommended new semantic branch: `true`
- suggested branch name: `research/final360m-thesis-mainline-promotion`

## Health check
- `python tools/current/show_mainline.py`: `pass`
- `python tools/current/check_current_artifacts.py`: `pass`
- old metrics diff: `none`

## Recommended next task
- `POST_FINAL360M_commit_small_final_assets`
