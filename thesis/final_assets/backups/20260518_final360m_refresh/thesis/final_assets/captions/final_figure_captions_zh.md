# Final Figure Captions (ZH)

## thesis/final_assets/figures/main_results/pair_level_main_results_bar.png
- 标题：pair-level 主结果条形图
- 图注：本图基于 `reports/RESULTS360_main_results_table.json, reports/TRAIN360E_metrics_test.json, reports/RESULTS360_main_results_table.json, reports/TRAIN360E_metrics_test.json, reports/FINAL360I_metrics_test.json, reports/BASE360D_metrics_test.json, reports/RESULTS360_main_results_table.json` 整理，用于展示 pair-level 主结果条形图。T57b rot_mean bar is omitted because the cleaned source set does not retain that value.

## thesis/final_assets/figures/ablations/structural_ablation_multimetric.png
- 标题：结构消融多指标图
- 图注：本图基于 `reports/ABLDVO2_metrics_test.json, reports/ABLDVO3_metrics_test.json` 整理，用于展示 结构消融多指标图。Covers rotation, translation direction, anti-parallel rate, scale ratio, and path ratio together.

## thesis/final_assets/figures/trajectories/trajectory_metric_comparison.png
- 标题：trajectory ATE 与 path_ratio 对比图
- 图注：本图基于 `reports/TRAIN360E_metrics_test.json, reports/SEQ360B_trajectory_metrics_test.json, external_baselines/results/odom360a_lightweight_trajectory_fusion/test, external_baselines/results/odom360b_local_pose_graph_kstep/test, reports/BASE360D_metrics_test.json` 整理，用于展示 trajectory ATE 与 path_ratio 对比图。ODOM360A/B values are computed from existing TUM exports only.

## thesis/final_assets/figures/trajectories/trajectory_overlay_ridge_to_lake.png
- 标题：代表性序列轨迹叠加图
- 图注：本图基于 `external_baselines/results/odom360a_lightweight_trajectory_fusion, external_baselines/results/odom360b_local_pose_graph_kstep, external_baselines/results/base360_hkust_360dvo_official` 整理，用于展示 代表性序列轨迹叠加图。Uses `ridge_to_lake` and only retained TUM exports that still exist in the cleaned worktree; direct TRAIN360E/SEQ360B TUM files are unavailable, so the overlay emphasizes export-available sequence backends plus a direct-composition reference.

## thesis/final_assets/figures/training_curves/final360i_validation_curves.png
- 标题：FINAL360I 验证曲线
- 图注：本图基于 `checkpoints/FINAL360I_struct360b_final/seed0/train_log.jsonl` 整理，用于展示 FINAL360I 验证曲线。Validation curves extracted from retained train_log.jsonl.

## thesis/final_assets/figures/training_curves/no_cross_interaction_validation_curves.png
- 标题：NoCrossInteraction 验证曲线
- 图注：本图基于 `checkpoints/FINAL360I_struct360b_final/seed0/train_log.jsonl` 整理，用于展示 NoCrossInteraction 验证曲线。Representative ablation curve from retained minival logs.

## thesis/final_assets/figures/training_curves/single_stage_validation_curves.png
- 标题：SingleStage 验证曲线
- 图注：本图基于 `checkpoints/FINAL360I_struct360b_final/seed0/train_log.jsonl` 整理，用于展示 SingleStage 验证曲线。Representative ablation curve from retained minival logs.

## thesis/final_assets/figures/training_curves/final360i_joint_vs_single_best.png
- 标题：joint score 与单指标 best 对比图
- 图注：本图基于 `checkpoints/FINAL360I_struct360b_final/seed0/train_log.jsonl` 整理，用于展示 joint score 与单指标 best 对比图。Compares checkpoint selection by joint score against minimum signed_tdir epoch.

## thesis/final_assets/figures/diagnostics/data360a_tmag_bucket_signed_tdir.png
- 标题：DATA360A tmag bucket signed_tdir 图
- 图注：本图基于 `external_baselines/results/data360a_tdir_regime_diagnostic` 整理，用于展示 DATA360A tmag bucket signed_tdir 图。Derived from retained final360i test pair rows.

## thesis/final_assets/figures/diagnostics/data360a_rotation_bucket_signed_tdir.png
- 标题：DATA360A rotation bucket signed_tdir 图
- 图注：本图基于 `external_baselines/results/data360a_tdir_regime_diagnostic` 整理，用于展示 DATA360A rotation bucket signed_tdir 图。Derived from retained final360i test pair rows.

## thesis/final_assets/figures/diagnostics/anti_parallel_regime_distribution.png
- 标题：anti_parallel regime 分布图
- 图注：本图基于 `external_baselines/results/data360a_tdir_regime_diagnostic` 整理，用于展示 anti_parallel regime 分布图。Shows anti-parallel remains a broad failure regime rather than a single-bucket issue.

## thesis/final_assets/figures/diagnostics/final360i_j_k_l_validation_compare.png
- 标题：FINAL360I / J / K / L 对比图
- 图注：本图基于 `external_baselines/results/data360a_tdir_regime_diagnostic` 整理，用于展示 FINAL360I / J / K / L 对比图。Validation-only comparison because standalone J/K/L test json files are unavailable.

## thesis/final_assets/figures/diagnostics/final360i_vs_seq360b_bucket_compare.png
- 标题：FINAL360I vs SEQ360B bucket 对比图
- 图注：本图基于 `external_baselines/results/data360a_tdir_regime_diagnostic` 整理，用于展示 FINAL360I vs SEQ360B bucket 对比图。Shows SEQ360B mainly changes scale/path behavior rather than direction error.

## thesis/final_assets/figures/matching_geometry/thesis360_cross_image_interaction_sample01_normal_mountains.png
- 标题：跨图像特征交互可视化
- 图注：图中展示了两帧全景图像之间的 token-level relation heatmap。颜色越亮表示跨图像 token 关系响应越强。该图说明模型在相对位姿预测前进行了真实的跨图像关系建模，而不是简单的全局特征拼接。这里展示的是 cross-image interaction，不是显式 keypoint matching。

## thesis/final_assets/figures/matching_geometry/thesis360_residual_gate_diagnostic.png
- 标题：粗到细残差精化中的门控响应分布
- 图注：该图展示 fine residual 分支中的 fine gate 与 residual gate 以及对应的残差修正幅度。结果表明模型倾向于在保持 coarse pose 稳定的前提下进行保守的小幅精化。

## thesis/final_assets/figures/matching_geometry/thesis360_cross_image_interaction_sample02_normal_downhill.png
- 标题：跨图像特征交互可视化
- 图注：该图展示另一组验证样本上的 token-level cross-image relation heatmap，用于说明跨图像关系建模在不同场景下稳定存在。这里展示的是跨图像特征交互，而不是显式 keypoint correspondence。

## thesis/final_assets/figures/matching_geometry/thesis360_cross_image_interaction_sample03_scale_path_k5.png
- 标题：跨图像特征交互可视化（k-step 样本）
- 图注：该图展示较大时间间隔样本上的跨图像 token relation heatmap，用于辅助观察 scale/path 相关样本中关系建模的变化。该图不是显式匹配图，而是 relation-level 可视化。

## thesis/final_assets/figures/matching_geometry/thesis360_coarse_final_pose_diagnostic.png
- 标题：coarse pose 与 final pose 诊断图
- 图注：该图比较 fixed eval-only 样本上 coarse pose 与 final pose 相对 GT 的误差，展示残差精化分支对旋转、平移方向和尺度的影响。该图用于说明模型主要是在 coarse 估计基础上做受控修正。

## thesis/final_assets/figures/matching_geometry/thesis360_token_layout_visualization.png
- 标题：球面 token 布局可视化
- 图注：该图展示 coarse 与 fine 球面 token 在 ERP 坐标中的布局，用于说明模型在全景图像上采用的球面 tokenization 结构，而不是传统关键点检测与匹配流程。
