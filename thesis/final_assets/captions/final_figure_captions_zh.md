# Final Figure Captions (ZH)

## thesis/final_assets/figures/main_results/pair_level_main_results_bar.png
- 标题：FINAL360M 与旧 FINAL360I / BASE360D 的 pair-level 方向误差对比
- 图注：`FINAL360M` 已替代 `FINAL360I` 成为论文主模型；旧 `FINAL360I` 仅保留为 subset-trained candidate 对比项。该主模型切换基于训练协议合规性，而不是“FINAL360M 全面优于 FINAL360I”。

## thesis/final_assets/figures/trajectories/trajectory_metric_comparison.png
- 标题：FINAL360M trajectory backend 与旧后端对比
- 图注：展示 `FINAL360M-direct`、`FINAL360M-ODOM360A`、`FINAL360M-ODOM360B` 与 `TRAIN360E` / `SEQ360B` / `BASE360D` 的 ATE 和 path_ratio 对比。`FINAL360M-ODOM360A` 改善了 `FINAL360M-direct`，但仍弱于旧 subset-model-based `ODOM360A`。

## thesis/final_assets/figures/trajectories/trajectory_overlay_ridge_to_lake.png
- 标题：FINAL360M 代表性序列轨迹叠加图
- 图注：代表性测试序列 `ridge_to_lake` 上，展示 `FINAL360M` 的 direct / ODOM360A / ODOM360B 与 GT、BASE360D 的轨迹形状差异。该图用于说明后端融合有效，但不足以完全解决轨迹形状误差。
