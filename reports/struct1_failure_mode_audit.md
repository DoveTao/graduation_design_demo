# STRUCT1 失败模式审计

## 1. 执行摘要
本次审计结论为 `STRUCT1A_DIRECTION_AND_SCALE_BOTH_FAILED`。STRUCT1 的 geometry-token pose solver 确实真实接入，但 scale/path 与 direction 同时失败，S5E15 仍应保留为 best candidate。

## 2. STRUCT1 为什么失败
失败不是因为没有接入 geometry token，而是接入后没有形成可用的方向与尺度约束。训练状态显示 `real_training_executed=true`、`W_ab_used_in_pose_solver=true`、`tdir_from_geometry_tokens=true`、`fine_stage_used=true`，因此这是一次真实的负结果。

## 3. scale/path 爆炸分析
- tmag 中位数比例=23.3454，p95=188.6088，max=12453.0508。
- path_ratio=12.9702，属于明显全局爆炸，`scale_explosion_global=True`。
- 超过 10 倍 GT 的边比例=0.6689，超过 20 倍 GT 的边比例=0.5497。
- 前 10 个最长预测步贡献了总路径的 0.6368，说明不是纯 outlier，但也存在重尾放大。
- bounded_train_prior 名义上存在，但 `bounded_prior_effective=False`，核心原因是它只做 `base_log_tmag + bounded_delta`，没有更强的上界约束。

## 4. tdir/anti_parallel 退化分析
- signed_tdir=76.6451，anti_parallel_rate=0.3377。
- geometry_token_tdir_failed=True。
- 高置信度边 tdir 均值=77.1950，低置信度边 tdir 均值=76.0927。
- 高置信度 anti_parallel_rate=0.3656，低置信度 anti_parallel_rate=0.3097。

## 5. geometry token confidence / entropy 与错误关系
- confidence vs tmag Spearman=-0.26713402857149504.
- confidence vs tdir Spearman=0.022756558294849797.
- entropy vs tmag Spearman=0.26713402857149504.
- entropy vs tdir Spearman=-0.022756558294849797.
这些相关性整体偏弱，说明现有 confidence / entropy 不能可靠地区分方向好坏；对 tmag 仅有弱相关，无法作为有效 scale guard。

## 6. export / scale guard 是否可疑
- export_used_guarded_tmag=True，说明导出没有明显绕过 guard。
- path_loss_entered_total=True，权重=0.2.
- final_tmag 使用 `exp(log_tmag)`，并且缺少 upper clamp：`final_tmag_has_upper_clamp=False`。
- 训练中还冻结了 geometry backbone / layer：`froze_geometry_backbone_during_training=True`，这会进一步削弱 geometry token 主干对方向和尺度的联合学习能力。

## 7. 是否值得 STRUCT2
`continue_to_struct2=False`。如果没有全新的尺度先验设计与更强的导出上界保护，不建议继续沿同一路线推进 STRUCT2。

## 8. 是否保留 S5E15
应保留。S5E15 仍然在 signed_tdir、anti_parallel、tmag_median_ratio、path_ratio、sim3 ATE 上明显优于 STRUCT1。

## 9. Git hygiene 处理
本次审计新增只读审计脚本、报告、checkpoint 与静态测试；不训练新模型，不刷新官方指标，不修改 S5 locked metrics/policy。
