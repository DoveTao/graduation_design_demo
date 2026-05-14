# S5E20 对比报告

## 执行摘要
S5E20 的关键不是再堆 head，而是确认 fake multiframe loss 已被真实 contiguous k-step composition supervision 替换，并且 delta_tdir 比 S5E19C 更保守。

## 方法对比
### S5E15
- status: S5E15_SCALE_DEUNDERFIT_IMPROVED
- coverage: 1.0
- signed_tdir_mean_deg: None
- anti_parallel_rate: None
- tmag_median_ratio: None
- path_ratio: None
- delta_tdir_norm_mean: None
- se3_ate: 3.9807674584476227
- sim3_ate: 3.968240516140872

### S5E19C
- status: S5E19C_REAL_TRAINING_NO_IMPROVEMENT
- coverage: 1.0
- signed_tdir_mean_deg: 56.17528018993624
- anti_parallel_rate: 0.16556291390728478
- tmag_median_ratio: 0.9879222977344607
- path_ratio: 0.12240164100086463
- delta_tdir_norm_mean: None
- se3_ate: 4.016260541940681
- sim3_ate: 3.9958080586973814

### S5E20
- status: S5E20_REAL_TRAINING_NO_IMPROVEMENT
- coverage: 1.0
- signed_tdir_mean_deg: 50.36741404926226
- anti_parallel_rate: 0.15011037527593818
- tmag_median_ratio: 1.1480486240737986
- path_ratio: 0.142260065005239
- delta_tdir_norm_mean: 0.004182390277601052
- se3_ate: 3.9887164161733195
- sim3_ate: 3.968171243900437

### ORB-SLAM3
- status: external_baseline
- coverage: 0.6013215859030837
- signed_tdir_mean_deg: None
- anti_parallel_rate: None
- tmag_median_ratio: None
- path_ratio: 0.2998258665660257
- delta_tdir_norm_mean: None
- se3_ate: 0.30854441069248173
- sim3_ate: 0.224292165986624

## 结论说明
本对比的重点不是再证明 S5E19C 的 no-op 问题，而是确认 S5E20 是否真的把 fake multiframe loss 换成了真实 contiguous k-step supervision，并且是否在不重新放大 delta_tdir 的情况下超过 S5E15。
当前结果显示：S5E20 成功修复了 S5E19C 的 aggressive delta，但没有形成超过 S5E15 的实质几何收益；相对 ORB-SLAM3 仍保留 full coverage 优势，但方向与路径长度质量差距仍然明显。
