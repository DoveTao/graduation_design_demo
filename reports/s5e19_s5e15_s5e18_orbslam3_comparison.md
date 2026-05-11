# S5E19 comparison report

## 执行摘要
S5E19 回到 spherical token + coarse-to-fine + multiframe geometry 主线，但是否真正优于 S5E15，要看 signed tdir、anti_parallel 和 path_ratio 是否一起受益。

## 比较表
{'method': 'S5 official locked reference', 'status': 'official_locked', 'coverage': None, 'rot_mean_deg': None, 'signed_tdir_mean_deg': None, 'tdir_abs_mean_deg': None, 'anti_parallel_rate': None, 'tmag_median_ratio': None, 'tmag_p95_ratio': None, 'path_ratio': 0.932379, 'none_ate': None, 'se3_ate': 7.352288, 'sim3_ate': 7.352288, 'caveat': 'official_locked'}
{'method': 'S5E15', 'status': 'S5E15_SCALE_DEUNDERFIT_IMPROVED', 'coverage': 1.0, 'rot_mean_deg': 0.9134395040767528, 'signed_tdir_mean_deg': 50.353041365033235, 'tdir_abs_mean_deg': 43.16064629037026, 'anti_parallel_rate': 0.1479028697571744, 'tmag_median_ratio': 1.268517401538299, 'tmag_p95_ratio': 4.66860108610276, 'path_ratio': 0.1571668166639846, 'none_ate': 9.328085906618083, 'se3_ate': 3.9807674584476227, 'sim3_ate': 3.968240516140872, 'caveat': 'S5E15_SCALE_DEUNDERFIT_IMPROVED'}
{'method': 'S5E18', 'status': 'S5E18_SPHERICAL_SIGNAL_INSUFFICIENT', 'coverage': 1.0, 'rot_mean_deg': 0.9134395040767528, 'signed_tdir_mean_deg': 50.353041365033235, 'tdir_abs_mean_deg': 43.16064629037026, 'anti_parallel_rate': 0.1479028697571744, 'tmag_median_ratio': 1.0329334322687342, 'tmag_p95_ratio': None, 'path_ratio': 0.1287350014311331, 'none_ate': 9.409128115948798, 'se3_ate': 4.001167934325995, 'sim3_ate': 3.9726845099015224, 'caveat': 'S5E18_SPHERICAL_SIGNAL_INSUFFICIENT'}
{'method': 'S5E19', 'status': 'S5E19_TRAINING_SMOKE_ONLY', 'coverage': 1.0, 'rot_mean_deg': 0.9134395040767528, 'signed_tdir_mean_deg': 50.353041365033235, 'tdir_abs_mean_deg': 43.16064629037026, 'anti_parallel_rate': 0.1479028697571744, 'tmag_median_ratio': 1.26853220360632, 'tmag_p95_ratio': 4.668572667067984, 'path_ratio': 0.15674640990008198, 'none_ate': 9.328785504501196, 'se3_ate': 3.9809703358574935, 'sim3_ate': 3.9682105001578156, 'caveat': 'S5E19_TRAINING_SMOKE_ONLY'}
{'method': 'ORB-SLAM3', 'status': 'external_baseline', 'coverage': 0.6013215859030837, 'rot_mean_deg': None, 'signed_tdir_mean_deg': None, 'tdir_abs_mean_deg': None, 'anti_parallel_rate': None, 'tmag_median_ratio': None, 'tmag_p95_ratio': None, 'path_ratio': 0.2998258665660257, 'none_ate': 11.46685113827757, 'se3_ate': 0.30854441069248173, 'sim3_ate': 0.224292165986624, 'caveat': 'external_baseline'}

## 解释
- S5E19 是否保留主线，要看 spherical token / coarse-to-fine / geometry constraint 是否都还在主模型内部。
- 如果没有优于 S5E15，主要 blocker 需要区分训练不足、结构接入失败还是 tdir 信号不足。
- 即便 full coverage 仍然领先 ORB-SLAM3，也不能把它说成 official result replacement。
