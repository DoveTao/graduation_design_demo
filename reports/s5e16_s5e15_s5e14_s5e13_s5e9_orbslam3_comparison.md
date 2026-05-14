# S5E16 comparison

## 执行摘要
S5E16 通过 confidence-calibrated route mixing 尝试在不爆炸的前提下改善 scale/path 与 anti_parallel。

## 比较表
{'name': 'S5 official locked result', 'coverage': None, 'rot_mean_deg': None, 'signed_tdir_mean_deg': None, 'anti_parallel_rate': None, 'tmag_median_ratio': None, 'tmag_p95_ratio': None, 'none_ate': None, 'se3_ate': 7.352288, 'sim3_ate': 7.352288, 'path_ratio': 0.932379, 'caveat': 'official_locked'}
{'name': 'S5E9 stable scale/path baseline', 'coverage': 1.0, 'rot_mean_deg': 0.9134395040767528, 'signed_tdir_mean_deg': 53.65013421168108, 'anti_parallel_rate': 0.10596026490066225, 'tmag_median_ratio': 0.39908888763230593, 'tmag_p95_ratio': 1.6560804642823275, 'none_ate': 9.659702088912063, 'se3_ate': 4.063931954977213, 'sim3_ate': 3.928931860681025, 'path_ratio': 0.05272511562778335, 'caveat': 'S5E9_SCALE_RAW_IMPROVED'}
{'name': 'S5E13', 'coverage': 1.0, 'rot_mean_deg': 0.9134395040767528, 'signed_tdir_mean_deg': 50.35304145887563, 'anti_parallel_rate': 0.1479028697571744, 'tmag_median_ratio': 0.399088892744088, 'tmag_p95_ratio': 1.6560806175064953, 'none_ate': 9.715160677827399, 'se3_ate': 4.07075013110235, 'sim3_ate': 3.9479012852111164, 'path_ratio': 0.05272511597296368, 'caveat': 'S5E13_OBSERVABLE_TDIR_IMPROVED_OVERALL_STILL_BAD'}
{'name': 'S5E14', 'coverage': 1.0, 'rot_mean_deg': 0.9134395040767528, 'signed_tdir_mean_deg': 50.353041365033235, 'anti_parallel_rate': 0.1479028697571744, 'tmag_median_ratio': 0.399088892744088, 'tmag_p95_ratio': 1.6560806175064955, 'none_ate': 9.71516068025357, 'se3_ate': 4.070750130887251, 'sim3_ate': 3.947901282749505, 'path_ratio': 0.052725115603500676, 'caveat': 'S5E14_OBSERVABLE_IMPROVED_OVERALL_STILL_BAD'}
{'name': 'S5E15', 'coverage': 1.0, 'rot_mean_deg': 0.9134395040767528, 'signed_tdir_mean_deg': 50.353041365033235, 'anti_parallel_rate': 0.1479028697571744, 'tmag_median_ratio': 1.268517401538299, 'tmag_p95_ratio': 4.66860108610276, 'none_ate': 9.328085906618083, 'se3_ate': 3.9807674584476227, 'sim3_ate': 3.968240516140872, 'path_ratio': 0.1571668166639846, 'caveat': 'S5E15_SCALE_DEUNDERFIT_IMPROVED'}
{'name': 'S5E16', 'coverage': 1.0, 'rot_mean_deg': 0.9134395040767528, 'signed_tdir_mean_deg': 50.353041365033235, 'anti_parallel_rate': 0.1479028697571744, 'tmag_median_ratio': 0.9966922440658065, 'tmag_p95_ratio': 3.6681865676521705, 'none_ate': 9.433622952683827, 'se3_ate': 4.001601345385457, 'sim3_ate': 3.968240516152594, 'path_ratio': 0.12348821309313078, 'caveat': 'S5E16_ROUTER_NO_IMPROVEMENT'}
{'name': 'ORB-SLAM3 external baseline', 'coverage': 0.6013215859030837, 'rot_mean_deg': None, 'signed_tdir_mean_deg': None, 'anti_parallel_rate': None, 'tmag_median_ratio': None, 'tmag_p95_ratio': None, 'none_ate': 11.46685113827757, 'se3_ate': 0.30854441069248173, 'sim3_ate': 0.224292165986624, 'path_ratio': 0.2998258665660257, 'caveat': 'external_baseline'}

## 解释
- S5E16 关注 confidence router 是否降低 anti_parallel 并维持 scale/path 改善。
- 不能将 S5E16 视作 official S5 替代。
