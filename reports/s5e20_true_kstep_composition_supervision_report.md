# S5E20 true k-step composition supervision report

## 执行摘要
S5E20 的目标是用真实 contiguous k-step composition supervision 替换 S5E19C 的 fake multiframe loss，并把 delta_tdir 控制在更保守的范围内。最终分类：`S5E20_REAL_TRAINING_NO_IMPROVEMENT`。

## S5E19D failure recap
S5E19D 已确认：S5E19C 的关键问题不是 no-op，而是 delta_tdir 过激、sign head 无效，以及 multiframe supervision 其实是随机 batch proxy。

## 为什么要替换 fake multiframe loss
S5E20 只使用真实连续窗口 i->i+k，不再允许 random batch proxy 伪装成 multiframe composition。

## true k-step contiguous window 构建
{'k_values': [1, 2, 3, 5], 'num_windows_by_k': {'1': 786, '2': 784, '3': 782, '5': 778}, 'all_windows_contiguous': True, 'uses_eval_scene': False}

## composition supervision 公式
T_chain = T_i,i+1 ∘ ... ∘ T_i+k-1,i+k，并对 direct GT relative pose 做 direction/path consistency 监督。

## conservative delta_tdir 设计
{'delta_tdir_norm_mean': 0.004182390277601052, 'delta_tdir_norm_p90': 0.005322978125939969, 'delta_tdir_norm_max': 0.005484306695765912, 'delta_tdir_too_aggressive': False}

## training status
{'real_training_executed': True, 'optimizer_step_count': 120, 'learned_weights_saved': True, 'smoke_policy_only': False, 'uses_true_contiguous_kstep_windows': True, 'uses_random_batch_proxy': False, 'uses_eval_gt_for_training': False, 'uses_orbslam3_teacher': False}

## traceable dense export
{'available': True, 'trajectory_path': 'external_baselines/results/s5e20_traceable_dense/scene01_seq03_s5e20_traceable_dense_tum.txt', 'edge_provenance': 'external_baselines/results/s5e20_traceable_dense/edge_provenance.jsonl', 'num_poses': 454, 'num_edges': 453, 'coverage': 1.0, 'all_edges_traceable': True}

## delta_tdir norm audit
{'delta_tdir_norm_mean': 0.004182390277601052, 'delta_tdir_norm_p90': 0.005322978125939969, 'delta_tdir_norm_max': 0.005484306695765912, 'delta_tdir_too_aggressive': False}

## component metrics
{'rot_mean_deg': 0.9134395040767528, 'rot_median_deg': 0.797895569319923, 'rot_p90_deg': 1.3218302317546538, 'signed_tdir_mean_deg': 50.36741404926226, 'signed_tdir_median_deg': 41.274023059345325, 'signed_tdir_p90_deg': 101.70150065318339, 'tdir_abs_mean_deg': 43.181392090891364, 'tdir_abs_median_deg': 39.927776639129625, 'tdir_abs_p90_deg': 78.66839390935559, 'tdir_mean_cosine': 0.5589339115543146, 'anti_parallel_rate': 0.15011037527593818, 'severe_wrong_sign_rate': 0.05518763796909492, 'direction_abs_good_but_signed_bad_rate': 0.024282560706401765, 'tmag_median_ratio': 1.1480486240737986, 'tmag_mean_ratio': 1.5281745108536944, 'tmag_p90_ratio': 3.4861365861929112, 'tmag_p95_ratio': 4.2256227408548, 'tmag_p99_ratio': 6.132733934572176, 'tmag_max_ratio': 17.93408364941292, 'path_ratio': 0.142260065005239}

## k-step metrics
{'k1_tdir': 50.36741416480564, 'k2_tdir': 48.07009344266454, 'k3_tdir': 47.7738353646015, 'k5_tdir': 47.292594034948706, 'composition_error': 47.713059694683146, 'path_length_consistency_error': 0.14473679496192643}

## external evaluator
{'none': {'ate': 9.370056222149003, 'drift': 0.12863110274642053, 'path_ratio': 0.14226006494698443}, 'se3': {'ate': 3.9887164161733195, 'drift': 0.12982531428143024, 'path_ratio': 0.14226006494698443}, 'sim3': {'ate': 3.968171243900437, 'drift': 0.13012736836662467, 'path_ratio': 0.14226006494698443}}

## 与 S5E15 / S5E19C / ORB-SLAM3 对比
{'tdir_improved': False, 'anti_parallel_reduced': False, 'path_ratio_improved': False, 'tmag_median_in_range': True, 'sim3_ate_improved_or_not_worse': True, 'delta_tdir_not_aggressive': True}

## 是否提升为 best candidate
{'keep_s5e15_as_best_candidate': True, 'promote_s5e20_as_best_candidate': False, 'continue_direction_training': False, 'recommended_next_stage': 'stop_direction_head_training_and_freeze_S5E15'}

## caveats
- S5E20 是 experimental candidate。
- 不替代 official S5 locked result。
- 没有使用 eval GT calibration。
- 没有使用 ORB-SLAM3 trajectory label。
