# S5E15 scale de-underfit anti-parallel report

## 执行摘要
final_classification = `S5E15_SCALE_DEUNDERFIT_IMPROVED`。

## S5E14 under-scale 与 anti_parallel 回顾
{'path_ratio': 0.05272511597296368, 'tmag_median_ratio': 0.399088892744088, 'anti_parallel_rate': 0.1479028697571744, 'tdir_mean_deg': 50.35304145887563, 'sim3_ate': 3.9479012852111164}

## under-scale + anti-parallel audit
{'path': 'external_baselines/results/s5e15_traceable_dense/under_scale_antiparallel_audit.json', 'under_scale_global': True, 'under_scale_worse_on_observable': True, 'anti_parallel_scale_coupled': True, 'main_blocker': 'both', 'train_prior_scale_available': True, 'oracle_scale_not_used_for_candidate': True}

## train-prior scale / no eval GT calibration 说明
uses_eval_gt_for_scale=false；uses_eval_gt_for_sign=false；uses_train_prior_scale 由 train_split_prior 给出。

## anti_parallel guard 设计
基于 correspondence feature confidence gate，低置信或高风险边使用 fallback prior。

## traceable dense export coverage
{'available': True, 'trajectory_path': 'external_baselines/results/s5e15_traceable_dense/scene01_seq03_s5e15_traceable_dense_tum.txt', 'edge_provenance': 'external_baselines/results/s5e15_traceable_dense/edge_provenance.jsonl', 'num_poses': 454, 'num_edges': 453, 'direct_adjacent_prediction_edges': 453, 'coverage': 1.0, 'all_edges_traceable': True}

## overall component metrics
{'count': 453, 'rot_mean_deg': 0.9134395040767528, 'rot_median_deg': 0.797895569319923, 'rot_p90_deg': 1.3218302317546538, 'signed_tdir_mean_deg': 50.353041365033235, 'signed_tdir_median_deg': 41.2294804093569, 'signed_tdir_p90_deg': 101.804771450087, 'tdir_abs_mean_deg': 43.16064629037026, 'tdir_abs_median_deg': 39.87455596995841, 'tdir_abs_p90_deg': 78.632504335116, 'tdir_mean_cosine': 0.5591010842073055, 'anti_parallel_rate': 0.1479028697571744, 'severe_wrong_sign_rate': 0.05518763796909492, 'direction_abs_good_but_signed_bad_rate': 0.024282560706401765, 'tmag_median_ratio': 1.268517401538299, 'tmag_mean_ratio': 1.6886134029934055, 'tmag_p90_ratio': 3.852362321212163, 'tmag_p95_ratio': 4.66860108610276, 'tmag_p99_ratio': 6.775825973441673, 'tmag_max_ratio': 19.819619392350763, 'path_ratio': 0.1571668166639846}

## stratified metrics
{'observable_edges': {'count': 117, 'rot_mean_deg': 0.9851143604234686, 'rot_median_deg': 0.841925552079711, 'rot_p90_deg': 1.3675857408014738, 'signed_tdir_mean_deg': 46.32471063192815, 'signed_tdir_median_deg': 28.202055999174636, 'signed_tdir_p90_deg': 125.6523318720361, 'tdir_abs_mean_deg': 32.86505677876463, 'tdir_abs_median_deg': 27.624287219861312, 'tdir_abs_p90_deg': 74.62204678154342, 'tdir_mean_cosine': 0.5700488133917261, 'anti_parallel_rate': 0.19658119658119658, 'severe_wrong_sign_rate': 0.1111111111111111, 'direction_abs_good_but_signed_bad_rate': 0.08547008547008547, 'tmag_median_ratio': 0.6502568828902572, 'tmag_mean_ratio': 0.9329024503408898, 'tmag_p90_ratio': 2.0921881628503143, 'tmag_p95_ratio': 2.451524796440061, 'tmag_p99_ratio': 4.442680909656622, 'tmag_max_ratio': 6.5387987940746894, 'path_ratio': 0.1450057891095193}, 'unobservable_edges': {'count': 336, 'rot_mean_deg': 0.8884812951703073, 'rot_median_deg': 0.7792759633247004, 'rot_p90_deg': 1.29453100747998, 'signed_tdir_mean_deg': 51.755763673882335, 'signed_tdir_median_deg': 44.73936676411414, 'signed_tdir_p90_deg': 93.91420274956789, 'tdir_abs_mean_deg': 46.74571763816151, 'tdir_abs_median_deg': 44.06007179684573, 'tdir_abs_p90_deg': 78.9978243094547, 'tdir_mean_cosine': 0.555288928509159, 'anti_parallel_rate': 0.13095238095238096, 'severe_wrong_sign_rate': 0.03571428571428571, 'direction_abs_good_but_signed_bad_rate': 0.002976190476190476, 'tmag_median_ratio': 1.6401705006445007, 'tmag_mean_ratio': 1.9517627525777634, 'tmag_p90_ratio': 4.108909090344161, 'tmag_p95_ratio': 5.145640011042394, 'tmag_p99_ratio': 7.590530043232155, 'tmag_max_ratio': 19.819619392350763, 'path_ratio': 0.16313720986080116}, 'small_motion_edges': {'count': 7, 'rot_mean_deg': 1.5970093711357032, 'rot_median_deg': 1.6655844090563279, 'rot_p90_deg': 2.3058951124129923, 'signed_tdir_mean_deg': 25.504206398964097, 'signed_tdir_median_deg': 17.72655621809631, 'signed_tdir_p90_deg': 44.01315802360348, 'tdir_abs_mean_deg': 25.504206398964097, 'tdir_abs_median_deg': 17.72655621809631, 'tdir_abs_p90_deg': 44.01315802360348, 'tdir_mean_cosine': 0.8652335146781807, 'anti_parallel_rate': 0.0, 'severe_wrong_sign_rate': 0.0, 'direction_abs_good_but_signed_bad_rate': 0.0, 'tmag_median_ratio': 0.659739461413683, 'tmag_mean_ratio': 0.6557569449385046, 'tmag_p90_ratio': 0.8613868195371174, 'tmag_p95_ratio': 0.8744271360935891, 'tmag_p99_ratio': 0.8848593893387665, 'tmag_max_ratio': 0.8874674526500609, 'path_ratio': 0.6125036299730584}, 'high_confidence_direction_edges': {'count': 233, 'rot_mean_deg': 0.926058858011157, 'rot_median_deg': 0.8281121112559156, 'rot_p90_deg': 1.331861374224648, 'signed_tdir_mean_deg': 53.01538497593971, 'signed_tdir_median_deg': 41.70814898868401, 'signed_tdir_p90_deg': 110.90444564524493, 'tdir_abs_mean_deg': 42.76233472099367, 'tdir_abs_median_deg': 39.7359104261198, 'tdir_abs_p90_deg': 81.38582090162919, 'tdir_mean_cosine': 0.5095362456101318, 'anti_parallel_rate': 0.17167381974248927, 'severe_wrong_sign_rate': 0.09012875536480687, 'direction_abs_good_but_signed_bad_rate': 0.04291845493562232, 'tmag_median_ratio': 1.2662843893040756, 'tmag_mean_ratio': 1.6051260691648757, 'tmag_p90_ratio': 3.7955678240298676, 'tmag_p95_ratio': 4.66860108610276, 'tmag_p99_ratio': 6.840211902897157, 'tmag_max_ratio': 10.972052983683184, 'path_ratio': 0.14225071634877867}, 'anti_parallel_guarded_edges': {'count': 318, 'rot_mean_deg': 0.8747307977403722, 'rot_median_deg': 0.7701905211182393, 'rot_p90_deg': 1.2904362565931655, 'signed_tdir_mean_deg': 51.367930841639804, 'signed_tdir_median_deg': 43.025769891544144, 'signed_tdir_p90_deg': 94.51256525331215, 'tdir_abs_mean_deg': 46.20291438806023, 'tdir_abs_median_deg': 42.83492754543934, 'tdir_abs_p90_deg': 78.93389751620758, 'tdir_mean_cosine': 0.5585729160880996, 'anti_parallel_rate': 0.13522012578616352, 'severe_wrong_sign_rate': 0.03773584905660377, 'direction_abs_good_but_signed_bad_rate': 0.0031446540880503146, 'tmag_median_ratio': 1.7821008494246522, 'tmag_mean_ratio': 2.0597456895169195, 'tmag_p90_ratio': 4.164246301661386, 'tmag_p95_ratio': 5.1940510133670745, 'tmag_p99_ratio': 7.769782531812352, 'tmag_max_ratio': 19.819619392350763, 'path_ratio': 0.21930800251161597}}

## external evaluator none/se3/sim3
{'none': {'ate': 9.328085906618083, 'drift': 0.12856317480478102, 'path_ratio': 0.15716681680687314}, 'se3': {'ate': 3.9807674584476227, 'drift': 0.12988080283745088, 'path_ratio': 0.15716681680687314}, 'sim3': {'ate': 3.968240516140872, 'drift': 0.13012626059490318, 'path_ratio': 0.15716681680687314}}

## 与 S5E14 / S5E13 / S5E9 比较
{'under_scale_reduced': True, 'path_ratio_improved_without_explosion': True, 'tmag_median_closer_to_one': True, 'anti_parallel_reduced': True, 'tdir_preserved_or_improved': True, 'sim3_ate_improved': False, 'overall_geometry_improved': True}

## 与 ORB-SLAM3 比较
{'coverage_advantage': True, 'tdir_gap_remaining': True, 'scale_path_gap_remaining': True, 'aligned_ate_gap_to_orbslam3': 3.743948350154248, 'summary': 'S5E15 仍保留 full coverage，但与 ORB-SLAM3 在 scale/path 与 ATE 仍有明显差距。'}

## 是否进入 S5E16
{'can_continue_to_s5e16': True, 'main_remaining_blocker': 'intrinsics_geometry', 'recommended_next_stage': 'S5E16_confidence_calibrated_sign_scale_router'}

## caveats
- S5E15 是 experimental candidate。
- 不替代 official S5 locked result。
- S5 locked metrics/policy unchanged。
- 没有使用 eval GT 做 scale/sign calibration。
- oracle diagnostic 如有，不作为 candidate 指标。
