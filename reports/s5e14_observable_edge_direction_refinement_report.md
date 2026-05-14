# S5E14 observable edge direction refinement report

## 执行摘要
final_classification = `S5E14_OBSERVABLE_IMPROVED_OVERALL_STILL_BAD`。S5E14 作为 experimental candidate，目标是在不破坏 rot 和 traceability 的前提下，做 observable-edge direction refinement，并降低 anti_parallel 风险。

## S5E13 结果回顾
- overall signed tdir = `50.35304145887563`
- observable signed tdir = `46.32471099526666`
- reliable signed tdir = `47.07029256818264`
- overall anti_parallel_rate = `0.1479028697571744`
- reliable anti_parallel_rate = `0.20535714285714285`
- path_ratio = `0.05272511597296368`

## S5E13 direction error audit
{'path': 'external_baselines/results/s5e14_traceable_dense/direction_error_audit.json', 'reliable_mask_too_loose': True, 'observable_edges_include_noisy_direction_cases': True, 'inference_time_blending_recommended': True, 'main_direction_error_source': 'unknown'}

## 为什么 observable 不等于 reliable signed direction
observable 仅表示局部 correspondence 可见；但在 low_parallax / near_static / high_angle_dispersion / small_motion 下，signed direction 依然可能噪声高或符号不稳定，因此需要 strict reliable mask。

## refined mask / strict reliable mask 设计
通过 reliable_original + low_parallax/near_static/small_motion 过滤 + angle_dispersion/parallax/flow/inlier/gt_tmag 阈值得到 reliable_strict。

## inference-time blending 或 refinement 策略
采用 inference-time feature gating：reliable_strict 主要使用 correspondence_head，其他边退回 fallback_prior 或 blended，且明确标记 direction_source/scale_source/inference_blend_weight。

## no GT leakage / no eval GT calibration 说明
uses_scene01_seq03_gt_for_training=false；uses_eval_gt_for_calibration=false。scene01/seq03 GT 仅用于 evaluation/diagnostics。

## traceable dense export coverage
{'available': True, 'trajectory_path': 'external_baselines/results/s5e14_traceable_dense/scene01_seq03_s5e14_traceable_dense_tum.txt', 'edge_provenance': 'external_baselines/results/s5e14_traceable_dense/edge_provenance.jsonl', 'num_poses': 454, 'num_edges': 453, 'direct_adjacent_prediction_edges': 453, 'coverage': 1.0, 'all_edges_traceable': True}

## overall component metrics
{'count': 453, 'rot_mean_deg': 0.9134395040767528, 'rot_median_deg': 0.797895569319923, 'rot_p90_deg': 1.3218302317546538, 'signed_tdir_mean_deg': 50.353041365033235, 'signed_tdir_median_deg': 41.229480409356896, 'signed_tdir_p90_deg': 101.804771450087, 'tdir_abs_mean_deg': 43.16064629037027, 'tdir_abs_median_deg': 39.87455596995842, 'tdir_abs_p90_deg': 78.63250433511601, 'tdir_mean_cosine': 0.5591010842073055, 'anti_parallel_rate': 0.1479028697571744, 'severe_wrong_sign_rate': 0.05518763796909492, 'direction_abs_good_but_signed_bad_rate': 0.024282560706401765, 'tmag_median_ratio': 0.399088892744088, 'tmag_mean_ratio': 0.5858264348350569, 'tmag_p90_ratio': 1.3689528097606414, 'tmag_p95_ratio': 1.6560806175064955, 'tmag_p99_ratio': 2.4199378476577404, 'tmag_max_ratio': 7.078435497268129, 'path_ratio': 0.052725115603500676}

## stratified metrics: observable / reliable_original / reliable_strict / unobservable / small-motion
{'observable_edges': {'count': 117, 'rot_mean_deg': 0.9851143604234686, 'rot_median_deg': 0.841925552079711, 'rot_p90_deg': 1.3675857408014738, 'signed_tdir_mean_deg': 46.32471063192815, 'signed_tdir_median_deg': 28.202055999174636, 'signed_tdir_p90_deg': 125.6523318720361, 'tdir_abs_mean_deg': 32.86505677876464, 'tdir_abs_median_deg': 27.624287219861337, 'tdir_abs_p90_deg': 74.62204678154342, 'tdir_mean_cosine': 0.5700488133917261, 'anti_parallel_rate': 0.19658119658119658, 'severe_wrong_sign_rate': 0.1111111111111111, 'direction_abs_good_but_signed_bad_rate': 0.08547008547008547, 'tmag_median_ratio': 0.1835846648476164, 'tmag_mean_ratio': 0.2656077947506881, 'tmag_p90_ratio': 0.5906798878741712, 'tmag_p95_ratio': 0.692130094985901, 'tmag_p99_ratio': 1.2542859711057663, 'tmag_max_ratio': 1.8460753230024536, 'path_ratio': 0.04133888496932256}, 'reliable_original_edges': {'count': 112, 'rot_mean_deg': 0.965044032843611, 'rot_median_deg': 0.8394672342118747, 'rot_p90_deg': 1.3436689149487966, 'signed_tdir_mean_deg': 47.07029218862366, 'signed_tdir_median_deg': 28.693579381055685, 'signed_tdir_p90_deg': 128.12429115269902, 'tdir_abs_mean_deg': 33.00976093130106, 'tdir_abs_median_deg': 27.64497005447692, 'tdir_abs_p90_deg': 75.62826202135851, 'tdir_mean_cosine': 0.5586450926460694, 'anti_parallel_rate': 0.20535714285714285, 'severe_wrong_sign_rate': 0.11607142857142858, 'direction_abs_good_but_signed_bad_rate': 0.08928571428571429, 'tmag_median_ratio': 0.17780579696473353, 'tmag_mean_ratio': 0.26723899938626866, 'tmag_p90_ratio': 0.59094933733763, 'tmag_p95_ratio': 0.7017755920947653, 'tmag_p99_ratio': 1.2686856028202502, 'tmag_max_ratio': 1.8460753230024536, 'path_ratio': 0.039905626427282626}, 'reliable_strict_edges': {'count': 0, 'rot_mean_deg': None, 'rot_median_deg': None, 'rot_p90_deg': None, 'signed_tdir_mean_deg': None, 'signed_tdir_median_deg': None, 'signed_tdir_p90_deg': None, 'tdir_abs_mean_deg': None, 'tdir_abs_median_deg': None, 'tdir_abs_p90_deg': None, 'tdir_mean_cosine': None, 'anti_parallel_rate': None, 'severe_wrong_sign_rate': None, 'direction_abs_good_but_signed_bad_rate': None, 'tmag_median_ratio': None, 'tmag_mean_ratio': None, 'tmag_p90_ratio': None, 'tmag_p95_ratio': None, 'tmag_p99_ratio': None, 'tmag_max_ratio': None, 'path_ratio': 0.0}, 'unobservable_edges': {'count': 336, 'rot_mean_deg': 0.8884812951703073, 'rot_median_deg': 0.7792759633247004, 'rot_p90_deg': 1.29453100747998, 'signed_tdir_mean_deg': 51.755763673882335, 'signed_tdir_median_deg': 44.73936676411414, 'signed_tdir_p90_deg': 93.91420274956789, 'tdir_abs_mean_deg': 46.74571763816151, 'tdir_abs_median_deg': 44.06007179684573, 'tdir_abs_p90_deg': 78.9978243094547, 'tdir_mean_cosine': 0.555288928509159, 'anti_parallel_rate': 0.13095238095238096, 'severe_wrong_sign_rate': 0.03571428571428571, 'direction_abs_good_but_signed_bad_rate': 0.002976190476190476, 'tmag_median_ratio': 0.5857751788016075, 'tmag_mean_ratio': 0.6973311398644354, 'tmag_p90_ratio': 1.4674675322657718, 'tmag_p95_ratio': 1.8377285753722838, 'tmag_p99_ratio': 2.7109035868686266, 'tmag_max_ratio': 7.078435497268129, 'path_ratio': 0.05831512628403073}, 'small_motion_edges': {'count': 7, 'rot_mean_deg': 1.5970093711357032, 'rot_median_deg': 1.6655844090563279, 'rot_p90_deg': 2.3058951124129923, 'signed_tdir_mean_deg': 25.5042063989641, 'signed_tdir_median_deg': 17.72655621809631, 'signed_tdir_p90_deg': 44.013158023603495, 'tdir_abs_mean_deg': 25.5042063989641, 'tdir_abs_median_deg': 17.72655621809631, 'tdir_abs_p90_deg': 44.013158023603495, 'tdir_mean_cosine': 0.8652335146781805, 'anti_parallel_rate': 0.0, 'severe_wrong_sign_rate': 0.0, 'direction_abs_good_but_signed_bad_rate': 0.0, 'tmag_median_ratio': 0.24104474293521486, 'tmag_mean_ratio': 0.250985041241813, 'tmag_p90_ratio': 0.3341742297026383, 'tmag_p95_ratio': 0.35352985700469325, 'tmag_p99_ratio': 0.36901435884633726, 'tmag_max_ratio': 0.3728854843067483, 'path_ratio': 0.23244953497323023}, 'low_parallax_edges': {'count': 170, 'rot_mean_deg': 0.9211682652783666, 'rot_median_deg': 0.7892281553385228, 'rot_p90_deg': 1.3737662509777313, 'signed_tdir_mean_deg': 53.36861882737824, 'signed_tdir_median_deg': 43.31692095685946, 'signed_tdir_p90_deg': 108.56351199766037, 'tdir_abs_mean_deg': 45.39240018333365, 'tdir_abs_median_deg': 42.83492754543934, 'tdir_abs_p90_deg': 75.90589946097506, 'tdir_mean_cosine': 0.5234938132938769, 'anti_parallel_rate': 0.17647058823529413, 'severe_wrong_sign_rate': 0.058823529411764705, 'direction_abs_good_but_signed_bad_rate': 0.0058823529411764705, 'tmag_median_ratio': 0.3768217271292188, 'tmag_mean_ratio': 0.5991381020046169, 'tmag_p90_ratio': 1.43454595725944, 'tmag_p95_ratio': 1.6638721426913359, 'tmag_p99_ratio': 2.739925238558994, 'tmag_max_ratio': 7.078435497268129, 'path_ratio': 0.05737924060041926}, 'near_static_edges': {'count': 318, 'rot_mean_deg': 0.8747307977403722, 'rot_median_deg': 0.7701905211182393, 'rot_p90_deg': 1.2904362565931655, 'signed_tdir_mean_deg': 51.3679308416398, 'signed_tdir_median_deg': 43.02576989154413, 'signed_tdir_p90_deg': 94.51256525331215, 'tdir_abs_mean_deg': 46.20291438806023, 'tdir_abs_median_deg': 42.83492754543934, 'tdir_abs_p90_deg': 78.93389751620758, 'tdir_mean_cosine': 0.5585729160880994, 'anti_parallel_rate': 0.13522012578616352, 'severe_wrong_sign_rate': 0.03773584905660377, 'direction_abs_good_but_signed_bad_rate': 0.0031446540880503146, 'tmag_median_ratio': 0.636464589080233, 'tmag_mean_ratio': 0.7359119281050231, 'tmag_p90_ratio': 1.4872308220219237, 'tmag_p95_ratio': 1.8550182190596696, 'tmag_p99_ratio': 2.774922332790126, 'tmag_max_ratio': 7.078435497268129, 'path_ratio': 0.07839792705817461}}

## external evaluator none/se3/sim3
{'none': {'ate': 9.71516068025357, 'drift': 0.12919590089886548, 'path_ratio': 0.05272511558013559}, 'se3': {'ate': 4.070750130887251, 'drift': 0.12960713653119316, 'path_ratio': 0.05272511558013559}, 'sim3': {'ate': 3.947901282749505, 'drift': 0.1301594288373733, 'path_ratio': 0.05272511558013559}}

## 与 S5E13 / S5E12 / S5E9 比较
{'overall_tdir_improved': True, 'observable_tdir_improved': True, 'reliable_anti_parallel_reduced': False, 'overall_anti_parallel_reduced': False, 'under_scale_reduced': False, 'sim3_ate_improved': True, 'overall_geometry_improved': False}

## 与 ORB-SLAM3 比较
{'coverage_advantage': True, 'tdir_gap_remaining': True, 'scale_path_gap_remaining': True, 'aligned_ate_gap_to_orbslam3': 3.723609116762881, 'summary': 'S5E14 保持 full coverage，但与 ORB-SLAM3 在 tdir、scale/path、ATE 上仍有明显差距。'}

## 是否进入 S5E15
{'can_continue_to_s5e15': True, 'main_remaining_blocker': 'under_scale', 'recommended_next_stage': 'S5E15_scale_direction_joint_refinement'}

## caveats
- S5E14 是 experimental candidate。
- 不替代 official S5 locked result。
- S5 locked metrics/policy unchanged。
- strict essential geometry 仍受 intrinsics 限制。
- 若 train reliable signed edges 缺失，不声称 supervised signed-direction training 完整成功。
