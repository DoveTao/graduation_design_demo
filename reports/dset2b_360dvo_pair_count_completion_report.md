# DSET2B 360DVO pair count completion report

## 1. 为什么做 DSET2B
DSET2 已经完成 fuller split 建档，但当前 blocker 是 pair 数量不足而不是 small-motion 风险，所以 DSET2B 的任务是补齐更多 360DVO sequence、重审完整性，并判断是否可以进入 TRAIN360_spherical_pose_baseline。

## 2. 远端 sequence inventory
- repo_id: chris1004336379/360DVO
- inventory_method: huggingface_hub
- num_sequences: 20
- dependency_missing: False
- blockers: []

## 3. 下载/补齐 summary
- selected_sequences_total: ['bridge_night', 'canyon_line', 'city_driving', 'downhill_biking', 'dragon_boat', 'drone_racetrack', 'field', 'hongkong_central', 'mountains', 'ridge_to_lake', 'shanghai_street', 'snowmobile', 'wingsuit']
- new_sequences_downloaded: ['downhill_biking', 'drone_racetrack', 'ridge_to_lake', 'shanghai_street', 'snowmobile']
- repaired_existing_sequences: ['bridge_night', 'canyon_line', 'field', 'mountains', 'wingsuit']
- oversized_sequences_skipped_for_full_sync: ['city_driving', 'dragon_boat', 'hongkong_central', 'london_bridge']
- synthetic_timestamps_written: ['bridge_night', 'canyon_line', 'city_driving', 'downhill_biking', 'dragon_boat', 'drone_racetrack', 'field', 'hongkong_central', 'london_bridge', 'mountains', 'ridge_to_lake', 'shanghai_street', 'snowmobile', 'snowy_mountain_road', 'tokyo_citywalk', 'wingsuit']
- total_bytes_downloaded: 3463378995
- download_complete: True
- blockers: []

## 4. sequence completeness audit
- num_sequences_scanned: 16
- top_complete_sequences: ['canyon_line', 'mountains', 'snowmobile', 'shanghai_street', 'downhill_biking', 'ridge_to_lake', 'bridge_night', 'drone_racetrack', 'field', 'city_driving', 'hongkong_central', 'dragon_boat']
- top_complete_sequence_stats: {"canyon_line": {"selected_for_dset2b_download": true, "image_count": 786, "pose_row_count": 786, "timestamp_count": 786, "first_image": "0001.jpg", "last_image": "0786.jpg", "has_groundtruth": true, "has_timestamps": true, "valid_pose_fraction": 1.0, "valid_adjacent_pairs_possible": 785, "valid_kstep_pairs_possible": 2348, "current_manifest_pairs_if_present": {"all": 6, "train": 6, "val": 0, "test": 0}, "missing_frames_suspected": false, "pose_limited_suspected": false, "timestamp_limited_suspected": false, "adapter_limit_suspected": true, "usable_frame_count": 786, "completeness_score": 3134.0}, "mountains": {"selected_for_dset2b_download": true, "image_count": 765, "pose_row_count": 765, "timestamp_count": 765, "first_image": "0001.jpg", "last_image": "0765.jpg", "has_groundtruth": true, "has_timestamps": true, "valid_pose_fraction": 1.0, "valid_adjacent_pairs_possible": 764, "valid_kstep_pairs_possible": 2285, "current_manifest_pairs_if_present": {"all": 137, "train": 0, "val": 137, "test": 0}, "missing_frames_suspected": false, "pose_limited_suspected": false, "timestamp_limited_suspected": false, "adapter_limit_suspected": true, "usable_frame_count": 765, "completeness_score": 3050.0}, "snowmobile": {"selected_for_dset2b_download": true, "image_count": 719, "pose_row_count": 830, "timestamp_count": 830, "first_image": "0001.jpg", "last_image": "0719.jpg", "has_groundtruth": true, "has_timestamps": true, "valid_pose_fraction": 0.8662650602409638, "valid_adjacent_pairs_possible": 718, "valid_kstep_pairs_possible": 2147, "current_manifest_pairs_if_present": null, "missing_frames_suspected": true, "pose_limited_suspected": false, "timestamp_limited_suspected": false, "adapter_limit_suspected": false, "usable_frame_count": 719, "completeness_score": 2865.866265060241}, "shanghai_street": {"selected_for_dset2b_download": true, "image_count": 685, "pose_row_count": 685, "timestamp_count": 685, "first_image": "0001.jpg", "last_image": "0685.jpg", "has_groundtruth": true, "has_timestamps": true, "valid_pose_fraction": 1.0, "valid_adjacent_pairs_possible": 684, "valid_kstep_pairs_possible": 2045, "current_manifest_pairs_if_present": null, "missing_frames_suspected": false, "pose_limited_suspected": false, "timestamp_limited_suspected": false, "adapter_limit_suspected": false, "usable_frame_count": 685, "completeness_score": 2730.0}, "downhill_biking": {"selected_for_dset2b_download": true, "image_count": 576, "pose_row_count": 576, "timestamp_count": 576, "first_image": "0001.jpg", "last_image": "0576.jpg", "has_groundtruth": true, "has_timestamps": true, "valid_pose_fraction": 1.0, "valid_adjacent_pairs_possible": 575, "valid_kstep_pairs_possible": 1718, "current_manifest_pairs_if_present": null, "missing_frames_suspected": false, "pose_limited_suspected": false, "timestamp_limited_suspected": false, "adapter_limit_suspected": false, "usable_frame_count": 576, "completeness_score": 2294.0}, "ridge_to_lake": {"selected_for_dset2b_download": true, "image_count": 552, "pose_row_count": 552, "timestamp_count": 552, "first_image": "0001.jpg", "last_image": "0552.jpg", "has_groundtruth": true, "has_timestamps": true, "valid_pose_fraction": 1.0, "valid_adjacent_pairs_possible": 551, "valid_kstep_pairs_possible": 1646, "current_manifest_pairs_if_present": null, "missing_frames_suspected": false, "pose_limited_suspected": false, "timestamp_limited_suspected": false, "adapter_limit_suspected": false, "usable_frame_count": 552, "completeness_score": 2198.0}}

## 5. train/val/test split
- train_sequences: ['canyon_line', 'shanghai_street', 'bridge_night', 'drone_racetrack', 'field', 'city_driving', 'hongkong_central', 'dragon_boat']
- val_sequences: ['mountains', 'downhill_biking']
- test_sequences: ['snowmobile', 'ridge_to_lake']

## 6. pair manifest summary
- num_sequences: 12
- num_pairs_train: 12540
- num_pairs_val: 5342
- num_pairs_test: 5062
- num_adjacent_pairs_train: 3149
- num_adjacent_pairs_val: 1339
- num_adjacent_pairs_test: 1269

## 7. motion distribution summary
- motion_distribution: {"all": {"num_pairs": 22944, "num_adjacent_pairs": 5757, "num_kstep_pairs": 17187, "gt_step_mean": 1.238818924705704, "gt_step_median": 0.826486159627815, "gt_step_p10": 0.22642682835332423, "gt_step_p25": 0.4104652138959404, "gt_step_p75": 1.6316650078249533, "gt_step_p90": 2.8663710765950214, "gt_step_p95": 3.774753701878289, "small_motion_fraction": 0.0002615062761506276, "very_small_motion_fraction": 0.0001307531380753138, "path_length": 28423.46140844767, "rotation_step_mean": 1.7893387118540989}, "train": {"num_pairs": 12540, "num_adjacent_pairs": 3149, "num_kstep_pairs": 9391, "gt_step_mean": 1.1319347645649451, "gt_step_median": 0.7383465950213741, "gt_step_p10": 0.2285895105188174, "gt_step_p25": 0.39553144286681086, "gt_step_p75": 1.4785651035907899, "gt_step_p90": 2.702818897875926, "gt_step_p95": 3.348410920121954, "small_motion_fraction": 0.0004784688995215311, "very_small_motion_fraction": 0.00023923444976076556, "path_length": 14194.461947644413, "rotation_step_mean": 2.010681966491894}, "val": {"num_pairs": 5342, "num_adjacent_pairs": 1339, "num_kstep_pairs": 4003, "gt_step_mean": 1.137338008945414, "gt_step_median": 0.5242541488407756, "gt_step_p10": 0.1294159302718717, "gt_step_p25": 0.2819850331580699, "gt_step_p75": 1.3068121615233526, "gt_step_p90": 3.097980192190403, "gt_step_p95": 4.427909264371944, "small_motion_fraction": 0.0, "very_small_motion_fraction": 0.0, "path_length": 6075.659643786401, "rotation_step_mean": 0.7694169425717885}, "test": {"num_pairs": 5062, "num_adjacent_pairs": 1269, "num_kstep_pairs": 3793, "gt_step_mean": 1.610695341172828, "gt_step_median": 1.2670413875783233, "gt_step_p10": 0.4518312842734925, "gt_step_p25": 0.69119848869515, "gt_step_p75": 2.178866277104222, "gt_step_p90": 3.2179406416457987, "gt_step_p95": 3.8766406197222523, "small_motion_fraction": 0.0, "very_small_motion_fraction": 0.0, "path_length": 8153.339817016855, "rotation_step_mean": 2.3173469454274214}, "train_val_similarity": {"available": true, "median_ratio": 1.4083753016623657, "mean_ratio": 0.9952492184926811, "small_motion_fraction_delta": 0.0004784688995215311}, "train_test_similarity": {"available": true, "median_ratio": 0.5827328154075255, "mean_ratio": 0.7027615562237405, "small_motion_fraction_delta": 0.0004784688995215311}, "val_test_similarity": {"available": true, "median_ratio": 0.41376245005127615, "mean_ratio": 0.7061161598178219, "small_motion_fraction_delta": 0.0}}

## 8. 是否达到 TRAIN360 readiness
- readiness: {"num_sequences_ok": true, "train_pairs_ok": true, "val_pairs_ok": true, "test_pairs_ok": true, "sequence_split_ok": true, "small_motion_risk_low": true, "adjacent_train_ok": true, "adjacent_val_ok": true, "adjacent_test_ok": true, "train360_ready": true}
- final_classification: DSET2B_TRAIN360_READY

## 9. 是否还需要更多 sequence
- download_more_sequences: False

## 10. 下一步建议
- recommendation: {"do_train360_baseline": true, "do_base360_baseline": true, "download_more_sequences": false, "keep_s5e15_as_legacy_best_candidate": true, "use_360dvo_as_main_dataset": true}
- 本轮不训练、不 fine-tune、不使用 360DVO GT 做 calibration。
- S5 locked metrics/policy were not changed.
