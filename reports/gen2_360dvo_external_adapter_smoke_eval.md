# GEN2 360DVO external adapter smoke eval

## 1. 为什么做 GEN2
GEN1 已将 360DVO 选为 primary external panoramic dataset，GEN2 的目标是在不训练模型的前提下验证最小 adapter、pair manifest 和外部 smoke eval feasibility。

## 2. 360DVO dataset inspection
{
  "dataset": "360DVO",
  "dataset_found": true,
  "local_root": "data/360DVO",
  "sequences_found": [
    "Sequences/field"
  ],
  "image_count_by_sequence": {
    "Sequences/field": 101
  },
  "pose_files_found": [
    "data/360DVO/GroundTruth/field.txt"
  ],
  "timestamp_files_found": [
    "data/360DVO/Timestamps/field_timestamps.txt"
  ],
  "sample_image_resolution": [
    3840,
    1920
  ],
  "pose_format_detected": "tum_like_tx_ty_tz_qx_qy_qz_qw_no_timestamp",
  "quaternion_order_detected": "xyzw",
  "inspection_ready": true,
  "blockers": [],
  "source_pose_convention_detected": "T_w_c_assumed_from_tum_without_timestamp"
}

## 3. adapter schema
{
  "dataset": "360DVO",
  "target_pose_convention": "T_w_c",
  "relative_pose_terms": [
    "R_BA",
    "t_BA_B",
    "tdir_B"
  ],
  "pair_types": [
    "adjacent",
    "kstep"
  ],
  "k_values": [
    1,
    2,
    3,
    5
  ]
}

## 4. pair manifest summary
{
  "dataset": "360DVO",
  "manifest_ready": true,
  "pose_convention_ready": true,
  "split_by_sequence": true,
  "forbid_random_pair_split": true,
  "max_smoke_pairs": 100,
  "num_pairs": 100,
  "num_adjacent_pairs": 25,
  "num_kstep_pairs": 75,
  "valid_pose_fraction": 1.0,
  "k_values": [
    1,
    2,
    3,
    5
  ],
  "sequence_summaries": [
    {
      "seq_id": "Sequences/field",
      "num_images": 101,
      "pose_file": "data/360DVO/GroundTruth/field.txt",
      "timestamp_file": "data/360DVO/Timestamps/field_timestamps.txt",
      "pose_rows": 101,
      "pairs_built": 100
    }
  ],
  "blockers": [],
  "target_schema_terms": [
    "R_BA",
    "t_BA_B",
    "tdir_B",
    "T_w_c",
    "adjacent",
    "kstep"
  ]
}

## 5. motion distribution vs DATA2 当前数据
{
  "num_pairs": 100,
  "gt_step_median": 1.1221688852924325,
  "gt_step_mean": 1.1227872974709823,
  "gt_step_p10": 0.3778633395147681,
  "gt_step_p25": 0.7444670891145597,
  "gt_step_p50": 1.1221688852924325,
  "gt_step_p75": 1.8753056274114026,
  "gt_step_p90": 1.8908075712689083,
  "gt_step_p95": 2.262509924036204,
  "small_motion_fraction": 0.0,
  "very_small_motion_fraction": 0.0,
  "path_length": 112.27872974709823,
  "rotation_step_mean": 0.632855073162311,
  "rotation_step_p90": 1.40334445167258,
  "train_eval_sequence_split_feasibility": false,
  "compared_to_data2_seq03": {
    "data2_seq03_gt_step_median": 0.00750599760191659,
    "data2_seq03_small_motion_fraction": 0.38852097130242824,
    "data2_seq03_path_length": 25.833639521294945,
    "gt_step_median_ratio_over_data2_seq03": 149.50296347095778,
    "small_motion_fraction_delta_vs_data2_seq03": -0.38852097130242824,
    "path_length_ratio_over_data2_seq03": 4.346221896242907
  },
  "vs_data2_seq03": {
    "data2_seq03_gt_step_median": 0.00750599760191659,
    "data2_seq03_small_motion_fraction": 0.38852097130242824,
    "data2_seq03_path_length": 25.833639521294945,
    "gt_step_median_ratio_over_data2_seq03": 149.50296347095778,
    "small_motion_fraction_delta_vs_data2_seq03": -0.38852097130242824,
    "path_length_ratio_over_data2_seq03": 4.346221896242907
  },
  "larger_or_more_stable_than_data2_seq03": true,
  "suitable_for_tdir_generalization_eval": false,
  "small_motion_risk_like_data2": false,
  "blockers": []
}

## 6. external smoke eval 是否可行
{
  "model_eval_attempted": true,
  "model_eval_available": false,
  "eval_blocker": "ADAPTER_NOT_MODEL_READY",
  "num_pairs": 100,
  "component_metrics_available": false,
  "rot_mean_deg": null,
  "tdir_mean_deg": null,
  "tmag_median_ratio": null,
  "path_ratio": null,
  "ate_available": false,
  "dataset_only_feasibility": true,
  "no_training": true
}

## 7. blockers
{
  "inspection_blockers": [],
  "manifest_blockers": [],
  "eval_blocker": "ADAPTER_NOT_MODEL_READY"
}

## 8. 是否进入 GEN3
{
  "do_gen3_external_eval": true,
  "do_external_training": false,
  "keep_s5e15_as_best_candidate": true,
  "main_next_step": "wire_external_inference_bridge_for_360dvo"
}

## 9. caveats
- 本轮不训练、不 fine-tune、不生成新 candidate。
- 不提交下载数据、原始图像、raw trajectory、大日志或模型权重。
- 如果 360DVO 本地数据缺失，本报告只给出 adapter/blocker 审计，不假造 prediction。
