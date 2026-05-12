# GEN2 360DVO external adapter smoke eval

## 1. 为什么做 GEN2
GEN1 已将 360DVO 选为 primary external panoramic dataset，GEN2 的目标是在不训练模型的前提下验证最小 adapter、pair manifest 和外部 smoke eval feasibility。

## 2. 360DVO dataset inspection
{
  "dataset": "360DVO",
  "dataset_found": false,
  "local_root": "/home/dovetao/datasets/360DVO",
  "sequences_found": [],
  "image_count_by_sequence": {},
  "pose_files_found": [],
  "timestamp_files_found": [],
  "sample_image_resolution": null,
  "pose_format_detected": "unknown",
  "quaternion_order_detected": "unknown",
  "inspection_ready": false,
  "blockers": [
    "DATA_NOT_FOUND",
    "DOWNLOAD_REQUIRED"
  ]
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
  "manifest_ready": false,
  "pose_convention_ready": false,
  "split_by_sequence": true,
  "forbid_random_pair_split": true,
  "max_smoke_pairs": 100,
  "num_pairs": 0,
  "num_adjacent_pairs": 0,
  "num_kstep_pairs": 0,
  "valid_pose_fraction": 0.0,
  "k_values": [
    1,
    2,
    3,
    5
  ],
  "sequence_summaries": [],
  "blockers": [
    "DATA_NOT_FOUND",
    "INSPECTION_NOT_READY"
  ],
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
  "num_pairs": 0,
  "gt_step_median": null,
  "gt_step_mean": null,
  "gt_step_p10": null,
  "gt_step_p25": null,
  "gt_step_p50": null,
  "gt_step_p75": null,
  "gt_step_p90": null,
  "gt_step_p95": null,
  "small_motion_fraction": null,
  "very_small_motion_fraction": null,
  "path_length": null,
  "rotation_step_mean": null,
  "rotation_step_p90": null,
  "train_eval_sequence_split_feasibility": false,
  "vs_data2_seq03": {},
  "larger_or_more_stable_than_data2_seq03": false,
  "suitable_for_tdir_generalization_eval": false,
  "small_motion_risk_like_data2": null,
  "blockers": [
    "EMPTY_MANIFEST"
  ]
}

## 6. external smoke eval 是否可行
{
  "model_eval_attempted": false,
  "model_eval_available": false,
  "eval_blocker": "DATASET_NOT_FOUND_LOCALLY",
  "num_pairs": 0,
  "component_metrics_available": false,
  "rot_mean_deg": null,
  "tdir_mean_deg": null,
  "tmag_median_ratio": null,
  "path_ratio": null,
  "ate_available": false,
  "dataset_only_feasibility": false,
  "no_training": true
}

## 7. blockers
{
  "inspection_blockers": [
    "DATA_NOT_FOUND",
    "DOWNLOAD_REQUIRED"
  ],
  "manifest_blockers": [
    "DATA_NOT_FOUND",
    "INSPECTION_NOT_READY"
  ],
  "eval_blocker": "DATASET_NOT_FOUND_LOCALLY"
}

## 8. 是否进入 GEN3
{
  "do_gen3_external_eval": false,
  "do_external_training": false,
  "keep_s5e15_as_best_candidate": true,
  "main_next_step": "download_or_mount_360dvo_then_rerun_gen2"
}

## 9. caveats
- 本轮不训练、不 fine-tune、不生成新 candidate。
- 不提交下载数据、原始图像、raw trajectory、大日志或模型权重。
- 如果 360DVO 本地数据缺失，本报告只给出 adapter/blocker 审计，不假造 prediction。
