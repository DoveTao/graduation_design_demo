# DSET2C 360DVO dataset hygiene report

## 1. 为什么做 DSET2C
DSET2B 已经达到 TRAIN360_READY，但本地 data/360DVO 仍混有 partial download、空图像目录、pose-only 元数据和未进入训练的旁支 sequence，因此 DSET2C 的目标是把可用数据与 quarantine 元数据化，并固化 canonical manifest。

## 2. 本地 data/360DVO 为什么看起来乱
本地目录同时承载了多轮下载、partial 补齐、HF cache 和不同长度 sequence，因此仅看文件树会同时出现完整序列、半下载序列、纯 pose/timestamp 残留和空目录。

## 3. 哪些乱是正常的：sequence 长度不同
不同 sequence 本来就有不同长度；只要 image/pose/timestamp 可以形成稳定 pair，并且不直接依赖原始目录 glob 训练，这种长度差异不是问题。

## 4. 哪些乱是风险：空图像目录、pose-only、image-only、低 pair 数
风险主要来自 QUARANTINE_* sequence：空图像目录、图像缺失、pose/timestamp 缺失、以及 pair 数不足以稳定进入 TRAIN360 的序列。

## 5. 每类 sequence 的数量
- status_counts: {"USABLE_FOR_TRAIN360": 10, "USABLE_PARTIAL_HIGH_YIELD": 1, "QUARANTINE_EMPTY_IMAGE_DIR": 1, "QUARANTINE_IMAGE_MISSING": 2, "QUARANTINE_POSE_MISSING": 0, "QUARANTINE_TIMESTAMP_MISSING": 0, "QUARANTINE_LOW_PAIR_COUNT": 2, "QUARANTINE_UNKNOWN_FORMAT": 0}

## 6. 被纳入 canonical manifest 的 sequence
- included_sequences: ['canyon_line', 'shanghai_street', 'bridge_night', 'drone_racetrack', 'field', 'city_driving', 'mountains', 'downhill_biking', 'snowmobile', 'ridge_to_lake']

## 7. 被排除的 sequence 及原因
- excluded_sequences: {"dragon_boat": {"recommended_status": "QUARANTINE_LOW_PAIR_COUNT", "used_by_dset2b_train_val_test": ["train"], "current_dset2b_manifest_pairs": 189}, "hongkong_central": {"recommended_status": "QUARANTINE_LOW_PAIR_COUNT", "used_by_dset2b_train_val_test": ["train"], "current_dset2b_manifest_pairs": 197}, "london_bridge": {"recommended_status": "QUARANTINE_EMPTY_IMAGE_DIR", "used_by_dset2b_train_val_test": [], "current_dset2b_manifest_pairs": 0}, "snowy_mountain_road": {"recommended_status": "QUARANTINE_IMAGE_MISSING", "used_by_dset2b_train_val_test": [], "current_dset2b_manifest_pairs": 0}, "tokyo_citywalk": {"recommended_status": "QUARANTINE_IMAGE_MISSING", "used_by_dset2b_train_val_test": [], "current_dset2b_manifest_pairs": 0}, "wingsuit": {"recommended_status": "USABLE_FOR_TRAIN360", "used_by_dset2b_train_val_test": [], "current_dset2b_manifest_pairs": 0}}

## 8. canonical train/val/test pair 数
- num_pairs_train: 12154
- num_pairs_val: 5342
- num_pairs_test: 5062
- num_adjacent_pairs_train: 3049
- num_adjacent_pairs_val: 1339
- num_adjacent_pairs_test: 1269

## 9. 是否仍 TRAIN360_READY
- readiness: {"train360_ready_after_hygiene": true, "train_pairs_ok": true, "val_pairs_ok": true, "test_pairs_ok": true, "sequence_split_ok": true, "no_quarantine_sequence_used": true, "adjacent_train_ok": true, "adjacent_val_ok": true, "adjacent_test_ok": true, "small_motion_risk_low": true}
- final_classification: DSET2C_TRAIN360_READY_AFTER_HYGIENE

## 10. 后续所有训练必须只读 canonical manifest
- 后续 TRAIN360 / BASE360 训练必须只读取 external_baselines/results/dset2c_360dvo_canonical 下的 canonical manifest，不再直接 glob data/360DVO 原始目录。

## 11. caveats
- DSET2B 曾引用 partial 但高收益的 sequence；DSET2C 已将低于 hygiene 阈值的序列排除出 canonical manifest。
- 本轮不训练、不 fine-tune、不删除 raw data。
- S5 locked metrics/policy were not changed.
