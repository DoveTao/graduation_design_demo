# S5E2 adjacent_dense dataset audit

## 执行摘要

S5E2 已构建 experimental adjacent edge dataset。现有默认 split 为 `scene_seq`，`split_seed=3407`，train split 是 `scene01/seq01` 与 `scene01/seq02`，eval/test sequence 是 `scene01/seq03`。本数据集不使用 `scene01/seq03` GT 训练，只在后续 evaluation / diagnostics 中使用。

## S5E1 blocker 回顾

S5E1 was blocked by missing engineering interfaces: no adjacent-dense dataloader/head/export path existed. The final S5 traceable artifact was selected_k1 only, so 321 adjacent edges remained unavailable.

## adjacent_dense dataset 构建

- num_train_pairs = 707
- num_val_pairs = 79
- eval_sequence = scene01/seq03
- eval_pairs = 453
- split_respected = True
- test_gt_used_for_training = False

## no GT leakage caveat

`scene01/seq03` 的 groundtruth 只允许用于 evaluation / diagnostics；训练 targets 仅来自 train split labels。

## classification

`final_classification = S5E2_ADJACENT_DATASET_READY`
