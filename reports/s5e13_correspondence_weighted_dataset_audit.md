# S5E13 correspondence weighted dataset audit

## 执行摘要
S5E13 weighted dataset 构建状态：`S5E13_TRAIN_CORRESPONDENCE_MISSING_PARTIAL`。

## train/eval split 与 no GT leakage
- train scenes 使用 `scene01/seq01` 和 `scene01/seq02` 的 adjacent pairs。
- eval scene 使用 `scene01/seq03`，其 GT 不用于训练。
- uses_scene01_seq03_gt_for_training = `false`

## 权重统计
- train_weighted_edges = `707`
- val_weighted_edges = `79`
- eval_edges = `453`
- train_correspondence_features_available = `true`
- eval_correspondence_features_available = `true`
- observable_fraction_train = `0.0`
- observable_fraction_eval = `0.2582781456953642`
- signed_direction_reliable_fraction_train = `0.0`
- signed_direction_reliable_fraction_eval = `0.24724061810154527`

## 说明
- observable / reliable edges 会获得更高的 signed tdir 监督权重。
- near-static / low-parallax / small-motion edges 会降低 signed direction loss 权重。
- strict essential geometry 没有用于训练监督。
