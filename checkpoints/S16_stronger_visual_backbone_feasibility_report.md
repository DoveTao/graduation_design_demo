# S16 Stronger Visual Backbone Feasibility Report

## Executive summary

- final classification: `PRETRAINED-WEIGHTS-UNAVAILABLE`
- S16 is a feasibility diagnostic, not a final clean candidate replacement.
- S5 remains the final clean candidate with locked metrics drift `1.327343`, ATE `7.352288`, path_ratio `0.932379`.
- S16b full backbone integration recommended: `False`
- interpretation: No local pretrained torchvision/timm weights were available, and no internet download was attempted.

## Motivation from S13/S15 closure

- S13 quantified the practical gap and identified `R-TDIR-COUPLED-LIMITED` as the primary bottleneck.
- S15 closed the tiny trajectory-level training route under the current harness because it remained unstable.
- S16 therefore checks whether stronger frozen visual features show diagnostic signal for R/tdir coupling before any full backbone replacement is considered.

## Baseline gate

- passed: `True`
- current-architecture load missing / unexpected: `14` / `0`
- ridge_calib buffers missing: `2`
- coupled_pose_head params missing: `12`
- other missing: `0`
- S5 locked metrics preserved: `True`

## Current representation audit

- status: `completed`
- device: `cuda`
- unique extracted train-split rows: `1024`
- feature inputs: coarse pooled features, fine pooled features, spherical bearing summaries, `pred_tmag`, `dt`, `k`.
- leakage policy: GT rotation/tdir/tmag are labels only; they are not included in probe features.

Mean two-fold CV probe results:

| family | rot R2 | rot ang err | tdir R2 | tdir ang err | tdir cos | log tmag MAE | high AUC | high recall | joint AUC | gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regime_only | -4.0341 | 6.3958 | -6.1174 | 118.4661 | -0.4206 | 0.0045 | 0.4506 | 0.1935 | 0.3689 | 3.6342 |
| current_model_features_only | -22.2815 | 15.0542 | -4.4660 | 95.0187 | -0.0850 | 0.2860 | 0.4073 | 0.8377 | 0.7083 | 9.6060 |

## Encoder availability audit

- internet download attempted: `False`
- any pretrained weights available: `False`

| encoder | pkg | pretrained | params | dim | download | frozen |
| --- | --- | --- | --- | --- | --- | --- |
| torchvision_resnet50 | True | False | 25557032.0000 | 2048.0000 | False | True |
| torchvision_convnext_tiny | True | False | 28589128.0000 | 768.0000 | False | True |
| torchvision_vit_b_16 | True | False | 86567656.0000 | 768.0000 | False | True |
| timm_cached | False | False | None | None | False | True |

## Frozen feature extraction protocol

- For a locally cached pretrained encoder, the intended pair feature is `feat_A`, `feat_B`, `feat_B - feat_A`, `feat_A * feat_B`, concatenated.
- Safe regime features are restricted to `pred_tmag`, `dt`, and `k`.
- GT pose, GT tmag, GT tdir, and pair error are labels only and are never inference features.
- This run did not execute stronger frozen feature extraction because no local pretrained weights were available.

## Probe families

- Family A: `current_model_features_only`.
- Family B: `stronger_encoder_features_only`.
- Family C: `stronger_encoder_plus_regime_features`.
- Family D: `current_model_features_plus_stronger_encoder_features`.
- Family E: `current_model_features_plus_stronger_encoder_plus_regime_features`.
- Families B-E were not run when pretrained weights were unavailable.

## CV results

| fold | family | rot R2 | tdir R2 | tmag MAE | high AUC | joint AUC |
| --- | --- | --- | --- | --- | --- | --- |
| heldout_scene01_seq01 | regime_only | -3.8374 | -7.3686 | 0.0045 | 0.5803 | 0.6702 |
| heldout_scene01_seq01 | current_model_features_only | -3.7507 | -6.5438 | 0.1282 | 0.5935 | 0.5932 |
| heldout_scene01_seq02 | regime_only | -4.2308 | -4.8661 | 0.0045 | 0.3210 | 0.0676 |
| heldout_scene01_seq02 | current_model_features_only | -40.8123 | -2.3882 | 0.4437 | 0.2210 | 0.8235 |

## Regime-level analysis

| fold | type | bucket | n | rot err | tdir err | log tmag err |
| --- | --- | --- | --- | --- | --- | --- |
| heldout_scene01_seq01 | dt_bucket | <0.3 | 179.0000 | 23.0979 | 31.5425 | 0.7294 |
| heldout_scene01_seq01 | dt_bucket | >=2.0 | 15.0000 | 24.1881 | 64.7707 | 3.2504 |
| heldout_scene01_seq01 | dt_bucket | [0.3,0.5) | 18.0000 | 24.0400 | 49.0606 | 1.0707 |
| heldout_scene01_seq01 | dt_bucket | [0.5,1.0) | 27.0000 | 22.8713 | 48.9133 | 1.5639 |
| heldout_scene01_seq01 | dt_bucket | [1.0,2.0) | 17.0000 | 23.1166 | 62.0558 | 2.3712 |
| heldout_scene01_seq01 | k_bucket | k=1 | 39.0000 | 20.9475 | 41.6655 | 0.9105 |
| heldout_scene01_seq01 | k_bucket | k=10 | 49.0000 | 24.1398 | 37.8003 | 1.0961 |
| heldout_scene01_seq01 | k_bucket | k=2 | 42.0000 | 21.3171 | 45.6347 | 1.3038 |
| heldout_scene01_seq01 | k_bucket | k=20 | 41.0000 | 28.3382 | 30.0998 | 0.8140 |
| heldout_scene01_seq01 | k_bucket | k=3 | 44.0000 | 21.8341 | 37.7770 | 1.3023 |
| heldout_scene01_seq01 | k_bucket | k=5 | 41.0000 | 22.5091 | 38.6890 | 1.1334 |
| heldout_scene01_seq02 | dt_bucket | <0.3 | 176.0000 | 19.4652 | 28.2835 | 0.7730 |
| heldout_scene01_seq02 | dt_bucket | >=2.0 | 12.0000 | 19.3098 | 18.1988 | 3.2907 |
| heldout_scene01_seq02 | dt_bucket | [0.3,0.5) | 16.0000 | 20.1378 | 19.0063 | 0.9928 |
| heldout_scene01_seq02 | dt_bucket | [0.5,1.0) | 36.0000 | 20.3016 | 17.8715 | 1.6518 |
| heldout_scene01_seq02 | dt_bucket | [1.0,2.0) | 16.0000 | 19.9361 | 17.6753 | 2.3486 |
| heldout_scene01_seq02 | k_bucket | k=1 | 40.0000 | 20.7633 | 33.0241 | 1.0064 |
| heldout_scene01_seq02 | k_bucket | k=10 | 30.0000 | 18.7306 | 17.0490 | 1.1398 |
| heldout_scene01_seq02 | k_bucket | k=2 | 57.0000 | 20.7167 | 41.8883 | 1.2521 |
| heldout_scene01_seq02 | k_bucket | k=20 | 46.0000 | 16.7725 | 14.8161 | 0.7276 |
| heldout_scene01_seq02 | k_bucket | k=3 | 46.0000 | 20.5337 | 18.6422 | 1.5337 |
| heldout_scene01_seq02 | k_bucket | k=5 | 37.0000 | 20.0067 | 18.0380 | 1.0440 |

## R/tdir coupling proxy analysis

- stronger-feature R/tdir coupling proxy result: `not_evaluable_without_pretrained_stronger_features`
- Probe AUC/R2 is interpreted only as diagnostic readability, not as odometry improvement.

## Leakage audit

- S16 does not modify `checkpoints/S5_clean_tmag_calibration_policy.json`.
- S16 does not overwrite S5 locked metrics.
- S16 does not train a residual pose head.
- S16 does not use the test set for encoder, probe, or threshold selection.
- S16 does not use GT pose, GT tmag, GT tdir, or pair error as inference features.
- S16 does not download pretrained weights and does not write heavy feature dumps.
- Frozen encoder probing is not described as deployable final inference.

## Whether S16b is recommended

- recommended: `False`
- reason: No local pretrained torchvision/timm weights were available, and no internet download was attempted.

## Whether S5 remains final clean candidate

- `True`. S16 is not a final clean candidate replacement and makes no clean gain claim.

## Final classification

- `PRETRAINED-WEIGHTS-UNAVAILABLE`
