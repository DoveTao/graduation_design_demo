# S19 Geometry Aware Pretraining Feasibility Report

## Executive summary

- final classification: `NO-STABLE-GEOMETRY-PRETRAINING-GAIN`
- S19b full integration recommended: `False`
- S19 is a feasibility diagnostic only; it does not replace S5 and does not claim clean gain.
- interpretation: No geometry-aware pretraining candidate produced stable R/tdir/joint probe gains beyond the current task-specific baseline.

## Motivation from S13/S16/S17/S18

- S13 quantified the practical gap and identified `R-TDIR-COUPLED-LIMITED` as the primary bottleneck.
- S16/S16b showed that generic frozen ImageNet features are weaker than the current task-specific representation for this geometric task.
- S17 cleaned up supervision/convention doubts, and S18 retained the historical protocol with representativeness caveat.
- S19 therefore tests whether lightweight geometry-aware pretraining can make the existing current-feature representation more linearly readable before any full integration.

## Baseline gate

- passed: `True`
- current-architecture load missing / unexpected: `14` / `0`
- ridge_calib buffers missing: `2`
- coupled_pose_head params missing: `12`
- other missing: `0`
- S5 locked metrics preserved: `True`

## Current task-specific feature baseline

- locked S16 reference: `rot_R2=-22.281483, tdir_R2=-4.465965, joint_AUC=0.708327`
- this S19 run baseline: `rot_R2=-16.342585, tdir_R2=-28.474333, joint_AUC=0.581828`

## ImageNet ResNet50 secondary baseline

- locked S16b reference: `rot_R2=-258.785889, tdir_R2=-5.509770, joint_AUC=0.383036`

## Geometry pretraining target definitions

- Relative rotation target: GT relative rotation log-vector with SmoothL1 loss.
- Translation direction target: GT local tdir with cosine loss.
- Optional log_tmag auxiliary: GT log magnitude with SmoothL1 loss.
- Optional joint high-error auxiliary: train-only binary label from combined current-model R/tdir difficulty bucket.

## Candidate families

- A. `current_feature_probe_baseline` mode=`probe_head_only` losses=`{}`
- B. `rot_tdir_pretrain_small` mode=`probe_head_only` losses=`{"rot": 1.0, "tdir": 1.0, "log_tmag": 0.0, "joint": 0.0}`
- E. `geometry_multitask_small` mode=`probe_head_only` losses=`{"rot": 1.0, "tdir": 1.0, "log_tmag": 0.5, "joint": 0.5}`

## Trainable parameter audit

| candidate | mode | trainable | frozen | base changed | S5 touched |
| --- | --- | --- | --- | --- | --- |
| rot_tdir_pretrain_small | probe_head_only | 572040.0000 | 11023858.0000 | False | False |
| geometry_multitask_small | probe_head_only | 572040.0000 | 11023858.0000 | False | False |

- explicit trainable names: `['backbone.0.weight', 'backbone.0.bias', 'backbone.2.weight', 'backbone.2.bias', 'rot_head.weight', 'rot_head.bias', 'tdir_head.weight', 'tdir_head.bias', 'tmag_head.weight', 'tmag_head.bias', 'joint_head.weight', 'joint_head.bias']`

## Training budget

- trainable mode: `probe_head_only`
- train cap per sequence: `1024`
- val cap per sequence: `256`
- updates per candidate per fold: `100`
- batch_size: `8`
- folds: `['heldout_scene01_seq01', 'heldout_scene01_seq02']`

## Probe protocol

- Extract frozen current-model pair features on the train split only.
- Train lightweight geometry heads on held-in train sequence supervision only.
- Re-extract learned embedding from the trained head and evaluate it with the same ridge/AUC probe family used for the current baseline.
- No final test sequence is used for candidate selection.

## CV/probe results

| family | rot R2 | tdir R2 | log tmag MAE | high AUC | joint AUC | gap |
| --- | --- | --- | --- | --- | --- | --- |
| regime_only | -3.8774 | -36.8026 | 0.0027 | 0.7105 | 0.4313 | 13.8296 |
| current_feature_probe_baseline | -16.3426 | -28.4743 | 0.0845 | 0.5066 | 0.5818 | 15.6041 |
| rot_tdir_pretrain_small | -7.0086 | -35.5798 | 2.6316 | 0.5820 | 0.5103 | 16.5054 |
| geometry_multitask_small | -24.7734 | -17.6171 | 6.7945 | 0.2295 | 0.5133 | 27.8958 |

## Primary comparison against current features

- current baseline: `rot_R2=-16.342585, tdir_R2=-28.474333, joint_AUC=0.581828`
- best geometry candidate: `geometry_multitask_small: rot_R2=-24.773435, tdir_R2=-17.617115, joint_AUC=0.513306`

## Secondary comparison against ResNet50

- ImageNet frozen ResNet50 reference: `rot_R2=-258.785889, tdir_R2=-5.509770, joint_AUC=0.383036`
- geometry result vs ImageNet conclusion: `not_better_than_imagenet_locked_reference`

## R/tdir coupling proxy analysis

- coupling proxy result: `geometry_multitask_small: rot_R2=-24.773435, tdir_R2=-17.617115, joint_AUC=0.513306`

## Tmag analysis

- regime_only log_tmag_MAE: `0.002735`
- current_model_features_only log_tmag_MAE: `0.084496`
- best geometry log_tmag_MAE: `6.794462`

## Leakage audit

- no test-set candidate selection: `True`
- no test-set threshold selection: `True`
- gt labels used only for train/CV supervision: `True`
- no gt pose / gt_tmag / gt_tdir used as inference feature: `True`
- S5 policy unchanged: `True`
- no heavy checkpoint committed: `True`
- no pretrained/fine-tuned weights committed: `True`

## Whether S19b is recommended

- `False`

## Whether S5 remains final clean candidate

- `True`

## Final classification

- `NO-STABLE-GEOMETRY-PRETRAINING-GAIN`
