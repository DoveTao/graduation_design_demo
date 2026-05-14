# S16b Frozen Pretrained Backbone Probe Report

## Executive summary

- final classification: `NO-STABLE-BACKBONE-FEATURE-GAIN`
- S16b is a feasibility diagnostic only; it is not a clean gain claim and does not replace S5.
- S5 remains the final clean candidate.
- encoder used: `torchvision_resnet50`
- pretrained weights available: `True`
- internet download attempted: `True`
- interpretation: Frozen pretrained ResNet50 features did not produce stable R/tdir/joint probe gains beyond the current/regime baselines.

## Baseline gate

- passed: `True`
- current-architecture load missing / unexpected: `14` / `0`
- S5 locked metrics preserved: `True`

## Probe constraints

- encoder frozen: `True`
- feature probe only: `True`
- full integration performed: `False`
- no heavy feature dump committed: `True`
- S5 final policy modified: `False`

## Families compared

- A. `current_model_features_only`
- B. `resnet50_pretrained_features_only`
- C. `resnet50_pretrained_features_plus_regime_features`
- D. `current_model_features_plus_resnet50_pretrained_features`
- E. `current_model_features_plus_resnet50_pretrained_features_plus_regime_features`

## Mean CV results

| family | rot R2 | tdir R2 | log tmag R2 | log tmag MAE | high AUC | joint AUC | gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regime_only | -4.0341 | -6.1174 | 1.0000 | 0.0045 | 0.4506 | 0.3689 | 3.6342 |
| current_model_features_only | -22.2815 | -4.4660 | 0.9233 | 0.2860 | 0.4073 | 0.7083 | 9.6060 |
| resnet50_pretrained_features_only | -258.7859 | -5.5098 | -129.3412 | 9.4317 | 0.6538 | 0.3830 | 132.2104 |
| resnet50_pretrained_features_plus_regime_features | -194.7359 | -5.3163 | -19.6930 | 3.6431 | 0.6524 | 0.3753 | 74.2481 |
| current_model_features_plus_resnet50_pretrained_features | -115.7265 | -5.0811 | -17.0007 | 3.2833 | 0.6562 | 0.3789 | 46.9358 |
| current_model_features_plus_resnet50_pretrained_features_plus_regime_features | -101.6272 | -4.9709 | -10.1337 | 2.7438 | 0.6520 | 0.3787 | 39.9104 |

## Locked comparison target

- S16 current baseline rot_R2: `-22.281483`
- S16 current baseline tdir_R2: `-4.465965`
- S16 current baseline joint_AUC: `0.708327`

## Fold details

| fold | family | rot R2 | tdir R2 | log tmag R2 | log tmag MAE | high AUC | joint AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heldout_scene01_seq01 | regime_only | -3.8374 | -7.3686 | 1.0000 | 0.0045 | 0.5803 | 0.6702 |
| heldout_scene01_seq01 | current_model_features_only | -3.7507 | -6.5438 | 0.9837 | 0.1282 | 0.5935 | 0.5932 |
| heldout_scene01_seq01 | resnet50_pretrained_features_only | -439.0311 | -6.8467 | -38.9747 | 5.1294 | 0.5608 | 0.2050 |
| heldout_scene01_seq01 | resnet50_pretrained_features_plus_regime_features | -332.0037 | -6.9024 | -3.4732 | 1.1833 | 0.5750 | 0.2547 |
| heldout_scene01_seq01 | current_model_features_plus_resnet50_pretrained_features | -187.3817 | -6.5572 | -3.0864 | 1.1687 | 0.5687 | 0.2263 |
| heldout_scene01_seq01 | current_model_features_plus_resnet50_pretrained_features_plus_regime_features | -161.6372 | -6.5647 | -2.8129 | 1.0713 | 0.5700 | 0.2330 |
| heldout_scene01_seq02 | regime_only | -4.2308 | -4.8661 | 1.0000 | 0.0045 | 0.3210 | 0.0676 |
| heldout_scene01_seq02 | current_model_features_only | -40.8123 | -2.3882 | 0.8628 | 0.4437 | 0.2210 | 0.8235 |
| heldout_scene01_seq02 | resnet50_pretrained_features_only | -78.5407 | -4.1728 | -219.7077 | 13.7341 | 0.7468 | 0.5611 |
| heldout_scene01_seq02 | resnet50_pretrained_features_plus_regime_features | -57.4681 | -3.7301 | -35.9128 | 6.1030 | 0.7297 | 0.4959 |
| heldout_scene01_seq02 | current_model_features_plus_resnet50_pretrained_features | -44.0714 | -3.6051 | -30.9150 | 5.3978 | 0.7436 | 0.5316 |
| heldout_scene01_seq02 | current_model_features_plus_resnet50_pretrained_features_plus_regime_features | -41.6171 | -3.3770 | -17.4546 | 4.4163 | 0.7339 | 0.5245 |

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

## Final interpretation

- current-feature baseline result: `rot_R2=-22.281483, tdir_R2=-4.465965, joint_AUC=0.708327`
- pretrained-feature result: `resnet50_pretrained_features_only: rot_R2=-258.785889, tdir_R2=-5.509770, joint_AUC=0.383036`
- R/tdir coupling proxy result: `resnet50_pretrained_features_only: rot_R2=-258.785889, tdir_R2=-5.509770, joint_AUC=0.383036`
- S16c full integration recommended: `False`
- S5 remains final clean candidate: `True`
