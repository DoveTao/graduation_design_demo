# S17 Pose Supervision And Dataset Quality Audit

## Executive summary

- final classification: `CV-SPLIT-NOT-REPRESENTATIVE`
- secondary classifications: `DATASET-SUPERVISION-CLEAN`
- S5 remains final clean candidate: `True`
- interpretation: Algebra and conventions are clean, but the final test sequence is not representative of the prior train/CV folds.

## Project state and final S5 candidate

- git branch: `optimize/s17-pose-supervision-dataset-quality-audit`
- git commit: `8f520764749d41235629537cd666ed66df2cb9cc`
- git status: `?? checkpoints/S17_pose_supervision_dataset_quality_audit_candidates.json
?? checkpoints/S17_pose_supervision_dataset_quality_audit_report.md
?? checkpoints/S17_pose_supervision_dataset_quality_figures/
?? reports/final_s17_pose_supervision_dataset_quality_summary.md
?? tools/s17_pose_supervision_dataset_quality_audit.py`
- final candidate: `S5_clean_tmag_calibration_policy`
- locked drift / ATE / path_ratio: `1.327343` / `7.352288` / `0.932379`

## Dataset / split summary

- dataset root: `data`
- split_by / train_ratio / split_seed: `scene_seq` / `0.8` / `3407`
- eval_k_list: `[1, 2, 3, 5, 10, 20]`
- eval min/max dt: `0.020000` / `5.000000`
- eval pair_step: `1`
- train sequences: `['scene01/seq01', 'scene01/seq02']`
- test sequences: `['scene01/seq03']`
- prior CV folds: `[{'fold': 'heldout_scene01_seq01', 'train': ['scene01/seq02'], 'val': ['scene01/seq01']}, {'fold': 'heldout_scene01_seq02', 'train': ['scene01/seq01'], 'val': ['scene01/seq02']}]`

| split | pairs | seqs | dt median | tmag median | speed median |
| --- | --- | --- | --- | --- | --- |
| train | 3286 | 2 | 0.1109 | 0.1109 | 1.0000 |
| test | 1695 | 1 | 0.0947 | 0.0947 | 1.0000 |

## GT relative pose algebra audit

- status: `pass`
- rotation diff max: `0.000004` deg
- tdir diff max: `0.000000` deg
- tmag relative diff max: `0.000000`
- dt relative diff max: `0.000000`
- interpretation: Absolute-pose-derived relative targets and manifest metadata are algebraically self-consistent.

## Frame order / inverse pair audit

- reversed pair count: `0`
- non-positive dt count: `0`
- k mismatch count: `0`
- inverse pair count in deterministic eval manifest: `0`
- interpretation: Pair ordering is monotonic with positive dt/tmag and consistent k; the deterministic eval manifest does not materially expose reversed pairs.

## Composition consistency audit

- triple count: `4346`
- rotation composition max error: `0.000005` deg
- tdir composition max error: `0.000016` deg
- tmag composition relative max error: `0.000000`
- interpretation: Direct pairs and composed GT chains are self-consistent under the official accumulation order.

## Coordinate convention audit

- dataset convention: `R_gt = R_wB^T R_wA`, `t_gt = R_wB^T (p_A - p_B)`
- train target convention: `{'supervised_tensor': "aux['t_dir_local']", 'pred_frame': 'A', 'loss_mapping': "losses._gt_translation_in_pred_frame(..., pred_t_frame='A') uses R_gt^T * t_gt_dir"}`
- eval accumulation convention: `{'evaluated_tensor': "aux['t_vec_out']", 'output_frame': 'B', 'compose_function': 'train_mvp._compose_rel_pose_np(R_rel, t_rel, R_cur0, t_cur0)', 'camera_center': 'train_mvp._camera_center_from_T_c0_np'}`
- all match: `True`
- interpretation: Dataset, training supervision, and eval accumulation use a consistent A-local training / B-frame output convention.

## dt/k/tmag distribution audit

- train dt L1 vs test: `0.097312`
- train k L1 vs test: `0.101080`
- train gt_tmag L1 vs test: `0.092416`
- train pred_tmag L1 vs test: `0.558594`
- train dt×k L1 vs test: `0.211423`
- high-risk bucket mass train / test: `0.034084` / `0.034218`

## Train/test split audit

- interpretation: Final test uses only scene01/seq03 while prior train/CV operated on scene01/seq01 and scene01/seq02; pred_tmag shift uses deterministic sampled eval pairs, while GT regime statistics use full split manifests. Also, dt_world is not an independent temporal variable here: it numerically equals gt_tmag, which is why the derived speed proxy collapses to ~1.0.
- train sequences are only `scene01/seq01` and `scene01/seq02`, while final test is only `scene01/seq03`.

## Label noise / ambiguity audit

- sampled examples: `6`
- interpretation: High-risk examples are numerically self-consistent; ambiguity risk should be interpreted as data difficulty rather than label inconsistency unless future manual review finds otherwise.

| category | split | scene | seq | i | j | k | dt | gt_tmag | speed | rot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_risk_dt>=1_k20 | test | scene01 | seq03 | 13 | 33 | 20 | 4.9921 | 4.9921 | 1.0000 | 8.2609 |
| high_speed | train | scene01 | seq02 | 358 | 359 | 1 | 0.2931 | 0.2931 | 1.0000 | 0.3585 |
| low_motion | test | scene01 | seq03 | 152 | 155 | 3 | 0.0200 | 0.0200 | 1.0000 | 1.6886 |
| large_rotation | train | scene01 | seq02 | 315 | 335 | 20 | 0.0541 | 0.0541 | 1.0000 | 12.2746 |
| large_dt | test | scene01 | seq03 | 13 | 33 | 20 | 4.9921 | 4.9921 | 1.0000 | 8.2609 |
| large_tmag | test | scene01 | seq03 | 13 | 33 | 20 | 4.9921 | 4.9921 | 1.0000 | 8.2609 |

## S13 component error cross-check

- status: `matched`
- recomputed rot mean / median / p90: `20.715418` / `20.433718` / `21.371955`
- recomputed tdir mean / median / p90: `72.349468` / `102.517961` / `110.918112`
- recomputed tmag log mean / median / p90: `0.899289` / `0.793141` / `1.742507`
- interpretation: S13 component errors are reproducible under the verified S17 convention.

## Sanity baselines / oracle sanity

| baseline | ATE | drift | path_ratio |
| --- | --- | --- | --- |
| constant_identity | 12.7575 | 1.9924 | 0.9628 |
| gt_scale_only | 7.5699 | 1.2503 | 1.0000 |
| oracle_R | 8.8837 | 1.4185 | 0.6949 |
| oracle_tdir | 7.4157 | 1.3497 | 0.6949 |
| oracle_R_tdir | 3.2599 | 0.5615 | 0.6949 |
| oracle_tmag | 7.5699 | 1.2503 | 1.0000 |

## Final classification

- primary: `CV-SPLIT-NOT-REPRESENTATIVE`
- secondary: `['DATASET-SUPERVISION-CLEAN']`
- rationale: Algebra and conventions are clean, but the final test sequence is not representative of the prior train/CV folds.

## Recommended next direction

- Redesign split evaluation and collect more representative data before further model scaling; if backbone work resumes, prefer task-specific geometric or multi-frame pretraining.

## Leakage / no-modification audit

- No model training was run.
- No S5 policy was modified.
- No test-set tuning was performed.
- No dataset labels or split files were modified.
- No heavy feature dumps were written.
