# AUDIT3 历史分支结果可信度审计

## 执行摘要
本次 AUDIT3 对 18 个历史 branch 做了只读 inventory、claim extraction 和 validity risk sweep。最终分类：`AUDIT3_HISTORICAL_RESULTS_REQUIRE_REPAIR`。

## 为什么要审历史 branch
S5E19B 已经证明，历史实验链中可能出现 smoke-only、fallback、mixed-ref 和 evaluator reuse 风险。为了避免把这些风险结果误写进论文主表，本轮只做 branch-level 审计，不训练、不刷新指标。

## branch inventory table
### main
- commit: bb8fadd387ee0bb2ada02bd6541e26905824d5a3
- latest_commit_message: Add S1d5 cleanup and staging audit reports
- exists_local/remote: True / True
- reports: 16
- checkpoints: 2
- result_jsons: 0
- training_status_files: 0
- raw_artifacts_present: True

### experiment/s5e1-traceable-adjacent-dense
- commit: 332b7324c0747f676515656f2261be28013cd1d2
- latest_commit_message: S5E20: add reports checkpoints and lightweight summaries
- exists_local/remote: True / True
- reports: 143
- checkpoints: 146
- result_jsons: 199
- training_status_files: 24
- raw_artifacts_present: True

### curation/final-report-archive
- commit: 0b8b54a3582e160f371f310a497c1f1b05ac503b
- latest_commit_message: Materialize curated reports archive
- exists_local/remote: True / True
- reports: 75
- checkpoints: 70
- result_jsons: 1
- training_status_files: 0
- raw_artifacts_present: True

### experiment/s5d2-dense-export-convention-audit
- commit: 9de06321507218657d0bbc8ece970566cf52a74f
- latest_commit_message: Add curated report archive plan
- exists_local/remote: True / True
- reports: 75
- checkpoints: 70
- result_jsons: 1
- training_status_files: 0
- raw_artifacts_present: True

### experiment/orbslam3-fisheye-strong-baseline
- commit: 797865b23679c428ef3ea4c21e1723b5f4b4204d
- latest_commit_message: Add S5 dense same-evaluator comparison artifacts
- exists_local/remote: True / True
- reports: 75
- checkpoints: 73
- result_jsons: 8
- training_status_files: 0
- raw_artifacts_present: True

### experiment/mf1-multi-frame-chain-refiner
- commit: ea559d523413b7cfa5e596d89cb9d2e21e8bcb47
- latest_commit_message: Add MF1 multi-frame chain refiner artifacts
- exists_local/remote: True / True
- reports: 75
- checkpoints: 76
- result_jsons: 1
- training_status_files: 0
- raw_artifacts_present: True

### experiment/jrt1-joint-rtdir-coupled-refiner
- commit: 6765af3ba47c30faed5b4865a004e1960f5d696b
- latest_commit_message: Add JRT1 branch remote sync audit
- exists_local/remote: True / True
- reports: 72
- checkpoints: 71
- result_jsons: 1
- training_status_files: 0
- raw_artifacts_present: True

### polish/final-reproducibility-guardrails
- commit: 7cb0a2c4b8ce352d340722617615f5837d3a802f
- latest_commit_message: Add repository hygiene audit before JRT1
- exists_local/remote: True / True
- reports: 67
- checkpoints: 65
- result_jsons: 1
- training_status_files: 0
- raw_artifacts_present: True

### optimize/s19-geometry-aware-pretraining-feasibility
- commit: f90d16768a167a498b39c1d25b5d47d30f14e213
- latest_commit_message: Add git branch remote sync audit
- exists_local/remote: True / True
- reports: 54
- checkpoints: 57
- result_jsons: 0
- training_status_files: 0
- raw_artifacts_present: True

### optimize/s17-pose-supervision-dataset-quality-audit
- commit: e020e11902a3443ff70b41d6c2b880c581ad7bba
- latest_commit_message: Add S18 split representativeness evaluation
- exists_local/remote: True / True
- reports: 51
- checkpoints: 56
- result_jsons: 0
- training_status_files: 0
- raw_artifacts_present: True

### optimize/s16-stronger-visual-backbone-feasibility
- commit: 8f520764749d41235629537cd666ed66df2cb9cc
- latest_commit_message: Close S16 backbone feasibility after ResNet50 probe
- exists_local/remote: True / True
- reports: 49
- checkpoints: 54
- result_jsons: 0
- training_status_files: 0
- raw_artifacts_present: True

### optimize/s15-trajectory-level-training-objective
- commit: 8a2857735cdccef4287de70055e8d87d7bfe0993
- latest_commit_message: Close S15 trajectory training line after stability retest
- exists_local/remote: True / True
- reports: 46
- checkpoints: 52
- result_jsons: 0
- training_status_files: 0
- raw_artifacts_present: True

### optimize/s12-regime-balanced-sampling
- commit: 34edd743fe6c26c3226b66ce5386b5bf0ed19531
- latest_commit_message: Add S13 practical usability gap analysis
- exists_local/remote: True / True
- reports: 37
- checkpoints: 48
- result_jsons: 0
- training_status_files: 0
- raw_artifacts_present: True

### optimize/s14-local-window-pose-graph
- commit: 36e68fe7a38fc3c337164142d8c04eaecd4663cf
- latest_commit_message: Close code optimization after S14 pose graph diagnostic
- exists_local/remote: True / True
- reports: 39
- checkpoints: 50
- result_jsons: 0
- training_status_files: 0
- raw_artifacts_present: True

### optimize/s11-tmag-scale-consistency-training
- commit: b00eab4de6784b63a958a440df9c96941588485b
- latest_commit_message: Close code optimization after S10 S11 diagnostics
- exists_local/remote: True / True
- reports: 35
- checkpoints: 35
- result_jsons: 0
- training_status_files: 0
- raw_artifacts_present: True

### optimize/s10-chain-pathratio-smoother
- commit: 0ae9aee1ca5c53728884ad4f4b42c4d7bd1dd28c
- latest_commit_message: Add S10 chain-level path-ratio preserving smoother diagnostic
- exists_local/remote: True / True
- reports: 33
- checkpoints: 34
- result_jsons: 0
- training_status_files: 0
- raw_artifacts_present: True

### optimize/s3a0-coupled-pose-residual-head
- commit: f232fa334d4ef222120d4c2c0efbcc44a31d8722
- latest_commit_message: Close post-S5 optimization after S8 S9 diagnostics
- exists_local/remote: True / True
- reports: 32
- checkpoints: 33
- result_jsons: 0
- training_status_files: 0
- raw_artifacts_present: True

### optimize/s2-fine-refinement-on-s1d5
- commit: c1f4b3653b96870412c12728f51c2bce2f329003
- latest_commit_message: Add S3 coupled pose head design plan
- exists_local/remote: True / True
- reports: 20
- checkpoints: 23
- result_jsons: 0
- training_status_files: 0
- raw_artifacts_present: True

## branch claim extraction table
### main
- claim_type: unknown
- representative_path: checkpoints/S1d5_clean_dt_anchor_policy.json
- final_classification: None
- claimed_results: {'ate': None, 'drift': None, 'path_ratio': None, 'rot': None, 'tdir': None, 'tdir_abs': None, 'tmag': None, 'coverage': None}

### experiment/s5e1-traceable-adjacent-dense
- claim_type: performance
- representative_path: checkpoints/S5E15_scale_deunderfit_antiparallel_candidate.json
- final_classification: S5E15_SCALE_DEUNDERFIT_IMPROVED
- claimed_results: {'ate': 9.328085906618083, 'drift': 0.12856317480478102, 'path_ratio': 0.15716681680687314, 'rot': 0.9134395040767528, 'tdir': 50.35304145887563, 'tdir_abs': 43.16064629037026, 'tmag': 0.399088892744088, 'coverage': 1.0}

### curation/final-report-archive
- claim_type: report_only
- representative_path: checkpoints/REPORT1_reports_inventory_audit.json
- final_classification: None
- claimed_results: {'ate': None, 'drift': None, 'path_ratio': None, 'rot': None, 'tdir': None, 'tdir_abs': None, 'tmag': None, 'coverage': None}

### experiment/s5d2-dense-export-convention-audit
- claim_type: diagnostic
- representative_path: checkpoints/S5D2_dense_export_convention_audit.json
- final_classification: S5D2-EVAL-SCOPE-MISMATCH
- claimed_results: {'ate': 7.352288, 'drift': 1.327343, 'path_ratio': 2.777267572676944, 'rot': None, 'tdir': None, 'tdir_abs': None, 'tmag': None, 'coverage': None}

### experiment/orbslam3-fisheye-strong-baseline
- claim_type: baseline
- representative_path: checkpoints/ORB0_orbslam3_local_setup.json
- final_classification: ORB0_BUILD_FAILED
- claimed_results: {'ate': None, 'drift': None, 'path_ratio': None, 'rot': None, 'tdir': None, 'tdir_abs': None, 'tmag': None, 'coverage': None}

### experiment/mf1-multi-frame-chain-refiner
- claim_type: unknown
- representative_path: checkpoints/MF1_final_closeout_summary.json
- final_classification: NO_STABLE_MF1_CHAIN_GAIN
- claimed_results: {'ate': 7.352288, 'drift': 1.327343, 'path_ratio': 1.4299451510111492, 'rot': 20.61405372619629, 'tdir': 19.020076751708984, 'tdir_abs': None, 'tmag': None, 'coverage': None}

### experiment/jrt1-joint-rtdir-coupled-refiner
- claim_type: unknown
- representative_path: checkpoints/JRT1_final_closeout_summary.json
- final_classification: NO_STABLE_JRT1_TRAJECTORY_GAIN
- claimed_results: {'ate': 7.352288, 'drift': 1.327343, 'path_ratio': 0.932379, 'rot': None, 'tdir': None, 'tdir_abs': None, 'tmag': None, 'coverage': None}

### polish/final-reproducibility-guardrails
- claim_type: unknown
- representative_path: checkpoints/REPO1_remote_sync_and_hygiene_audit.json
- final_classification: None
- claimed_results: {'ate': None, 'drift': None, 'path_ratio': None, 'rot': None, 'tdir': None, 'tdir_abs': None, 'tmag': None, 'coverage': None}

### optimize/s19-geometry-aware-pretraining-feasibility
- claim_type: feasibility
- representative_path: checkpoints/S19_geometry_aware_pretraining_feasibility_candidates.json
- final_classification: NO-STABLE-GEOMETRY-PRETRAINING-GAIN
- claimed_results: {'ate': 7.352288, 'drift': 1.327343, 'path_ratio': 0.932379, 'rot': 1.0, 'tdir': 1.0, 'tdir_abs': None, 'tmag': None, 'coverage': None}

### optimize/s17-pose-supervision-dataset-quality-audit
- claim_type: diagnostic
- representative_path: checkpoints/S17_pose_supervision_dataset_quality_audit_candidates.json
- final_classification: CV-SPLIT-NOT-REPRESENTATIVE
- claimed_results: {'ate': 7.352288, 'drift': 1.327343, 'path_ratio': 0.932379, 'rot': None, 'tdir': None, 'tdir_abs': None, 'tmag': None, 'coverage': None}

### optimize/s16-stronger-visual-backbone-feasibility
- claim_type: feasibility
- representative_path: checkpoints/S16_stronger_visual_backbone_feasibility_candidates.json
- final_classification: PRETRAINED-WEIGHTS-UNAVAILABLE
- claimed_results: {'ate': 7.352288, 'drift': 1.327343, 'path_ratio': 0.932379, 'rot': None, 'tdir': None, 'tdir_abs': None, 'tmag': None, 'coverage': None}

### optimize/s15-trajectory-level-training-objective
- claim_type: unknown
- representative_path: checkpoints/S15_trajectory_level_training_objective_candidates.json
- final_classification: INCONCLUSIVE
- claimed_results: {'ate': 7.352288, 'drift': 1.327343, 'path_ratio': 0.932379, 'rot': 4.801319305623525, 'tdir': 33.47919443007318, 'tdir_abs': 17.6821562935927, 'tmag': None, 'coverage': None}

### optimize/s12-regime-balanced-sampling
- claim_type: unknown
- representative_path: checkpoints/S12_regime_balanced_sampling_candidates.json
- final_classification: NO-STABLE-REGIME-SAMPLING-GAIN
- claimed_results: {'ate': 7.352288, 'drift': 1.327343, 'path_ratio': 0.932379, 'rot': 19.852126089474, 'tdir': 37.726716288547806, 'tdir_abs': 19.357324309034116, 'tmag': None, 'coverage': None}

### optimize/s14-local-window-pose-graph
- claim_type: unknown
- representative_path: checkpoints/S14_local_window_pose_graph_optimization_candidates.json
- final_classification: NO-STABLE-POSE-GRAPH-GAIN
- claimed_results: {'ate': 2.325955951641413, 'drift': 1.227873151918006, 'path_ratio': 4.930620961065616, 'rot': None, 'tdir': None, 'tdir_abs': None, 'tmag': None, 'coverage': None}

### optimize/s11-tmag-scale-consistency-training
- claim_type: unknown
- representative_path: checkpoints/S11_tmag_scale_consistency_candidates.json
- final_classification: NO-STABLE-TMAG-CONSISTENCY-GAIN
- claimed_results: {'ate': 7.352288, 'drift': 1.327343, 'path_ratio': 0.932379, 'rot': 19.852126089474, 'tdir': 37.726716288547806, 'tdir_abs': 19.357324309034116, 'tmag': None, 'coverage': None}

### optimize/s10-chain-pathratio-smoother
- claim_type: unknown
- representative_path: checkpoints/S10_chain_level_path_ratio_preserving_smoother_candidates.json
- final_classification: NO-STABLE-CHAIN-SMOOTHER-GAIN
- claimed_results: {'ate': 7.351221328788474, 'drift': 1.3270823574188497, 'path_ratio': 0.9349468349052762, 'rot': 21.481789469718933, 'tdir': None, 'tdir_abs': 29.967745549976826, 'tmag': None, 'coverage': None}

### optimize/s3a0-coupled-pose-residual-head
- claim_type: unknown
- representative_path: checkpoints/S5_tmag_regime_aware_calibration_candidates.json
- final_classification: TMAG-CALIBRATION-CLEAN-GAIN
- claimed_results: {'ate': 6.1805456603493365, 'drift': 1.1188390214986763, 'path_ratio': 1.0064212927193779, 'rot': nan, 'tdir': None, 'tdir_abs': nan, 'tmag': None, 'coverage': None}

### optimize/s2-fine-refinement-on-s1d5
- claim_type: unknown
- representative_path: checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p2/policy_used.json
- final_classification: None
- claimed_results: {'ate': None, 'drift': None, 'path_ratio': None, 'rot': None, 'tdir': None, 'tdir_abs': None, 'tmag': None, 'coverage': None}

## validity risk table
### main
- validity_classification: BRANCH_REPORT_ONLY
- risks: {'smoke_only': None, 'inference_only': None, 'derived_candidate': True, 'fallback_risk': False, 'hardcoded_REF': True, 'evaluator_mixed_ref': True, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': False, 'eval_gt_calibration_risk': False, 'orb_teacher_risk': False, 'missing_trajectory': True, 'missing_training_status': True, 'missing_eval_json': True, 'protocol_mismatch': False}

### experiment/s5e1-traceable-adjacent-dense
- validity_classification: BRANCH_VALID_PERFORMANCE_CANDIDATE
- risks: {'smoke_only': True, 'inference_only': False, 'derived_candidate': True, 'fallback_risk': False, 'hardcoded_REF': False, 'evaluator_mixed_ref': False, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': False, 'eval_gt_calibration_risk': True, 'orb_teacher_risk': False, 'missing_trajectory': False, 'missing_training_status': False, 'missing_eval_json': False, 'protocol_mismatch': False}

### curation/final-report-archive
- validity_classification: BRANCH_REPORT_ONLY
- risks: {'smoke_only': None, 'inference_only': None, 'derived_candidate': True, 'fallback_risk': False, 'hardcoded_REF': True, 'evaluator_mixed_ref': True, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': True, 'eval_gt_calibration_risk': False, 'orb_teacher_risk': True, 'missing_trajectory': False, 'missing_training_status': True, 'missing_eval_json': True, 'protocol_mismatch': False}

### experiment/s5d2-dense-export-convention-audit
- validity_classification: BRANCH_VALID_DIAGNOSTIC
- risks: {'smoke_only': None, 'inference_only': None, 'derived_candidate': False, 'fallback_risk': False, 'hardcoded_REF': False, 'evaluator_mixed_ref': False, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': True, 'eval_gt_calibration_risk': False, 'orb_teacher_risk': False, 'missing_trajectory': False, 'missing_training_status': True, 'missing_eval_json': True, 'protocol_mismatch': False}

### experiment/orbslam3-fisheye-strong-baseline
- validity_classification: BRANCH_BASELINE_ONLY
- risks: {'smoke_only': None, 'inference_only': None, 'derived_candidate': True, 'fallback_risk': False, 'hardcoded_REF': False, 'evaluator_mixed_ref': False, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': True, 'eval_gt_calibration_risk': False, 'orb_teacher_risk': False, 'missing_trajectory': False, 'missing_training_status': True, 'missing_eval_json': False, 'protocol_mismatch': False}

### experiment/mf1-multi-frame-chain-refiner
- validity_classification: BRANCH_PROTOCOL_MISMATCH
- risks: {'smoke_only': None, 'inference_only': None, 'derived_candidate': True, 'fallback_risk': False, 'hardcoded_REF': False, 'evaluator_mixed_ref': False, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': False, 'eval_gt_calibration_risk': False, 'orb_teacher_risk': False, 'missing_trajectory': True, 'missing_training_status': True, 'missing_eval_json': True, 'protocol_mismatch': True}

### experiment/jrt1-joint-rtdir-coupled-refiner
- validity_classification: BRANCH_INSUFFICIENT_EVIDENCE
- risks: {'smoke_only': None, 'inference_only': None, 'derived_candidate': True, 'fallback_risk': False, 'hardcoded_REF': False, 'evaluator_mixed_ref': False, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': False, 'eval_gt_calibration_risk': False, 'orb_teacher_risk': False, 'missing_trajectory': False, 'missing_training_status': True, 'missing_eval_json': True, 'protocol_mismatch': False}

### polish/final-reproducibility-guardrails
- validity_classification: BRANCH_MIXED_REF_RISK
- risks: {'smoke_only': None, 'inference_only': None, 'derived_candidate': True, 'fallback_risk': False, 'hardcoded_REF': True, 'evaluator_mixed_ref': True, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': False, 'eval_gt_calibration_risk': False, 'orb_teacher_risk': False, 'missing_trajectory': False, 'missing_training_status': True, 'missing_eval_json': True, 'protocol_mismatch': False}

### optimize/s19-geometry-aware-pretraining-feasibility
- validity_classification: BRANCH_FEASIBILITY_ONLY
- risks: {'smoke_only': None, 'inference_only': None, 'derived_candidate': False, 'fallback_risk': False, 'hardcoded_REF': False, 'evaluator_mixed_ref': False, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': False, 'eval_gt_calibration_risk': False, 'orb_teacher_risk': False, 'missing_trajectory': True, 'missing_training_status': True, 'missing_eval_json': True, 'protocol_mismatch': True}

### optimize/s17-pose-supervision-dataset-quality-audit
- validity_classification: BRANCH_VALID_DIAGNOSTIC
- risks: {'smoke_only': None, 'inference_only': None, 'derived_candidate': True, 'fallback_risk': False, 'hardcoded_REF': False, 'evaluator_mixed_ref': False, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': False, 'eval_gt_calibration_risk': False, 'orb_teacher_risk': False, 'missing_trajectory': True, 'missing_training_status': True, 'missing_eval_json': True, 'protocol_mismatch': True}

### optimize/s16-stronger-visual-backbone-feasibility
- validity_classification: BRANCH_FEASIBILITY_ONLY
- risks: {'smoke_only': None, 'inference_only': None, 'derived_candidate': False, 'fallback_risk': False, 'hardcoded_REF': False, 'evaluator_mixed_ref': False, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': False, 'eval_gt_calibration_risk': False, 'orb_teacher_risk': False, 'missing_trajectory': True, 'missing_training_status': True, 'missing_eval_json': True, 'protocol_mismatch': True}

### optimize/s15-trajectory-level-training-objective
- validity_classification: BRANCH_PROTOCOL_MISMATCH
- risks: {'smoke_only': None, 'inference_only': None, 'derived_candidate': False, 'fallback_risk': False, 'hardcoded_REF': False, 'evaluator_mixed_ref': False, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': False, 'eval_gt_calibration_risk': False, 'orb_teacher_risk': False, 'missing_trajectory': True, 'missing_training_status': True, 'missing_eval_json': True, 'protocol_mismatch': True}

### optimize/s12-regime-balanced-sampling
- validity_classification: BRANCH_FALLBACK_RISK
- risks: {'smoke_only': None, 'inference_only': None, 'derived_candidate': False, 'fallback_risk': True, 'hardcoded_REF': False, 'evaluator_mixed_ref': False, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': False, 'eval_gt_calibration_risk': False, 'orb_teacher_risk': False, 'missing_trajectory': True, 'missing_training_status': True, 'missing_eval_json': True, 'protocol_mismatch': True}

### optimize/s14-local-window-pose-graph
- validity_classification: BRANCH_PROTOCOL_MISMATCH
- risks: {'smoke_only': None, 'inference_only': None, 'derived_candidate': False, 'fallback_risk': False, 'hardcoded_REF': False, 'evaluator_mixed_ref': False, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': False, 'eval_gt_calibration_risk': False, 'orb_teacher_risk': False, 'missing_trajectory': True, 'missing_training_status': True, 'missing_eval_json': True, 'protocol_mismatch': True}

### optimize/s11-tmag-scale-consistency-training
- validity_classification: BRANCH_FALLBACK_RISK
- risks: {'smoke_only': None, 'inference_only': None, 'derived_candidate': False, 'fallback_risk': True, 'hardcoded_REF': False, 'evaluator_mixed_ref': False, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': False, 'eval_gt_calibration_risk': False, 'orb_teacher_risk': False, 'missing_trajectory': True, 'missing_training_status': True, 'missing_eval_json': True, 'protocol_mismatch': True}

### optimize/s10-chain-pathratio-smoother
- validity_classification: BRANCH_PROTOCOL_MISMATCH
- risks: {'smoke_only': None, 'inference_only': None, 'derived_candidate': False, 'fallback_risk': False, 'hardcoded_REF': False, 'evaluator_mixed_ref': False, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': False, 'eval_gt_calibration_risk': False, 'orb_teacher_risk': False, 'missing_trajectory': True, 'missing_training_status': True, 'missing_eval_json': True, 'protocol_mismatch': True}

### optimize/s3a0-coupled-pose-residual-head
- validity_classification: BRANCH_PROTOCOL_MISMATCH
- risks: {'smoke_only': None, 'inference_only': None, 'derived_candidate': True, 'fallback_risk': False, 'hardcoded_REF': False, 'evaluator_mixed_ref': False, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': False, 'eval_gt_calibration_risk': False, 'orb_teacher_risk': False, 'missing_trajectory': True, 'missing_training_status': True, 'missing_eval_json': True, 'protocol_mismatch': True}

### optimize/s2-fine-refinement-on-s1d5
- validity_classification: BRANCH_PROTOCOL_MISMATCH
- risks: {'smoke_only': None, 'inference_only': None, 'derived_candidate': True, 'fallback_risk': False, 'hardcoded_REF': False, 'evaluator_mixed_ref': False, 'old_metrics_reuse_risk': False, 'trajectory_copy': False, 'restored_dense_leakage': False, 'eval_gt_calibration_risk': False, 'orb_teacher_risk': False, 'missing_trajectory': True, 'missing_training_status': True, 'missing_eval_json': True, 'protocol_mismatch': True}

## 20° tdir 等历史 claim 来源说明
下面这些历史 claim 需要特别小心区分它们来自 dense trajectory 还是 pairwise/local-window 协议；只有同 evaluator、同 split、同 traceable dense 导出协议的结果，才适合和 S5E15 / ORB-SLAM3 直接并表。
- experiment/mf1-multi-frame-chain-refiner: tdir=19.020076751708984, rot=20.61405372619629, path_ratio=1.4299451510111492, protocol=unknown, directly_comparable_to_s5e15=False

## 哪些 branch 可以引用
- main table: ['experiment/orbslam3-fisheye-strong-baseline', 'experiment/s5e1-traceable-adjacent-dense']
- diagnostic: ['experiment/s5d2-dense-export-convention-audit', 'optimize/s17-pose-supervision-dataset-quality-audit']
- feasibility: ['optimize/s16-stronger-visual-backbone-feasibility', 'optimize/s19-geometry-aware-pretraining-feasibility']

## 哪些 branch 不可作为性能结果
['experiment/mf1-multi-frame-chain-refiner', 'optimize/s10-chain-pathratio-smoother', 'optimize/s11-tmag-scale-consistency-training', 'optimize/s12-regime-balanced-sampling', 'optimize/s14-local-window-pose-graph', 'optimize/s15-trajectory-level-training-objective', 'optimize/s2-fine-refinement-on-s1d5', 'optimize/s3a0-coupled-pose-residual-head', 'polish/final-reproducibility-guardrails']

## 当前 best candidate 是否仍为 S5E15
- best_current_candidate_remains: S5E15_scale_deunderfit_antiparallel_candidate

## 是否需要恢复某个 branch 重新跑
若未来要恢复历史 20° tdir 一类 claim，优先重审 protocol mismatch 的 optimize branch，而不是直接把它们并入当前 dense trajectory 主表。
