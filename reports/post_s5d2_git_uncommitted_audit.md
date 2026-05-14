# Post-S5D2 Git Uncommitted / Remote Sync Audit

## Scope

S5D2 was committed and pushed separately. This report audits the remaining uncommitted and remote-sync state only. It does not modify algorithms, S5 policy, locked metrics, split, or official eval convention.

## Current Branch

- branch: `experiment/s5d2-dense-export-convention-audit`
- latest commit: `d808e79 Add S5D2 dense export convention audit`
- upstream: `origin/experiment/s5d2-dense-export-convention-audit`
- ahead/behind: `0/0`

## S5D2 Commit Status

- commit hash: `d808e79`
- push status: `pushed to origin/experiment/s5d2-dense-export-convention-audit`
- S5D2 audit: `PASS`
- verify_final_candidate: `PASS`
- project_health_check: `PASS`
- unittest: `PASS, 112 tests`

## Remaining Working Tree Changes

| file | status | likely group | recommendation |
|---|---|---|---|
| (none) | clean tracked diff | n/a | no action |

## Untracked Files

| file | likely group | recommendation |
|---|---|---|
| `checkpoints/MF1_chain_refiner_config.json` | MF1 | commit on MF1 branch or keep pending per user decision |
| `checkpoints/MF1_final_closeout_summary.json` | MF1 | commit on MF1 branch or keep pending per user decision |
| `checkpoints/MF1a_chain_refiner_smoke_results.json` | MF1 | commit on MF1 branch or keep pending per user decision |
| `checkpoints/MF1b_train_cv_config.json` | MF1 | commit on MF1 branch or keep pending per user decision |
| `checkpoints/MF1b_train_cv_results.json` | MF1 | commit on MF1 branch or keep pending per user decision |
| `checkpoints/ORB0_orbslam3_local_setup.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `checkpoints/ORB0b_orbslam3_build_remediation.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `checkpoints/ORB1a_fisheye_dataset_audit.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `checkpoints/ORB1b_calibration_conversion_audit.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `checkpoints/ORB1c_orbslam3_fisheye_run.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `checkpoints/ORB1d_orbslam3_fisheye_evaluation_results.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `checkpoints/S5D_dense_external_export.json` | S5D3 | review for dense/export baseline branch; do not mix with S5D2 audit commit |
| `checkpoints/S5D_same_evaluator_comparison_results.json` | S5D3 | review for dense/export baseline branch; do not mix with S5D2 audit commit |
| `external_baselines/config/orbslam3_fisheye_cam0_scene01_seq03.yaml` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/results/orbslam3_fisheye_cam0/eval_alignment_none.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/results/orbslam3_fisheye_cam0/eval_alignment_se3.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/results/orbslam3_fisheye_cam0/eval_alignment_sim3.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_images.txt` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_orbslam3_runtime.yaml` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_timestamps.txt` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/results/orbslam3_fisheye_cam0/sequence_export_stdout.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/results/s5_dense/eval_alignment_none.json` | S5D3 | review for dense/export baseline branch; do not mix with S5D2 audit commit |
| `external_baselines/results/s5_dense/eval_alignment_se3.json` | S5D3 | review for dense/export baseline branch; do not mix with S5D2 audit commit |
| `external_baselines/results/s5_dense/eval_alignment_sim3.json` | S5D3 | review for dense/export baseline branch; do not mix with S5D2 audit commit |
| `external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt` | S5D3 | review for dense/export baseline branch; do not mix with S5D2 audit commit |
| `external_baselines/runners/orbslam3_mono_fisheye_runner` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/runners/orbslam3_mono_fisheye_runner.cc` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/runners/run_orbslam3_fisheye_cam0.sh` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `reports/MF1_final_closeout_summary.md` | MF1 | commit on MF1 branch or keep pending per user decision |
| `reports/MF1a_chain_refiner_smoke_report.md` | MF1 | commit on MF1 branch or keep pending per user decision |
| `reports/MF1b_train_cv_report.md` | MF1 | commit on MF1 branch or keep pending per user decision |
| `reports/orbslam3_build_remediation_report.md` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `reports/orbslam3_fisheye_calibration_conversion_report.md` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `reports/orbslam3_fisheye_dataset_audit.md` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `reports/orbslam3_fisheye_evaluation_report.md` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `reports/orbslam3_fisheye_run_report.md` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `reports/orbslam3_local_setup_report.md` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `reports/s5_dense_external_export_report.md` | S5D3 | review for dense/export baseline branch; do not mix with S5D2 audit commit |
| `reports/s5_orbslam3_same_evaluator_comparison.md` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `scripts/remediate_orbslam3_build_orb0b.sh` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `scripts/run_mf1_closeout_summary.sh` | MF1 | commit on MF1 branch or keep pending per user decision |
| `scripts/run_mf1a_chain_refiner_smoke.sh` | MF1 | commit on MF1 branch or keep pending per user decision |
| `scripts/run_mf1b_train_cv.sh` | MF1 | commit on MF1 branch or keep pending per user decision |
| `scripts/setup_orbslam3_local.sh` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `tests/test_mf1_closeout_static.py` | MF1 | commit on MF1 branch or keep pending per user decision |
| `tests/test_mf1_static.py` | MF1 | commit on MF1 branch or keep pending per user decision |
| `tests/test_mf1b_static.py` | MF1 | commit on MF1 branch or keep pending per user decision |
| `tests/test_orbslam3_build_remediation_static.py` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `tests/test_orbslam3_fisheye_calibration_static.py` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `tests/test_orbslam3_fisheye_evaluation_static.py` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `tests/test_orbslam3_fisheye_run_static.py` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `tests/test_orbslam3_fisheye_static.py` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `tests/test_orbslam3_local_setup_static.py` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `tests/test_s5_dense_external_export_static.py` | S5D3 | review for dense/export baseline branch; do not mix with S5D2 audit commit |
| `tools/audit_raw_fisheye_dataset.py` | unknown / needs user review | needs user review; do not auto-delete or stash |
| `tools/convert_cam_infos_to_orbslam3_yaml.py` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `tools/export_orbslam3_fisheye_sequence.py` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `tools/export_s5_dense_trajectory.py` | S5D3 | review for dense/export baseline branch; do not mix with S5D2 audit commit |
| `tools/mf1_build_chain_dataset.py` | MF1 | commit on MF1 branch or keep pending per user decision |
| `tools/mf1_eval_chain_refiner.py` | MF1 | commit on MF1 branch or keep pending per user decision |
| `tools/mf1_eval_train_cv_chain_refiner.py` | MF1 | commit on MF1 branch or keep pending per user decision |
| `tools/mf1_train_chain_refiner.py` | MF1 | commit on MF1 branch or keep pending per user decision |
| `tools/mf1_train_cv_chain_refiner.py` | MF1 | commit on MF1 branch or keep pending per user decision |
| `tools/run_same_evaluator_comparison.py` | S5D3 | review for dense/export baseline branch; do not mix with S5D2 audit commit |

## Generated Outputs

| file | likely group | recommendation |
|---|---|---|
| `checkpoints/MF1_chain_refiner_config.json` | MF1 | commit on MF1 branch or keep pending per user decision |
| `checkpoints/MF1_final_closeout_summary.json` | MF1 | commit on MF1 branch or keep pending per user decision |
| `checkpoints/MF1a_chain_refiner_smoke_results.json` | MF1 | commit on MF1 branch or keep pending per user decision |
| `checkpoints/MF1b_train_cv_config.json` | MF1 | commit on MF1 branch or keep pending per user decision |
| `checkpoints/MF1b_train_cv_results.json` | MF1 | commit on MF1 branch or keep pending per user decision |
| `checkpoints/ORB0_orbslam3_local_setup.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `checkpoints/ORB0b_orbslam3_build_remediation.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `checkpoints/ORB1a_fisheye_dataset_audit.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `checkpoints/ORB1b_calibration_conversion_audit.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `checkpoints/ORB1c_orbslam3_fisheye_run.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `checkpoints/ORB1d_orbslam3_fisheye_evaluation_results.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `checkpoints/S5D_dense_external_export.json` | S5D3 | review for dense/export baseline branch; do not mix with S5D2 audit commit |
| `checkpoints/S5D_same_evaluator_comparison_results.json` | S5D3 | review for dense/export baseline branch; do not mix with S5D2 audit commit |
| `external_baselines/results/orbslam3_fisheye_cam0/eval_alignment_none.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/results/orbslam3_fisheye_cam0/eval_alignment_se3.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/results/orbslam3_fisheye_cam0/eval_alignment_sim3.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_images.txt` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_orbslam3_runtime.yaml` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_timestamps.txt` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/results/orbslam3_fisheye_cam0/sequence_export_stdout.json` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `external_baselines/results/s5_dense/eval_alignment_none.json` | S5D3 | review for dense/export baseline branch; do not mix with S5D2 audit commit |
| `external_baselines/results/s5_dense/eval_alignment_se3.json` | S5D3 | review for dense/export baseline branch; do not mix with S5D2 audit commit |
| `external_baselines/results/s5_dense/eval_alignment_sim3.json` | S5D3 | review for dense/export baseline branch; do not mix with S5D2 audit commit |
| `external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt` | S5D3 | review for dense/export baseline branch; do not mix with S5D2 audit commit |
| `reports/MF1_final_closeout_summary.md` | MF1 | commit on MF1 branch or keep pending per user decision |
| `reports/MF1a_chain_refiner_smoke_report.md` | MF1 | commit on MF1 branch or keep pending per user decision |
| `reports/MF1b_train_cv_report.md` | MF1 | commit on MF1 branch or keep pending per user decision |
| `reports/orbslam3_build_remediation_report.md` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `reports/orbslam3_fisheye_calibration_conversion_report.md` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `reports/orbslam3_fisheye_dataset_audit.md` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `reports/orbslam3_fisheye_evaluation_report.md` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `reports/orbslam3_fisheye_run_report.md` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `reports/orbslam3_local_setup_report.md` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `reports/s5_dense_external_export_report.md` | S5D3 | review for dense/export baseline branch; do not mix with S5D2 audit commit |
| `reports/s5_orbslam3_same_evaluator_comparison.md` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |

## Suspicious Unrelated Files

| file | likely group | recommendation |
|---|---|---|
| `external_baselines/runners/orbslam3_mono_fisheye_runner` | ORB-SLAM3 baseline | commit on ORB-SLAM3 baseline branch or keep pending per user decision |
| `tools/audit_raw_fisheye_dataset.py` | unknown / needs user review | needs user review; do not auto-delete or stash |

## Branch Remote Sync

| branch | upstream | ahead | behind | recommendation |
|---|---|---:|---:|---|
| `experiment/jrt1-joint-rtdir-coupled-refiner` | `origin/experiment/jrt1-joint-rtdir-coupled-refiner` | `0` | `0` | synced |
| `experiment/mf1-multi-frame-chain-refiner` | `(none)` | `n/a` | `n/a` | no upstream; do not auto-push |
| `experiment/orbslam3-fisheye-strong-baseline` | `(none)` | `n/a` | `n/a` | no upstream; do not auto-push |
| `experiment/s5d2-dense-export-convention-audit` | `origin/experiment/s5d2-dense-export-convention-audit` | `0` | `0` | synced |
| `main` | `origin/main` | `0` | `0` | synced |
| `optimize/s10-chain-pathratio-smoother` | `origin/optimize/s10-chain-pathratio-smoother` | `0` | `0` | synced |
| `optimize/s11-tmag-scale-consistency-training` | `origin/optimize/s11-tmag-scale-consistency-training` | `0` | `0` | synced |
| `optimize/s12-regime-balanced-sampling` | `origin/optimize/s12-regime-balanced-sampling` | `0` | `0` | synced |
| `optimize/s14-local-window-pose-graph` | `origin/optimize/s14-local-window-pose-graph` | `0` | `0` | synced |
| `optimize/s15-trajectory-level-training-objective` | `origin/optimize/s15-trajectory-level-training-objective` | `0` | `0` | synced |
| `optimize/s16-stronger-visual-backbone-feasibility` | `origin/optimize/s16-stronger-visual-backbone-feasibility` | `0` | `0` | synced |
| `optimize/s17-pose-supervision-dataset-quality-audit` | `origin/optimize/s17-pose-supervision-dataset-quality-audit` | `0` | `0` | synced |
| `optimize/s19-geometry-aware-pretraining-feasibility` | `origin/optimize/s19-geometry-aware-pretraining-feasibility` | `0` | `0` | synced |
| `optimize/s2-fine-refinement-on-s1d5` | `origin/optimize/s2-fine-refinement-on-s1d5` | `0` | `0` | synced |
| `optimize/s3a0-coupled-pose-residual-head` | `origin/optimize/s3a0-coupled-pose-residual-head` | `0` | `0` | synced |
| `polish/final-reproducibility-guardrails` | `origin/polish/final-reproducibility-guardrails` | `0` | `0` | synced |

## Protected Files Check

- `checkpoints/S5_clean_tmag_calibration_policy.json`: `no diff`
- `checkpoints/final_clean_candidate_manifest.json`: `no diff`

## Recommended Next Actions

- Keep S5D2 as a separate pushed audit branch.
- Ask the user whether to commit, discard, or keep each remaining untracked experiment group.
- If a branch has no upstream, set/push it only after explicit user request.
- If future branches are ahead, push branch-by-branch rather than batching unrelated work.
