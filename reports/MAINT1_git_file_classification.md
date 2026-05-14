# MAINT1 Git File Classification

## Category 1: Should Commit As Code

- `datasets/dset2c_manifest_dataset.py`
- `datasets/__init__.py`
- `tools/train360b_forward_sanity.py`
- `tools/train360c_spherical_pose_baseline.py`
- `tools/eval_train360_pose.py`
- `tools/run_base360_hkust_360dvo_official.py`
- `tools/base360d_component_metric_alignment.py`
- `tools/base360c_official_python`
- `tools/base360c_ittnotify_shim.c`
- `train360_pose_losses.py`
- `configs/train360_v0_baseline.yaml`
- `configs/train360_v0_manifest_sanity.yaml`

## Category 2: Should Commit As Reports / Metric Summaries

- `reports/TRAIN360A_architecture_inventory_and_reusable_module_audit.md`
- `reports/TRAIN360A_reusable_module_matrix.json`
- `reports/TRAIN360B_manifest_native_dataloader_adapter_and_forward_sanity.md`
- `reports/TRAIN360B_forward_sanity.json`
- `reports/TRAIN360C_spherical_pose_baseline.md`
- `reports/TRAIN360C_metrics_val.json`
- `reports/TRAIN360C_metrics_test.json`
- `reports/BASE360B_env_inventory.md`
- `reports/BASE360B_fix_HKUST_360DVO_official_env_and_rerun.md`
- `reports/BASE360C_cuda_ba_build_inventory.md`
- `reports/BASE360C_fix_cuda_ba_extension_and_rerun_official_smoke.md`
- `reports/BASE360D_component_metric_alignment.md`
- `reports/BASE360D_metrics_val.json`
- `reports/BASE360D_metrics_test.json`
- `reports/BASE360_HKUST_360DVO_official_baseline_eval.md`
- `reports/BASE360_metrics_val.json`
- `reports/BASE360_metrics_test.json`
- `reports/RESULTS360_main_results_table.md`
- `reports/RESULTS360_main_results_table.json`
- `reports/RESULTS360_experiment_narrative.md`
- `reports/TRAIN360_vs_BASE360_vs_T57b_summary.md`
- `reports/MAINT1_conda_env_inventory.md`
- `reports/MAINT1_conda_env_inventory.json`
- `reports/MAINT1_conda_cleanup_recommendation.md`
- `reports/MAINT1_git_inventory.md`
- `reports/MAINT1_git_inventory.json`
- `reports/MAINT1_git_file_classification.md`
- `reports/MAINT1_conda_and_git_cleanup_summary.md`

## Category 3: Usually Keep Local, Do Not Commit By Default

- `checkpoints/TRAIN360C_spherical_pose_baseline/final.pt`
- `checkpoints/TRAIN360C_spherical_pose_baseline/best_val.pt`
- `external_baselines/results/base360_hkust_360dvo_official/` generated image inputs under `smoke/` and `official_demo_input_x0p5/`
- large local logs under `logs/`

## Category 4: Check Before Deciding

- `envs/pytorch.lock.yml`
- `envs/pytorch.pip-freeze.txt`
- `envs/base360dvo_rebuild.lock.yml`
- `envs/base360dvo_rebuild.pip-freeze.txt`
- `envs/base360dvo.candidate_old.lock.yml`
- `envs/base360dvo.candidate_old.pip-freeze.txt`
- `external_baselines/results/base360_hkust_360dvo_official/component_metrics/*.json`
- `external_baselines/results/base360_hkust_360dvo_official/*/split_metrics.json`
- `external_baselines/results/base360_hkust_360dvo_official/*/*/per_sequence_metrics.json`
- `external_baselines/results/base360_hkust_360dvo_official/*/*/pred_tum.txt` and `gt_tum.txt`: currently ignored, keep local unless a reviewer explicitly wants raw trajectories in Git.

## .gitignore Notes

- `reports/` stays tracked.
- `data/`, Python cache directories, `wandb/`, `runs/`, `tmp/`, and `.DS_Store` are already ignored.
- This maintenance pass additionally ignored generated `official_demo_input*`, `smoke/`, and `train_log.jsonl` outputs so the working tree better reflects reviewable changes.

## Size Snapshot

- `checkpoints/TRAIN360C_spherical_pose_baseline`: 175M	checkpoints/TRAIN360C_spherical_pose_baseline
- `external_baselines/results/base360_hkust_360dvo_official`: 1.3G	external_baselines/results/base360_hkust_360dvo_official
- `envs`: 56K	envs
