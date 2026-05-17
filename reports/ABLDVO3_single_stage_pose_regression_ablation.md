# ABLDVO3 single-stage pose regression ablation

## 1. Executive summary
- executed true/false: `false`
- model trained: `ABLDVO3_SingleStagePoseRegression`
- train subset count: `512`
- mini-val count: `1024`
- classification: `blocked`
- main conclusion: `training did not start because precheck failed on disk capacity before any checkpoint or metric artifact was produced.`

## 2. Why ABLDVO3
- `ABLDVO2` already established the contribution trend for `PlainPairVO`, `NoSphericalGeometry`, and `NoCrossImageInteraction`.
- the missing remaining large-module question is whether `coarse-to-fine residual refinement` itself contributes beyond `spherical-aware representation + cross-image interaction`.
- this task was prepared to isolate that factor with `ABLDVO3_SingleStagePoseRegression`, but execution stopped before training due to environment capacity.

## 3. Blocker
- blocker: `insufficient disk free space: 4.96 GiB`
- configured minimum free space: `5.0 GiB`
- branch at blocker time: `research/abldvo3-single-stage-pose-regression-ablation`
- CUDA check: `pass`
- manifests / ABLDVO2 reports / ablation code presence: `pass`
- current artifact check: `pass`
- py_compile: `pass`

## 4. What was prepared
- new branch created from `research/abldvo2-larger-subset-big-module-ablation`
- `ABLDVO3_SingleStagePoseRegression` entry enabled in [models/ablvo360_big_module_ablation.py](/home/dovetao/graduation_design_demo/models/ablvo360_big_module_ablation.py:749)
- ABLDVO3-only config added at [configs/abldvo3_single_stage_pose_regression_ablation.yaml](/home/dovetao/graduation_design_demo/configs/abldvo3_single_stage_pose_regression_ablation.yaml:1)
- subset strategy prepared to match ABLDVO2:
  - train subset count `512`
  - mini-val count `1024`
  - seed `1`
  - same stratified subset rule as ABLDVO2

## 5. Next recovery step
- free at least `0.5-1.0 GiB` beyond the current threshold so the run can start safely
- rerun:
  - `conda run -n pytorch python tools/train_ablvo360_big_module_ablation.py configs/abldvo3_single_stage_pose_regression_ablation.yaml`
- after disk is available, this task should only train one model, so it is substantially smaller than ABLDVO2

## 6. Compliance checklist
- training_executed = false
- test_used_for_selection = false
- existing_metrics_modified = false
- checkpoints_committed = false
- raw_data_committed = false
- explicit_matching_used = false
- ransac_used = false
- pnp_used = false
- ba_used = false
