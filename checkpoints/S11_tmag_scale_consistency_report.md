# S11 Tmag Scale Consistency Report

## Executive summary

- final classification: `INCONCLUSIVE`
- S11 replaces S5: `False`
- selected candidate: `None`
- final test run: `False`

## Motivation from S4/S5/S8/S9/S10

- S4/S5 showed that magnitude regime dominates much of the current error structure.
- S8/S9/S10 indicated that post-processing, routing, and chain smoothing provide limited clean improvement space.
- S11 therefore shifts the optimization target back into training-time tmag scale learning.

## Why training-time tmag consistency instead of post-processing

- S11 keeps the R / tdir path unchanged and only adjusts tmag-related supervision or calibration inside the model.
- gt_tmag is used only as a training label, never as an inference-time feature.

## Candidate loss definitions

- `loss_tmag_log`: existing log-space Huber supervision.
- `loss_speed`: Huber on `log(pred_tmag/dt) - log(gt_tmag/dt)`.
- `loss_ratio`: Huber on adjacent-pair log-ratio consistency inside train triplets.
- `loss_chain_sum`: Huber on short-chain log-sum consistency proxy.
- high-regime reweighting is train-only and uses pred/gt/dt/k labels without leaking them into inference.

## Training protocol

- mode: `smoke`
- max_steps per fold: `5`
- train-CV split uses `scene01/seq01` and `scene01/seq02` only for launch-stage selection.
- all candidates initialize from the coarse-to-fine base checkpoint and freeze non-tmag parameters.

Smoke note: this launch-stage run disables odometry evaluation to validate training/loss plumbing first, so ATE/drift/path_ratio are expected to remain unavailable (`nan`).

## Train-CV selection

| candidate | family | cv_mean_ATE | cv_mean_drift | cv_mean_path_ratio | clean_gate |
| --- | --- | ---: | ---: | ---: | --- |
| A_tmag_head_only_baseline | baseline_tmag_head_only | nan | nan | nan | False |
| B_dt_aware_affine_calib | dt_aware_shallow_calibrator | nan | nan | nan | False |

## Hard-gate audit

- `A_tmag_head_only_baseline`: mean_ATE `nan` vs baseline `nan`, mean_drift `nan` vs baseline `nan`, path_ratio `nan`, path_ok=`False`, init_ok=`False`, gate=`False`
- `B_dt_aware_affine_calib`: mean_ATE `nan` vs baseline `nan`, mean_drift `nan` vs baseline `nan`, path_ratio `nan`, path_ok=`False`, init_ok=`False`, gate=`False`

## Final test result, if selected

- not run in this launch stage.

## Compare against S5

- locked S5 drift / ATE / path_ratio = `1.327343` / `7.352288` / `0.932379`
- S5 remains final clean candidate: `True`

## Leakage audit

- passed: `True`
- test_used_for_selection: `False`
- gt_tmag_as_inference_feature: `False`
- gt_pose_as_inference_feature: `False`
- post_processing_router_used: `False`
- train_cv_selection_only: `True`

## Final classification

- `INCONCLUSIVE`
