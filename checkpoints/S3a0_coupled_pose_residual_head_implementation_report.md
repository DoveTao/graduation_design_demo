# S3a0 Coupled Pose Residual Head Implementation Report

## 1. Modified Files

Modified source files:

- `config.py`
- `interaction.py`
- `model.py`
- `train_mvp.py`
- `losses.py`

New tools / scripts:

- `tools/s3a0_audit_coupled_pose_head.py`
- `scripts/run_s3a0_coupled_pose_residual_head_smoke.sh`
- `scripts/run_s3a0_coupled_pose_residual_head_train.sh`

Generated reports:

- `checkpoints/S3a0_coupled_pose_head_smoke_audit.md`
- `checkpoints/S3a0_coupled_pose_residual_head_smoke_report.md`

## 2. New Config Additions

Added coupled-head feature flags with defaults off:

- `use_coupled_pose_residual_head`
- `coupled_pose_residual_trainable`
- `coupled_pose_residual_hidden_dim`
- `coupled_pose_residual_dropout`
- `coupled_pose_residual_rot_scale`
- `coupled_pose_residual_tdir_scale`
- `coupled_pose_residual_gate_init`
- `coupled_pose_residual_use_dt_embed`
- `coupled_pose_residual_use_confidence`

Added coupled loss weights:

- `coupled_pose_rot_loss_w`
- `coupled_pose_tdir_loss_w`
- `coupled_pose_joint_loss_w`
- `coupled_pose_residual_reg_w`
- `coupled_pose_chain_loss_w`

Added coupled-only training controls:

- `train_coupled_pose_residual_only`
- `freeze_backbone_for_coupled_pose`
- `freeze_tmag_for_coupled_pose`
- `freeze_dt_anchor_for_coupled_pose`

Default behavior note:

- All new fields default to disabled or inactive, so old forward behavior stays unchanged when `use_coupled_pose_residual_head=False`.

## 3. Coupled Head Architecture

Implemented `CoupledPoseResidualHead` in `interaction.py`.

Inputs:

- fine pooled feature statistics from `Ff_t`
- base `R` flattened to 9 values
- base local-frame `tdir`
- optional confidence summary from fine matching weights
- optional log-`dt_world`

Outputs:

- `delta_rot_vec` with small-angle scaling
- `delta_tdir_vec` with small residual scaling
- `gate` via sigmoid, initialized near zero with `gate_init=-4.0`

Initialization:

- rotation, tdir, and gate heads are initialized to zero-update behavior
- forward sanity shows the enabled head initially reproduces baseline outputs exactly

## 4. Forward Integration Point

Integration is in `model.py`, after the existing S2b-style coarse-to-fine fused pose is formed.

Flow:

1. Compute the original fine-rot / fine-tdir / fine-tmag fused result.
2. Save:
   - `R_before_coupled`
   - `tdir_before_coupled`
   - `tmag_before_coupled`
3. If `use_coupled_pose_residual_head=True`:
   - predict `delta_rot_vec`, `delta_tdir_vec`, `gate`
   - convert `delta_rot_vec` to `delta_R` through a small-angle `so(3)` exponential map
   - apply `R_after_coupled = delta_R @ R_before_coupled`
   - apply `tdir_after_coupled = normalize(tdir_before_coupled + gate * delta_tdir_vec)`
4. Keep `tmag_after_coupled = tmag_before_coupled`

Exposed audit fields:

- `R_before_coupled`
- `tdir_before_coupled`
- `R_after_coupled`
- `tdir_after_coupled`
- `tmag_before_coupled`
- `tmag_after_coupled`
- `delta_rot_vec`
- `delta_tdir_vec`
- `coupled_gate`
- `coupled_delta_rot_norm`
- `coupled_delta_tdir_norm`
- `coupled_gate_mean`

## 5. Loss Integration Point

Added in `train_mvp.py` with helpers in `losses.py`.

New losses:

- `L_coupled_rot`: geodesic rotation loss on `R_after_coupled`
- `L_coupled_tdir`: local-frame translation-direction loss on `tdir_after_coupled`
- `L_coupled_joint`: B-frame consistency loss using `R_after_coupled @ tdir_after_coupled`
- `L_coupled_reg`: residual regularization on `delta_rot_vec`, `delta_tdir_vec`, and `gate`
- `L_coupled_chain`: currently wired as zero for S3a0

Frame convention used for the joint loss:

- local-frame direction is supervised in A-frame
- joint consistency is checked in B-frame by transforming the local direction through `R_pred`

## 6. Freeze / Optimizer Audit Result

Implemented `train_coupled_pose_residual_only=True` mode in `train_mvp.py`.

Observed smoke audit result:

- trainable parameter names are only under `coupled_pose_head.*`
- trainable parameter count: `119463`
- frozen parameter count: `10572485`
- optimizer parameter count: `119463`
- forbidden trainable params: `[]`

Trainable names:

- `coupled_pose_head.backbone.0.weight`
- `coupled_pose_head.backbone.0.bias`
- `coupled_pose_head.backbone.1.weight`
- `coupled_pose_head.backbone.1.bias`
- `coupled_pose_head.backbone.4.weight`
- `coupled_pose_head.backbone.4.bias`
- `coupled_pose_head.rot_head.weight`
- `coupled_pose_head.rot_head.bias`
- `coupled_pose_head.tdir_head.weight`
- `coupled_pose_head.tdir_head.bias`
- `coupled_pose_head.gate_head.weight`
- `coupled_pose_head.gate_head.bias`

Conclusion:

- optimizer isolation works as intended
- encoder / coarse / existing fine / tmag are not trained in coupled-only mode

## 7. tmag Before / After Sanity

From `checkpoints/S3a0_coupled_pose_head_smoke_audit.md`:

- baseline vs coupled `t_mag` max abs diff: `0.000000000000`
- `tmag_before_coupled` vs `tmag_after_coupled` max abs diff: `0.000000000000`

Conclusion:

- the coupled residual head does not alter `tmag`
- the forward-path `tmag` preservation requirement is satisfied

## 8. Smoke Training Result

Smoke run:

- script: `scripts/run_s3a0_coupled_pose_residual_head_smoke.sh`
- updates attempted: `30`
- `bad_forward=0`
- `skip_updates=0`
- `total_updates=30`

Numerical result:

- no NaN-like failures after fixing the `so(3)` small-angle exponential map
- coupled-only optimizer path trains stably for the smoke window

Latest smoke eval metrics:

- `ATE = 8.738240317016485`
- `drift = 7.3170828428151955`
- `path_ratio = 0.28612668773627314`

Interpretation:

- Wiring and optimization are valid.
- `tmag` preservation inside the model is valid.
- The smoke run does **not** preserve the S2b clean trajectory-shape behavior.

## 9. Recommendation on Next Step

Recommendation:

- Do **not** enter S3a0 train-CV small training yet.

Reason:

- Although the coupled head is integrated and trainable in isolation, the smoke run does not preserve clean `path_ratio`.
- Current evidence says implementation is technically ready, but experiment framing is not yet faithful enough to the S2b clean candidate.
- The likely next fix is to align smoke / train-time evaluation with the actual S2b clean policy path more faithfully before any train-CV sweep.

## 10. Current Mainline Status

- S2b remains the current clean candidate.
- S3a0 implementation is in place.
- S3a0 smoke audit passed.
- S3a0 smoke training passed numerically and optimizer-wise.
- S3a0 smoke training failed the trajectory-shape preservation requirement, so it is not ready to replace S2b.
