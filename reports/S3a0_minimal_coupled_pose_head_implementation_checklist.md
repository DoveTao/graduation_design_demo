# S3a0 Minimal Coupled Pose Head Implementation Checklist

## Goal

Implement `S3a0_minimal_coupled_pose_residual_head` as a controlled S2b-top residual experiment without changing S1d5/S2b policy and without touching `tmag`.

## Checklist

1. Add config fields

- Add a master enable flag such as `use_coupled_pose_residual_head=False`.
- Add head-width / hidden-dim fields.
- Add gate enable and max-strength fields.
- Add residual regularization weights.
- Add joint consistency loss weight.
- Add a strict freeze mode flag for coupled-head-only training.
- Ensure defaults keep current behavior identical to S2b when disabled.

2. Add module class

- Create a tiny `CoupledPoseResidualHead` in `interaction.py`.
- Input should use pooled fine features and confidence.
- Output should include `delta_R_residual`, `delta_tdir_residual`, and optional `coupled_pose_gate`.
- Initialize the module near zero residual / identity residual.

3. Wire forward pass

- Instantiate the new head in `model.py` behind the config gate.
- Feed it the existing fine features, confidence, and optional base pose summary.
- Compose residuals on top of the current S2b fine output.
- Keep `tmag_final = S2b_tmag` unchanged.
- Export audit tensors in `aux`.

4. Add loss terms

- Keep current pose loss as the main supervision.
- Add rotation residual supervision on final `R`.
- Add local-frame or `tdir_abs` supervision on final `tdir`.
- Add explicit `R/tdir` joint consistency loss.
- Add small residual regularization on residual magnitude.
- Keep any trajectory-shape term off or very low for the first pass.

5. Add freeze mode

- Extend `train_mvp.py` freeze handling so only the new coupled head and optional gate are trainable.
- Freeze `module2`, `coarse`, current `fine` heads, depth modules, and all `tmag` parameters.
- Fail early if no parameters remain trainable.

6. Add optimizer audit

- Print trainable parameter names and counts before optimizer creation.
- Assert optimizer params only come from the allowed coupled-head prefixes.
- Save the freeze summary into checkpoint / final summary metadata.

7. Add eval-only sanity

- Run eval-only loading with the new config disabled and confirm S2b compatibility remains unchanged.
- Run eval-only loading with the new head enabled but untrained and confirm forward pass works.
- Confirm `unexpected=0` for intended load path or document any allowed compatibility exception.
- Confirm `t_mag` and `t_vec` stay sourced from S2b path.

8. Add train-CV script

- Add a separate S3a0 launch script instead of modifying `scripts/eval_s2b_clean_policy.sh`.
- Keep train-CV selection on train split only.
- Use conservative LR and explicit run naming.
- Do not launch test-selection logic during hyperparameter search.

9. Add report template

- Prepare a small markdown template for each S3a0 run.
- Record config, trainable param count, freeze audit, train-CV metrics, final test metrics, and failure notes.
- Include explicit checks for `path_ratio`, `ATE`, `drift`, and `unexpected`.

10. Add rollback criteria

- Stop if `path_ratio < 0.90`.
- Stop if `ATE >= 7.352371`.
- Stop if optimizer includes frozen backbone params.
- Stop if train-CV improves but test degrades in a clear overfit pattern.
- Roll back to S2b if the coupled head cannot beat baseline without destabilizing path shape.

## Acceptance Targets

- `path_ratio >= 0.90`
- `ATE < 7.352371`
- `drift <= 1.35`
- `tmag` behavior unchanged from S2b policy
- trainable params remain small and explicitly audited
- no training or selection leakage from test labels
