# S3a Coupled Pose Head Design Plan

## 1. Motivation

S2 has converged to a clean candidate: S2b train-CV selected S1d5 dt-anchor + `fine_rot=0.45` policy. Its behavior is already healthy on scale and trajectory length, so the next step should not be another fusion-strength sweep.

Current clean candidate:

- S2b metrics: `drift=1.327402`, `ATE=7.352371`, `path_ratio=0.934984`
- S1d5 baseline: `drift=1.396358`, `ATE=7.632463`, `path_ratio=0.934982`

What S2 already told us:

- `fine_rot=0.45` gives a robust clean gain.
- `fine_tdir` fusion did not help in the current fusion family.
- dt-aware / global-rot diagnostics did not generalize cleanly under train-CV.
- S2b remains the current clean fine-rot candidate.
- More policy sweep is low value.

The key remaining signal is from the S2c2 oracle:

- `oracle_R ATE=10.590301`
- `oracle_tdir ATE=7.658319`
- `oracle_R_tdir ATE=0.911024`
- `oracle_tmag ATE=7.569829`

This is a strong coupled-error diagnosis. Replacing rotation alone or translation direction alone is not enough, but replacing them together is extremely powerful. Therefore S3 should move toward coupled pose modeling or better fine-token representation, not more policy tuning.

## 2. Current Bottleneck Summary

### S2b Status

- `drift=1.327402`
- `ATE=7.352371`
- `path_ratio=0.934984`

Interpretation:

- Path-length behavior is healthy enough that scale is not the main blocker.
- The remaining error is mostly in pose shape / direction quality after scale is already stabilized by the S1d5/S2b dt-anchor policy.

### R / tdir Oracle Diagnostic

- `oracle_R ATE=10.590301`
- `oracle_tdir ATE=7.658319`
- `oracle_R_tdir ATE=0.911024`

Interpretation:

- Independent rotation or tdir fixes are not sufficient.
- Independent fine fusion strengths are likely saturating.
- The remaining error likely lives in the joint structure between `R` and `tdir`, or in the fine token representation feeding them.
- This supports a controlled coupled residual design before broader representation surgery.

## 3. Candidate S3 Directions

### Option S3a: Coupled Fine Pose Residual Head

Idea:

- Predict a small joint residual for rotation and translation direction from shared fine features.
- Treat S2b as the stable base prediction and learn only a tiny corrective layer.

Constraints:

- Freeze coarse backbone.
- Freeze S1d5/S2b dt-anchor scale policy.
- Do not train `tmag`.
- Only train a small coupled head.

Outputs:

- `delta_R_residual`
- `delta_tdir_residual`
- optional residual confidence / gate

Loss:

- joint pose loss with `R` and `tdir` consistency
- small residual regularization
- optional low-weight trajectory-consistency term

Why it matches the diagnostic:

- It directly targets the observed coupled `R+tdir` gap while keeping the healthy S2b scale path fixed.

Risk:

- Can over-correct and hurt `path_ratio` indirectly through direction shape.
- Can overfit train-CV if the residual head is too expressive.

### Option S3b: Frame-Consistent tdir Head

Idea:

- Re-derive fine `tdir` in a clearly defined frame, such as A-local or output/B frame, and enforce consistency with `R`.

Motivation:

- Current `fine_tdir` fusion is harmful, which may indicate frame mismatch, target mismatch, or an incompatible supervision path rather than a lack of tdir capacity itself.

Loss:

- local-frame tdir loss
- `R`-conditioned tdir loss
- sign-invariant `tdir_abs` if needed

Why it matters:

- It is the best focused diagnostic if S3a still shows tdir instability.

Risk:

- Easy to make silent frame-convention mistakes.
- Cleaner as a diagnostic branch than as the first mainline experiment.

### Option S3c: Coupled SE(3) Local Refinement Head

Idea:

- Predict a small Lie-algebra-like residual `xi = [omega, v_dir]` or a constrained SE(3)-style local residual.

Constraint:

- `tmag` remains from S2b dt-anchor.

Goal:

- Refine local trajectory shape while keeping overall scale policy unchanged.

Why it is interesting:

- Gives a more principled joint parameterization than separate `R` and `tdir` heads.

Risk:

- Higher implementation complexity.
- Harder to debug than a simpler residual head.
- More thesis overhead for the first S3 experiment.

### Option S3d: Better Fine Token Representation / Cross-Attention

Idea:

- Improve the fine-token branch before adding a new head.

Possible methods:

- stronger local geometric token-pair features
- epipolar-aware fine attention
- relative spherical coordinate features
- token confidence / entropy features

Why it may help:

- The current bottleneck may be representational, not just head-level.

Risk:

- Too many moving parts at once.
- Harder to attribute gains.
- Less controlled than first validating whether a tiny coupled head already captures the oracle gap.

### Option S3e: Train-CV Teacher Distillation from S2b / Oracle Diagnostics

Idea:

- Use train-only oracle or pseudo-target diagnostics to teach a coupled head.

Must avoid:

- any test-label leakage
- any oracle built from test GT during model selection or target construction

Why it may help:

- Can shape the residual head toward the coupled error mode highlighted by the oracle.

Risk:

- Easy to become invalid if oracle targets use test labels.
- More reporting burden to prove clean train-only construction.

## 4. Recommendation

### Ranking

1. `S3a0_minimal_coupled_pose_residual_head`
2. `S3b` frame-consistent tdir diagnostic
3. `S3c` constrained SE(3) local refinement
4. `S3d` representation upgrade
5. `S3e` teacher distillation

### Rationale

Expected benefit:

- S3a best matches the oracle evidence with the least modeling jump.

Implementation risk:

- S3a is lower risk than S3c/S3d because it can reuse the existing fine pooled features and coarse-to-fine wiring.

Likelihood to preserve S2b `path_ratio`:

- S3a is favorable because `tmag` stays unchanged and the residual can be gated small.

Cleanliness for thesis / report:

- S3a is the cleanest controlled experiment: same S2b base policy, same scale path, only a tiny coupled residual added.

Recommendation:

- First implement `S3a0_minimal_coupled_pose_residual_head` as a controlled experiment.
- Keep `S3b` as the immediate diagnostic fallback if the tdir frame issue still appears.
- Do not jump directly to large representation changes before checking whether a small coupled residual already unlocks the main gain.

## 5. Minimal Implementation Plan: `S3a0_minimal_coupled_pose_residual_head`

### Name

`S3a0_minimal_coupled_pose_residual_head`

### Goal

Learn a tiny coupled residual head on top of S2b, targeting `R/tdir` consistency while preserving S2b scale.

### Freeze Policy

Freeze:

- encoder / `module2` unless absolutely needed
- coarse module
- existing fine heads
- `tmag` head
- S1d5/S2b dt-anchor policy

Trainable:

- only new coupled residual head
- optionally one small scalar gate or one tiny per-branch gate

Recommended strict freeze interpretation for S3a0:

- Do not update `module2.patch_embed_f`, `module2.enc_f`, `fine.pose_head`, `fine.t_head`, `fine.mag_head`, `coarse.*`, `direct_head`, depth modules, or tmag calibration params.
- Only allow names under a new dedicated prefix such as `coupled_pose_head.*` plus optional `coupled_pose_gate`.

### Inputs

Primary inputs:

- fine pooled features from current fine stage
- coarse / fine matching confidence
- existing `pred_R` / `pred_tdir` if available
- optional dt bucket embedding

Must not use:

- GT labels as model input

Minimal input recommendation:

- pooled `Ff_t` or pooled fine feature summary
- `token_weight_f`
- coarse prediction summary derived from `Rc` and `tc_dir`

### Outputs

- `delta_R_residual`
- `delta_tdir_residual`
- residual confidence / gate

Minimal parameterization recommendation:

- rotation residual as tiny 6D residual mapped near identity
- tdir residual as 3D additive local-frame delta, then renormalize
- scalar gate constrained to a small interval, for example via `sigmoid * max_strength`

### Application

- `R_final = compose(delta_R, S2b_R)`
- `tdir_final = normalize(tdir_residual_update(S2b_tdir, delta_tdir))`
- `tmag_final = S2b_tmag` unchanged

Expected behavior:

- `path_ratio` should stay near S2b because `tmag` is unchanged
- any gain should come from better coupled local pose shape, not scale manipulation

Recommended concrete composition:

- Build residual on top of coarse-to-fine S2b outputs, not on top of coarse-only outputs.
- Use local-frame residual application for `tdir`, then let existing output-frame conversion remain unchanged.

### Loss

Required terms:

- rotation geodesic loss
- `tdir_abs` or local-frame tdir loss
- joint consistency loss between `R` and `tdir` frame
- small residual regularization

Optional term:

- low-weight train-chain trajectory ATE / shape loss after core behavior is stable

Recommended first-pass weighting:

- keep the existing pose loss as the main supervision
- add small explicit residual penalties to keep updates near identity / zero
- keep trajectory-consistency off or very low for the first pass

Suggested loss decomposition:

- `L_rot_res = geodesic(R_final, R_gt)`
- `L_tdir_res = translation_direction_loss(tdir_final, t_gt, R_gt, pred_t_frame=A or chosen frame)`
- `L_joint_consistency = translation_direction_loss(R_final @ tdir_local_final, t_gt, R_gt, pred_t_frame=B)` or an equivalent frame-consistency construction
- `L_res_reg = ||logmap(delta_R)|| + ||delta_tdir||`

### Training Protocol

- train split only
- train-CV selection
- no test labels for hyperparameter selection
- start with very small updates
- early stop by train-CV `ATE` / `drift` / `path_ratio`
- final test only once

Practical training protocol:

- initialize residual head near zero-update / identity-update
- use lower LR than normal fine training if needed
- audit optimizer param list before the first training run
- save explicit freeze summary and trainable param count into final summary

### Success Criteria

- `path_ratio >= 0.90`
- `ATE < 7.352371`
- `drift <= 1.35`
- `explicit-cfg / unexpected=0`
- trainable params remain small and audited
- no `tmag` degradation

### Failure Criteria

- `path_ratio < 0.90`
- ATE worsens vs S2b
- optimizer includes frozen backbone params
- coupled head clearly overfits train-CV

### Why S3a0 Is the Right First Experiment

- It is the smallest change that directly targets the strongest diagnostic signal.
- It preserves the validated S2b scale behavior.
- It is easy to rollback.
- It gives a clean thesis narrative: S2 tuned fusion strengths, S3 tests whether a tiny coupled residual can exploit the `R+tdir` oracle gap.

## 6. Code Impact Audit

This section is read-only planning only. No source changes are made in this task.

### `config.py`

Role:

- Add new config flags and loss weights for S3a0.

Current relevant hooks:

- fine-stage toggles at [config.py](/home/dovetao/graduation_design_demo/config.py:88)
- loss and translation controls at [config.py](/home/dovetao/graduation_design_demo/config.py:159)

Planned additions:

- enable flag for coupled residual head
- hidden dimension / head width
- max residual strength or init scale
- use gate flag
- residual regularization weights
- joint consistency loss weight
- optional frame-selection flag for coupled tdir supervision
- dedicated freeze mode name or boolean

### `interaction.py`

Role:

- Most natural place to add the new module class because fine-stage heads already live here.

Current relevant hooks:

- fine head construction at [interaction.py](/home/dovetao/graduation_design_demo/interaction.py:748)
- fine feature outputs and token confidence at [interaction.py](/home/dovetao/graduation_design_demo/interaction.py:908)
- fine outputs dictionary at [interaction.py](/home/dovetao/graduation_design_demo/interaction.py:960)

Planned additions:

- new small module class, e.g. `CoupledPoseResidualHead`
- pooled feature path from `Ff_t` plus confidence pooling
- outputs for `delta_R_residual`, `delta_tdir_residual`, `coupled_gate`
- optional helper to compose residual with base `R` and base local `tdir`

Why here:

- It already owns fine pooled features, `token_weight_f`, and current fine head outputs.

### `model.py`

Role:

- Instantiate the coupled head and compose it with S2b outputs in the forward pass.

Current relevant hooks:

- model assembly at [model.py](/home/dovetao/graduation_design_demo/model.py:273)
- coarse-to-fine fusion path at [model.py](/home/dovetao/graduation_design_demo/model.py:446)
- final output writeback at [model.py](/home/dovetao/graduation_design_demo/model.py:463)

Forward outputs that likely need extension:

- `aux["R_coupled_final"]` or reuse final `R`
- `aux["t_dir_coupled_final"]` or reuse final `t_dir`
- `aux["delta_R_residual"]`
- `aux["delta_tdir_residual"]`
- `aux["coupled_pose_gate"]`
- optionally `aux["t_dir_base_s2b"]` and `aux["R_base_s2b"]` for audit

Compatibility note:

- Keep existing `t_mag` behavior unchanged.
- Keep the existing S2b blend path available when the new config is off.

### `train_mvp.py`

Role:

- Add freeze mode, optimizer audit, new loss hookup, and summary logging.

Current relevant hooks:

- freeze policy at [train_mvp.py](/home/dovetao/graduation_design_demo/train_mvp.py:72)
- optimizer param filtering at [train_mvp.py](/home/dovetao/graduation_design_demo/train_mvp.py:3530)
- main pose loss at [train_mvp.py](/home/dovetao/graduation_design_demo/train_mvp.py:3653)
- final weighted objective at [train_mvp.py](/home/dovetao/graduation_design_demo/train_mvp.py:4366)

Planned changes:

- add a stricter freeze function or extend `_set_train_fine_only` with a new mode for coupled-head-only training
- print / save named trainable parameter audit
- attach coupled residual losses after current pose losses
- save coupled-head config and trainable param count into final summaries

Freeze / optimizer audit to do:

- verify only the new head and optional gate have `requires_grad=True`
- assert optimizer param names match the allowed prefixes
- fail early if backbone or old fine heads appear in the optimizer

### `losses.py`

Role:

- Best place to add reusable coupled residual regularization and joint consistency helpers.

Current relevant hooks:

- `pose_loss` at [losses.py](/home/dovetao/graduation_design_demo/losses.py:55)
- `translation_direction_loss` at [losses.py](/home/dovetao/graduation_design_demo/losses.py:99)

Planned additions:

- small residual regularization helper
- optional joint `R/tdir` consistency helper if we want explicit reusable logic

Loss attachment point:

- call from `train_mvp.py`, not from inside the model
- keep the current loss API style: model returns tensors in `aux`, training script computes losses

### `tools/eval_clean_policy.py`

Role:

- Preserve compatibility with the existing S2b frozen clean policy evaluation.

Current relevant hooks:

- fixed fine-stage restore logic at [tools/eval_clean_policy.py](/home/dovetao/graduation_design_demo/tools/eval_clean_policy.py:161)
- summary payload at [tools/eval_clean_policy.py](/home/dovetao/graduation_design_demo/tools/eval_clean_policy.py:283)

Compatibility plan:

- default S2b path must remain unchanged when coupled residual head is disabled
- if a future S3 config adds new keys, loading old S2b checkpoints must still yield `unexpected=0` or benign `missing` only under explicit compatibility logic
- do not change this tool now unless future S3 evaluation needs extra reporting fields

### `scripts/eval_s2b_clean_policy.sh`

Role:

- Keep as the exact S2b reproducibility entrypoint.

Current relevant hook:

- fixed S2b policy path at [scripts/eval_s2b_clean_policy.sh](/home/dovetao/graduation_design_demo/scripts/eval_s2b_clean_policy.sh:4)

Compatibility plan:

- no policy changes
- no script behavior changes for S2b
- future S3 evaluation should use a separate script, not overwrite this one

## 7. Minimal Agent-Ready Execution Order

1. Add S3a0 config fields with safe defaults that keep all current behavior off.
2. Add a tiny `CoupledPoseResidualHead` in `interaction.py`.
3. Instantiate it in `model.py` behind a config gate.
4. Use existing fine pooled features and confidence to produce residual outputs.
5. Compose residuals with the existing S2b pose path while leaving `tmag` untouched.
6. Expose residual tensors and gates in `aux`.
7. Add coupled residual loss terms in `train_mvp.py`.
8. Add strict freeze / optimizer audit for head-only training.
9. Add summary logging for trainable parameter count and unexpected keys.
10. Run eval-only sanity before any real training launch.

## 8. Final Recommendation

Recommended first S3 experiment:

- `S3a0_minimal_coupled_pose_residual_head`

Recommended implementation stance:

- Treat it as a controlled residual-on-S2b experiment.
- Preserve `tmag` and the current clean dt-anchor policy exactly.
- Keep S3b ready as the next diagnostic if the residual head exposes a frame-consistency problem rather than solving it.
