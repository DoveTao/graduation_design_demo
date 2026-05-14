# S3a Residual Head Summary

## A. Executive Summary

- `S3a` explored coupled / residual pose-head training after `S2b`.
- `S3a0` code path was implemented.
- `S3a0b` showed the initial smoke failure was confounded by missing `S2b` magnitude policy alignment.
- `S3a0c` fixed the policy-path alignment and restored the correct `S2b` baseline behavior under explicit policy loading.
- `S3a1` rot-only residual smoke passed with safe invariance and optimizer isolation.
- `S3a1` train-CV small run failed legal selection because every candidate was blocked by the hard gate on held-out `scene01/seq02` drift.
- Therefore `S2b` remains the current clean candidate.

## B. Timeline

1. `S3a` design:
   residual / coupled pose-head line proposed as a post-`S2b` extension.
2. `S3a0` implementation:
   coupled residual head, losses, freeze controls, and smoke scripts were added.
3. `S3a0b` failure attribution:
   the large smoke collapse was traced to evaluation-path mismatch, not yet to trained residual behavior itself.
4. `S3a0c` policy alignment fix:
   smoke / eval path was corrected to explicitly load `checkpoints/S2b_clean_fine_rot_policy.json`.
5. `S3a1` rot-only smoke:
   rot-only residual head preserved `tdir/tmag` and stayed numerically close to `S2b`.
6. `S3a1` train-CV small run:
   small grid finished, but no candidate satisfied the legal selection gates.

## C. Results Table

| item | drift | ATE | path_ratio | extra | status |
| --- | ---: | ---: | ---: | --- | --- |
| `S2b` baseline | 1.327402 | 7.352371 | 0.934984 | current clean candidate | valid |
| `S3a1` smoke | 1.328421 | 7.356588 | 0.935035 | `tdir/tmag unchanged`, optimizer only rot residual/gate | smoke pass |
| `S3a1` closest invalid CV candidate | mean CV drift `1.363408` | mean CV ATE `6.652671` | mean CV path_ratio `1.005759` | `A_lr5e-05_upd100_rot0.01_gate0.02`; held-out `seq02 drift = 1.608224` | invalid by hard gate |
| `S3a1` final test | N/A | N/A | N/A | no legally selected config | not run |

## D. Interpretation

- Residual-head infrastructure is safe.
- Rot-only residual can preserve `tdir/tmag` in smoke.
- However train-CV stability is not yet sufficient.
- Bigger training is not justified yet.
- `S2b` remains simpler and more reliable.

## E. Why Not Continue S3a1 Long Training Now

- No legally selected config exists from the small train-CV sweep.
- Held-out drift instability remains the blocking issue.
- The observed pattern is consistent with overfitting risk on the small residual head line.
- Current gains are not clean, because they do not survive the legal selection rule.

## F. Recommended Next Options

### Option 1

- Stop at `S2b` for final project delivery.
- Treat `S2b` as the current clean candidate and avoid expanding `S3a1`.

### Option 2

- `S3b_fine_token_representation_diagnostic`
- Inspect whether fine features contain stable pose-residual signal.
- Do no training first.

### Option 3

- `S3a2_gate_clamped_or_chain_safe_residual`
- Only consider this if there is a strong reason to continue residual-head training.
- It must include explicit path / chain safety loss and a stricter gate regime.

## G. Final Verdict

- `S3a` residual-head line is not ready to replace `S2b`.
- `S2b` remains the current clean candidate.
- Do not continue `S3a1` train-CV without redesign.

## Safety Audit Recap

- `tdir before/after diff = 0.0`
- `tmag before/after diff = 0.0`
- optimizer only:
  `coupled_pose_head.backbone.*`
  `coupled_pose_head.rot_head.*`
  `coupled_pose_head.gate_head.*`
- forbidden trainable params count = `0`

## Current Recommendation

- Do not continue `S3a1` larger training now.
- Do not promote `S3a1` to current clean candidate.
- Keep `S2b` as the clean mainline for current reporting and delivery.
