# S3a0b Coupled Head Failure Attribution Report

## 1. S3a0 Smoke Failure Summary

- S2b clean candidate: `drift=1.327402`, `ATE=7.352371`, `path_ratio=0.934984`
- S3a0 smoke: `drift=7.317083`, `ATE=8.738240`, `path_ratio=0.286127`
- smoke audit pass: `True`
- `tmag_before/after` max diff: `0.000000`
- optimizer only coupled head: `True`

## 2. Artifact Availability

- smoke checkpoint directory exists locally: `False`
- smoke final checkpoint exists locally: `False`
- implication: trained residual branch replay is not fully available; full logged smoke metrics can be read, but trained rot-only / tdir-only replay cannot be executed exactly

## 3. Extracted Signals

- gate init: `-4.0`
- coupled gate mean from audit: `0.017986`
- coupled delta rot norm from audit: `0.000000`
- coupled delta tdir norm from audit: `0.000000`
- final trained gate mean from persisted artifacts: `NA`
- final trained coupled delta norms from persisted artifacts: `NA`
- train losses over updates from persisted artifacts: `NA`
- optimizer params: `119463` trainable vs `119463` trainable count

## 4. Residual Source Attribution

- raw baseline without S2b policy scaling evaluates to: `drift=1.326834`, `ATE=7.522779`, `path_ratio=0.496344`
- S2b policy baseline evaluates to: `drift=1.327402`, `ATE=7.352371`, `path_ratio=0.934984`
- raw-baseline vs smoke-full gap: `drift=5.990248`, `ATE=1.215461`, `path_ratio=0.210217`
- interpretation: the smoke failure is already reproduced by the raw base checkpoint path, before any trained coupled residual replay is required
- gate-zero proxy matches the raw baseline path, which means the coupled-head plumbing itself is not what creates the huge path-ratio drop

## 5. Variant Table

| variant | status | drift | ATE | path_ratio | RPE_rot | RPE_tdir | RPE_tmag | rot | tdir_abs | tdir_local_A_abs | tmag_p10 | tmag_p50 | tmag_p90 | tmag_diff | d_rot_norm | d_tdir_norm | gate_mean | k | pairs | chains | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S2b_policy_baseline | ok | 1.327402 | 7.352371 | 0.934984 | 20.715353 | 72.349483 | 0.073926 | 21.481086 | 29.944641 | 23.152590 | 0.127012 | 0.183018 | 0.402103 | 0.000000 | NA | NA | NA | 1 | 132 | 19 | official clean candidate with bucket-scaled dt-anchor policy |
| baseline_raw_no_policy | ok | 1.326834 | 7.522779 | 0.496344 | 20.715353 | 72.349483 | 0.122433 | 21.095084 | 38.412443 | 30.558884 | 0.123481 | 0.151557 | 0.156023 | 0.000000 | NA | NA | NA | 1 | 132 | 19 | raw base checkpoint, no S2b bucket scaling policy; load_missing=14, load_unexpected=0 |
| S3a0_smoke_full_logged | logged_only | 7.317083 | 8.738240 | 0.286127 | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0.000000 | 0.000000 | 0.000000 | 0.017986 | NA | NA | NA | smoke checkpoint removed locally; using persisted markdown metrics only |
| rot_only_residual | unavailable | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | smoke checkpoint not present locally after cleanup; cannot replay trained residual parameters |
| tdir_only_residual | unavailable | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | smoke checkpoint not present locally after cleanup; cannot replay trained residual parameters |
| gate_zero_proxy | ok | 1.326834 | 7.522779 | 0.496344 | 20.715353 | 72.349483 | 0.122433 | 21.095084 | 38.412443 | 30.558884 | 0.123541 | 0.151557 | 0.156094 | 0.000000 | NA | NA | 0.000000 | 1 | 132 | 19 | proxy gate-zero wrapper over current zero-init head; load_missing=14, load_unexpected=0 |
| residual_scale_0.0 | ok | 1.326834 | 7.522779 | 0.496344 | 20.715353 | 72.349483 | 0.122433 | 21.095084 | 38.412443 | 30.558884 | 0.123541 | 0.151557 | 0.156094 | 0.000000 | NA | NA | 0.000000 | 1 | 132 | 19 | equivalent to gate-zero proxy under current artifact availability |
| residual_scale_0.1 | unavailable | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | smoke checkpoint not present locally after cleanup; cannot replay trained residual parameters |
| residual_scale_0.25 | unavailable | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | smoke checkpoint not present locally after cleanup; cannot replay trained residual parameters |
| residual_scale_0.5 | unavailable | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | smoke checkpoint not present locally after cleanup; cannot replay trained residual parameters |
| residual_scale_1.0 | logged_only | 7.317083 | 8.738240 | 0.286127 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0.017986 | NA | NA | NA | same as S3a0_smoke_full_logged |

## 6. Loss / Gate / Norm Analysis

- pairwise smoke training was numerically stable, but the persisted artifacts do not include the per-update loss trace
- audit-time residual norms were exactly zero and gate mean was only about `0.018`, so the architecture was initialized in a near-identity regime
- because the logged smoke metrics are nearly identical to the freshly recomputed raw-baseline metrics, the dominant collapse cannot currently be attributed to a large trained residual norm or a rapidly opened gate
- the stronger available signal is `RPE_trans_mag`: S2b clean policy is `0.0739`, while the raw baseline path is around `0.7139`; that is consistent with path-length collapse driven by wrong magnitude calibration in odometry composition
- therefore pairwise pose loss and coupled residual loss are not sufficient to protect chain geometry when the evaluation path omits the clean magnitude policy

## 7. tmag Invariance Confirmation

- inside the coupled head forward path, `tmag_before_coupled` and `tmag_after_coupled` match exactly
- this does not mean the smoke run matched S2b scale behavior; it only means the coupled residual did not change the raw model's magnitude output

## 8. Why Path Ratio Can Collapse Despite Unchanged tmag

- `unchanged tmag` in S3a0 means unchanged relative to the raw base checkpoint output, not unchanged relative to the S2b clean policy trajectory
- S2b clean candidate depends on bucket-scaled dt-anchor magnitude policy from `checkpoints/S2b_clean_fine_rot_policy.json`
- S3a0 smoke used the raw base checkpoint path inside `train_mvp.py` evaluation, which does not apply those bucket factors
- once odometry composition uses the raw, under-corrected magnitude path, `RPE_trans_mag` blows up and `path_ratio` collapses even if the coupled head preserves its own before/after `tmag` exactly

## 9. Candidate Fixes

### A. S3a1_rot_only_residual_head
- safest next modeling probe after restoring the S2b policy path
- only allow `delta_R`, keep `delta_tdir=0`

### B. S3a2_gate_clamped_coupled_head
- clamp gate or residual scale hard
- useful only after the evaluation path is made S2b-equivalent

### C. S3a3_chain_safe_loss
- add explicit path-ratio / chain-shape preservation loss
- this addresses the known pairwise-vs-chain mismatch

### D. S3a4_tdir_frame_retarget
- revisit tdir frame before enabling `delta_tdir` again
- especially relevant once trained residual replay becomes available

### E. Abort S3a0 current coupled design
- not recommended yet, because available evidence does not show the trained coupled residual as the primary collapse source

## 10. Recommended Next Experiment

- recommended next experiment: `S3a1_rot_only_residual_head after first restoring the exact S2b clean policy path inside smoke/train evaluation`
- before any new smoke or train-CV run, make the smoke/eval path use the exact S2b clean magnitude policy or an equivalent integrated wrapper

## 11. Final Verdict

- final verdict: `INSUFFICIENT-ARTIFACTS`
- path_ratio collapse root cause: The dominant available cause is evaluation-path mismatch: S3a0 smoke used the raw base checkpoint without the S2b clean dt-anchor bucket scaling policy, so `tmag` was preserved by the coupled head but preserved at the wrong raw scale.
- secondary note: pairwise losses still lack direct chain-geometry protection, so `S3a3_chain_safe_loss` remains a relevant follow-up once the baseline path is corrected

