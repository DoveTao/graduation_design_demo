# Final Negative Results Summary

## S3a residual-head line
- Result: did not replace S2b/S5.
- Reason: no legally selected train-CV candidate passed hard-gate stability for final promotion.
- Practical meaning: residual-head infrastructure is usable, but current setting did not deliver clean generalization gains.

## S3b fine-token representation diagnostic
- Result: no stable residual-pose signal sufficient for immediate clean replacement path.
- Evidence: probe generalization instability and DT/K-dominated explanatory pattern in the final diagnostic report.
- Practical meaning: fine token features may still help in redesigned settings, but not validated as a stable current mainline.

## S8 fine/spherical token reliability router
- Result: diagnostic negative (`TOKEN-NO-ADDED-VALUE`).
- Evidence: token-only and token-plus-regime probes both failed to beat the simpler regime-only reliability baseline built from `pred_tmag`, `dt`, and `k`.
- Practical meaning: under the current representation, fine/spherical token features do not add deployable reliability signal beyond regime features, so S8c is not justified.

## S9 regime-only reliability router
- Result: diagnostic negative (`NO-STABLE-REGIME-ROUTER-GAIN`).
- Evidence: the best diagnostic candidate was `R3_dt1_k20_fallback`, but no candidate passed the train-CV clean gate. The best diagnostic line still had mean CV `ATE=6.647316`, `drift=1.363353`, `path_ratio=0.994282`, with path_ratio outside the safe clean range.
- Practical meaning: regime-only routing may expose bucket-level diagnostic signal, but it does not currently provide a clean deployable replacement for S5.

## S10 chain-level path-ratio preserving smoother
- Result: diagnostic negative (`NO-STABLE-CHAIN-SMOOTHER-GAIN`).
- Evidence: chain-level tmag smoothing and preserve-sum renormalization did not produce a candidate that safely improved the clean CV line enough to replace S5.
- Practical meaning: the remaining post-S5 improvement space is not easily unlocked by lightweight chain-level smoothing alone.

## S11 tmag scale consistency training
- Result: diagnostic negative (`NO-STABLE-TMAG-CONSISTENCY-GAIN`).
- Evidence: in lightweight two-fold CV, `C_loss_only_consistency` and `D_high_regime_weighted_consistency` slightly improved tmag/speed/chain-sum proxy metrics and slightly improved fold ATE/drift relative to `A_tmag_head_only_baseline`, but no stable path_ratio-supported clean evidence was obtained.
- Practical meaning: training-time tmag consistency may provide weak proxy signal, but it still does not justify replacing S5 or promoting S11b full clean CV under the final project constraints.

## Fine_tdir sweep
- Result: no clean gain.
- Evidence: S2d `NO-TDIR-GAIN` conclusion; increasing fine_tdir harmed or failed to improve drift/ATE under clean constraints.
- Practical meaning: current final line keeps `fine_tdir=0.0`.

## DT-aware rot diagnostic
- Result: diagnostic signals existed (S2f), but clean train-CV selection failed to convert them into a superior final candidate (S2g failed).
- Practical meaning: keep S2b global fine_rot policy as the robust clean refinement.

## Oracle tmag diagnostic
- Result: better raw metrics can be observed in oracle-style diagnostic.
- Limitation: not deployable and carries path_ratio risk (`0.898234`), plus oracle variable usage is disallowed for inference-time policy.
- Practical meaning: use only as an upper-bound diagnostic reference in thesis discussion.

## Final negative-results takeaway
Post-S5 optimization attempts did not produce a new deployable replacement. S8 showed that token reliability probing does not outperform regime-only features, S9 showed that regime-only routing does not stably pass the clean CV gate, S10 showed that chain-level smoothing does not recover a safe clean gain, and S11 showed only weak proxy improvement without path_ratio-supported clean evidence. These negative findings justify why the final candidate remains S5 clean calibration and why the thesis should emphasize reliability, reproducibility, and marginal gain rather than aggressive post-lockdown optimization claims.
