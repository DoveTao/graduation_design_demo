# Final Project Mainline Summary

## Final mainline
S1d5 scale repair -> S2b clean fine-rot policy -> S3a/S3b negative or diagnostic exploration -> S4 TMAG-REGIME-DOMINANT diagnostic -> S5 clean pred_tmag regime-aware calibration -> S6/S7 lockdown + reproducibility -> S8/S9 post-lockdown optimization attempts, both negative -> final candidate remains S5.

## Stage-by-stage contribution
1. S1d5 fixed the core scale/path-ratio collapse under corrected evaluation scope. Relative to the true raw multiscale baseline (drift=1.494, ATE=9.644, path_ratio=0.496), S1d5 reached drift=1.396358, ATE=7.632463, path_ratio=0.934982, establishing the first clean and deployable trajectory-scale repair.
2. S2b provided the main clean refinement on top of S1d5 using train-CV selected fine_rot=0.45 (fine_tdir=0, fine_tmag=0): drift=1.327402, ATE=7.352371, path_ratio=0.934984.
3. S3a/S3b did not replace S2b/S5 because they did not produce stable clean gains under legal train-CV selection gates. S3a residual-head line had no legally selected final candidate; S3b showed weak fine-token residual signal generalization.
4. S4 localized remaining error mass to magnitude-regime structure, with pred_tmag/gt_tmag regime effects stronger than simple dt or k-only explanations, motivating calibration rather than extra head training.
5. S5 validated a clean inference-time pred_tmag regime-aware calibration candidate (`highpred_q90_mid0p95_high0p80`) selected by train-CV and fit by train-only quantiles, yielding drift=1.327343, ATE=7.352288, path_ratio=0.932379.
6. S6/S7 locked this candidate through reproduction, leakage audit, and current-architecture reproducibility checks, confirming that `S5_clean_tmag_calibration_policy` is still the valid deployable clean mainline.
7. S8 tested whether fine/spherical token features could be reused as a reliability estimator rather than a residual pose regressor. The result was negative: token-only and token-plus-regime probes both failed to beat a simple regime-only baseline built from `pred_tmag`, `dt`, and `k`, so no S8c follow-up was recommended.
8. S9 tested a post-S5 regime-only reliability router using only deployable inference-visible regime features. Although some diagnostic candidates slightly improved specific high-risk buckets, no candidate passed the train-CV clean gate because path_ratio stayed outside the safe clean range, so no final test was promoted and S5 was not replaced.

## Final status
- Final clean candidate: `checkpoints/S5_clean_tmag_calibration_policy.json`
- Predecessor: `checkpoints/S2b_clean_fine_rot_policy.json`
- Post-lockdown optimization attempts: `S8 = TOKEN-NO-ADDED-VALUE`, `S9 = NO-STABLE-REGIME-ROUTER-GAIN`
- Final message for thesis: S5 replaces S2b with a clean but marginal gain; later token-reliability and regime-router optimization attempts did not yield a new clean deployable replacement.
