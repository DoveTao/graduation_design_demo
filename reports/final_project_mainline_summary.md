# Final Project Mainline Summary

## Final mainline
S1d5 scale repair -> S2b clean fine-rot policy -> S3a/S3b negative or diagnostic exploration -> S4 TMAG-REGIME-DOMINANT diagnostic -> S5 clean pred_tmag regime-aware calibration -> S6/S7 lockdown + reproducibility -> S8/S9 post-lockdown optimization attempts, both negative -> S10/S11/S12 later post-S5 optimization attempts, all negative -> S13 practical-gap quantification -> S14 local-window pose-graph diagnostic, still negative -> final candidate remains S5.

## Stage-by-stage contribution
1. S1d5 fixed the core scale/path-ratio collapse under corrected evaluation scope. Relative to the true raw multiscale baseline (drift=1.494, ATE=9.644, path_ratio=0.496), S1d5 reached drift=1.396358, ATE=7.632463, path_ratio=0.934982, establishing the first clean and deployable trajectory-scale repair.
2. S2b provided the main clean refinement on top of S1d5 using train-CV selected fine_rot=0.45 (fine_tdir=0, fine_tmag=0): drift=1.327402, ATE=7.352371, path_ratio=0.934984.
3. S3a/S3b did not replace S2b/S5 because they did not produce stable clean gains under legal train-CV selection gates. S3a residual-head line had no legally selected final candidate; S3b showed weak fine-token residual signal generalization.
4. S4 localized remaining error mass to magnitude-regime structure, with pred_tmag/gt_tmag regime effects stronger than simple dt or k-only explanations, motivating calibration rather than extra head training.
5. S5 validated a clean inference-time pred_tmag regime-aware calibration candidate (`highpred_q90_mid0p95_high0p80`) selected by train-CV and fit by train-only quantiles, yielding drift=1.327343, ATE=7.352288, path_ratio=0.932379.
6. S6/S7 locked this candidate through reproduction, leakage audit, and current-architecture reproducibility checks, confirming that `S5_clean_tmag_calibration_policy` is still the valid deployable clean mainline.
7. S8 tested whether fine/spherical token features could be reused as a reliability estimator rather than a residual pose regressor. The result was negative: token-only and token-plus-regime probes both failed to beat a simple regime-only baseline built from `pred_tmag`, `dt`, and `k`, so no S8c follow-up was recommended.
8. S9 tested a post-S5 regime-only reliability router using only deployable inference-visible regime features. Although some diagnostic candidates slightly improved specific high-risk buckets, no candidate passed the train-CV clean gate because path_ratio stayed outside the safe clean range, so no final test was promoted and S5 was not replaced.
9. S10 tested a chain-level path-ratio-preserving tmag smoother on top of S5. This line also ended negative: no candidate produced a stable clean gain, and the final classification was `NO-STABLE-CHAIN-SMOOTHER-GAIN`.
10. S11 moved the optimization target back into training-time tmag scale learning, using tmag-only fine-tuning plus consistency losses and train-only high-regime weighting. The lightweight two-fold CV showed weak proxy improvement and slight fold ATE/drift improvement for C/D over A, but no stable path_ratio-supported clean evidence, so the final classification remained `NO-STABLE-TMAG-CONSISTENCY-GAIN` and no S11b full clean CV was recommended.
11. S12 shifted to data-centric regime-balanced sampling. Although some samplers improved high-risk proxy metrics and slightly improved lightweight odometry summaries, they still failed to provide clean-eligible path_ratio evidence, so the final classification was `NO-STABLE-REGIME-SAMPLING-GAIN`.
12. S13 quantified the remaining practical usability gap and showed that S5 is still the best clean candidate but not practical-ready. The key conclusion was that the dominant bottleneck is `R-TDIR-COUPLED-LIMITED`, with chain accumulation as an important secondary factor.
13. S14 tested whether local-window multi-edge consistency could help once graph redundancy was confirmed to exist. Graph constraints were indeed sufficient, but the lightweight pose-graph formulation failed to produce a stable gain: the selected candidate `D_joint_w7_s1` worsened ATE/drift and expanded path_ratio, so the final classification was `NO-STABLE-POSE-GRAPH-GAIN`.

## Final status
- Final clean candidate: `checkpoints/S5_clean_tmag_calibration_policy.json`
- Predecessor: `checkpoints/S2b_clean_fine_rot_policy.json`
- Post-lockdown / post-S5 optimization attempts: `S8 = TOKEN-NO-ADDED-VALUE`, `S9 = NO-STABLE-REGIME-ROUTER-GAIN`, `S10 = NO-STABLE-CHAIN-SMOOTHER-GAIN`, `S11 = NO-STABLE-TMAG-CONSISTENCY-GAIN`, `S12 = NO-STABLE-REGIME-SAMPLING-GAIN`, `S14 = NO-STABLE-POSE-GRAPH-GAIN`
- Final message for thesis: S5 replaces S2b with a clean but marginal gain; later router, smoother, sampling, training-time tmag-consistency, and lightweight local-window pose-graph optimization attempts did not yield a new clean deployable replacement.
