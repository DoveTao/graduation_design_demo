# Final Project Mainline Summary

## Final mainline
S1d5 scale repair -> S2b clean fine-rot policy -> S3a/S3b negative or diagnostic exploration -> S4 TMAG-REGIME-DOMINANT diagnostic -> S5 clean pred_tmag regime-aware calibration -> S6 lockdown audit passed.

## Stage-by-stage contribution
1. S1d5 fixed the core scale/path-ratio collapse under corrected evaluation scope. Relative to the true raw multiscale baseline (drift=1.494, ATE=9.644, path_ratio=0.496), S1d5 reached drift=1.396358, ATE=7.632463, path_ratio=0.934982, establishing the first clean and deployable trajectory-scale repair.
2. S2b provided the main clean refinement on top of S1d5 using train-CV selected fine_rot=0.45 (fine_tdir=0, fine_tmag=0): drift=1.327402, ATE=7.352371, path_ratio=0.934984.
3. S3a/S3b did not replace S2b/S5 because they did not produce stable clean gains under legal train-CV selection gates. S3a residual-head line had no legally selected final candidate; S3b showed weak fine-token residual signal generalization.
4. S4 localized remaining error mass to magnitude-regime structure, with pred_tmag/gt_tmag regime effects stronger than simple dt or k-only explanations, motivating calibration rather than extra head training.
5. S5 validated a clean inference-time pred_tmag regime-aware calibration candidate (`highpred_q90_mid0p95_high0p80`) selected by train-CV and fit by train-only quantiles, yielding drift=1.327343, ATE=7.352288, path_ratio=0.932379.
6. S6 locked this candidate through reproduction and leakage audit (`S5-LOCKED-FINAL-CLEAN-CANDIDATE`), confirming no test-statistics leakage and no gt_tmag usage at inference.

## Final status
- Final clean candidate: `checkpoints/S5_clean_tmag_calibration_policy.json`
- Predecessor: `checkpoints/S2b_clean_fine_rot_policy.json`
- Final message for thesis: S5 replaces S2b with a clean but marginal gain; no claim of large model-level improvement.
