# Final Post-S9 Code Optimization Closure

## Closure decision

Code optimization stops after S9. No further S8c, S10, router, residual-head, or fine-token follow-up is recommended for the current final delivery line.

## Why optimization stops

The project already reached a valid locked final clean candidate at S5 and confirmed that status through the S6/S7 lockdown and reproducibility path. After that point, the remaining optimization work was intentionally constrained to post-lockdown clean diagnostics. Those post-lockdown attempts did not produce a superior deployable replacement.

## S8 result

S8 tested whether fine/spherical token features could still be useful as a reliability estimator even though they had already failed to show stable residual-pose value. The result was negative: token-only and token-plus-regime probes failed to outperform the simpler regime-only baseline built from `pred_tmag`, `dt`, and `k`. Final classification: `TOKEN-NO-ADDED-VALUE`.

## S9 result

S9 tested whether a regime-only router, using only deployable inference-visible features, could cleanly improve over S5. The baseline gate passed, but no candidate passed the train-CV clean gate. The best diagnostic candidate, `R3_dt1_k20_fallback`, still failed because path_ratio stayed outside the safe clean range, so no final test was promoted. Final classification: `NO-STABLE-REGIME-ROUTER-GAIN`.

## Final candidate

The final candidate remains [S5_clean_tmag_calibration_policy.json](/home/dovetao/graduation_design_demo/checkpoints/S5_clean_tmag_calibration_policy.json), with locked metrics:

- drift = `1.327343`
- ATE = `7.352288`
- path_ratio = `0.932379`

## Final project interpretation

After S5, the remaining optimization attempts were still useful, but only as negative results. They strengthen the final thesis position that the clean deployable line is already saturated enough that later token-reliability or regime-only routing ideas do not currently justify replacement.

## Next step

No S10 is recommended. The next project step is thesis writing, defense preparation, and presentation material built around the locked S5 final clean candidate plus the S8/S9 negative-result discussion.
