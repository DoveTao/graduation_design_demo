# Final Post-S11 Code Optimization Closure

## Closure decision
Code optimization stops after S11. The final clean candidate remains `S5_clean_tmag_calibration_policy`, and no S12 continuation is recommended.

## Why optimization stops after S11
The post-S5 search space has now been tested along four different directions:

- S8: token-based reliability probing
- S9: regime-only deployable router
- S10: chain-level path-ratio-preserving smoother
- S11: training-time tmag scale consistency

None of these lines produced a clean deployable replacement for S5 under the final project constraints.

## S8/S9/S10/S11 closure stack
- S8 ended with `TOKEN-NO-ADDED-VALUE`: fine/spherical token features did not beat a simpler regime-only reliability baseline.
- S9 ended with `NO-STABLE-REGIME-ROUTER-GAIN`: regime-only routing showed some diagnostic signal but no safe clean gate pass.
- S10 ended with `NO-STABLE-CHAIN-SMOOTHER-GAIN`: chain-level smoothing did not recover a stable clean improvement over S5.
- S11 ended with `NO-STABLE-TMAG-CONSISTENCY-GAIN`: training-time tmag consistency produced weak proxy improvement, but not path_ratio-supported clean evidence for replacement.

## Final candidate status
- Final clean candidate remains: `checkpoints/S5_clean_tmag_calibration_policy.json`
- Locked metrics remain: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`
- No later branch replaced S5.

## Practical interpretation
The project has already exhausted the most plausible low-risk optimization directions after S5:

- inference-time token routing did not help,
- regime-only fallback/routing did not help,
- chain-level smoothing did not help,
- training-time tmag consistency did not provide enough clean evidence.

This makes further code optimization lower-value than consolidating the existing result into a clear thesis narrative.

## Final recommendation
- No S12 recommended.
- No further post-processing, routing, smoothing, or tmag-consistency optimization should be treated as part of the final code mainline.
- The next step is thesis writing, result consolidation, and defense material preparation.
