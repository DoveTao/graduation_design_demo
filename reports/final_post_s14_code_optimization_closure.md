# Final Post-S14 Code Optimization Closure

## Closure decision
Code optimization stops after S14. The final clean candidate remains `S5_clean_tmag_calibration_policy`, and no S15 follow-up is recommended within the current pairwise pipeline.

## Why optimization stops here
S8, S9, S10, S11, S12, and S14 collectively tested the remaining low-risk post-S5 optimization space:
- S8: token reliability did not add value beyond regime features.
- S9: regime-only routing had diagnostic signal but no stable clean gain.
- S10: chain-level smoothing did not produce a stable safe gain.
- S11: tmag consistency training showed weak proxy improvement but no stable clean evidence.
- S12: regime-balanced sampling improved some high-risk proxy metrics but still lacked clean-eligible path_ratio evidence.
- S14: graph constraints were sufficient, but the lightweight local-window pose graph still worsened ATE/drift/path_ratio on the final diagnostic.

These results are consistent with the S13 conclusion that the remaining practical gap is not likely to be closed by another local tweak layered on top of the current S5 pairwise pipeline.

## Final candidate status
- Final clean candidate: `S5_clean_tmag_calibration_policy`
- Locked metrics: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`
- Replacement status after S8/S9/S10/S11/S12/S14: unchanged

## No S15 recommendation
No S15 code-optimization line is recommended in the current project scope. Another router, smoother, lightweight sampler tweak, or local-window graph variant is unlikely to produce a clean deployable replacement without changing the problem level.

## If practical-ready performance is required
The next major direction should move to a larger-method change rather than another incremental patch:
- stronger backbone or pretrained visual encoder
- trajectory-level training objective instead of purely pair-level optimization
- full multi-frame or global optimization rather than lightweight local-window consistency only
- data or supervision redesign for difficult high-tmag / large-dt / k=20 regimes

## Project close-out
For thesis and defense material, the right final message is:
- S5 is the best clean and reproducible deployable candidate.
- Later code-optimization attempts were informative and worth reporting, but they did not replace S5.
- Practical-ready performance likely requires a higher-level trajectory formulation, stronger representation, or redesigned data/supervision.
