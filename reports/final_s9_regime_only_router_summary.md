# Final S9 Regime-Only Router Summary

- Final classification: `NO-STABLE-REGIME-ROUTER-GAIN`
- S9 replaces S5: `False`
- S5 remains final clean candidate: `True`
- Selected candidate: `R3_dt1_k20_fallback`

S9 stopped the token/fine reliability direction and tested only deployable regime features. The router was restricted to `pred_tmag`, `dt`, and `k`, with train-CV-only thresholds/actions and one optional final test for the selected candidate.

No candidate was strong enough to justify a final replacement test, so S9 remains a diagnostic result rather than a new final candidate.

Interpretation: if S9 fails to replace S5, then the current removable error mass is already largely captured by the fixed S5 magnitude-regime policy, and further clean gains will likely require a stronger representation or a different routing/evaluation design rather than simply adding more regime heuristics.
