# Final Post-S15 Trajectory Training Closure

## Why S15 stops
S15 is closed because the final tiny stability retest still failed at the earliest acceptable gate. After the harness parity fix, `A_pair_only_baseline_fixed_harness` remained unstable with `odom_ATE=21.710394`, `drift=35.126991`, and `path_ratio=2.647609`. Under the project rules, that means the trajectory-level training line should stop before any S15f or later extension.

## S15 attribution chain
1. S15 introduced the trajectory-level training objective as a future-facing training-time direction rather than an inference-time post-processor.
2. S15a showed that the original tiny trajectory smoke collapsed badly, which made the immediate failure attribution a harness question before it could be a trajectory-loss tuning question.
3. S15b found that the training harness did not apply the locked S5/S2b wrapped policy lineage, so the trajectory line was not even training the same policy family as the final clean candidate.
4. S15c fixed the wrapped policy lineage and restored exact zero-update alignment, but train/eval drift and one-update destructive behavior still remained.
5. S15d found and fixed the real parity bug: frozen-subtree mode control was being recursively overwritten by a trailing `model.train(True)`. After that fix, train/eval deterministic forward parity became exact and one-update R/tdir perturbation dropped to near zero.
6. S15e reran the tiny trajectory retest under the corrected harness. The parity bug was gone, but pair-only tiny training still destabilized odometry, and the full light trajectory variant did not recover a usable trajectory improvement.

## Final S15 conclusion
- S15 does not replace S5.
- S15f lightweight CV is not recommended.
- S16 is not recommended under the current project scope.
- The final clean candidate remains `S5_clean_tmag_calibration_policy`.

## Interpreting the negative result
The S15 line is still useful as evidence. It shows that:
- the earlier trajectory-training failure was not purely a train/eval parity artifact;
- fixing the harness was necessary, but not sufficient;
- the current tiny-update route is still too unstable to be reported as a valid result line.

The post-fix D candidate did improve proxy numbers over the original pre-fix smoke (`val_ate_proxy 15.708141 -> 5.528377`, `val_path_proxy 2.515539 -> 1.742522`), but odometry still worsened (`ATE 19.286121 -> 21.995400`, `drift 24.674397 -> 35.549201`). That is not acceptable evidence for continuation.

## Final candidate status
- Final candidate changed: `no`
- Final candidate: `S5_clean_tmag_calibration_policy`
- Locked metrics: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`

## Future work only
If trajectory-level training is revisited in future work, it should not continue from the current tiny-update recipe unchanged. More promising directions would be:
- redesign the trajectory supervision and training harness more fundamentally;
- add stronger stability controls instead of relying on very small update budgets;
- consider stronger backbone, more data, or richer supervision rather than only local tiny updates on the existing line.
