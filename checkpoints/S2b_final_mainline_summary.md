# S2b Final Mainline Summary

## Executive Summary
- S2b is a train-CV selected clean fine-rot policy candidate.
- It inherits the S1d5 dt-anchor scale policy.
- It changes only `fine_rot` from `0.40` to `0.45`.
- It does not train any model parameter.
- It does not use test labels for selection.
- It improves S1d5 ATE/drift while preserving path_ratio.

## Policy Details
- base checkpoint: `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- inherited policy: `checkpoints/S1d5_clean_dt_anchor_policy.json`
- fine_rot = `0.45`
- fine_tdir = `0.0`
- fine_tmag = `0.0`
- use_geometry_refine = `False`
- selection source: `train_only_leave_one_train_seq_out_cv`

## Reproduction Metrics
- drift = `1.327402`
- ATE = `7.352371`
- path_ratio = `0.934984`
- RPE_rot = `20.715353`
- RPE_trans_dir = `72.349483`
- RPE_trans_mag = `0.073926`
- rot = `21.481086`
- tdir_abs = `29.944641`
- tdir_local_A_abs = `23.152590`
- tmag P10/P50/P90 = `0.127012 / 0.183018 / 0.402103`
- selected_k / pairs / chains = `1 / 132 / 19`
- missing/unexpected = `2 / 0`

## Comparison Against S1d5
- drift: `1.396358 -> 1.327402`
- ATE: `7.632463 -> 7.352371`
- path_ratio: `0.934982 -> 0.934984`
- interpretation: fine-rot refinement improves trajectory accuracy without damaging scale repair.

## Limitations
- S2b is still an eval/calibration/fusion policy, not an end-to-end trained fine model.
- ATE≈7.35 is still high in absolute units.
- fine_tdir/fine_tmag remain unused.
- Broader validation on more scenes/sequences is still needed.

## Verdict
- SUCCESS
- S2b can be considered the current clean fine-rot policy candidate.
- It may replace S1d5 as the latest clean candidate on this branch.
- S1d5 remains the tagged stable baseline.
