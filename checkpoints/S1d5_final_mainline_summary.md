# S1d5 final mainline summary

## Executive summary

- S1d5 is the current clean exported mainline.
- It uses the true multiscale checkpoint with explicit-cfg restore and `unexpected=0`.
- It uses a train-only selected dt-anchor + rot policy.
- It does not rely on the historical scalar-load artifact.

## Policy details

- checkpoint path: `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- bucket edges: `[0.1, 0.3, 0.5, 1.0]`
- effective bucket factors: `{'[0.1,0.3)': 1.302368139324791, '[0.3,0.5)': 2.6003610319428594, '[0.5,1)': 3.9662143503139955}`
- alpha: `1.05`
- fine_rot: `0.4`
- fine_tdir: `0.0`
- fine_tmag: `0.0`
- geometry refine: `False`
- selection source: `train_only_leave_one_train_seq_out_cv`

## Reproduction metrics

- drift = 1.396358
- ATE = 7.632463
- path_ratio = 0.934982
- direction_only path_ratio = 0.999997
- RPE_trans_mag = 0.073926
- tmag P10/P50/P90 = 0.126964 / 0.183179 / 0.402137
- selected_k / pairs / chains = 1 / 132 / 19
- missing/unexpected = 2 / 0

## Comparison against baselines

- clean baseline table: [S1d5_final_baseline_comparison.md](/home/dovetao/graduation_design_demo/checkpoints/S1d5_final_baseline_comparison.md)
- historical scalar-load results remain explicitly marked invalid as true baselines.

## Interpretation

- S1d5 fixes scale collapse: `path_ratio 0.496 -> 0.935`.
- S1d5 also improves ATE: `9.644 -> 7.632`.
- Drift remains strong: `1.494 -> 1.396`.
- Direction-only path ratio near 1 confirms the remaining scale behavior is controlled.

## Limitations

- ATE is still not small in absolute units.
- S1d5 is an eval / calibration policy, not yet a fully internalized end-to-end learned model.
- It depends on a dt-bucket scale anchor.
- It still needs broader validation if more scenes / sequences become available.

## Next-step recommendation

- Option A: stop here and use S1d5 as the final clean mainline.
- Option B: later do `S1d6_minimal_trainable_dt_anchor_head` to internalize the policy.
- F1d remains paused.

## Figures

- figures folder: [S1d5_final_figures](/home/dovetao/graduation_design_demo/checkpoints/S1d5_final_figures)
- trajectory plot generated: True
