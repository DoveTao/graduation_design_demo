# CURRENT_MAINLINE FINAL360M Promotion

## Executive Summary

- promotion executed: `true`
- new mainline model: `FINAL360M_fulltrain_struct360b_thesis_main_guarded`
- selected checkpoint: `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded/best_full_val.pt`
- old `FINAL360I` status: `subset-trained candidate only`
- training executed in this task: `false`
- metrics modified in this task: `false`

## Why FINAL360M is promoted

- `FINAL360M` is the only `full-train protocol-qualified thesis main result` in the current worktree.
- it satisfies the required protocol chain:
  - `train_subset_disabled = true`
  - `train_sample_count = 12154`
  - `selected_by_full_val = true`
  - `test_used_for_selection = false`
  - `best_full_val.pt` exists and is traceable to the guarded full-train run

## Why old FINAL360I is downgraded

- the audit found that old `FINAL360I` used `train_subset_max = 256`
- effective train sample count was `256`, not the full train manifest
- therefore old `FINAL360I` is not eligible as the final thesis main model
- allowed thesis status: `subset-trained candidate only`

## FINAL360M vs old FINAL360I

- relationship: `mixed`
- `FINAL360M` should not be described as uniformly better
- `FINAL360M` is promoted because it is protocol-correct, not because every pair metric improved

## Trajectory Narrative

- `FINAL360M-direct` remains weak
- `FINAL360M-ODOM360A` improves `FINAL360M-direct`
- `FINAL360M-ODOM360B` does not improve over refreshed `FINAL360M-ODOM360A`
- refreshed `FINAL360M` trajectory backend is still weaker than old subset-model-based `ODOM360A`

## Thesis Narrative Policy

- pair-level full-train thesis main model: `FINAL360M`
- old `FINAL360I`: historical subset-trained candidate only
- translation direction (`tdir`) remains the main bottleneck
- backend fusion helps trajectory behavior, but does not fully solve trajectory shape error
