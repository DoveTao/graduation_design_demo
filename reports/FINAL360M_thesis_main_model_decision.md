# FINAL360M thesis main model decision

## Decision

- recommended thesis main model: `FINAL360M_fulltrain_struct360b_thesis_main_guarded`
- selected checkpoint: `checkpoints/FINAL360M_fulltrain_struct360b_thesis_main_guarded/best_full_val.pt`
- decision type: `accept as final thesis main model`

## Why

- `FINAL360M` satisfies the required thesis protocol:
  - full train manifest used
  - train subset disabled
  - train/val/test counts correct
  - selected by full validation
  - test not used for selection
  - full test executed
  - no nonfinite detected in the guarded run
- old `FINAL360I` does not satisfy the same protocol because it was trained on a subset-limited run.

## Performance interpretation

- compared to old `FINAL360I subset-trained candidate`, `FINAL360M` is `mixed`
- positive:
  - slightly better `rot_mean`
  - better `tmag_median_ratio`
  - better `path_ratio`
  - zero `scale_collapse_rate`
- negative:
  - worse `signed_tdir_mean`
  - slightly worse `anti_parallel_rate`

## Thesis wording guidance

- the thesis should not claim `FINAL360M` is uniformly better than old `FINAL360I`
- the thesis should claim:
  - `FINAL360M` is the final full-train, protocol-valid main model
  - it improves scale/path behavior and keeps similar rotation quality
  - it trades off some translation-direction quality relative to the earlier subset-trained candidate

## Old FINAL360I status

- recommended label in thesis: `subset-trained candidate`
- not acceptable labels:
  - `final thesis main model`
  - `full-train final result`

