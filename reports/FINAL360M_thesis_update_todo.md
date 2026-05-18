# FINAL360M thesis update todo

## Required

- update `THESIS362` final result tables to replace old `FINAL360I` main-result row with `FINAL360M`
- explicitly relabel old `FINAL360I` as `subset-trained candidate`
- update pair-level comparison paragraphs to state:
  - `FINAL360M` is protocol-valid full-train main result
  - it improves `tmag/path`
  - it does not improve `signed_tdir`
- update any figure/table captions that still call `FINAL360I` the final main model

## Recommended

- rerun trajectory-level evaluation for `FINAL360M` using the existing `TRAIN360E` evaluation path
- add a concise caveat in thesis text that the final main-model switch is protocol-driven and accompanied by a direction/scale tradeoff
- update summary tables to include:
  - `FINAL360M full-train guarded`
  - `FINAL360I subset-trained candidate`
  - `TRAIN360D`
  - `TRAIN360H`
  - `BASE360D`

## Nice to have

- add one short note in the experiment section documenting why the original `FINAL360M` AMP run was rejected and why the guarded restart was used
- keep watchdog / NaN audit artifacts as reproducibility appendix material, not as core thesis results

