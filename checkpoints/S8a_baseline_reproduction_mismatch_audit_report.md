# S8a Baseline Reproduction Mismatch Audit Report

## Executive summary

- final classification: `HISTORY-REPORT-MISMATCH`
- root cause: Locked S2b predecessor metrics/load_missing were inherited from the legacy e7ba870 summary (load_missing=2), while later code/reporting already validated a benign load_missing=14 path against the same checkpoint and policy.
- no model training, no S8 router selection, and no S5/S2b policy mutation were performed.
- S8 remains paused in this audit turn.

## Reproduction mismatch details

- expected S2b: drift=`1.327402`, ATE=`7.352371`, path_ratio=`0.934984`, load_missing=`2`
- observed blocker run: drift=`1.3270823574`, ATE=`7.3512213288`, path_ratio=`0.9349468349`, load_missing=`14`
- expected S5: drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`

## Expected vs observed metrics

| item | drift | ATE | path_ratio | load_missing | load_unexpected |
| --- | ---: | ---: | ---: | ---: | ---: |
| S2b expected | 1.327402 | 7.352371 | 0.934984 | 2 | 0 |
| S2b observed blocker | 1.327082 | 7.351221 | 0.934947 | 14 | 0 |
| S3a0c S2b_policy_baseline | 1.327402 | 7.352371 | 0.934984 | 14 | 0 |
| legacy `e7ba870` S2b summary | 1.327402 | 7.352371 | 0.934984 | 2 | 0 |

## Git/file provenance audit

- relevant path history:

```text
719705f Add final project reproducibility and thesis summaries
0d37fd4 Lock down S5 final clean calibration candidate
15dc1a0 Add S3a1 rot-only residual head smoke
57c5eb2 Align S3a0 smoke eval with S2b clean policy
939ef3c Implement S3a0 minimal coupled pose residual head smoke
e7ba870 Finalize S2b clean fine-rot policy on S1d5
c2d7146 Finalize S1d5 clean mainline artifacts and gitignore
7a8d18c feat: add optional odom chain vector direction loss
f6d0e0e feat: add optional odom chain length loss
371593a feat: add optional tmag under regularization
1342a71 feat: add optional tmag scale regularization
a81b973 fix: save eval checkpoints and ungated eval selection
2efd354 feat: add dt-conditioned translation magnitude head
8331469 feat: add O49 stable rerun modes and device-aware summary
6964ff8 add O49 stability source probes
2243d0a add robust O49 odom selection probes
c144627 stabilize high-weight O49 seq-turn loss
26fb3fe stage: add trajectory shape diagnostics
0e3ccd4 stage: add dt-bin scale calibration diagnostic
1c8e930 stage: add odometry trajectory debug exports
90761cd stage25: add learned tmag bias calibration
a077848 stage24: add odom scale-fit oracle diagnostics
45bf240 stage23: add odom tmag smoothing diagnostics
e68de0b stage22: add continuous tiny-dt tdir ramp
3bab8e3 stage20: ignore tiny-dt samples for tdir loss
9848d56 stage19: add small-k odom checkpoint selection
4ec85ab stage18: add odom-aware checkpoint selection
06a00f6 stage17: add eval-only candidate comparison
553001b stage16: add tdir anchor distillation
ecde5e6 stage14: add odometry curriculum checkpoints
4dcaac8 stage11: isolate translation magnitude gradients
97d0ba7 stage11: add translation magnitude loss schedule
da5458e stage7: add eval geometry refinement
ea6d037 stage5: export matching diagnostics and bucket csv
3a72499 stage3: add sequence odometry evaluation
2311216 stage2-4: add scale head and experiment scripts
0273b28 fix: avoid duplicate coarse pose aux for fuse-zero fine stage
655c24a tune: disable color augmentation by default
0560ee8 tune: reduce color augmentation strength
2eb2233 chore: add training config overrides
10076e2 feat: adopt stats pooled pose baseline
375e63d chore: clean repository artifacts and add module headers
ee1c3be 0429 new_tbranch
96a5126 0429 faster_ver
82f7610 0429 color_and_full_dt
250ece8 0428 fine_begin
551592e 0426 train and eval
5941fec 0425 cnn patch
bb760d2 0425 begin
63c85ad 0414 bucket and new trans head
7091ea6 0414 add depth vis
9ebcaf1 0414 add depth branch
0eb5661 0413 fix vis json
3c8b235 0413 fix vis json
c34bcef 0411 fixed_pairs + deterministic
2633f6f 0330 complete todo
f407001 0325final
81edbf4 piecewise LR
edc6174 config_low_peak_early_stop
620e9fc add comparison and ablation
255449f 3.24
8b93cdb v1.6
499b9a5 v1.5 beta atan2
dbf695c v1.4 info_detail
b7871a7 v1.3 new model
2ee3ff6 v1.2 solve nan
4c289f5 v1.1 add eps
3391530 Initial commit
```
- changed files from `c471b21` to `719705f` within the audit scope: `['checkpoints/S5_clean_tmag_calibration_policy.json', 'checkpoints/final_clean_candidate_manifest.json', 'scripts/eval_s5_clean_policy.sh']`
- changed files from `0d37fd4` to `719705f` within the audit scope: `['checkpoints/final_clean_candidate_manifest.json']`
- diff stat `c471b21..719705f`:

```text
checkpoints/S5_clean_tmag_calibration_policy.json | 44 +++++++++++++++++++++++
 checkpoints/final_clean_candidate_manifest.json   | 24 +++++++++++++
 scripts/eval_s5_clean_policy.sh                   | 18 ++++++++++
 3 files changed, 86 insertions(+)
```
- diff stat `0d37fd4..719705f`:

```text
checkpoints/final_clean_candidate_manifest.json | 24 ++++++++++++++++++++++++
 1 file changed, 24 insertions(+)
```
- result: no relevant eval/model/config/script drift was found after S6 within the audited file set; the only scoped file added after S6 is `checkpoints/final_clean_candidate_manifest.json`.

## Checkpoint loading audit

- policy path: `checkpoints/S2b_clean_fine_rot_policy.json`
- checkpoint path: `/home/dovetao/graduation_design_demo/checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- strict loading: `False`
- current missing key count: `14`
- current unexpected key count: `0`
- current missing keys:
  - `coarse.mag_head.ridge_calib_raw_center`
  - `fine.mag_head.ridge_calib_raw_center`
  - `coupled_pose_head.backbone.0.weight`
  - `coupled_pose_head.backbone.0.bias`
  - `coupled_pose_head.backbone.1.weight`
  - `coupled_pose_head.backbone.1.bias`
  - `coupled_pose_head.backbone.4.weight`
  - `coupled_pose_head.backbone.4.bias`
  - `coupled_pose_head.rot_head.weight`
  - `coupled_pose_head.rot_head.bias`
  - `coupled_pose_head.tdir_head.weight`
  - `coupled_pose_head.tdir_head.bias`
  - `coupled_pose_head.gate_head.weight`
  - `coupled_pose_head.gate_head.bias`

## Missing/unexpected key analysis

- historical legacy summary `e7ba870` reported `load_missing=2` and the legacy model source does not contain `coupled_pose_head`: `False`
- current benign base missing keys: `['coarse.mag_head.ridge_calib_raw_center', 'fine.mag_head.ridge_calib_raw_center']`
- current coupled-head missing keys: `['coupled_pose_head.backbone.0.weight', 'coupled_pose_head.backbone.0.bias', 'coupled_pose_head.backbone.1.weight', 'coupled_pose_head.backbone.1.bias', 'coupled_pose_head.backbone.4.weight', 'coupled_pose_head.backbone.4.bias', 'coupled_pose_head.rot_head.weight', 'coupled_pose_head.rot_head.bias', 'coupled_pose_head.tdir_head.weight', 'coupled_pose_head.tdir_head.bias', 'coupled_pose_head.gate_head.weight', 'coupled_pose_head.gate_head.bias']`
- current critical missing keys after filtering: `[]`
- interpretation: `load_missing=14` decomposes cleanly into the old 2 ridge-calibration buffers plus 12 later-added coupled-head parameters.
- S3a0c already recorded that a `load_missing=14` S2b wrapper path matched the locked S2b metrics exactly, so the 12 extra keys are historical evidence of a benign unused-head mismatch rather than immediate forward corruption.

## Eval parameter/scope audit

- locked/historical summary and current policy both use:
  - `fine_rot=0.45`
  - `fine_tdir=0.0`
  - `fine_tmag=0.0`
  - `selected_k=1`
  - `num_pairs=132`
  - `num_chains=19`
- `Config` defaults for coupled-head enablement remain `False` in the current branch, so the extra coupled-head parameters are instantiated but not supposed to be active in the S2b eval path.
- `S2c1` already documented that official `path_ratio` comes from the debug-chain summary path with `odom_trajectory_debug_max_chains=1`; no new post-S6 change was found in the audited script/code set that would alter that scope.

## Historical command comparison

- `scripts/eval_s2b_clean_policy.sh`: unchanged in the audited commits after S5.
- `scripts/eval_s5_clean_policy.sh`: introduced at S6/S7 for the frozen S5 candidate; it does not affect the S2b script path.
- `tools/eval_clean_policy.py`: unchanged in the audited commits after S5.
- locked predecessor summary provenance: `checkpoints/S2b_final_repro/s1d5_policy_eval_summary.json` traces back to `e7ba870`, not to S6/S7.

## Root cause classification

- `HISTORY-REPORT-MISMATCH`
- explanation: Locked S2b predecessor metrics/load_missing were inherited from the legacy e7ba870 summary (load_missing=2), while later code/reporting already validated a benign load_missing=14 path against the same checkpoint and policy.

## Fix applied, if any

- No reproduction-path fix was applied in this audit turn.
- Recommended next safe fix if needed: add an explicit legacy-compatible S2b reproduction wrapper that prints both the current 14-key load audit and the legacy predecessor provenance, without mutating any locked metrics or policies.

## Post-fix reproduction result, if any

- none in this audit turn.

## Whether S8 can resume

- `False`
- reason: the predecessor reproduction gate is still mixed between a legacy locked summary (`load_missing=2`) and a later benign current architecture (`load_missing=14`), so S8 should stay paused until a single authoritative legacy-compatible reproduction path is frozen.

## Whether S5 remains final clean candidate

- `True`
- manifest path: `checkpoints/S5_clean_tmag_calibration_policy.json` with final metrics drift=`1.327343`, ATE=`7.352288`, path_ratio=`0.932379`.
