# MAINT2 Batch Commit Experiment Artifacts

## 1. Executive Summary

- commits created: 3 experiment commits so far plus this pending MAINT2 report commit
- pushed to remote: false at report generation time; push is the next step
- large artifacts excluded: true
- checkpoints preserved: true
- working tree status after experiment commits: only MAINT2 documentation plus local-only BASE360 provenance files remain untracked

## 2. Commit List

- `8e0af9c` `TRAIN360: add manifest-native baseline pipeline and reports`
  included categories: TRAIN360 code, configs, forward sanity tooling, training/eval helpers, TRAIN360A/B/C reports, TRAIN360 val/test metric summaries
- `b840e89` `BASE360: add HKUST official baseline runner and aligned metrics`
  included categories: BASE360 runner code, component metric alignment tooling, cuda_ba shim source, BASE360B/C/D reports, aligned metrics JSON, compact external summary JSONs
- `fb5b9ad` `RESULTS360: add main comparison tables and experiment narrative`
  included categories: main comparison table markdown/json, experiment narrative, TRAIN360 vs BASE360 vs T57b summary
- `0f00dfd` `MAINT: add conda environment inventory and git cleanup notes`
  included categories: environment lock files already committed during MAINT1, so no new ENV commit was needed in MAINT2

## 3. Excluded Artifacts

- TRAIN360C `.pt` checkpoints stayed local and were not staged
- raw data under `data/360DVO/`, `data/raw/`, and `data/external/` was not staged
- large logs under `logs/` were not staged
- large BASE360 generated outputs, resized input images, smoke images, and TUM trajectories were not staged
- official `360dvo.pth` weight was not staged or committed
- local BASE360 provenance helpers such as `image_list.txt`, `input_manifest.json`, and `run_metadata.json` remain untracked for now

## 4. Git Status

- branch: `maintenance/env-git-cleanup`
- upstream: `origin/maintenance/env-git-cleanup`
- ahead/behind: `* maintenance/env-git-cleanup                         fb5b9ad [origin/maintenance/env-git-cleanup: 领先 3] RESULTS360: add main comparison tables and experiment narrative`
- remaining untracked/modified files:
  - `?? external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/image_list.txt`
  - `?? external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/input_manifest.json`
  - `?? external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/run_metadata.json`
  - `?? external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/image_list.txt`
  - `?? external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/input_manifest.json`
  - `?? external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/run_metadata.json`
  - `?? external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/image_list.txt`
  - `?? external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/input_manifest.json`
  - `?? external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/run_metadata.json`
  - `?? external_baselines/results/base360_hkust_360dvo_official/val/mountains/image_list.txt`
  - `?? external_baselines/results/base360_hkust_360dvo_official/val/mountains/input_manifest.json`
  - `?? external_baselines/results/base360_hkust_360dvo_official/val/mountains/run_metadata.json`
  - `?? reports/MAINT2_git_precommit_inventory.md`

## 5. Next Recommendation

- finish push for this branch
- then create a clean follow-up plan for `TRAIN360D`
- recommended next task: `proceed_to_TRAIN360D_observability_kstep_scale_stabilization`
