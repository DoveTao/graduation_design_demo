# MAINT3 Base360 Provenance Branch Prep

## 1. Current State

- branch: `maintenance/env-git-cleanup`
- upstream: `origin/maintenance/env-git-cleanup`
- pending untracked files before MAINT3:
  - `external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/image_list.txt`
  - `external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/input_manifest.json`
  - `external_baselines/results/base360_hkust_360dvo_official/test/ridge_to_lake/run_metadata.json`
  - `external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/image_list.txt`
  - `external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/input_manifest.json`
  - `external_baselines/results/base360_hkust_360dvo_official/test/snowmobile/run_metadata.json`
  - `external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/image_list.txt`
  - `external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/input_manifest.json`
  - `external_baselines/results/base360_hkust_360dvo_official/val/downhill_biking/run_metadata.json`
  - `external_baselines/results/base360_hkust_360dvo_official/val/mountains/image_list.txt`
  - `external_baselines/results/base360_hkust_360dvo_official/val/mountains/input_manifest.json`
  - `external_baselines/results/base360_hkust_360dvo_official/val/mountains/run_metadata.json`
  - `reports/MAINT1B_conda_env_finalize.md`

## 2. Size Check

- `image_list.txt` files:
  - `ridge_to_lake`: `24840` bytes
  - `snowmobile`: `30198` bytes
  - `downhill_biking`: `27072` bytes
  - `mountains`: `31365` bytes
- `input_manifest.json` files:
  - `ridge_to_lake`: `1877` bytes
  - `snowmobile`: `1847` bytes
  - `downhill_biking`: `1890` bytes
  - `mountains`: `1830` bytes
- `run_metadata.json` files:
  - `ridge_to_lake`: `1871` bytes
  - `snowmobile`: `1841` bytes
  - `downhill_biking`: `1884` bytes
  - `mountains`: `1824` bytes

## 3. Content Check

- `image_list.txt` contains plain text image paths such as `data/360DVO/Sequences/<sequence>/0001.jpg`.
- `input_manifest.json` contains canonical split, input directory, frame count, pair count, TUM path references, and official demo input metadata.
- `run_metadata.json` contains the same canonical input metadata plus final `run_status`.
- no file contains raw image bytes, checkpoint payloads, `360dvo.pth`, credentials, or large logs.

## 4. Commit Decision

- provenance files are safe to commit
- reason:
  - text-only
  - small
  - useful for canonical BASE360 reproducibility
  - no prohibited large artifact content
- additional cleanup needed before TRAIN360D branch creation:
  - commit `reports/MAINT1B_conda_env_finalize.md` so the maintenance branch can become clean

## 5. Planned Sequence

1. commit BASE360 provenance files
2. commit MAINT3 final branch-preparation report and the pending MAINT1B report
3. push `maintenance/env-git-cleanup`
4. confirm clean working tree
5. create and push `experiment/train360d-observability-kstep-scale`
