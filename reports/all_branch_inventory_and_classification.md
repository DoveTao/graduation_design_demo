# All-Branch Inventory and Classification

## Scope

This is a read-only branch inventory. No merges, no deletes, no cherry-picks, and no file moves were performed.

## Branch Count Summary

- total local branches: `16`
- total remote branches: `16`
- total unique branches: `16`
- branches with upstream: `16`
- branches without upstream: `0`
- branches ahead: `0`
- branches behind: `0`

## Branch Table

| branch | latest commit | upstream | ahead/behind | reports | checkpoints | likely group | likely status | recommended action |
|---|---|---|---|---:|---:|---|---|---|
| `experiment/jrt1-joint-rtdir-coupled-refiner` | `6765af3 Add JRT1 branch remote sync audit` | `origin/experiment/jrt1-joint-rtdir-coupled-refiner` | `0/0` | `74` | `157` | `archived_negative_experiments` | `negative_experiment_branch` | must include in REPORT2; keep, do not merge by default |
| `experiment/mf1-multi-frame-chain-refiner` | `ea559d5 Add MF1 multi-frame chain refiner artifacts` | `origin/experiment/mf1-multi-frame-chain-refiner` | `0/0` | `77` | `162` | `archived_negative_experiments` | `negative_experiment_branch` | must include in REPORT2; keep, do not merge by default |
| `experiment/orbslam3-fisheye-strong-baseline` | `797865b Add S5 dense same-evaluator comparison artifacts` | `origin/experiment/orbslam3-fisheye-strong-baseline` | `0/0` | `77` | `159` | `external_baseline_experiments` | `external_baseline_branch` | must include in REPORT2; keep separate from core clean candidate branches |
| `experiment/s5d2-dense-export-convention-audit` | `9fcdd5d Add reports inventory and index` | `origin/experiment/s5d2-dense-export-convention-audit` | `0/0` | `74` | `153` | `diagnostics` | `active_diagnostic_branch` | must include in REPORT2; cherry-pick report files later only if explicitly desired |
| `main` | `bb8fadd Add S1d5 cleanup and staging audit reports` | `origin/main` | `0/0` | `18` | `11` | `final_or_polish` | `mainline_reference` | include in REPORT2 only if final report inventory needs main branch context |
| `optimize/s10-chain-pathratio-smoother` | `0ae9aee Add S10 chain-level path-ratio preserving smoother diagnostic` | `origin/optimize/s10-chain-pathratio-smoother` | `0/0` | `35` | `94` | `historical_optimize_experiments` | `historical_experiment_branch` | include if reports/checkpoints present; likely archive-style read-only branch |
| `optimize/s11-tmag-scale-consistency-training` | `b00eab4 Close code optimization after S10 S11 diagnostics` | `origin/optimize/s11-tmag-scale-consistency-training` | `0/0` | `37` | `96` | `historical_optimize_experiments` | `historical_experiment_branch` | include if reports/checkpoints present; likely archive-style read-only branch |
| `optimize/s12-regime-balanced-sampling` | `34edd74 Add S13 practical usability gap analysis` | `origin/optimize/s12-regime-balanced-sampling` | `0/0` | `39` | `111` | `historical_optimize_experiments` | `historical_experiment_branch` | include if reports/checkpoints present; likely archive-style read-only branch |
| `optimize/s14-local-window-pose-graph` | `36e68fe Close code optimization after S14 pose graph diagnostic` | `origin/optimize/s14-local-window-pose-graph` | `0/0` | `41` | `114` | `historical_optimize_experiments` | `historical_experiment_branch` | include if reports/checkpoints present; likely archive-style read-only branch |
| `optimize/s15-trajectory-level-training-objective` | `8a28577 Close S15 trajectory training line after stability retest` | `origin/optimize/s15-trajectory-level-training-objective` | `0/0` | `48` | `122` | `historical_optimize_experiments` | `historical_experiment_branch` | include if reports/checkpoints present; likely archive-style read-only branch |
| `optimize/s16-stronger-visual-backbone-feasibility` | `8f52076 Close S16 backbone feasibility after ResNet50 probe` | `origin/optimize/s16-stronger-visual-backbone-feasibility` | `0/0` | `51` | `126` | `historical_optimize_experiments` | `historical_experiment_branch` | include if reports/checkpoints present; likely archive-style read-only branch |
| `optimize/s17-pose-supervision-dataset-quality-audit` | `e020e11 Add S18 split representativeness evaluation` | `origin/optimize/s17-pose-supervision-dataset-quality-audit` | `0/0` | `53` | `137` | `historical_optimize_experiments` | `historical_experiment_branch` | include if reports/checkpoints present; likely archive-style read-only branch |
| `optimize/s19-geometry-aware-pretraining-feasibility` | `f90d167 Add git branch remote sync audit` | `origin/optimize/s19-geometry-aware-pretraining-feasibility` | `0/0` | `56` | `143` | `historical_optimize_experiments` | `historical_experiment_branch` | include if reports/checkpoints present; likely archive-style read-only branch |
| `optimize/s2-fine-refinement-on-s1d5` | `c1f4b36 Add S3 coupled pose head design plan` | `origin/optimize/s2-fine-refinement-on-s1d5` | `0/0` | `22` | `44` | `historical_optimize_experiments` | `historical_experiment_branch` | include if reports/checkpoints present; likely archive-style read-only branch |
| `optimize/s3a0-coupled-pose-residual-head` | `f232fa3 Close post-S5 optimization after S8 S9 diagnostics` | `origin/optimize/s3a0-coupled-pose-residual-head` | `0/0` | `34` | `89` | `historical_optimize_experiments` | `historical_experiment_branch` | include if reports/checkpoints present; likely archive-style read-only branch |
| `polish/final-reproducibility-guardrails` | `7cb0a2c Add repository hygiene audit before JRT1` | `origin/polish/final-reproducibility-guardrails` | `0/0` | `69` | `151` | `final_or_polish` | `active_polish_baseline` | must include in REPORT2; keep as authoritative clean baseline branch |

## Branch Groups

### final_or_polish

- `main`
- `polish/final-reproducibility-guardrails`

### historical_optimize_experiments

- `optimize/s10-chain-pathratio-smoother`
- `optimize/s11-tmag-scale-consistency-training`
- `optimize/s12-regime-balanced-sampling`
- `optimize/s14-local-window-pose-graph`
- `optimize/s15-trajectory-level-training-objective`
- `optimize/s16-stronger-visual-backbone-feasibility`
- `optimize/s17-pose-supervision-dataset-quality-audit`
- `optimize/s19-geometry-aware-pretraining-feasibility`
- `optimize/s2-fine-refinement-on-s1d5`
- `optimize/s3a0-coupled-pose-residual-head`

### archived_negative_experiments

- `experiment/jrt1-joint-rtdir-coupled-refiner`
- `experiment/mf1-multi-frame-chain-refiner`

### external_baseline_experiments

- `experiment/orbslam3-fisheye-strong-baseline`

### diagnostics

- `experiment/s5d2-dense-export-convention-audit`

### hygiene_or_reporting

- `(none)`

### unknown

- `(none)`

## Branches Recommended for Multibranch Reports Inventory

### must include

- `polish/final-reproducibility-guardrails`
- `experiment/jrt1-joint-rtdir-coupled-refiner`
- `experiment/mf1-multi-frame-chain-refiner`
- `experiment/orbslam3-fisheye-strong-baseline`
- `experiment/s5d2-dense-export-convention-audit`

### include if reports present

- `optimize/s10-chain-pathratio-smoother`
- `optimize/s11-tmag-scale-consistency-training`
- `optimize/s12-regime-balanced-sampling`
- `optimize/s14-local-window-pose-graph`
- `optimize/s15-trajectory-level-training-objective`
- `optimize/s16-stronger-visual-backbone-feasibility`
- `optimize/s17-pose-supervision-dataset-quality-audit`
- `optimize/s19-geometry-aware-pretraining-feasibility`
- `optimize/s2-fine-refinement-on-s1d5`
- `optimize/s3a0-coupled-pose-residual-head`

### optional

- `(none)`

## Remote Sync Issues

- local-only: `[]`
- remote-only: `[]`
- no upstream: `[]`
- ahead: `[]`
- behind: `[]`
- divergent: `[]`

## Recommended Next Step

Next task should be `REPORT2_multibranch_reports_inventory`, using the branch list generated here rather than a hard-coded small subset.

## No-Op Confirmation

- no files moved = true
- no files deleted = true
- no branches merged = true
- no branches deleted = true

