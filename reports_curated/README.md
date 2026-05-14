# Curated Reports Archive

This directory contains copied canonical report snapshots from multiple branches.
Original reports remain in their source branches.
No source reports were deleted or moved.
This archive is for thesis / final review convenience.

## Source Branches

- `experiment/jrt1-joint-rtdir-coupled-refiner`
- `experiment/mf1-multi-frame-chain-refiner`
- `experiment/orbslam3-fisheye-strong-baseline`
- `experiment/s5d2-dense-export-convention-audit`
- `optimize/s10-chain-pathratio-smoother`
- `optimize/s11-tmag-scale-consistency-training`
- `optimize/s12-regime-balanced-sampling`
- `optimize/s14-local-window-pose-graph`
- `optimize/s15-trajectory-level-training-objective`
- `optimize/s16-stronger-visual-backbone-feasibility`
- `optimize/s17-pose-supervision-dataset-quality-audit`
- `optimize/s19-geometry-aware-pretraining-feasibility`
- `polish/final-reproducibility-guardrails`

## Directory Structure

- `reports_curated/final/`
- `reports_curated/ablations/`
- `reports_curated/baselines/`
- `reports_curated/diagnostics/`
- `reports_curated/negative_experiments/jrt1/`
- `reports_curated/negative_experiments/mf1/`
- `reports_curated/audits/`
- `reports_curated/environment/`

## Provenance

Use `reports_curated/MANIFEST.json` to verify the source branch, source path, source commit, and source blob hash for each copied file.

## Materialization Summary

- copied canonical reports: `28`
- missing sources: `0`
- skipped for review: `8`

## S5 Locked Metrics

- ATE = `7.352288`
- drift = `1.327343`
- path_ratio = `0.932379`

S5 remains final clean candidate.

## Caveats

- S5 official locked metrics and dense external / same-evaluator metrics have scope differences.
- ORB-SLAM3 is an external baseline and is not a replacement for S5.
- JRT1 and MF1 are negative experiments and remain archived as such.
- This archive is a copied snapshot set; branch-local reports still live on their source branches.

## Missing or Skipped Sources

- `reports_curated/negative_experiments/optimize/final_s10_chain_smoother_summary.md` <- `optimize/s10-chain-pathratio-smoother:reports/final_s10_chain_smoother_summary.md` (`skipped_needs_review`)
- `reports_curated/negative_experiments/optimize/final_s11_tmag_scale_consistency_summary.md` <- `optimize/s11-tmag-scale-consistency-training:reports/final_s11_tmag_scale_consistency_summary.md` (`skipped_needs_review`)
- `reports_curated/negative_experiments/optimize/final_s12_regime_balanced_sampling_summary.md` <- `optimize/s12-regime-balanced-sampling:reports/final_s12_regime_balanced_sampling_summary.md` (`skipped_needs_review`)
- `reports_curated/negative_experiments/optimize/final_s14_local_window_pose_graph_summary.md` <- `optimize/s14-local-window-pose-graph:reports/final_s14_local_window_pose_graph_summary.md` (`skipped_needs_review`)
- `reports_curated/negative_experiments/optimize/final_s15_trajectory_level_training_objective_summary.md` <- `optimize/s15-trajectory-level-training-objective:reports/final_s15_trajectory_level_training_objective_summary.md` (`skipped_needs_review`)
- `reports_curated/negative_experiments/optimize/final_s16_stronger_visual_backbone_feasibility_summary.md` <- `optimize/s16-stronger-visual-backbone-feasibility:reports/final_s16_stronger_visual_backbone_feasibility_summary.md` (`skipped_needs_review`)
- `reports_curated/negative_experiments/optimize/final_s17_pose_supervision_dataset_quality_summary.md` <- `optimize/s17-pose-supervision-dataset-quality-audit:reports/final_s17_pose_supervision_dataset_quality_summary.md` (`skipped_needs_review`)
- `reports_curated/negative_experiments/optimize/final_s19_geometry_aware_pretraining_feasibility_summary.md` <- `optimize/s19-geometry-aware-pretraining-feasibility:reports/final_s19_geometry_aware_pretraining_feasibility_summary.md` (`skipped_needs_review`)

