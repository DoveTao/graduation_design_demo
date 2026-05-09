# Reports Inventory Audit

- total reports count: `72`
- markdown report count: `70`
- json-like report count: `2`
- top-level report count: `72`
- subdirectory report count: `0`

## Grouped Counts

| group | count | notes |
|---|---:|---|
| `final` | 14 | final/thesis-facing |
| `ablation` | 4 | ablation and baseline summaries |
| `diagnostics` | 3 | diagnostic and audit reports |
| `negative_experiments` | 27 | negative or diagnostic experiment lines |
| `external` | 7 | external baseline / ORB-SLAM3 / environment |
| `repo_audits` | 17 | repository hygiene / cleanup / git audits |

## Duplicate / Overlap Candidates

### External baseline comparison framing

keep: protocol + comparison + one diagnostic summary
assessment: `overlapping_but_not_duplicate`
reports:
- `reports/external_algorithm_baseline_protocol.md`
- `reports/external_algorithm_baseline_comparison.md`
- `reports/alignment_consistent_strong_baseline_comparison.md`
- `reports/current_valid_baselines.md`

### S5 external export and dense audit chain

keep: S5D2 as current active dense-export audit; older compatibility audit remains diagnostic history
assessment: `same_theme_different_stage`
reports:
- `reports/s5_external_export_compatibility_audit.md`
- `reports/S5D2_dense_export_convention_audit.md`

### Final summary overlap

keep: all; they serve different thesis-facing roles
assessment: `complementary_final_set`
reports:
- `reports/final_project_mainline_summary.md`
- `reports/final_clean_results_table.md`
- `reports/final_negative_results_summary.md`
- `reports/final_thesis_claims_and_limitations.md`

### Cleanup and staging documentation

keep: retain for auditability now; good future archive target
assessment: `archive_candidate_cluster`
reports:
- `reports/cleanup_inventory_s1d5.md`
- `reports/cleanup_summary_s1d5.md`
- `reports/git_staging_s1d5_precheck.md`
- `reports/git_staging_s1d5_summary.md`
- `reports/post_commit_workspace_cleanup_plan.md`

## Active Final Reports

- `reports/final_project_mainline_summary.md`
- `reports/final_clean_results_table.md`
- `reports/final_reproducibility_guide.md`
- `reports/final_negative_results_summary.md`
- `reports/final_thesis_claims_and_limitations.md`
- `reports/thesis_experiment_section_draft.md`
- `reports/final_figure_index.md`
- `reports/final_s1d5_english_abstract.md`
- `reports/final_s1d5_mainline_report.md`
- `reports/final_s1d5_results_table.csv`
- `reports/final_s1d5_thesis_summary.md`
- `reports/final_s2_fine_refinement_summary.md`
- `reports/final_s5_clean_candidate_summary.md`
- `reports/final_s7_project_delivery_summary.md`

## Stale / Superseded Candidates

- `reports/post_s5d2_git_uncommitted_audit.md`: superseded in scope by `reports/REPO2_untracked_experiment_artifact_resolution.md`.
- `reports/current_valid_baselines.md`: useful baseline registry, but final-facing summary role overlaps with `reports/final_clean_results_table.md`.
- Cleanup and git staging notes are historical and are stronger archive candidates than active top-level references.

## Branch-Local Gaps

- No JRT1 closeout reports are present on this branch inventory.
- No MF1 closeout reports are present on this branch inventory.
- No `reports/s5_orbslam3_same_evaluator_comparison.md` is present on this branch inventory because it lives on the ORB-SLAM3 experiment branch.

## Recommended Future Directory Structure

- `reports/final/`
- `reports/ablations/`
- `reports/baselines/`
- `reports/diagnostics/`
- `reports/experiments/jrt1/`
- `reports/experiments/mf1/`
- `reports/experiments/orbslam3/`
- `reports/audits/`
- `reports/environment/`

## Retention Rules

- no files moved: `true`
- no files deleted: `true`
- no report bodies rewritten: `true`
