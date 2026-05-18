# THESIS362 Final Experiment Master Table and Visual Pack

## Scope
- this task reorganized retained experiment outputs into thesis-ready tables, figures, captions, and manifests
- no training was executed
- no full-test model re-evaluation was executed
- no metrics json file was modified
- no checkpoint was modified

## Matching / Geometry Visuals
- matching / geometry figure export was attempted in eval-only mode against the retained FINAL360I mainline
- no training was executed
- no checkpoint or metric file was modified
- default model behavior changed: `false`
- debug hooks added: `false`
- matching / geometry figures written: `false`
- status: `unavailable` for the retained FINAL360I STRUCT360B mainline because the requested `Wf_ab_raw / Wf_ab / allowed_mask / epi_bias` tensors are not emitted without intrusive model changes

## Missing / Unavailable Items
- `reports/ODOM360A_metrics_test.json`: unavailable in current worktree; trajectory metrics were derived from retained TUM exports instead
- `reports/ODOM360B_metrics_test.json`: unavailable in current worktree; trajectory metrics were derived from retained TUM exports instead
- `reports/DATA360A_regime_buckets.json`: unavailable in current worktree; bucket diagnostics were derived from retained jsonl row exports instead
- `reports/DATA360A_antiparallel_analysis.json`: unavailable in current worktree; anti-parallel analysis was derived from retained jsonl row exports instead
- `reports/FINAL360J_metrics_test.json`, `reports/FINAL360K_metrics_test.json`, `reports/FINAL360L_metrics_test.json`: unavailable in current worktree; appendix summaries use retained validation train logs instead
- `reports/TRAIN360C_metrics_test.json`: unavailable in current worktree; retained TRAIN360C numbers were recovered from aggregate results reports

## Outputs
- tables: `thesis/final_assets/tables`
- figures: `thesis/final_assets/figures`
- captions: `thesis/final_assets/captions`
- manifests: `thesis/final_assets/manifests`
