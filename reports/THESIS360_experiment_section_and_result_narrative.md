# THESIS360 experiment section and result narrative

## Files generated
- `thesis/THESIS360_experiment_section.md`
- `thesis/THESIS360_results_tables.md`
- `thesis/THESIS360_ablation_analysis.md`
- `thesis/THESIS360_limitations.md`
- `thesis/THESIS360_method_result_narrative.md`
- `thesis/THESIS360_chinese_experiment_section.md`
- `thesis/THESIS360_english_experiment_section.md`
- `thesis/THESIS360_defense_talking_points.md`
- `reports/THESIS360_source_metrics_manifest.json`

## Metrics source manifest
- all quoted thesis-facing numbers were recorded in `reports/THESIS360_source_metrics_manifest.json`
- each entry tracks metric name, source file, model, split, and caveat

## Main conclusion
- `FINAL360I` is the current pair-level thesis main model.
- `FINAL360I` is clearly stronger than `T57b` and the trajectory-derived `BASE360D` component baseline on pair-level translation component metrics.
- `TRAIN360E` shows that adjacent-pair composition is feasible with full reported coverage, but sequence-level drift remains substantial.
- `SEQ360B` shows that scale/log_tmag correction improves path ratio and SE3 ATE, but not Sim3 ATE, indicating that trajectory shape remains the harder problem.

## Caveats
- `BASE360D` is an official sequence VO pipeline and should not be treated as fully homogeneous with pair-level learned inference.
- `BASE360D` component metrics are trajectory-derived, not native pair-level outputs.
- the thesis should not claim that `FINAL360I` has already surpassed official 360DVO on full trajectory ATE.
- `SEQ360A` and `STRUCT360C` remain status-only diagnostic lines and are not positive main results.

## Compliance
- no training executed
- no fine-tuning executed
- no metrics modified
- no checkpoint modified

## Recommended next task
- prepare the final thesis chapter layout and integrate these materials into the manuscript template

