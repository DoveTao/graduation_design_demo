# AUDIT1 experimental candidate validity and noop sweep

## 执行摘要
- final_classification = `AUDIT1_S5E15_VALID_BEST_CANDIDATE`
- best_self_developed_candidate = `S5E15`

## 为什么需要本审计
S5E19B 已确认 S5E19 存在 smoke-only、direction no-op、fallback_to_s5e15 与 evaluator 旧 reference 混入风险，因此需要系统回扫 S5E13-S5E19。

## S5E19B 触发背景
- S5E19 被确认不是可直接解释为结构无效，而是实现路径存在 smoke/no-op/fallback 风险。

## Candidate Inventory
- `S5E13` `S5E13_real_correspondence_signed_direction_candidate`: missing_files = `[]`
- `S5E14` `S5E14_observable_edge_direction_refinement_candidate`: missing_files = `[]`
- `S5E15` `S5E15_scale_deunderfit_antiparallel_candidate`: missing_files = `[]`
- `S5E16` `S5E16_confidence_calibrated_sign_scale_router`: missing_files = `[]`
- `S5E17` `S5E17_router_activation_threshold_repair_candidate`: missing_files = `[]`
- `S5E18` `S5E18_equirectangular_bearing_flow_candidate`: missing_files = `['training_status']`
- `S5E19` `S5E19_rotation_compensated_multiframe_geometry_candidate`: missing_files = `[]`

## Training Authenticity Table
- `S5E13`: training_authenticity=`real`, smoke_only=`False`, optimizer_step_count=`None`, learned_weights_saved=`None`
- `S5E14`: training_authenticity=`inference_only`, smoke_only=`False`, optimizer_step_count=`None`, learned_weights_saved=`None`
- `S5E15`: training_authenticity=`inference_only`, smoke_only=`False`, optimizer_step_count=`None`, learned_weights_saved=`None`
- `S5E16`: training_authenticity=`inference_only`, smoke_only=`False`, optimizer_step_count=`None`, learned_weights_saved=`None`
- `S5E17`: training_authenticity=`inference_only`, smoke_only=`False`, optimizer_step_count=`None`, learned_weights_saved=`None`
- `S5E18`: training_authenticity=`inference_only`, smoke_only=`None`, optimizer_step_count=`None`, learned_weights_saved=`None`
- `S5E19`: training_authenticity=`smoke_only`, smoke_only=`True`, optimizer_step_count=`None`, learned_weights_saved=`None`

## Export Authenticity Table
- `S5E13`: export_authenticity=`derived_with_declared_base`, loads_base_candidate=`True`, fallback_to_previous=`False`, source_model_consistent=`True`
- `S5E14`: export_authenticity=`derived_with_declared_base`, loads_base_candidate=`True`, fallback_to_previous=`False`, source_model_consistent=`True`
- `S5E15`: export_authenticity=`derived_with_declared_base`, loads_base_candidate=`True`, fallback_to_previous=`False`, source_model_consistent=`True`
- `S5E16`: export_authenticity=`derived_with_declared_base`, loads_base_candidate=`True`, fallback_to_previous=`False`, source_model_consistent=`True`
- `S5E17`: export_authenticity=`derived_with_declared_base`, loads_base_candidate=`True`, fallback_to_previous=`False`, source_model_consistent=`True`
- `S5E18`: export_authenticity=`fallback_risk`, loads_base_candidate=`True`, fallback_to_previous=`True`, source_model_consistent=`True`
- `S5E19`: export_authenticity=`fallback_risk`, loads_base_candidate=`True`, fallback_to_previous=`True`, source_model_consistent=`True`

## Evaluator Authenticity Table
- `S5E13`: evaluator_authenticity=`mixed_ref`, hardcoded_previous_ref=`True`, metrics_from_eval_json=`True`
- `S5E14`: evaluator_authenticity=`mixed_ref`, hardcoded_previous_ref=`True`, metrics_from_eval_json=`True`
- `S5E15`: evaluator_authenticity=`mixed_ref`, hardcoded_previous_ref=`True`, metrics_from_eval_json=`True`
- `S5E16`: evaluator_authenticity=`mixed_ref`, hardcoded_previous_ref=`True`, metrics_from_eval_json=`True`
- `S5E17`: evaluator_authenticity=`fresh_eval`, hardcoded_previous_ref=`False`, metrics_from_eval_json=`True`
- `S5E18`: evaluator_authenticity=`fresh_eval`, hardcoded_previous_ref=`False`, metrics_from_eval_json=`True`
- `S5E19`: evaluator_authenticity=`mixed_ref`, hardcoded_previous_ref=`True`, metrics_from_eval_json=`True`

## No-op Pairwise Diff Table
- `S5E13_vs_S5E14`: trajectory_near_identical=`True`, edge_metrics_near_identical=`True`, likely_noop=`True`
- `S5E14_vs_S5E15`: trajectory_near_identical=`False`, edge_metrics_near_identical=`False`, likely_noop=`False`
- `S5E15_vs_S5E16`: trajectory_near_identical=`False`, edge_metrics_near_identical=`False`, likely_noop=`False`
- `S5E16_vs_S5E17`: trajectory_near_identical=`False`, edge_metrics_near_identical=`False`, likely_noop=`False`
- `S5E17_vs_S5E18`: trajectory_near_identical=`True`, edge_metrics_near_identical=`True`, likely_noop=`True`
- `S5E15_vs_S5E19`: trajectory_near_identical=`False`, edge_metrics_near_identical=`False`, likely_noop=`True`

## S5E15 deep validity conclusion
{
  "valid_as_best_self_developed_composite_candidate": true,
  "independent_of_s5e14": false,
  "scale_improvement_authentic": true,
  "uses_eval_gt_calibration": false,
  "fallback_risk": false,
  "hardcoded_metric_risk": false,
  "traceability_ok": true,
  "final_recommendation": "keep_as_best_candidate"
}

## 哪些实验可以作为 positive result
S5E13, S5E15

## 哪些实验只能作为 negative / diagnostic
S5E14, S5E16, S5E17, S5E18, S5E19

## 是否仍推荐 S5E15 作为 best self-developed composite candidate
`S5E15`

## 后续修复建议
- 先修复 S5E18 的 trajectory/metrics copy 路径，再把 bearing-flow 只作为真实增量输入。
- 不要再把 S5E19 类 smoke/no-op 结果解释成结构本身无效。
- 继续使用 S5E15 作为当前 best self-developed composite candidate，但明确其是 derived candidate，不是 full fresh training winner。
