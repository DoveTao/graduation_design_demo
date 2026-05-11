# S5D13 traceable dense export 报告

## 执行摘要

S5D13 的 final classification 为 `S5D13_TRACEABLE_DENSE_UNAVAILABLE_SELECTED_ONLY`。当前 final S5 可合法追溯的输出只有 selected_k1 pairwise artifact，不能生成 `453` 条 adjacent dense predictions，因此没有写入伪造的 454-frame dense TUM。

该结论直接支撑 S5D14 的分层：restored S5 dense artifact 只能作为 diagnostic artifact；traceable final S5 output 当前只有 `132/453` selected_k1 edges；S5 official locked result 不被替代。

## S5D12 recap

S5D12 发现 restored dense TUM 中 non-selected edges 的 provenance unknown，并且无法证明 dense fill/materialization 的具体 bug 机制，因为当前分支没有可验证的 fill code path 或 pre-materialization intermediate。

## Traceable export source audit

- `source_found`: true
- `source_type`: `selected_only`
- `source_location`: `external_baselines/results/s5_pairwise_replay_s5d7/final_s5_pairwise_vectors_scene01_seq03.jsonl`
- `can_generate_453_adjacent_edges`: false

当前可用 source 只包含 S5D7 selected_k1 pairwise predictions。没有使用 GT pose 生成 prediction，也没有使用 GT scale 修正 emitted_step_delta。

## Export coverage

- `num_poses`: 0
- `num_edges`: 453
- `coverage_vs_454`: 0.0
- `coverage_vs_453_edges`: 0.2913907284768212

输出的 `scene01_seq03_traceable_dense_tum.txt` 有意保持为空，因为 selected-only provenance 不能伪装成 dense trajectory。

## Edge provenance summary

| source_type | count |
| --- | ---: |
| direct_adjacent_prediction | 0 |
| selected_prediction | 132 |
| interpolated_fill | 0 |
| composed_fill | 0 |
| unavailable | 321 |

## Metrics by source type

| source_type | num_edges | path_length | tmag_median | tmag_p90 | tmag_max | tdir_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selected_prediction | 132 | 18.789319270944194 | 0.9819414718338607 | 5.680931529985728 | 7.2821664762474825 | 99.54956140857578 |
| unavailable | 0 | 0.0 | null | null | null | null |

## Dominant run 72-360 trace

- `available`: false
- `trace_path`: `external_baselines/results/s5_traceable_dense_s5d13/dominant_run_72_360_trace.json`
- `near_constant_0p134m_steps_detected`: null
- `gap_normalization_verified`: null

edges `72-360` 在 provenance 中存在记录，但 source_type 为 `unavailable`，因此不能确认 near-constant step 或 gap normalization。

## Comparison to restored dense TUM

- `compared`: false
- `matches_restored_dense`: false
- `path_length_ratio_traceable_over_restored`: null

selected-only provenance 不能复现 restored dense trajectory。

## External evaluator results if available

`external_eval.attempted=false`，因为没有生成合法 traceable dense TUM。

## Diagnosis

- `traceable_dense_available`: false
- `selected_only_blocker`: true
- `dense_fill_protocol_verified`: false
- `dense_fill_bug_confirmed`: null
- `most_likely_status`: `selected_only`

## Recommendations

- 不要把 selected-only provenance 当作 454-pose dense trajectory。
- 若后续需要 dense comparison，应新增 diagnostic-only adjacent dense inference/export，并为每条 edge 写入 provenance。
- future dense materialization path 必须暴露 `source_type`。
- 本诊断结果必须与 official S5 locked metrics 分开呈现。

## Caveats

- diagnostic only。
- traceable dense export does not replace official S5 locked result。
- S5 locked metrics/policy unchanged。
- no GT used for prediction。
