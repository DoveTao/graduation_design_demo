# S5 dense external export 诊断报告

## 执行摘要

本报告只记录 restored S5 dense diagnostic artifact 的外部 evaluator 指标和 traceability caveat。该 TUM 文件覆盖 `454/454` poses，但 S5D12/S5D13 已确认它不是 verified final S5 dense export：当前 final S5 可合法追溯的输出只有 `132/453` selected_k1 edges，其余 `321` edges unavailable。

因此，`external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt` 只能作为 diagnostic unverified dense artifact，不能作为 official S5 dense result，不能替代 `S5_clean_tmag_calibration_policy` 的 official locked result。

## restored dense artifact 指标

| alignment | ATE | drift | path_ratio | coverage | status |
| --- | ---: | ---: | ---: | ---: | --- |
| none | 21.681522044975264 | 0.21683137477814862 | 2.777267572676944 | 454/454 | diagnostic only |
| se3 | 8.231468716451547 | 0.23274388321733164 | 2.777267572676944 | 454/454 | diagnostic only |
| sim3 | 4.07912293550008 | 0.13382808429229665 | 2.777267572676944 | 454/454 | diagnostic only |

## traceability caveat

S5D11 显示，restored dense TUM 的 non-selected edges 有系统性 tmag over-scaling：

| subset | num_edges | path_length_fraction | tmag_median_ratio | tmag_p90_ratio | tmag_p95_ratio | tmag_max_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| non-selected | 321 | 0.606273 | 29.375292 | 71.221354 | 88.130943 | 335.512628 |

dominant run 是 edges `72-360`，length `289`，estimated path `39.022165`，GT path `1.557407`，median tmag ratio `32.228312`。S5D12 未找到可验证的 dense fill/materialization code path，也没有 pre-materialization intermediate；S5D13 确认当前 traceable final S5 output is selected_k1 only。

## 可用结论

- restored dense artifact 可用于诊断 non-selected edge over-scaling。
- restored dense artifact 可用于说明 dense materialization/provenance 风险。
- restored dense artifact 不可写成正式 dense S5 export。
- restored dense artifact 不可作为 official main result。
- S5 official locked result remains the primary S5 result。

## Caveats

- diagnostic only。
- no GT used for prediction。
- S5 locked metrics/policy unchanged。
- restored dense diagnostic metrics do not replace S5 locked metrics。
