# S5 与 ORB-SLAM3 最终比较：traceability caveats

## Executive summary / 执行摘要

S5 official locked result 仍是本文最终结果：`S5_clean_tmag_calibration_policy` 在 official clean evaluator 下的 locked metrics 为 `ATE=7.352288`、`drift=1.327343`、`path_ratio=0.932379`。该结果不被任何 external diagnostic 替代。

ORB-SLAM3 是 verified external strong baseline，使用 raw fisheye cam0 输入和 fitted KB8 compatibility approximation，在 `273/454` frames 上成功 tracking。它在成功 tracking 片段上显示出很强的几何精度，但 coverage 不完整，并且与 S5 不是 same-input-protocol。

restored S5 dense artifact 只能作为 diagnostic artifact。S5D11/S5D12/S5D13 已确认：该 dense TUM 中 `321` 条 non-selected edges provenance unavailable，dominant run edges `72-360` 存在系统性 tmag over-scaling，且当前无法验证它是 final S5 的 traceable dense export。

traceable S5 final output 当前只有 selected_k1 `132/453` edges。因此，S5 dense vs ORB-SLAM3 same-evaluator table 只能作为 diagnostic comparison，不能作为 official main result。

## Result hierarchy / 结果层级

| Result type | Method / artifact | Source | Coverage | Evaluator | Status | Can be used as official? | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S5 official locked result | `S5_clean_tmag_calibration_policy` | `checkpoints/S5_clean_tmag_calibration_policy.json` | official clean test protocol | official clean evaluator | official locked result | yes | S5 locked metrics/policy unchanged |
| ORB-SLAM3 external baseline | ORB-SLAM3 fisheye cam0 | `external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt` | `273/454` | external trajectory evaluator | verified external ORB-SLAM3 baseline | no | partial tracking; fitted KB8 calibration; raw fisheye cam0 input |
| restored S5 dense diagnostic artifact | restored S5 dense TUM | `external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt` | `454/454` | external trajectory evaluator | diagnostic unverified dense artifact | no | restored dense artifact is diagnostic-only; not traceable final S5 dense export |
| traceable S5 selected_k1 artifact | selected_k1 pairwise vectors | `external_baselines/results/s5_pairwise_replay_s5d7/final_s5_pairwise_vectors_scene01_seq03.jsonl` | `132/453` edges | pairwise diagnostic | traceable selected_k1 artifact | no | sparse/disconnected; cannot generate 453 adjacent dense predictions |

## S5 official locked result

| candidate | ATE | drift | path_ratio | evaluator | status |
| --- | ---: | ---: | ---: | --- | --- |
| `S5_clean_tmag_calibration_policy` | 7.352288 | 1.327343 | 0.932379 | official clean evaluator | official locked final |

该结果是本文 S5 方法的主结果。locked metrics unchanged，不被 restored dense diagnostic artifact、traceable selected_k1 artifact 或 ORB-SLAM3 external baseline 替代。

## ORB-SLAM3 external strong baseline

| alignment | ATE | drift | path_ratio | coverage |
| --- | ---: | ---: | ---: | ---: |
| none | 11.46685113827757 | 0.03737053832593786 | 0.2998258665660257 | 273/454 |
| se3 | 0.30854441069248173 | 0.0374760642069601 | 0.2998258665660257 | 273/454 |
| sim3 | 0.224292165986624 | 0.06452708470276013 | 0.2998258665660257 | 273/454 |

ORB-SLAM3 使用 raw fisheye cam0 输入，并带有 fitted KB8 compatibility approximation。它在成功 tracking 的片段上几何精度很强；但 coverage 为 `273/454=0.6013215859030837`，`path_ratio=0.299826`，说明覆盖路径不完整。因此它是 strong external baseline，不是 S5 替代物。

## Restored S5 dense diagnostic artifact

| alignment | ATE | drift | path_ratio | coverage |
| --- | ---: | ---: | ---: | ---: |
| none | 21.681522044975264 | 0.21683137477814862 | 2.777267572676944 | 454/454 |
| se3 | 8.231468716451547 | 0.23274388321733164 | 2.777267572676944 | 454/454 |
| sim3 | 4.07912293550008 | 0.13382808429229665 | 2.777267572676944 | 454/454 |

S5D11 定位到 full dense path length issue 主要来自 long non-selected runs，而不是少数 outliers。non-selected edges 的 `tmag_median_ratio=29.375292`，`tmag_p90_ratio=71.221354`，`tmag_p95_ratio=88.130943`，`tmag_max_ratio=335.512628`。dominant run edges `72-360` 长度为 `289`，estimated path 为 `39.022165`，GT path 为 `1.557407`，median tmag ratio 为 `32.228312`。

S5D12 的 final classification 为 `S5D12_DENSE_FILL_PROTOCOL_MISMATCH_IDENTIFIED`：dense export/fill code path found = false，pre-materialization intermediate available = false，nonselected source type = unknown。S5D13 的 final classification 为 `S5D13_TRACEABLE_DENSE_UNAVAILABLE_SELECTED_ONLY`：current final S5 can only be legally traced to selected_k1 pairwise predictions。

因此 restored S5 dense artifact 只用于诊断，不能作为 official dense S5 result。

## Traceable S5 selected_k1 artifact

| source | edges | coverage_vs_453_edges | rot_mean_deg | tdir_mean_deg | tdir_abs_mean_deg | tmag_median_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selected_k1 pairwise artifact | 132 | 0.2913907284768212 | 20.835182 | 99.549561 | 41.117887 | 0.981941 |

selected_k1 的 tmag 相对正常，但 tdir 较差。更重要的是，selected_k1 是 sparse/disconnected protocol，不能生成 453 adjacent dense predictions。因此它是 traceable selected_k1 artifact，不是 dense external trajectory。

## What can and cannot be claimed / 可以声称与不能声称

### 可以声称

- S5 official locked result remains the final method result。
- ORB-SLAM3 is a verified external strong baseline on raw fisheye cam0。
- restored S5 dense artifact is useful for diagnostic analysis。
- Final S5 traceable output currently selected_k1 only。
- Dense S5 vs ORB-SLAM3 same-external-evaluator table is diagnostic, not official。

### 不能声称

- S5 has a verified 454-frame traceable dense external export。
- restored S5 dense TUM is official final S5 dense result。
- S5 and ORB-SLAM3 have identical input protocol。
- ORB-SLAM3 replaces S5。
- restored dense diagnostic metrics replace S5 locked metrics。
- S5 dense same-evaluator comparison is a fair official main table。

## Thesis-ready paragraph / 毕设可用段落

本文最终方法结果以 official clean evaluator 下的 `S5_clean_tmag_calibration_policy` 为准，其 locked metrics 为 `ATE=7.352288`、`drift=1.327343`、`path_ratio=0.932379`。为提供外部几何方法参照，本文引入 ORB-SLAM3 作为 raw fisheye cam0 输入下的 verified external strong baseline。ORB-SLAM3 在成功 tracking 的片段上具有较强几何精度，但其 coverage 为 `273/454`，且与 S5 的 panorama/equirectangular 输入协议不同，因此不能替代 S5 official result。另一方面，restored S5 dense TUM 只作为 diagnostic artifact：S5D11-S5D13 的 traceability audit 表明，其 non-selected dense edges 存在 provenance unavailable 和 long-run tmag over-scaling 问题，当前 final S5 可合法追溯的输出只有 `132/453` selected_k1 edges。因此，S5 dense 与 ORB-SLAM3 的 same-external-evaluator 表格仅用于诊断分析，不作为 official main comparison；S5 locked metrics/policy unchanged。

## Recommendations / 建议

- 正文主表使用 S5 official locked result。
- ORB-SLAM3 放在 external strong baseline 小节，明确 partial tracking 和 same-input-protocol caveat。
- restored S5 dense artifact 仅放在诊断小节，不写成 verified final S5 dense export。
- selected_k1 artifact 用于 traceability 和 pairwise 诊断，不写成 dense trajectory。
- 后续若需要真正 dense comparison，应新增 diagnostic-only adjacent dense inference/export，并为每条 edge 写入 provenance。

## Caveats

- diagnostic only。
- no official S5 result replacement。
- S5 locked metrics/policy unchanged。
- S5 and ORB-SLAM3 are not same-input-protocol。
- restored dense artifact is diagnostic-only。
