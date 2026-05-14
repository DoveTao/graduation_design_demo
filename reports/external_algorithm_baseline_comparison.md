# 外部算法基线比较与 traceability caveat

## 执行摘要

本报告用于论文中的外部算法基线定位，但不把外部诊断结果替换为 S5 official locked result。`S5_clean_tmag_calibration_policy` 仍是 official locked result，locked metrics 为 `ATE=7.352288`、`drift=1.327343`、`path_ratio=0.932379`。

ORB-SLAM3 可以作为 verified external ORB-SLAM3 baseline，因为它在 raw fisheye cam0 输入上生成了可评估轨迹；但它与 S5 不是 same-input-protocol，且 coverage 只有 `273/454`。restored S5 dense artifact is diagnostic-only，不能写成正式 dense S5 export。S5D13 进一步确认，当前可追溯的 final S5 输出只有 traceable selected_k1 artifact：`132/453` edges 可追溯，`321` edges unavailable。

## 结果层级

| Result type | Method / artifact | Source | Coverage | Evaluator | Status | Can be used as official? | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S5 official locked result | `S5_clean_tmag_calibration_policy` | official clean evaluator | official test protocol | official clean evaluator | official locked result | yes | S5 locked metrics/policy unchanged |
| ORB-SLAM3 external baseline | ORB-SLAM3 fisheye cam0 | `external_baselines/results/orbslam3_fisheye_cam0/scene01_seq03_est_tum.txt` | `273/454` | external trajectory evaluator | verified external ORB-SLAM3 baseline | no | partial tracking, fitted KB8 compatibility approximation, raw fisheye cam0 input |
| restored S5 dense diagnostic artifact | restored dense TUM | `external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt` | `454/454` | external trajectory evaluator | diagnostic unverified dense artifact | no | restored dense artifact is diagnostic-only; not verified as final traceable S5 dense export |
| traceable S5 selected_k1 artifact | selected pairwise artifact | `external_baselines/results/s5_pairwise_replay_s5d7/final_s5_pairwise_vectors_scene01_seq03.jsonl` | `132/453` edges | pairwise diagnostic | traceable selected_k1 artifact | no | sparse/disconnected selected_k1 protocol; cannot generate 453 adjacent dense predictions |

## 定量结果

### S5 official locked result

| candidate | ATE | drift | path_ratio | status |
| --- | ---: | ---: | ---: | --- |
| `S5_clean_tmag_calibration_policy` | 7.352288 | 1.327343 | 0.932379 | official locked final |

### ORB-SLAM3 verified external baseline

| alignment | ATE | drift | path_ratio | coverage |
| --- | ---: | ---: | ---: | ---: |
| none | 11.46685113827757 | 0.03737053832593786 | 0.2998258665660257 | 0.6013215859030837 |
| se3 | 0.30854441069248173 | 0.0374760642069601 | 0.2998258665660257 | 0.6013215859030837 |
| sim3 | 0.224292165986624 | 0.06452708470276013 | 0.2998258665660257 | 0.6013215859030837 |

ORB-SLAM3 在成功 tracking 的片段上几何精度很强，特别是 `se3/sim3` alignment 后的 ATE 很低。但 `path_ratio=0.299826` 和 `273/454` coverage 说明它只覆盖了部分路径，因此是 strong external baseline，不是 S5 替代物。

### restored S5 dense diagnostic artifact

| alignment | ATE | drift | path_ratio | coverage |
| --- | ---: | ---: | ---: | ---: |
| none | 21.681522044975264 | 0.21683137477814862 | 2.777267572676944 | 1.000000 |
| se3 | 8.231468716451547 | 0.23274388321733164 | 2.777267572676944 | 1.000000 |
| sim3 | 4.07912293550008 | 0.13382808429229665 | 2.777267572676944 | 1.000000 |

该表只能称为 diagnostic same-external-evaluator comparison using restored S5 dense artifact。S5D11/S5D12/S5D13 已确认，restored dense TUM 中 `321` 条 non-selected edges provenance unavailable，并且 edges `72-360` 的 long run 驱动了 tmag over-scaling。因此它不能作为 official main result。

## 论文可用解释

外部算法基线用于补充说明传统几何方法在同一序列上的潜力与限制。ORB-SLAM3 在 raw fisheye cam0 输入下取得了较强的局部几何精度，但其 tracking coverage 仅为 `273/454`，且与 S5 的 panorama/equirectangular 输入协议不同。S5 的最终结果仍以 official clean evaluator 下的 locked metrics 为准。restored S5 dense artifact 只保留为诊断材料，用于分析 dense materialization 中 non-selected edges 的异常 scale 行为；它不替代 S5 official locked result，也不能作为 fair official main table。

## Caveats

- S5 and ORB-SLAM3 are not same-input-protocol。
- restored dense artifact is diagnostic-only。
- traceable final S5 output is selected_k1 only。
- restored dense diagnostic metrics do not replace S5 locked metrics。
- S5 locked metrics/policy unchanged。
- Published results from different datasets are not compared directly against S5.
