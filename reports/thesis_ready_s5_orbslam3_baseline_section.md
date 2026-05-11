# 毕设可用段落：S5 与 ORB-SLAM3 外部基线定位

本文最终方法结果以 official clean evaluator 下的 `S5_clean_tmag_calibration_policy` 为准，其 locked metrics 为 `ATE=7.352288`、`drift=1.327343`、`path_ratio=0.932379`。为补充外部几何方法参照，本文引入 ORB-SLAM3 作为 raw fisheye cam0 输入下的 verified external strong baseline。ORB-SLAM3 在成功 tracking 的片段上表现出较强的几何精度，例如 `se3` alignment 下 `ATE=0.308544`，但其 tracking coverage 为 `273/454`，`path_ratio=0.299826`，说明只覆盖了部分路径；同时 ORB-SLAM3 与 S5 的输入协议不同，不构成 same-input-protocol comparison。

S5 的 restored dense TUM 仅作为 diagnostic unverified dense artifact 使用。S5D11-S5D13 的 traceability audit 表明，该 dense artifact 中 `321` 条 non-selected edges provenance unavailable，dominant run edges `72-360` 存在明显 tmag over-scaling；当前 final S5 能合法追溯的输出只有 `132/453` selected_k1 pairwise predictions。因此，S5 dense 与 ORB-SLAM3 的 same-external-evaluator 表格只能作为诊断分析，不能作为 official main result，也不能替代 S5 locked metrics。S5 locked metrics/policy unchanged。
