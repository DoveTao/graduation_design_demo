# RESULTS360 experiment narrative

## Thesis-ready English paragraph
We evaluate the kept 360DVO mainline on the DSET2C canonical sequence-level split, which forbids random pair splitting and direct raw-directory split discovery. The current pair-level thesis main model is `FINAL360I_struct360b_final_selected`, which inherits the `STRUCT360B` match-free coarse-to-fine design and achieves test pair metrics of about `45.26 deg` signed translation-direction mean, `0.2017` anti-parallel rate, `0.8344` translation-magnitude median ratio, and `0.6404` path ratio. To expose sequence-level drift, we keep `TRAIN360E` as the trajectory composition and ATE evaluation path; it shows that strong adjacent-pair behavior does not automatically transfer to stable trajectories, with test ATE values of about `222.57 / 118.60 / 27.56` for none / SE3 / Sim3 alignment and a trajectory path ratio of `1.7563`. We retain `SEQ360B` as the only promoted sequence-scale variant because it improves the trajectory path ratio to about `1.3502` and lowers SE3 ATE to about `75.95`, although it over-corrects pair-level translation magnitude and does not improve Sim3 trajectory shape. For external context, we keep `BASE360D` as the HKUST official `360DVO` baseline and `T57b` as a legacy recovered ERP pair baseline reference; both remain important comparison anchors, but `BASE360D` component metrics are trajectory-derived and `T57b` is not a current mainline candidate. Two negative or unstable follow-up attempts are preserved only as status summaries: `SEQ360A` (`no_improvement`) and `STRUCT360C` (`evaluation_failed`).

## Thesis-ready Chinese paragraph
我们在 DSET2C canonical sequence-level split 上评估当前保留的 360DVO 主线结果。该划分明确禁止 random pair split，也不允许通过直接扫描原始序列目录来定义当前 split。当前 pair-level 论文主模型是 `FINAL360I_struct360b_final_selected`，其结构沿袭 `STRUCT360B` 的 match-free coarse-to-fine 设计，在 test 上取得约 `45.26 deg` 的 signed translation-direction mean、`0.2017` 的 anti-parallel rate、`0.8344` 的 translation-magnitude median ratio 和 `0.6404` 的 path ratio。为了暴露 sequence-level drift，我们保留 `TRAIN360E` 作为 trajectory composition 与 ATE 评估路径；结果表明，相邻 pair 上表现较强并不自动意味着轨迹稳定，其 test ATE none / SE3 / Sim3 约为 `222.57 / 118.60 / 27.56`，trajectory path ratio 约为 `1.7563`。我们保留 `SEQ360B` 作为唯一晋级的 sequence-scale variant，因为它能把 trajectory path ratio 改善到约 `1.3502`，并把 SE3 ATE 降到约 `75.95`；但它也会过度修正 pair-level translation magnitude，且没有改善 Sim3 轨迹形状误差。作为外部参照，我们继续保留 `BASE360D` 作为 HKUST official `360DVO` baseline，并保留 `T57b` 作为 legacy recovered ERP pair baseline reference；二者仍是重要比较锚点，但 `BASE360D` 的 component metrics 属于 trajectory-derived 指标，而 `T57b` 不是当前主线候选。两条失败或不稳定的后续尝试仅保留状态摘要：`SEQ360A`（`no_improvement`）和 `STRUCT360C`（`evaluation_failed`）。

## Concise bullet-point version
- visible comparison set: `FINAL360I`, `TRAIN360E`, `SEQ360B`, `BASE360D`, `T57b`
- pair-level main model: `FINAL360I_struct360b_final_selected`
- trajectory evaluation path: `TRAIN360E_sequence_trajectory_export_and_ATE_eval`
- retained sequence-scale variant: `SEQ360B_lightweight_scale_smoothing_head`
- negative or unstable experiments are not part of the visible comparison set:
  - `SEQ360A`: status summary only
  - `STRUCT360C`: status summary only
