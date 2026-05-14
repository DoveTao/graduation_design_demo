# Final S1d5 Mainline Report

## A. Executive Summary

当前最终 clean mainline 是 `S1d5 exported clean dt-anchor + rot policy`。它从 `true T57b multiscale baseline` 出发，严格使用 `explicit-cfg / unexpected=0` 口径，policy 由 `train_only_leave_one_train_seq_out_cv` 选择，不使用 test label 拟合，也不依赖 historical scalar-load artifact。

相对于真实基线，S1d5 在三个核心指标上同时取得改进：
- drift: `1.494 -> 1.396`
- ATE: `9.644 -> 7.632`
- path_ratio: `0.496 -> 0.935`

这说明 S1d5 不仅修复了原始的 scale collapse，也保持了较好的轨迹质量和可复现性。

## B. Problem Diagnosis

历史上记录的 `T57b/F1c path_ratio≈0.983` 已经被确认不是 `true T57b multiscale baseline`，而是 `scalar-load mismatch artifact`。在修复 eval harness 的 checkpoint cfg 恢复逻辑后，真实的 `T57b multiscale` 基线应为：
- drift = `1.494`
- ATE = `9.644`
- path_ratio = `0.496`

S1 系列审计进一步表明，当前 path_ratio 异常的主因是 `scale amplitude`，不是 eval pair list 偶然性，也不是 direction-only path length collapse。关键证据是：
- `direction_only path_ratio ≈ 1.0`
- `metric path_ratio = 0.496`

这意味着方向链路本身并未严重塌缩，真正塌的是 translation magnitude 的尺度。

## C. Method

S1d5 的方法可以概括为：
- `dt-bucket trajectory scale anchor`
- `rot-only fusion`

bucket 结构为：
- `[0.1,0.3)`
- `[0.3,0.5)`
- `[0.5,1.0)`

最终使用的有效 bucket factors 为：
- `[0.1,0.3) = 1.302368139324791`
- `[0.3,0.5) = 2.6003610319428594`
- `[0.5,1.0) = 3.9662143503139955`

其余策略参数为：
- `alpha = 1.05`
- `fine_rot = 0.40`
- `fine_tdir = 0.0`
- `fine_tmag = 0.0`
- `use_geometry_refine = False`

这些参数不是通过 test sweep 直接确定的，而是来自：
- `train_only_leave_one_train_seq_out_cv`

因此，S1d5 是一个 clean、可复用、可审计的 exported policy，而不是 test-selected diagnostic。

## D. Evaluation Protocol

评估使用的 base checkpoint 为：
- `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`

评估协议要求：
- explicit cfg restored
- `load missing/unexpected = 2 / 0`
- `selected_k = 1`
- `available_k = [1,2,3,5,10,20]`
- `num_pairs = 132`
- `num_chains = 19`
- no test-label selection

这保证了所有 clean 结果都处于统一且可解释的 eval harness 下。

## E. Results Table

| method | status | drift | ATE | path_ratio | explicit_cfg | unexpected_count | selected_by_train_only | test_labels_used_for_selection | notes |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| true T57b multiscale 0/0 | valid clean baseline | 1.494 | 9.644 | 0.496 | yes | 0 | n/a | false | true baseline |
| true T57b multiscale + rot-only | valid eval-only diagnostic | 1.310 | 7.611 | 0.496 | yes | 0 | n/a | false | improves drift/ATE but not scale |
| S1d2 0/0 | clean scale-anchor candidate | 1.579 | 11.487 | 0.901 | yes | 0 | yes | false | repairs scale but hurts ATE |
| S1d2 + rot-only | clean scale-anchor + rot diagnostic | 1.367 | 7.749 | 0.901 | yes | 0 | false | false | strong combined diagnostic |
| S1d3 best alpha+rot | stable diagnostic target, not clean fitted mainline | 1.433 | 7.971 | 0.970 | yes | 0 | false | true | test-swept diagnostic only |
| S1d4 train-selected policy | clean train-CV selected mainline | 1.396 | 7.632 | 0.935 | yes | 0 | yes | false | clean selected policy |
| S1d5 exported policy | current clean exported mainline policy | 1.396358 | 7.632463 | 0.934982 | yes | 0 | yes | false | exported and reproducible |
| historical scalar-load 0/0 | invalid as true baseline | 1.786 | 11.573 | 0.983 | no | 15 | no | n/a | scalar-load mismatch artifact |
| historical scalar-load rot-only | invalid as true baseline | 1.491 | 7.973 | 0.983 | no | 15 | no | n/a | scalar-load mismatch artifact |

## F. Figures Section

图像产物位于：
- [S1d5_final_figures](/home/dovetao/graduation_design_demo/checkpoints/S1d5_final_figures)

包含：
- [baseline_bar_path_ratio.png](/home/dovetao/graduation_design_demo/checkpoints/S1d5_final_figures/baseline_bar_path_ratio.png)
- [baseline_bar_ATE_drift.png](/home/dovetao/graduation_design_demo/checkpoints/S1d5_final_figures/baseline_bar_ATE_drift.png)
- [tmag_distribution_s1d5.png](/home/dovetao/graduation_design_demo/checkpoints/S1d5_final_figures/tmag_distribution_s1d5.png)
- [trajectory_comparison_s1d5.png](/home/dovetao/graduation_design_demo/checkpoints/S1d5_final_figures/trajectory_comparison_s1d5.png)
- [scale_repair_summary.png](/home/dovetao/graduation_design_demo/checkpoints/S1d5_final_figures/scale_repair_summary.png)

## G. Limitations

需要明确说明：
- `ATE≈7.63` 在绝对量纲上仍然偏高，不能被表述为工程高精度结果。
- S1d5 是一个 `exported calibration/eval policy`，不是已经完全 internalized 的 end-to-end 训练模型。
- 当前 `dt-bucket factors` 仍需在更多 scenes / sequences 上做更广泛验证。
- S1d5 不应与 historical scalar-load results 混淆。
- `S1d6` 只是 future work proposal，并未运行。

## H. Final Verdict

- `S1d5 = current clean exported mainline`
- `F1d remains paused`
- `S1d6 remains proposal only`
