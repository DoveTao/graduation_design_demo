# 第 X 章 实验与结果分析

## 1. 实验数据与划分
本文实验基于 DSET2C canonical split 展开。该划分建立在 360DVO/HKUST 数据基础之上，采用严格的 sequence-level train/val/test protocol，并通过 manifest-native 数据入口进行样本读取。也就是说，当前实验流程通过 canonical JSONL manifests 定义训练集、验证集和测试集，而不是通过直接扫描原始序列目录来发现 split，也不使用 random pair split。这样做的目的是保证实验协议稳定、可复现，并避免 sequence identity 在不同 split 之间泄漏。

在训练与实验约束方面，本文不使用 ORB-SLAM3 teacher，不使用 HKUST 360DVO teacher，也不使用 BASE360 输出作为训练输入。与此同时，本文方法始终保持 match-free 设定，不引入显式特征匹配、RANSAC、PnP 或 bundle adjustment。因此，本文的贡献边界被明确限定在全景图像对的相对位姿估计问题，而不是完整 sequence VO pipeline 的工程替代。

## 2. 对比方法
当前保留的主比较集合包含五项。第一，`FINAL360I_struct360b_final_selected` 是本文当前的 pair-level 主模型，其结构来源于 `STRUCT360B_match_free_coarse_to_fine`。第二，`TRAIN360E` 并不是独立模型，而是将 `FINAL360I` 的相邻 pair 相对位姿预测组合成完整轨迹后，再进行 trajectory-level ATE 评价的路径。第三，`SEQ360B_lightweight_scale_smoothing_head` 是唯一保留的 sequence-scale variant，通过轻量的 log-scale correction 研究 trajectory 中的 scale/path drift。第四，`BASE360D` 是 HKUST official 360DVO external baseline；需要强调的是，其 component metrics 属于 trajectory-derived 指标，并不是与本文完全同接口的 native pair-level prediction。第五，`T57b` 是一个 recovered legacy baseline，仅作为外部历史参考。

此外，`SEQ360A` 与 `STRUCT360C` 仅保留状态摘要，不进入主结果表。前者结论为 `no_improvement`，后者结论为 `evaluation_failed`。因此，二者只能作为诊断尝试进行简要讨论，不能被表述为本文的有效提升结果。

## 3. 评价指标
本文评价分为 pair-level 与 trajectory-level 两层。pair-level 部分主要关注 translation component metrics。`signed_tdir_mean` 衡量预测平移方向与真实平移方向之间的平均夹角，越小越好；`anti_parallel_rate` 衡量预测方向落入反向半球的比例，越低越好；`tmag_median_ratio` 衡量预测平移尺度与真实尺度之比的中位数，越接近 1 越理想；`path_ratio` 则使用全体 pair 的预测尺度和与真实尺度和之比来反映整体尺度平衡情况。

trajectory-level 部分则通过把相邻 pair 的相对位姿组合成完整 sequence trajectory 来评价累计误差。`ATE none` 衡量不做额外刚体对齐时的原始轨迹误差，`ATE SE3` 衡量刚体对齐后的误差，`ATE Sim3` 衡量相似变换对齐后的误差，`trajectory_path_ratio` 则反映预测轨迹长度与真实轨迹长度之间的整体偏差。需要明确的是，pair-level 指标与 trajectory-level 指标衡量的是两个不同层面的能力：前者关注局部相对位姿质量，后者关注长序列组合后的累计漂移。

## 4. Pair-level 实验结果
在 pair-level 指标上，`FINAL360I` 的 test `signed_tdir_mean` 为 `45.264702006380205`，`anti_parallel_rate` 为 `0.20169893322797314`，`tmag_median_ratio` 为 `0.8344251368086006`，`path_ratio` 为 `0.6403519796204528`。相比之下，`T57b` 的对应数值约为 `111.964932`、`0.674672`、`0.175156` 和 `0.148787`。这说明，本文主模型在 translation direction 与 scale-related component behavior 上都相对这一 legacy baseline 获得了显著改善。

若与 `BASE360D` trajectory-derived component baseline 比较，差异同样明显。`BASE360D` 在 test component 层面的 `signed_tdir_mean` 为 `128.402578`，`anti_parallel_rate` 为 `0.814895`，`tmag_median_ratio` 为 `0.039910`，`pair_component_path_ratio` 为 `0.043542`。从这些数值看，`FINAL360I` 在 pair-level translation component metrics 上明显优于 `BASE360D`。不过，这一比较必须保留 caveat：`BASE360D` 的 component metrics 来自 official sequence trajectory 的后处理，而不是与本文模型完全一致的 pair-level forward interface，因此两者只能视为 partial comparability，而不能视为完全同质对比。

综合而言，最稳健的结论是：`FINAL360I` 已经在 pair-level panoramic relative pose estimation 上取得了有意义且稳定的提升，这也是本文最核心、最可信的实验结论。

## 5. Trajectory-level 实验结果
为了分析 pair-level 成功是否能够自然传递到 sequence-level，本文保留了 `TRAIN360E` 作为 trajectory composition evaluation。该实验表明，`FINAL360I` 的相邻 pair 预测可以组成完整 trajectory，并在保留的 test split 上达到 `1.0` 的 reported coverage；但与此同时，direct adjacent-pair composition 仍会产生明显 drift。其 test `ATE none`、`ATE SE3` 与 `ATE Sim3` 分别为 `222.56856382785097`、`118.6037794846689` 和 `27.564661865900444`，`trajectory_path_ratio` 为 `1.756343083453392`。这说明，局部相对位姿质量较强，并不自动意味着长程序列组合后的全局轨迹稳定。

与之相比，`BASE360D` 作为 official sequence VO pipeline，在 trajectory-level 仍然保有明显优势。保留报告给出的 test `ATE none`、`ATE SE3`、`ATE Sim3` 分别为 `103.256252`、`79.295214` 和 `2.974466`。因此，本文不能声称 `FINAL360I` 在完整 trajectory ATE 上已经全面超过 HKUST official 360DVO。更准确的表述应该是：本文在 pair-level component 层面有清晰进展，但 sequence-level 误差控制仍然是尚未解决的问题。

## 6. SEQ360B 变体分析
`SEQ360B` 的设计目标不是改变旋转或平移方向本身，而是在冻结 `FINAL360I` 主模型的前提下，通过轻量化的 log-scale correction 缓解 scale/path drift。实验结果表明，这一思路在 trajectory-level 上具有一定效果：其 test `trajectory_path_ratio` 从 `TRAIN360E` 的 `1.756343083453392` 改善到 `1.3502369615185652`，`ATE SE3` 从 `118.6037794846689` 降到 `75.94691348103409`，`ATE none` 也从 `222.56856382785097` 降到 `139.22717463963536`。

然而，`SEQ360B` 的 `ATE Sim3` 基本没有改善，从 `27.564661865900444` 变为 `27.567211313835486`。这一点非常关键：它说明 scale/path correction 的确能改善尺度与路径长度相关的累计误差，但并没有解决 trajectory shape error。换言之，残余误差更可能来自 rotation accumulation、direction accumulation 或更高层的 sequence-level inconsistency，而不仅仅是 scale drift。

同时还应看到，`SEQ360B` 的 pair-level `tmag_median_ratio` 达到 `1.6978484631707425`，明显偏离理想的 `≈1` 区间。这意味着它在 trajectory path 层面的收益伴随着 pair-level scale over-correction 风险。因此，`SEQ360B` 更适合被解释为一个“揭示问题结构”的 partial variant，而不是最终主模型。

## 7. 诊断实验
除了 `SEQ360B` 以外，当前还保留两条诊断尝试。`SEQ360A` 的状态是 `no_improvement`，说明简单的局部 clip consistency 约束并没有有效降低 trajectory drift。`STRUCT360C` 的状态是 `evaluation_failed`，说明 rotation-aware fine refinement 这一结构挑战者在当前条件下尚未稳定到可作为主线结果讨论的程度。因此，这两项实验只能用于说明“哪些方向尝试过但没有形成有效提升”，而不能被当作积极结果纳入主比较表。

## 8. 结果讨论
从整体实验结果看，本文最可靠的贡献应聚焦于 match-free panoramic pair-level relative pose estimation。`FINAL360I` 明显提升了 pair-level translation component metrics，这一改进相较 `T57b` 与 trajectory-derived 的 `BASE360D` component baseline 都是清晰的。然而，`TRAIN360E` 与 `SEQ360B` 又共同表明：pair-level success 与 full trajectory superiority 之间存在明显鸿沟。`TRAIN360E` 证明直接序列组合可行，但 drift 显著；`SEQ360B` 进一步证明单纯 scale/path correction 只能部分缓解问题，而不能解决 trajectory shape。

因此，本文方法不应被表述为“完整 VO pipeline 的替代品”，而应被准确表述为“对全景图像对相对位姿 component estimation 的改进”。这一定位既符合当前实验事实，也更能凸显本文的研究边界与真实贡献。

## 9. 本章小结
本章围绕 DSET2C canonical split 下的实验结果展开分析。结果表明，`FINAL360I` 是当前最稳健的 pair-level 主模型，在 translation direction 与 scale-related component metrics 上显著优于 `T57b` 和 trajectory-derived 的 `BASE360D` component baseline。与此同时，`TRAIN360E` 说明 direct adjacent-pair composition 仍存在明显 sequence-level drift，`SEQ360B` 说明 scale/log_tmag correction 能改善 path ratio 和 SE3 ATE，但不能改善 Sim3 ATE。综合来看，本文已经在 match-free panoramic pair-level relative pose estimation 上取得了可信的进展，而未来工作应进一步转向 sequence-level refinement、global scale calibration 和 lightweight trajectory optimization。
