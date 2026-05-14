# THESIS360 中文实验章节

## 实验设置
本文实验基于 DSET2C canonical split 展开。该划分来源于 360DVO/HKUST 数据体系，并采用严格的 sequence-level train/val/test protocol。当前仓库中的主线实现使用 manifest-native 数据协议，即通过 canonical JSONL manifests 读取 pair 样本，而不是通过直接扫描原始序列目录来定义当前 split；同时明确禁止 random pair split，以避免 sequence leakage 或实验协议漂移。

本文评价分为两个层次。第一类是 pair-level component metrics，包括 signed translation-direction mean、anti-parallel rate、translation-magnitude median ratio 和 path ratio，用于评价相对位姿预测中的方向与尺度质量。第二类是 trajectory-level metrics，即通过相邻 pair 的相对位姿进行 sequential composition 后，再计算 ATE none、ATE SE3、ATE Sim3 与 trajectory path ratio，用于评价序列级误差累积行为。

在训练与实验协议方面，本文不使用 ORB-SLAM3 teacher，不使用 HKUST 360DVO teacher，不使用 BASE360 输出作为训练输入，也不引入 explicit matching、RANSAC、PnP 或 bundle adjustment。因此，本文的主要研究对象始终是 match-free panoramic pair-level relative pose estimation。

## 对比方法
当前保留的主比较集合包含五项。`FINAL360I_struct360b_final_selected` 是当前 pair-level 论文主模型；其结构来源是 `STRUCT360B_match_free_coarse_to_fine`。`TRAIN360E` 是将 `FINAL360I` 的相邻 pair 预测组合成完整 trajectory 后进行 ATE 评估的路径。`SEQ360B` 是唯一保留的 sequence-scale variant，通过轻量化的 log-scale correction 研究 trajectory 中的 scale/path drift。`BASE360D` 是 HKUST official 360DVO external baseline，其 component metrics 来自 trajectory-derived post-processing。`T57b` 是恢复得到的 legacy baseline，仅作为外部参考。

此外，`SEQ360A` 与 `STRUCT360C` 只保留状态摘要，不进入主比较表。前者结论为 `no_improvement`，后者结论为 `evaluation_failed`。

## 实验结果
在 pair-level 指标上，`FINAL360I` 的 test signed translation-direction mean 为 45.264702，anti-parallel rate 为 0.201699，translation-magnitude median ratio 为 0.834425，path ratio 为 0.640352。相比之下，`T57b` 仅达到约 111.964932、0.674672、0.175156、0.148787；trajectory-derived 的 `BASE360D` component baseline 则为 128.402578、0.814895、0.039910、0.043542。因此，在 pair-level translation component metrics 上，`FINAL360I` 明显优于这两个外部参考。

但 trajectory-level 结果显示，pair-level 强并不自动意味着 sequence-level 强。`TRAIN360E` 虽然在 test split 上实现了 `1.0` 的 reported coverage，并能够导出完整 trajectory，但其 test ATE none / SE3 / Sim3 分别为 222.568564 / 118.603779 / 27.564662，trajectory path ratio 为 1.756343，说明直接相邻 pair 组合仍存在明显 drift。

`SEQ360B` 的 trajectory test ATE none / SE3 / Sim3 分别为 139.227175 / 75.946913 / 27.567211，trajectory path ratio 为 1.350237。相较 `TRAIN360E`，它能把 path ratio 从 1.7563 改善到 1.3502，并把 SE3 ATE 从 118.60 降到 75.95；但 Sim3 ATE 基本不变（27.56 到 27.57）。这表明它主要改善的是 scale/path drift，而不是 trajectory shape error。

## 消融分析
从 pair-level 对比看，`FINAL360I` 相对 `T57b` 已取得显著 translation component improvement，这支撑了本文方法在 match-free panoramic relative pose estimation 方向上的主要贡献。相对 trajectory-derived 的 `BASE360D` component baseline，`FINAL360I` 也显示出明显更优的 pair-level direction/scale behavior；但必须说明，这里的 `BASE360D` component metrics 并不是 native pair-level predictions，而是由 official trajectory 输出后处理得到，因此只能视为 partial comparability。

从 sequence-level 诊断看，`TRAIN360E` 证明直接 composition 是可行的，但 drift 明显。`SEQ360B` 进一步说明，scale/log_tmag correction 可以有效改善 path ratio 和 SE3 ATE，但不能改善 Sim3 ATE，因此剩余误差更可能对应 trajectory shape、rotation accumulation 或 direction accumulation。`SEQ360A` 的状态为 `no_improvement`，说明简单的短 clip consistency 并不足以解决 drift；`STRUCT360C` 的状态为 `evaluation_failed`，因此未进入最终模型线。

## 局限性
本文当前结果仍有几项必须诚实说明的局限性。第一，pair-level success 并不等于完整 VO pipeline superiority。第二，direct sequential composition 会持续积累 drift。第三，ATE Sim3 仍然难以改善，说明 trajectory shape 问题尚未被解决。第四，`SEQ360B` 只能改善 scale/path，而不能改善 shape。第五，`BASE360D` 仍然是成熟的 official sequence VO pipeline，在 trajectory-level 评价中不能被简单宣称已经被本文方法全面超越。第六，本文方法有意不引入 explicit matching、RANSAC、PnP 或 BA，因此缺少完整 sequence refinement 机制。

## 小结
综上，本文最稳健的结论是：在 match-free panoramic pair-level relative pose estimation 这一问题上，`FINAL360I` 已经取得了有意义且稳定的改进；但在完整 trajectory-level 行为上，`TRAIN360E` 和 `SEQ360B` 共同表明 sequence-level drift 仍是主要瓶颈。后续工作应进一步转向 longer-horizon sequence refinement、global scale calibration 与 lightweight trajectory optimization，而不是夸大当前结果已经全面超越 official 360DVO pipeline。

