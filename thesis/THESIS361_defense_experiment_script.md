# THESIS361 Defense Experiment Script

## 1. 30 秒实验概述
本文实验围绕 DSET2C canonical split 展开，核心目标不是复现一个完整 VO pipeline，而是验证一个 match-free panoramic relative pose network 在 pair-level relative pose estimation 上能否取得稳定提升。主模型是 `FINAL360I`，trajectory-level 评价通过 `TRAIN360E` 完成，`SEQ360B` 作为 scale/path diagnostic variant 保留，外部参照包括 `BASE360D` 与 `T57b`。

## 2. 1 分钟主结果说明
最核心的实验结论是：`FINAL360I` 在 pair-level translation component metrics 上显著优于 `T57b` 和 trajectory-derived 的 `BASE360D` component baseline，例如 test signed translation-direction mean 为 `45.264702006380205`，anti-parallel rate 为 `0.20169893322797314`。但 trajectory-level 结果说明，强 pair-level 表现并不自动意味着强 sequence-level 表现。`TRAIN360E` 的 test ATE none / SE3 / Sim3 为 `222.56856382785097 / 118.6037794846689 / 27.564661865900444`，说明 direct adjacent-pair composition 仍有明显 drift。`SEQ360B` 能把 trajectory path ratio 从 `1.756343083453392` 改善到 `1.3502369615185652`，并降低 ATE SE3，但 ATE Sim3 几乎不变，说明它主要缓解的是 scale/path drift，而不是 trajectory shape error。

## 3. 如何解释 FINAL360I
如果老师问最终模型是哪一个，我会回答：最终 pair-level 主模型是 `FINAL360I_struct360b_final_selected`。它的意义在于全景图像对相对位姿的 component estimation，尤其是 translation direction 与 scale-related component behavior，而不是完整 sequence VO pipeline 的全局最优结果。

## 4. 如何解释 TRAIN360E
`TRAIN360E` 不是新的训练模型，而是把 `FINAL360I` 的相邻 pair 预测组合成完整 trajectory 后再做 ATE 评估。它的作用是揭示 pair-level success 到 sequence-level stability 之间的误差鸿沟。实验表明，这种 direct adjacent-pair composition 是可行的，但 drift 仍然明显。

## 5. 如何解释 SEQ360B
`SEQ360B` 是唯一保留的 sequence-scale variant。它不改变主模型的旋转与平移方向预测，而是通过轻量的 log-scale correction 调整尺度。因此它对 path ratio 和 ATE SE3 有帮助，但对 ATE Sim3 没有帮助。我的解释是：它主要修的是 scale/path drift，不是 trajectory shape。

## 6. 如果老师问“为什么 ATE 不如 360DVO”，怎么回答
因为本文主模型是 pair-level relative pose estimator，而官方 360DVO 是成熟的 official sequence VO pipeline。局部相对位姿预测做得更好，不自动等于长链组合后的轨迹更稳定。当前方法没有 sequence-level pose graph refinement、temporal memory 或 BA 一类的全局纠正机制，所以 ATE 不如 official 360DVO 是合理且需要诚实承认的。

## 7. 如果老师问“你的创新在哪里”，怎么回答
创新点不在于已经全面替代完整 360DVO pipeline，而在于提出并验证了一个面向全景图像对的 match-free panoramic relative pose estimation framework，在 pair-level translation component metrics 上取得了清晰进展，并系统揭示了从 pair-level success 到 trajectory-level drift 之间的误差结构。

## 8. 如果老师问“为什么不使用显式匹配/RANSAC/BA”，怎么回答
这是研究边界的主动选择。本文希望研究的是，在不依赖显式 correspondence list 的情况下，是否可以直接学习到稳定的全景相对位姿表征。因此方法刻意保持 match-free，并不引入显式匹配、RANSAC、PnP 或 BA，把贡献聚焦在 learned relative pose estimation 本身。

## 9. 如果老师问“最终模型是哪一个”，怎么回答
最终模型是 `FINAL360I_struct360b_final_selected`。如果进一步问保留的 trajectory-scale 变体是什么，则答案是 `SEQ360B`，但它不是最终主模型，只是一个揭示 scale/path drift 结构的 partial diagnostic variant。

## 10. 如果老师问“SEQ360A/STRUCT360C 为什么失败”，怎么回答
`SEQ360A` 的结论是 `no_improvement`，说明简单的局部 clip consistency 没有有效降低 trajectory drift。`STRUCT360C` 的结论是 `evaluation_failed`，说明 rotation-aware fine refinement 这一结构挑战者在当前条件下还不稳定。二者都帮助说明问题边界，但都不是正向主结果。
