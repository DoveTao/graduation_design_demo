# THESIS360 Defense Talking Points

## 1. 一句话讲研究问题
我的研究问题是：在不依赖显式匹配的前提下，如何实现面向全景图像对的高质量相对位姿估计。

## 2. 一句话讲方法
我的方法基于球面 token 表征，采用 match-free 的 coarse-to-fine pose residual refinement，直接预测相对旋转、平移方向和尺度。

## 3. 一句话讲主结果
主结果是：`FINAL360I` 在 pair-level translation component metrics 上明显优于保留的外部参考，但把相邻 pair 直接组合成完整轨迹后仍会出现显著 drift。

## 4. 如何解释为什么 pair-level 强但 ATE 不如 360DVO
因为我的主模型是 pair-level relative pose estimator，而 360DVO 是成熟的 sequence VO pipeline；局部相对位姿强，并不自动等于长链组合后的全局轨迹强。

## 5. 如何解释 SEQ360B
`SEQ360B` 说明 scale/log_tmag correction 能明显改善 path ratio 和 SE3 ATE，但 Sim3 ATE 几乎不变，所以它主要缓解的是 scale/path drift，而不是 trajectory shape error。

## 6. 如果老师问“为什么不用显式匹配/BA”，怎么答
这是研究边界的主动选择：我想证明在 match-free panoramic pair-level estimation 框架下，是否能直接学到稳定的相对位姿表征，因此没有引入显式匹配、RANSAC、PnP 或 BA，把问题限定在端到端的相对位姿学习上。

## 7. 如果老师问“你的方法比 360DVO 创新在哪里”，怎么答
创新点不在于已经全面替代完整 VO pipeline，而在于提出并验证了一个面向全景图像对的 match-free relative pose learning framework，在 pair-level translation component metrics 上取得了清晰改进，并系统分析了从 pair-level 到 sequence-level 的误差鸿沟。

## 8. 如果老师问“为什么 trajectory 不如 360DVO”，怎么答
因为当前方法没有全局优化、时序记忆或 pose graph refinement，直接相邻组合会积累 rotation、direction 和 scale drift，而 360DVO 本身就是 sequence-level pipeline。

## 9. 如果老师问“最终模型是哪一个”，怎么答
最终 pair-level 主模型是 `FINAL360I_struct360b_final_selected`；如果谈 trajectory-scale diagnostic variant，则唯一保留的是 `SEQ360B`，但它不是主模型。

