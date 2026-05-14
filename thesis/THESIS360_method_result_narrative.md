# THESIS360 Method-Result Narrative

## 中文段落 1：方法概述
本文方法面向全景图像对的相对位姿估计，核心思想是基于球面 token 表征进行 match-free 学习，在 coarse-to-fine pose residual refinement 框架下直接输出相对旋转 `R`、平移方向 `tdir` 与平移尺度 `tmag`。整个实验主线采用 DSET2C manifest-native 数据协议，通过 canonical train/val/test manifests 读取样本，而不是通过原始序列目录的随机发现方式定义 split，从而保证实验设置可追踪、可复现，并与 sequence-level protocol 保持一致。

## 中文段落 2：结果与解释
实验结果表明，`FINAL360I` 在 pair-level translation component metrics 上已经取得明显提升：其 test signed translation-direction mean 为 45.26 deg，anti-parallel rate 为 0.2017，明显优于 `T57b` 与 trajectory-derived 的 `BASE360D` component baseline。但 `TRAIN360E` 同时表明，直接将相邻 pair 组合成完整 trajectory 后仍会出现显著 drift；进一步的 `SEQ360B` 只能部分缓解 scale/path drift，使 trajectory path ratio 从 1.7563 改善到 1.3502，并降低 SE3 ATE，却没有改善 Sim3 ATE。这说明后续研究仍需要 sequence-level refinement，而不能仅依赖更强的局部 pair-level 预测。

## English Paragraph 1: Method Description
Our method targets panoramic relative pose estimation in a fully match-free manner. It operates on spherical token representations and adopts a coarse-to-fine pose residual refinement strategy to predict relative rotation `R`, translation direction `tdir`, and translation magnitude `tmag` directly from image pairs. The current experimental line is built on the DSET2C manifest-native protocol, where canonical train/val/test manifests define the data split explicitly and avoid random pair splitting or raw directory glob-based split discovery.

## English Paragraph 2: Result Interpretation and Limitations
The retained evidence shows that `FINAL360I` is a strong pair-level model: it clearly improves translation component metrics over both the legacy `T57b` reference and the trajectory-derived `BASE360D` component baseline. However, `TRAIN360E` shows that full sequential composition still accumulates drift, and `SEQ360B` shows that scale/path correction can improve trajectory path ratio and SE3 ATE without improving Sim3 trajectory shape. Therefore, the main contribution of this work should be framed as strong match-free panoramic pair-level relative pose estimation rather than as a complete VO pipeline that fully surpasses the official 360DVO system.

