# THESIS361 Final Insert Pack

## 1. 中文实验章节精简版
本文在 DSET2C canonical split 上评估当前保留的全景相对位姿主线。实验采用 manifest-native 数据协议，通过 canonical train/val/test manifests 读取样本，明确禁止 random pair split 与直接原始目录发现 split。当前 pair-level 主模型为 `FINAL360I_struct360b_final_selected`，trajectory-level 评价路径为 `TRAIN360E`，唯一保留的 sequence-scale variant 为 `SEQ360B`，外部对比包括 HKUST official `360DVO` baseline `BASE360D` 和 recovered legacy baseline `T57b`。

pair-level 结果显示，`FINAL360I` 在 test 上取得 `45.264702006380205` 的 signed translation-direction mean、`0.20169893322797314` 的 anti-parallel rate、`0.8344251368086006` 的 tmag median ratio 和 `0.6403519796204528` 的 path ratio，明显优于 `T57b` 与 trajectory-derived 的 `BASE360D` component baseline。trajectory-level 结果则表明，`TRAIN360E` 虽能实现完整轨迹导出与 `1.0` 的 reported coverage，但 direct adjacent-pair composition 仍存在明显 drift，其 test ATE none / SE3 / Sim3 分别为 `222.56856382785097`、`118.6037794846689` 和 `27.564661865900444`。`SEQ360B` 能把 trajectory path ratio 从 `1.756343083453392` 改善到 `1.3502369615185652`，并将 ATE SE3 从 `118.6037794846689` 降到 `75.94691348103409`，但 ATE Sim3 基本不变，说明它主要缓解的是 scale/path drift，而非 trajectory shape error。

## 2. 结果表
见下方“表 1 / 表 2 / 表 3”整表，可直接复制到论文中。

### 表 1. Pair-level / Component Translation Metrics
| Model | signed_tdir_mean ↓ | anti_parallel_rate ↓ | tmag_median_ratio ≈ 1 | path_ratio ≈ 1 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| T57b legacy baseline | 111.964932 | 0.674672 | 0.175156 | 0.148787 | Recovered legacy external reference only. |
| BASE360D trajectory-derived component baseline | 128.402578 | 0.814895 | 0.039910 | 0.043542 | Derived from official sequence trajectories rather than native pair-level predictions. |
| FINAL360I main model | 45.264702006380205 | 0.20169893322797314 | 0.8344251368086006 | 0.6403519796204528 | Current pair-level thesis main model. |
| SEQ360B scale variant | 45.32544604265589 | 0.20015760441292357 | 1.6978484631707425 | 1.3504765883324317 | Improves trajectory scale/path behavior but over-corrects pair-level magnitude. |

### 表 2. Trajectory-level Metrics
| Model | ATE none ↓ | ATE SE3 ↓ | ATE Sim3 ↓ | trajectory_path_ratio ≈ 1 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| TRAIN360E / FINAL360I direct composition | 222.56856382785097 | 118.6037794846689 | 27.564661865900444 | 1.756343083453392 | Full reported test coverage; direct adjacent-pair composition still drifts. |
| SEQ360B scale-smoothed trajectory | 139.22717463963536 | 75.94691348103409 | 27.567211313835486 | 1.3502369615185652 | Path ratio and SE3 improve; Sim3 remains unchanged. |
| BASE360D official 360DVO | 103.256252 | 79.295214 | 2.974466 | 0.043618 | Official sequence VO pipeline; trajectory-level comparison is meaningful, but not interface-identical to pair-only inference. |

### 表 3. Diagnostic / Ablation Outcomes
| Method | Purpose | Outcome | Classification | Used as final model? | Lesson |
| --- | --- | --- | --- | --- | --- |
| FINAL360I | Final pair-level main model selection | Best retained balanced pair-level result | final_balanced_model | Yes | Pair-level match-free panoramic relative pose estimation is the main contribution. |
| SEQ360B | Lightweight scale / log_tmag correction | Improves path ratio and ATE SE3, but not ATE Sim3 | partial | No | Scale/path drift can be reduced without solving trajectory shape. |
| SEQ360A | Local sequence consistency | No clear trajectory benefit | no_improvement | No | Short-clip consistency alone did not reduce accumulated drift. |
| STRUCT360C | Rotation-aware fine refinement challenger | Unstable evaluation, not promoted | evaluation_failed | No | Structural refinement requires much stronger stability before fair comparison. |

## 3. 表注
- `BASE360D` component metrics are trajectory-derived and therefore only partially comparable with native pair-level predictions.
- Pair-level metrics and trajectory-level ATE evaluate different aspects of the system; better pair-level translation components do not automatically imply better full-trajectory ATE.

## 4. 关键结论段
当前最稳健的结论是：`FINAL360I` 已经在 match-free panoramic pair-level relative pose estimation 上取得了清晰进展，并在 pair-level translation component metrics 上显著优于 `T57b` 与 trajectory-derived 的 `BASE360D` component baseline。但 `TRAIN360E` 同时表明，direct adjacent-pair composition 仍会造成明显 sequence-level drift；`SEQ360B` 则进一步说明，scale/log_tmag correction 可以改善 path ratio 与 ATE SE3，却不能改善 ATE Sim3，因此 trajectory shape error 仍然存在。

## 5. 局限性段
本文当前结果不能被解释为“已经全面超越 official 360DVO sequence pipeline”。pair-level success 不等于 trajectory-level superiority；direct sequential composition 仍会累积 drift；`SEQ360B` 只能改善 scale/path 而不能改善 shape；`BASE360D` 仍然是成熟的 official sequence VO pipeline。此外，本文方法有意不引入 explicit matching、RANSAC、PnP 或 BA，因此缺少全局 sequence-level correction 机制。

## 6. 未来工作段
后续工作应重点转向 sequence-level pose graph refinement、temporal memory、global scale calibration、lightweight trajectory optimization、longer clip consistency，以及 rotation/translation accumulation control，而不是仅继续强化 isolated pair-level prediction。

## 7. 不可夸大声明清单
- 不可声称 `FINAL360I` 已经在完整 trajectory ATE 上超过 HKUST official 360DVO。
- 不可声称 `BASE360D` component metrics 与 `FINAL360I` pair-level metrics 完全同质可比。
- 不可声称 `SEQ360B` 已经解决 trajectory drift。
- 不可把 `SEQ360A` 或 `STRUCT360C` 写成有效提升。
