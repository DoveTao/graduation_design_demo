# S5 与 ORB-SLAM3 same-external-evaluator 诊断比较

## 执行摘要

本报告不是 fair official main table，而是 diagnostic same-external-evaluator comparison using restored S5 dense artifact。ORB-SLAM3 是 verified external ORB-SLAM3 baseline；restored S5 dense TUM 是 diagnostic unverified dense artifact；S5 official locked result 仍是最终 S5 结果。

S5 和 ORB-SLAM3 不是 same-input-protocol：S5 使用项目内 panorama/equirectangular 输入和 official clean evaluator；ORB-SLAM3 使用 raw fisheye cam0，并带有 fitted KB8 compatibility approximation 和 partial tracking coverage caveat。

## diagnostic same-external-evaluator table

| method / artifact | alignment | ATE | drift | path_ratio | coverage | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ORB-SLAM3 fisheye cam0 | none | 11.46685113827757 | 0.03737053832593786 | 0.2998258665660257 | 273/454 | verified external baseline |
| ORB-SLAM3 fisheye cam0 | se3 | 0.30854441069248173 | 0.0374760642069601 | 0.2998258665660257 | 273/454 | verified external baseline |
| ORB-SLAM3 fisheye cam0 | sim3 | 0.224292165986624 | 0.06452708470276013 | 0.2998258665660257 | 273/454 | verified external baseline |
| restored S5 dense artifact | none | 21.681522044975264 | 0.21683137477814862 | 2.777267572676944 | 454/454 | diagnostic unverified dense artifact |
| restored S5 dense artifact | se3 | 8.231468716451547 | 0.23274388321733164 | 2.777267572676944 | 454/454 | diagnostic unverified dense artifact |
| restored S5 dense artifact | sim3 | 4.07912293550008 | 0.13382808429229665 | 2.777267572676944 | 454/454 | diagnostic unverified dense artifact |

## 解释

ORB-SLAM3 在成功 tracking 的片段上显示出很强的几何精度，尤其是 `se3` 和 `sim3` alignment 后的 ATE。但它的 coverage 为 `273/454`，`path_ratio=0.299826`，说明轨迹覆盖不完整。

restored S5 dense artifact 覆盖 `454/454`，但 `path_ratio=2.777268`，S5D11 进一步定位到 non-selected long runs 的 tmag over-scaling。S5D12/S5D13 确认该 dense artifact provenance 不完整，不能写成 verified final S5 dense export。

## 结论边界

可以声称：ORB-SLAM3 是 verified external strong baseline；restored S5 dense artifact 是有用的 diagnostic artifact；S5 official locked result 仍是最终方法结果。

不能声称：S5 dense vs ORB-SLAM3 是 fair official main table；restored S5 dense TUM 是 official S5 dense result；S5 和 ORB-SLAM3 是 same-input-protocol；ORB-SLAM3 替代 S5；restored dense diagnostic metrics 替代 S5 locked metrics。

## Caveats

- diagnostic comparison only。
- 不能作为 official main result。
- restored dense artifact is diagnostic-only。
- S5 locked metrics/policy unchanged。
