# S2 Final Fine Refinement Summary

## S2 executive summary
- S2 的目标是在 `S1d5` 已完成 scale repair 的前提下，继续通过轻量 fine-stage fusion policy 降低 `ATE` 和 `drift`，同时保持 `path_ratio` 健康。
- `S2b` 是本阶段的 clean 成功点：在不训练任何模型参数、不使用 test labels 的前提下，train-CV 选择得到 `fine_rot=0.45`，把 `ATE` 从 `7.632463` 降到 `7.352371`，把 `drift` 从 `1.396358` 降到 `1.327402`，同时保持 `path_ratio=0.934984`。
- `S2d` 的结论是 `NO-TDIR-GAIN`：在当前 family 内，`fine_tdir>0` 没有带来收益，反而会稳定恶化 `ATE/drift`。
- `S2f` 给出了一些 test-side diagnostic signal，说明更细或 dt-aware 的 `fine_rot` 可能还有小幅提升空间。
- 但 `S2g` 用 train-CV clean 化这些信号后失败：train-CV 选出的 `D0_conservative_small_dt` 在 final test 上没有超过 `S2b`。
- 因此，S2 阶段的最终 clean candidate 仍然是 `S2b`，后续不再继续 policy sweep 作为主线。

## Results table

| method | key policy | drift | ATE | path_ratio | status |
| --- | --- | ---: | ---: | ---: | --- |
| S1d5 | `fine_rot=0.40`, `fine_tdir=0.0`, `fine_tmag=0.0` | 1.396358 | 7.632463 | 0.934982 | previous clean exported baseline |
| S2b | `fine_rot=0.45`, `fine_tdir=0.0`, `fine_tmag=0.0` | 1.327402 | 7.352371 | 0.934984 | current clean fine-rot policy candidate |
| S2d best | same as `S2b` | 1.327402 | 7.352371 | 0.934984 | verdict=`NO-TDIR-GAIN` |
| S2f diagnostic global `0.475` | global `fine_rot=0.475` | 1.331278 | 7.309820 | 0.934984 | test diagnostic only |
| S2f diagnostic dt-aware `aggressive_large_dt` | `[0.1,0.3):0.45`, `[0.3,0.5):0.50`, `[0.5,1.0):0.55` | 1.332847 | 7.345742 | 0.934983 | test diagnostic only |
| S2g selected `D0_conservative_small_dt` | `[0.1,0.3):0.40`, `[0.3,0.5):0.45`, `[0.5,1.0):0.50` | 1.366864 | 7.463772 | 0.934983 | verdict=`FAIL` |

## Interpretation
- `fine_rot` 在当前 family 内存在稳定、可 clean 化的收益，但这个收益基本集中在从 `0.40` 提升到 `0.45` 这一段。
- `S2b` 说明全局 `fine_rot=0.45` 是一个简单且稳健的改进：比 `S1d5` 更好，同时不破坏 `path_ratio`。
- `fine_tdir` 融合在 `S2d` 中被系统性否定：随着 `fine_tdir` 增大，`ATE/drift` 稳定变差，因此当前 family 不再继续沿 `tdir` 方向做 policy sweep。
- `S2f` 虽然发现了更细 global rot 或 dt-aware rot 的 diagnostic signal，但 `S2g` 证明这些信号没有在 train-CV 下 clean 化成功。
- 这说明 `S2b` 不仅更简单，而且比 `S2g` 更稳健；在当前数据规模和评估配置下，继续做更复杂的 rot policy 已经接近收益上限。

## Limitations
- `ATE` 仍然约为 `7.35`，绝对值上依然偏高。
- `S2c2` 的 oracle 结果表明，`R/tdir` 耦合误差仍然存在，说明剩余误差并没有被当前 policy family 完全解释。
- 进一步提升很可能需要表征或模型结构层面的变化，而不仅是 eval-time policy sweep。
- `S2b` 仍然是 policy-level fusion，而不是端到端训练的 fine-stage 模型。

## Next research direction
- 当前建议停止继续做 policy sweep，把 `S2b` 作为 S2 阶段最终 clean candidate 固定下来。
- 如果项目时间紧，建议直接进入 final reporting。
- 如果还要继续研究，更合理的后续方向是：
  - `S3_fine_token_representation_or_coupled_pose_head`
  - 也就是从表示学习或耦合 pose head 设计入手，而不是继续扫 `fine_rot/fine_tdir`。
