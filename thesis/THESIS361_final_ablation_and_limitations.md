# THESIS361 Final Ablation and Limitations

## 1. FINAL360I vs T57b
The clearest retained gain appears in the comparison between `FINAL360I` and the recovered `T57b` legacy baseline. `FINAL360I` reduces the test signed translation-direction mean from approximately `111.96` deg to `45.26` deg, lowers the anti-parallel rate from `0.6747` to `0.2017`, raises the translation-magnitude median ratio from `0.1752` to `0.8344`, and increases the path ratio from `0.1488` to `0.6404`. This is the strongest evidence that the proposed match-free panoramic pair-level model materially improves translation component quality.

## 2. FINAL360I vs BASE360D Component Metrics
`FINAL360I` also clearly outperforms the trajectory-derived `BASE360D` component baseline on retained pair/component translation metrics. `BASE360D` reaches `128.402578` signed direction mean and `0.814895` anti-parallel rate, both far weaker than `FINAL360I`. However, the caveat must remain explicit: `BASE360D` component numbers are recovered from official sequence trajectories rather than produced by the same pair-level interface, so the comparison is informative but only partially homogeneous.

## 3. TRAIN360E Trajectory Drift
`TRAIN360E` establishes that direct adjacent-pair composition is feasible and achieves full reported test coverage. At the same time, it reveals a strong gap between pair-level quality and full-trajectory stability. The retained test ATE values are `222.56856382785097` / `118.6037794846689` / `27.564661865900444` for none / SE3 / Sim3 alignment, and the `trajectory_path_ratio` is `1.756343083453392`. Therefore, the current model should not be interpreted as a complete VO pipeline.

## 4. SEQ360B Scale Smoothing
`SEQ360B` demonstrates that scale/log-scale correction can improve path-related trajectory behavior. Relative to `TRAIN360E`, it reduces the `trajectory_path_ratio` from `1.756343083453392` to `1.3502369615185652` and lowers `ATE SE3` from `118.6037794846689` to `75.94691348103409`. However, `ATE Sim3` remains effectively unchanged, from `27.564661865900444` to `27.567211313835486`. This strongly suggests that the variant mainly addresses scale/path drift rather than deeper trajectory-shape error. Its pair-level `tmag_median_ratio` of `1.6978484631707425` further indicates clear scale over-correction risk.

## 5. SEQ360A No Improvement
`SEQ360A` is retained only as a negative diagnostic line with classification `no_improvement`. The current evidence indicates that local sequence consistency at short clip length did not reduce the main trajectory drift, and therefore it is not part of the promoted result line.

## 6. STRUCT360C Evaluation Failed
`STRUCT360C` is retained only as an unstable challenger with classification `evaluation_failed`. It is not used as the final model and should not be presented as an improving ablation.

## 7. Limitations
Several limitations remain central to interpreting the current results. First, pair-level success does not imply full VO pipeline superiority. Second, direct sequential composition still accumulates drift. Third, `ATE Sim3` remains difficult to improve even when scale/path behavior is partially corrected. Fourth, `SEQ360B` improves scale/path but not trajectory shape. Fifth, `BASE360D` remains a mature official sequence VO pipeline and still provides a stronger trajectory-level anchor. Sixth, the current work intentionally excludes explicit matching, RANSAC, PnP, and BA, which means it lacks a global geometric correction mechanism.

## 8. Future Work
The retained evidence suggests that future work should move toward sequence-level refinement rather than further over-optimizing isolated pair-level outputs. Promising directions include sequence-level pose graph refinement, temporal memory, global scale calibration, lightweight trajectory optimization, longer clip consistency, and explicit rotation/translation accumulation control.
