# THESIS360 Ablation Analysis

## 1. FINAL360I vs T57b
`FINAL360I` greatly improves translation direction and scale-related pair metrics over the recovered `T57b` baseline. Signed translation-direction mean drops from approximately 111.96 deg to 45.26 deg, anti-parallel rate drops from 0.6747 to 0.2017, translation-magnitude median ratio rises from 0.1752 to 0.8344, and path ratio rises from 0.1488 to 0.6404. This is the clearest evidence that the current method substantially improves pair-level translation behavior.

## 2. FINAL360I vs BASE360D component metrics
`FINAL360I` is also much stronger than the trajectory-derived `BASE360D` component baseline on pair/component translation metrics. `BASE360D` reaches 128.40 deg signed direction mean and 0.8149 anti-parallel rate, which are both far weaker than `FINAL360I`. The caveat is essential: `BASE360D` numbers are recovered from official sequence trajectories rather than from a native pair-level prediction interface, so the comparison is informative but only partially homogeneous.

## 3. TRAIN360E
`TRAIN360E` shows that direct adjacent-pair composition is feasible and reaches full reported coverage on the retained test split. However, the composed trajectory still exhibits substantial drift: test ATE none / SE3 / Sim3 are 222.57 / 118.60 / 27.56, and trajectory path ratio is 1.7563. Therefore, the method is pair-level strong but not yet a complete sequence-level VO solution.

## 4. SEQ360B
`SEQ360B` demonstrates a targeted scale/path correction effect. Relative to `TRAIN360E`, the trajectory path ratio improves from 1.7563 to 1.3502, and ATE SE3 improves from 118.60 to 75.95. However, ATE Sim3 remains effectively unchanged (27.56 to 27.57). The natural interpretation is that `SEQ360B` mainly addresses scale/path drift, not trajectory shape. Its pair-level `tmag_median_ratio` of 1.6978 also suggests over-correction risk.

## 5. SEQ360A
`SEQ360A` is a negative ablation with classification `no_improvement`. The retained status summary indicates that local clip consistency did not reduce the main trajectory drift, so it is not part of the final promoted result line.

## 6. STRUCT360C
`STRUCT360C` is an unstable challenger with classification `evaluation_failed`. It is not used as the final model and is mentioned only as a diagnostic branch rather than as evidence of improvement.

