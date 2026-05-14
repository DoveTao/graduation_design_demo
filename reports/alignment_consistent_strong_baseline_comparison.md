# Alignment-Consistent Strong Baseline Comparison

## Scope
This report aims to compare S5 and Pano-ORB-VO under a more alignment-consistent external trajectory evaluator when possible.
It does not train a new model and does not reselect the final candidate.

## Pano-ORB-VO Reminder
- none: ATE=`10.194695`, drift=`0.172360`, path_ratio=`0.811334`
- se3: ATE=`4.137858`, drift=`0.173159`, path_ratio=`0.811334`
- sim3: ATE=`4.133322`, drift=`0.233378`, path_ratio=`0.811334`

## S5 Reminder
- locked ATE=`7.352288`
- locked drift=`1.327343`
- locked path_ratio=`0.932379`

## Alignment-Consistent Results
| method | alignment | ATE | drift | path_ratio | tracking_success_rate | notes |
| --- | --- | --- | --- | --- | --- | --- |
| S5 | none | 8.150968 | 0.490792 | 1.319695 | 0.090308 | S5 clean policy exported to the same external trajectory evaluator. |
| S5 | se3 | 1.872819 | 0.352925 | 1.319695 | 0.090308 | S5 clean policy exported to the same external trajectory evaluator. |
| S5 | sim3 | 1.497803 | 0.487979 | 1.319695 | 0.090308 | S5 clean policy exported to the same external trajectory evaluator. |
| Pano-ORB-VO | none | 10.194695 | 0.172360 | 0.811334 | 1.000000 | Protocol-compatible classical panorama VO baseline. |
| Pano-ORB-VO | se3 | 4.137858 | 0.173159 | 0.811334 | 1.000000 | Protocol-compatible classical panorama VO baseline. |
| Pano-ORB-VO | sim3 | 4.133322 | 0.233378 | 0.811334 | 1.000000 | Protocol-compatible classical panorama VO baseline. |

## Interpretation
Pano-ORB-VO is a strong protocol-compatible classical baseline. It is competitive or better in alignment-corrected trajectory shape metrics, but its path_ratio is substantially lower than S5, indicating weaker path-length/scale stability. S5 remains the final clean candidate, but this comparison shows that it is not uniformly superior to classical VO.

## Thesis wording
Under an alignment-consistent external trajectory evaluator, the proposed S5 system does not dominate the protocol-compatible classical panorama VO baseline on every metric. The classical baseline can be competitive in alignment-corrected trajectory shape, whereas S5 retains an advantage in path-length/scale stability, which remains important for a clean final candidate under the historical protocol.

## Caveats
- path_ratio remains the scale-sensitive caveat even when alignment-corrected ATE looks favorable
- Pano-ORB-VO is a strong protocol-compatible classical baseline
- Pano-ORB-VO is not ORB-SLAM3
- Pano-ORB-VO is not an original fisheye baseline
- S5 remains the final clean candidate
