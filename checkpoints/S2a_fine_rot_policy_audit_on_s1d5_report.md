# S2a fine rot policy audit on S1d5

## Setup
- branch: `optimize/s2-fine-refinement-on-s1d5`
- base checkpoint: `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt`
- policy: `checkpoints/S1d5_clean_dt_anchor_policy.json`
- fixed dt-anchor effective factors: unchanged from S1d5
- fixed settings: `fine_tdir=0.0`, `fine_tmag=0.0`, `use_geometry_refine=False`
- protocol: `explicit-cfg / unexpected=0` required
- no test labels used for fitting or clean selection

## Baselines
| method | status | fine_rot | drift | ATE | path_ratio | notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| true T57b multiscale 0/0 | valid clean baseline | 0.000000 | 1.494000 | 9.644000 | 0.496000 | explicit-cfg / unexpected=0 |
| true T57b multiscale + rot-only | valid eval-only diagnostic | 0.250000 | 1.310000 | 7.611000 | 0.496000 | explicit-cfg / unexpected=0 |
| S1d3 diagnostic best | test-swept diagnostic target | 0.350000 | 1.433000 | 7.971000 | 0.970000 | alpha=1.10; not clean fitted mainline |
| S1d4 train-selected policy | clean train-CV selected mainline precursor | 0.400000 | 1.396000 | 7.632000 | 0.935000 | alpha=1.05; train-only selected |
| S1d5 current policy | current clean exported mainline | 0.400000 | 1.396358 | 7.632463 | 0.934982 | exported policy |

## Sweep Results
| fine_rot | drift | ATE | path_ratio | RPE_rot | RPE_trans_dir | RPE_trans_mag | rot | tdir_abs | tdir_local_A_abs | metric_path_ratio | direction_only_path_ratio | tmag_p10 | tmag_p50 | tmag_p90 | selected_k | num_pairs | num_chains | missing | unexpected | run_dir |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.200000 | 1.453738 | 8.121751 | 0.934982 | 8.866946 | 76.571788 | 0.073926 | 9.921858 | 25.367518 | 23.152590 | 0.934982 | 0.999997 | 0.126968 | 0.183142 | 0.402346 | 1 | 132 | 19 | 2 | 0 | `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p2` |
| 0.250000 | 1.372939 | 7.772662 | 0.934985 | 11.169737 | 75.629279 | 0.073926 | 12.112561 | 26.024257 | 23.152590 | 0.934985 | 1.000000 | 0.126998 | 0.183116 | 0.402010 | 1 | 132 | 19 | 2 | 0 | `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p25` |
| 0.300000 | 1.480150 | 7.878038 | 0.934984 | 13.513611 | 74.735917 | 0.073926 | 14.380658 | 26.815096 | 23.152590 | 0.934984 | 0.999998 | 0.126920 | 0.183143 | 0.401848 | 1 | 132 | 19 | 2 | 0 | `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p3` |
| 0.350000 | 1.425013 | 7.933934 | 0.934986 | 15.891158 | 73.896452 | 0.073926 | 16.707872 | 27.736026 | 23.152590 | 0.934986 | 1.000001 | 0.126998 | 0.183072 | 0.402276 | 1 | 132 | 19 | 2 | 0 | `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p35` |
| 0.400000 | 1.396358 | 7.632463 | 0.934982 | 18.294509 | 73.101235 | 0.073926 | 19.079097 | 28.782671 | 23.152590 | 0.934982 | 0.999997 | 0.126874 | 0.183217 | 0.402080 | 1 | 132 | 19 | 2 | 0 | `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p4` |
| 0.450000 | 1.327402 | 7.352371 | 0.934984 | 20.715353 | 72.349483 | 0.073926 | 21.481086 | 29.944641 | 23.152590 | 0.934984 | 0.999998 | 0.126919 | 0.183086 | 0.402373 | 1 | 132 | 19 | 2 | 0 | `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p45` |
| 0.500000 | 1.368983 | 7.365145 | 0.934983 | 23.145094 | 71.641407 | 0.073926 | 23.901710 | 31.210630 | 23.152590 | 0.934983 | 0.999997 | 0.127056 | 0.183203 | 0.402130 | 1 | 132 | 19 | 2 | 0 | `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p5` |
| 0.550000 | 1.377539 | 7.618992 | 0.934984 | 25.574969 | 70.976784 | 0.073926 | 26.329629 | 32.568510 | 23.152590 | 0.934984 | 0.999998 | 0.126961 | 0.183173 | 0.402263 | 1 | 132 | 19 | 2 | 0 | `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p55` |
| 0.600000 | 1.376361 | 7.736472 | 0.934983 | 27.996231 | 70.354911 | 0.073926 | 28.754144 | 34.005467 | 23.152590 | 0.934983 | 0.999997 | 0.126905 | 0.183285 | 0.402352 | 1 | 132 | 19 | 2 | 0 | `checkpoints/S2a_fine_rot_policy_audit_on_s1d5/rot_0p6` |

## Diagnostic Candidates
- Candidates that improve over S1d5 current policy on both ATE and drift while keeping `path_ratio >= 0.90`:
  - `fine_rot=0.45`: drift=1.327402, ATE=7.352371, path_ratio=0.934984
  - `fine_rot=0.50`: drift=1.368983, ATE=7.365145, path_ratio=0.934983
  - `fine_rot=0.55`: drift=1.377539, ATE=7.618992, path_ratio=0.934984

## Best Observed Settings
- best by ATE: `fine_rot=0.45` -> drift=1.327402, ATE=7.352371, path_ratio=0.934984
- best by drift: `fine_rot=0.45` -> drift=1.327402, ATE=7.352371, path_ratio=0.934984

## Interpretation
- This is an eval-only test sweep on top of the fixed S1d5 exported policy.
- Any better fine_rot found here is a `diagnostic candidate`, not a clean replacement for S1d5.
- If a better candidate exists, the next step should be `S2b` train-only / train-CV selection.

## Mainline Status
- `reports/current_valid_baselines.md` was not updated unless a clearly better clean policy existed.
- `S1d5` remains the current clean exported mainline after this diagnostic audit.
