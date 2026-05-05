# Current Valid Baselines

| label | checkpoint | eval status | drift | ATE | path_ratio | status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| S2b train-CV selected S1d5 dt-anchor + fine_rot=0.45 policy | `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt` | explicit-cfg / unexpected=0 / clean / train-CV selected / no test labels used for selection | 1.327402 | 7.352371 | 0.934984 | current clean fine-rot policy candidate |
| S2g train-CV selected `D0_conservative_small_dt` | `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt` | explicit-cfg / unexpected=0 / clean selection attempt / train-CV selected / no test labels used for selection | 1.366864 | 7.463772 | 0.934983 | failed train-CV diagnostic; does not replace S2b |
| S1d5 exported clean dt-anchor + rot policy | `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt` | explicit-cfg / `unexpected=0` / clean exported policy / train-only selected policy / test labels used for selection: false | 1.396358 | 7.632463 | 0.934982 | tagged clean exported mainline baseline |
| true T57b multiscale 0/0 | `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt` | explicit-cfg / `unexpected=0` | 1.494 | 9.644 | 0.496 | valid clean baseline |
| true T57b multiscale + rot-only | `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt` | explicit-cfg / `unexpected=0` | 1.310 | 7.611 | 0.496 | valid eval-only diagnostic |
| S1d2 best candidate | `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt` | explicit-cfg / unexpected=0 / clean / no test labels | 1.579 | 11.487 | 0.901 | clean scale-anchor candidate |
| S1d2 best + rot-only eval | `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt` | explicit-cfg / unexpected=0 / clean scale + eval-only rot diagnostic | 1.367 | 7.749 | 0.901 | diagnostic |
| S1d3 best alpha+rot candidate | `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt` | explicit-cfg / unexpected=0 / test-swept diagnostic / not clean fitted mainline | 1.433 | 7.971 | 0.970 | invalid as clean mainline; diagnostic only |
| S1d4 train-selected dt-anchor + rot policy | `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt` | explicit-cfg / unexpected=0 / clean / train-CV selected / no test labels used for selection | 1.396 | 7.632 | 0.935 | clean train-CV selected mainline |
| historical scalar-load 0/0 | `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt` | wrapper-default scalar-load mismatch / `unexpected=15` | 1.786 | 11.573 | 0.983 | invalid as true baseline / diagnostic only |
| historical scalar-load rot-only | `checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt` | wrapper-default scalar-load mismatch / `unexpected=15` | 1.491 | 7.973 | 0.983 | invalid as true baseline / diagnostic only |
| T55a / T51 dt-conditioned results | mixed | dt-conditioned / dt leak sensitive | - | - | - | dt-leak contaminated upper-bound |
| T57c / T57e / ridge / oracle | mixed | ridge / oracle / calibration / non-deployable | - | - | - | diagnostic / oracle / calibration-leak or non-deployable |
