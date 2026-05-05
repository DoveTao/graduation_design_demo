# S1d5 final baseline comparison

| method | drift | ATE | path_ratio | clean_or_diagnostic | selected_by_train_only | explicit_cfg | unexpected_count | test_labels_used_for_selection | notes |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- | --- |
| true T57b multiscale 0/0 | 1.494 | 9.644 | 0.496 | clean | n/a | yes | 0 | false | valid clean baseline |
| true T57b multiscale + rot-only | 1.310 | 7.611 | 0.496 | diagnostic | n/a | yes | 0 | false | valid eval-only diagnostic |
| S1d2 0/0 | 1.579 | 11.487 | 0.901 | clean | yes | yes | 0 | false | clean scale-anchor candidate |
| S1d2 + rot-only | 1.367 | 7.749 | 0.901 | diagnostic | false | yes | 0 | false | clean scale-anchor + rot diagnostic |
| S1d3 best alpha+rot | 1.433 | 7.971 | 0.970 | diagnostic | false | yes | 0 | true | stable diagnostic target, test-swept, not clean fitted mainline |
| S1d4 train-selected policy | 1.396 | 7.632 | 0.935 | clean | yes | yes | 0 | false | clean train-CV selected mainline |
| S1d5 exported policy | 1.396 | 7.632 | 0.935 | clean | yes | yes | 0 | false | current clean exported mainline policy |
| historical scalar-load 0/0 | 1.786 | 11.573 | 0.983 | invalid diagnostic | no | no | 15 | n/a | invalid as true baseline / diagnostic only |
| historical scalar-load rot-only | 1.491 | 7.973 | 0.983 | invalid diagnostic | no | no | 15 | n/a | invalid as true baseline / diagnostic only |
