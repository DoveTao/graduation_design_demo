# S2b Final Comparison

| method | status | fine_rot | drift | ATE | path_ratio | explicit_cfg | unexpected_count | selected_by_train_only | test_labels_used_for_selection | notes |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| true T57b multiscale 0/0 | valid baseline | 0.00 | 1.494000 | 9.644000 | 0.496000 | yes | 0 | false | false | true multiscale baseline |
| S1d5 exported clean policy | previous clean mainline | 0.40 | 1.396358 | 7.632463 | 0.934982 | yes | 0 | true | false | tagged clean exported mainline |
| S2a diagnostic best | test-swept diagnostic | 0.45 | 1.327402 | 7.352371 | 0.934984 | yes | 0 | false | true | diagnostic only; not clean selection |
| S2b train-CV selected policy | clean train-CV selected candidate | 0.45 | 1.327402 | 7.352371 | 0.934984 | yes | 0 | true | false | inherits S1d5 dt-anchor policy |
| historical scalar-load 0/0 | invalid as true baseline / diagnostic only | 0.00 | 1.786000 | 11.573000 | 0.983000 | no | 15 | false | false | scalar-load mismatch artifact |
| historical scalar-load rot-only | invalid as true baseline / diagnostic only | 0.25 | 1.491000 | 7.973000 | 0.983000 | no | 15 | false | false | scalar-load mismatch artifact |
