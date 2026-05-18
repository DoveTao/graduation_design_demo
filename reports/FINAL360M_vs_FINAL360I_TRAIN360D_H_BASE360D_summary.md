# FINAL360M vs FINAL360I TRAIN360D TRAIN360H BASE360D summary

| model | status | split | rot_mean_deg | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `FINAL360M` | full-train thesis-main candidate | test | 2.322842 | 56.071808 | 0.207625 | 0.883856 | 0.681470 | 1.000000 |
| `FINAL360I` | subset-trained candidate | test | 2.330109 | 45.264702 | 0.201699 | 0.834425 | 0.640352 | 1.000000 |
| `TRAIN360D` | prior direction-best baseline | test | 2.346054 | 44.986174 | 0.196958 | 0.757252 | 0.578794 | 1.000000 |
| `TRAIN360H` | prior scale-best baseline | test | 3.161681 | 53.733759 | 0.202094 | 0.850392 | 0.649376 | 1.000000 |
| `BASE360D` | official external baseline | test component | 0.856182 | 128.402578 | 0.814895 | 0.039910 | 0.043542 | 1.000000 |

## Key reading

- `FINAL360M` is the only result in this table that satisfies the required full-train thesis-main protocol.
- Versus old `FINAL360I subset-trained candidate`, `FINAL360M` is `mixed`:
  - better on `rot_mean`, `tmag_median_ratio`, and `path_ratio`
  - worse on `signed_tdir_mean` and `anti_parallel_rate`
- Versus `TRAIN360D`, `FINAL360M` improves `rot_mean`, `tmag_median_ratio`, and `path_ratio`, but is worse on `signed_tdir_mean` and `anti_parallel_rate`.
- Versus `TRAIN360H`, `FINAL360M` improves `rot_mean`, `signed_tdir_mean`, and `path_ratio`, but is slightly worse on `anti_parallel_rate`.
- Versus `BASE360D`, `FINAL360M` is far stronger on pair-level direction and scale/path behavior.

## Bottom line

- protocol-valid thesis main candidate: `FINAL360M`
- old `FINAL360I` should remain in the thesis only as a `subset-trained candidate` reference, not as the final main result

