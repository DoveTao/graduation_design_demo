# FINAL360I vs all baselines

| model | split | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| FINAL360I selected | test | 45.264702006380205 | 0.20169893322797314 | 0.8344251368086006 | 0.6403519796204528 |
| STRUCT360B | test | 45.924701790734524 | 0.2042670881074674 | 0.8509755191001558 | 0.656656775908536 |
| TRAIN360D | test | 44.986174454415895 | 0.19695772421967603 | 0.7572524310356576 | 0.5787942300950458 |
| TRAIN360H | test | 53.73375854133177 | 0.20209403397866457 | 0.8503917514492008 | 0.6493755813576354 |
| T57b | reference | 111.96493221327962 | 0.6746724890829694 | 0.1751560082454769 | 0.14878731297064046 |
| BASE360D | test component | 128.40257804384538 | 0.8148952983010668 | 0.039909732236488166 | 0.04354171873324947 |

- fallback used: `False`
- compared to STRUCT360B: `comparable`
- compared to TRAIN360D: `partial`
- compared to TRAIN360H: `partial`
- val/test discrepancy: `59.20516609915211`
- strong target met: `false`
- balanced target met: `true`
