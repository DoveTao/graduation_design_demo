# TRAIN360H vs TRAIN360C / TRAIN360D / BASE360D

| model | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage | notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TRAIN360C | 54.956992341554056 | 0.21572500987751878 | 0.7692854374461574 | 0.5888660831060318 | 1.0 | locked baseline |
| TRAIN360D | 44.986174454415895 | 0.19695772421967603 | 0.7572524310356576 | 0.5787942300950458 | 1.0 | successful enhancement |
| TRAIN360H best | 53.73375854133177 | 0.20209403397866457 | 0.8503917514492008 | 0.6493755813576354 | 1.0 | sweep-selected final config |
| BASE360D | 128.40257804384538 | 0.8148952983010668 | 0.039909732236488166 | 0.04354171873324947 | 1.0 | component baseline |

- val/test discrepancy for TRAIN360H best: `50.66378497573652`
- success classification: `partial_success`
