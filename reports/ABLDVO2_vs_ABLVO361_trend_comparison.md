# ABLDVO2 vs ABLVO361 trend comparison

| model | run | rot_mean_deg | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PlainPairVO | ABLVO361 | 6.0484925179507885 | 48.68811067066032 | 0.20071118135124458 | 0.6458483659657839 | 0.5082019066288274 |
| PlainPairVO | ABLDVO2 | 2.421761716753892 | 49.1263039873907 | 0.20683524298696168 | 0.8769573224669822 | 0.6854662049800878 |
| NoSphericalGeometry | ABLVO361 | 2.6369434359182793 | 47.32440502655299 | 0.20090873172659027 | 0.6150483309923794 | 0.48372296542035503 |
| NoSphericalGeometry | ABLDVO2 | 2.5360854690087167 | 45.10322350027338 | 0.199723429474516 | 0.7810938989168676 | 0.6144397784214122 |
| NoCrossImageInteraction | ABLVO361 | 3.9910885753755463 | 52.43926329996059 | 0.20229158435401026 | 0.5504093571251074 | 0.4328696029867613 |
| NoCrossImageInteraction | ABLDVO2 | 3.606171494231352 | 49.75859849091306 | 0.20169893322797314 | 0.6230075781570772 | 0.49003655986581823 |
| FINAL360I_full_model | reference | 2.330108616583517 | 45.264702006380205 | 0.20169893322797314 | 0.8344251368086006 | 0.6403519796204528 |

- stable trend retained:
  - `NoCrossImageInteraction` is still the weakest key ablation.
  - `FINAL360I` is still the most balanced overall model.
- changed trend:
  - `PlainPairVO` becomes much stronger on `rot`, `tmag`, and `path_ratio` once the train subset increases.
  - `NoSphericalGeometry` becomes nearly tied with FINAL360I on `signed_tdir_mean`, so the spherical-aware benefit is better interpreted as `scale/path + mild rotation support` rather than a universal direction win.
