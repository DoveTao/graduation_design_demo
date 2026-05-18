# Final Pair-level Main Results

| Model | split | rot_mean_deg | signed_tdir_mean_deg | anti_parallel_rate | tmag_median_ratio | path_ratio | coverage | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T57b | reference | N/A | 111.964932 | 0.674672 | 0.175156 | 0.148787 | 1.000000 | Recovered legacy external ERP baseline reference; rot_mean unavailable in retained sources. |
| TRAIN360C | test | N/A | N/A | N/A | N/A | 0.588866 | 1.000000 | Retained self-developed baseline recovered from aggregate reports; component metrics not retained as standalone json. |
| FINAL360I subset candidate | test | N/A | N/A | N/A | N/A | N/A | N/A | Subset-trained candidate only; no longer thesis main result. |
| FINAL360M thesis-main candidate | test | N/A | N/A | N/A | N/A | N/A | N/A | Full-train guarded main result selected by full val only. |
| BASE360D | test | 0.856182 | 128.402578 | 0.814895 | 0.039910 | 0.043542 | 1.000000 | Official 360DVO external baseline; pair-level component metrics are trajectory-derived proxies. |

Table note: `FINAL360M` replaces `FINAL360I` as the final thesis main pair-level model because it is the only protocol-valid full-train result.
Table note: `FINAL360I` must be labeled as a subset-trained candidate only.
