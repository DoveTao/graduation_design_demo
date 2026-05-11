# S5D6 Final S5 Pairwise Replay Report

## Executive summary

Final classification: `S5D6_FINAL_S5_PAIRWISE_EXPORT_UNAVAILABLE`

Default report path anchor: `reports/s5d6_final_s5_pairwise_replay_report.md`
Default checkpoint path anchor: `checkpoints/S5D6_final_s5_pairwise_vectors_and_replay.json`
Default pairwise jsonl anchor: `external_baselines/results/s5_pairwise_replay/final_s5_pairwise_vectors_scene01_seq03.jsonl`
Default pairwise npz anchor: `external_baselines/results/s5_pairwise_replay/final_s5_pairwise_vectors_scene01_seq03.npz`
Default replayed tum anchor: `external_baselines/results/s5_pairwise_replay/final_s5_replayed_dense_tum.txt`
Default pairwise diagnostics anchor: `external_baselines/results/s5_pairwise_replay/final_s5_pairwise_tdir_diagnostics.json`

## Caveats

- S5D6 is diagnostic only.
- S5D6 does not modify predictions.
- S5D6 does not replace official S5 locked result.
- S5 locked metrics/policy were not changed.
- Pairwise replay trajectories are diagnostic external artifacts.
- no GT used to generate predictions.
