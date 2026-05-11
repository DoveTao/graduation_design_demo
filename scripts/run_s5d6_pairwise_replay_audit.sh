#!/usr/bin/env bash
set -euo pipefail

/home/dovetao/miniconda3/envs/pytorch/bin/python tools/export_final_s5_pairwise_vectors.py \
  --scene scene01 \
  --seq seq03 \
  --timestamps external_baselines/dataset/scene01_seq03/timestamps.txt \
  --groundtruth external_baselines/dataset/scene01_seq03/groundtruth_tum.txt \
  --out-jsonl external_baselines/results/s5_pairwise_replay/final_s5_pairwise_vectors_scene01_seq03.jsonl \
  --out-npz external_baselines/results/s5_pairwise_replay/final_s5_pairwise_vectors_scene01_seq03.npz \
  --out-diagnostics external_baselines/results/s5_pairwise_replay/final_s5_pairwise_tdir_diagnostics.json

/home/dovetao/miniconda3/envs/pytorch/bin/python tools/replay_s5_pairwise_dense_chain.py \
  --pairwise-jsonl external_baselines/results/s5_pairwise_replay/final_s5_pairwise_vectors_scene01_seq03.jsonl \
  --timestamps external_baselines/dataset/scene01_seq03/timestamps.txt \
  --out-tum external_baselines/results/s5_pairwise_replay/final_s5_replayed_dense_tum.txt \
  --out-json external_baselines/results/s5_pairwise_replay/final_s5_replay_metadata.json \
  --initial-pose identity

/home/dovetao/miniconda3/envs/pytorch/bin/python tools/evaluate_s5_pairwise_replay_diagnostics.py \
  --pairwise-jsonl external_baselines/results/s5_pairwise_replay/final_s5_pairwise_vectors_scene01_seq03.jsonl \
  --replayed-tum external_baselines/results/s5_pairwise_replay/final_s5_replayed_dense_tum.txt \
  --existing-s5-dense external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt \
  --groundtruth external_baselines/dataset/scene01_seq03/groundtruth_tum.txt \
  --timestamps external_baselines/dataset/scene01_seq03/timestamps.txt \
  --out-json checkpoints/S5D6_final_s5_pairwise_vectors_and_replay.json \
  --out-report reports/s5d6_final_s5_pairwise_replay_report.md \
  --out-dir external_baselines/results/s5_pairwise_replay
