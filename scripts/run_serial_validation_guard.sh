#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYBIN="${PYBIN:-/home/dovetao/miniconda3/envs/pytorch/bin/python}"
LOCK_FILE="${S5D11_VALIDATION_LOCK:-/tmp/graduation_design_demo_s6_eval.lock}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs checkpoints reports external_baselines/results/s5d11_nonselected_edge_audit

record_smi() {
  local label="$1"
  nvidia-smi > "logs/s5d11_nvidia_smi_${label}.log" 2>&1 || true
}

update_checkpoint() {
  "$PYBIN" tools/audit_s5d11_nonselected_edge_failures.py \
    --dense-tum external_baselines/results/s5_dense/scene01_seq03_s5_dense_est_tum.txt \
    --pairwise-jsonl external_baselines/results/s5_pairwise_replay_s5d7/final_s5_pairwise_vectors_scene01_seq03.jsonl \
    --groundtruth external_baselines/dataset/scene01_seq03/groundtruth_tum.txt \
    --timestamps external_baselines/dataset/scene01_seq03/timestamps.txt \
    --out-dir external_baselines/results/s5d11_nonselected_edge_audit \
    --out-json checkpoints/S5D11_serial_validation_and_nonselected_edge_audit.json \
    --out-report reports/s5d11_serial_validation_and_nonselected_edge_audit.md >/dev/null
}

run_step() {
  local name="$1"
  local log="$2"
  shift 2
  echo "[s5d11-serial] START ${name}" | tee "$log"
  record_smi "${name}_before"
  set +e
  "$@" >> "$log" 2>&1
  local status=$?
  set -e
  record_smi "${name}_after"
  if [[ "$status" -eq 0 ]]; then
    echo "[s5d11-serial] PASS ${name}" | tee -a "$log"
  else
    echo "[s5d11-serial] FAIL ${name} status=${status}" | tee -a "$log"
  fi
  return "$status"
}

main_chain() {
  echo "[s5d11-serial] using lock: ${LOCK_FILE}"
  echo "[s5d11-serial] python: ${PYBIN}"
  "$PYBIN" tools/audit_s5d11_validation_concurrency.py \
    --out-json checkpoints/S5D11_validation_concurrency_audit.json \
    --out-report reports/s5d11_validation_concurrency_audit.md >/dev/null || true

  local failed=0
  run_step "verify_final_candidate_serial" "logs/s5d11_verify_final_candidate_serial.log" bash scripts/verify_final_candidate.sh || failed=1
  update_checkpoint || true
  run_step "project_health_check_serial" "logs/s5d11_project_health_check_serial.log" bash scripts/project_health_check.sh || failed=1
  update_checkpoint || true
  run_step "s6_eval_only_serial" "logs/s5d11_s6_eval_only_serial.log" "$PYBIN" tools/s6_final_clean_candidate_lockdown_audit.py --eval-only || failed=1
  update_checkpoint || true
  run_step "unittest" "logs/s5d11_unittest.log" "$PYBIN" -m unittest discover -s tests -q || failed=1
  update_checkpoint || true
  record_smi "after"
  return "$failed"
}

flock "$LOCK_FILE" bash -c "$(declare -f record_smi update_checkpoint run_step main_chain); export PYBIN='$PYBIN'; export LOCK_FILE='$LOCK_FILE'; export PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF'; main_chain"
