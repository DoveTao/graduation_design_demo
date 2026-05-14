#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_POLICY = REPO_ROOT / "checkpoints" / "S5_clean_tmag_calibration_policy.json"
DEFAULT_MANIFEST = REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json"
DEFAULT_S6_EVAL_TOOL = REPO_ROOT / "tools" / "s6_final_clean_candidate_lockdown_audit.py"
DEFAULT_EXPECTED_ATE = 7.352288
DEFAULT_EXPECTED_DRIFT = 1.327343
DEFAULT_EXPECTED_PATH_RATIO = 0.932379
DEFAULT_TOL = 1.0e-5


def _safe_float(v: Any) -> float:
    return float(v)


def _fmt(v: float, digits: int = 6) -> str:
    return "nan" if not math.isfinite(v) else f"{v:.{digits}f}"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (REPO_ROOT / path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify that the locked final S5 clean candidate still reproduces the expected metrics.")
    p.add_argument("--policy", default=str(DEFAULT_POLICY), help="Path to S5 policy json.")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to final clean candidate manifest json.")
    p.add_argument("--expected-ate", type=float, default=DEFAULT_EXPECTED_ATE, help="Expected locked ATE.")
    p.add_argument("--expected-drift", type=float, default=DEFAULT_EXPECTED_DRIFT, help="Expected locked drift.")
    p.add_argument("--expected-path-ratio", type=float, default=DEFAULT_EXPECTED_PATH_RATIO, help="Expected locked path_ratio.")
    p.add_argument("--tol-ate", type=float, default=DEFAULT_TOL, help="Absolute tolerance for ATE.")
    p.add_argument("--tol-drift", type=float, default=DEFAULT_TOL, help="Absolute tolerance for drift.")
    p.add_argument("--tol-path-ratio", type=float, default=DEFAULT_TOL, help="Absolute tolerance for path_ratio.")
    p.add_argument("--eval-variant", default="default", choices=["default", "max_eval_batches_off", "explicit_selected_k"], help="Evaluation protocol variant forwarded to eval_clean_policy.")
    return p.parse_args()


def _evaluate_policy(policy_path: Path, eval_variant: str) -> Dict[str, Any]:
    if policy_path.resolve() == DEFAULT_POLICY.resolve():
        proc = subprocess.run(
            [sys.executable, str(DEFAULT_S6_EVAL_TOOL), "--eval-only"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(proc.stdout)
        return {
            "ATE": float(payload["s5"]["ATE"]),
            "drift": float(payload["s5"]["drift"]),
            "metric_path_ratio": float(payload["s5"]["path_ratio"]),
        }

    from tools.eval_clean_policy import _run
    from train_mvp import _resolve_clean_policy_payload

    policy = _resolve_clean_policy_payload(str(policy_path))
    tmp_dir = Path(tempfile.mkdtemp(prefix="verify_final_s5_", dir=str(REPO_ROOT / "checkpoints")))
    try:
        return _run(policy, policy_path, tmp_dir, eval_variant)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _check_manifest(manifest: Dict[str, Any], policy_path: Path, expected: Dict[str, float]) -> list[str]:
    issues: list[str] = []
    if str(manifest.get("final_candidate_name", "")).strip() != "S5_clean_tmag_calibration_policy":
        issues.append("manifest final_candidate_name is not S5_clean_tmag_calibration_policy")
    manifest_policy = str(manifest.get("policy_path", "")).strip()
    if manifest_policy and Path(manifest_policy).as_posix() != policy_path.relative_to(REPO_ROOT).as_posix():
        issues.append("manifest policy_path does not match requested policy path")
    metrics = dict(manifest.get("final_metrics", {}))
    for key, want in [("ATE", expected["ATE"]), ("drift", expected["drift"]), ("path_ratio", expected["path_ratio"])]:
        got = metrics.get(key)
        if got is None or not math.isfinite(float(got)):
            issues.append(f"manifest final_metrics missing finite {key}")
            continue
        if abs(float(got) - float(want)) > 1.0e-12:
            issues.append(f"manifest final_metrics {key}={got} does not match expected locked value {want}")
    return issues


def main() -> int:
    args = parse_args()
    policy_path = _resolve_repo_path(args.policy)
    manifest_path = _resolve_repo_path(args.manifest)
    expected = {
        "ATE": _safe_float(args.expected_ate),
        "drift": _safe_float(args.expected_drift),
        "path_ratio": _safe_float(args.expected_path_ratio),
    }
    tolerance = {
        "ATE": _safe_float(args.tol_ate),
        "drift": _safe_float(args.tol_drift),
        "path_ratio": _safe_float(args.tol_path_ratio),
    }

    if not policy_path.is_file():
        raise FileNotFoundError(f"policy not found: {policy_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    manifest = _read_json(manifest_path)
    manifest_issues = _check_manifest(manifest, policy_path, expected)
    actual_eval = _evaluate_policy(policy_path, args.eval_variant)
    actual = {
        "ATE": _safe_float(actual_eval["ATE"]),
        "drift": _safe_float(actual_eval["drift"]),
        "path_ratio": _safe_float(actual_eval["metric_path_ratio"]),
    }

    metric_checks = {
        key: abs(actual[key] - expected[key]) <= tolerance[key]
        for key in ("ATE", "drift", "path_ratio")
    }
    status = "PASS" if all(metric_checks.values()) and not manifest_issues else "FAIL"

    print("[S5-final-verifier]")
    print(f"  policy: {policy_path.relative_to(REPO_ROOT).as_posix()}")
    print(f"  manifest: {manifest_path.relative_to(REPO_ROOT).as_posix()}")
    print("  expected:")
    print(f"    ATE: {_fmt(expected['ATE'])}")
    print(f"    drift: {_fmt(expected['drift'])}")
    print(f"    path_ratio: {_fmt(expected['path_ratio'])}")
    print("  actual:")
    print(f"    ATE: {_fmt(actual['ATE'])}")
    print(f"    drift: {_fmt(actual['drift'])}")
    print(f"    path_ratio: {_fmt(actual['path_ratio'])}")
    print("  tolerance:")
    print(f"    ATE: {_fmt(tolerance['ATE'], 8)}")
    print(f"    drift: {_fmt(tolerance['drift'], 8)}")
    print(f"    path_ratio: {_fmt(tolerance['path_ratio'], 8)}")
    print(f"  status: {status}")
    if manifest_issues:
        print("  manifest_issues:")
        for issue in manifest_issues:
            print(f"    - {issue}")
    if status != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
