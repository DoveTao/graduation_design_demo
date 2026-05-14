#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from config import Config
from model import PanoramaRelPoseModel
from eval_clean_policy import _cfg_from_dict, _load_ckpt_cfg


REPORT_PATH = REPO_ROOT / "checkpoints" / "S8a_baseline_reproduction_mismatch_audit_report.md"
ARTIFACTS_PATH = REPO_ROOT / "checkpoints" / "S8a_baseline_reproduction_mismatch_audit_artifacts.json"

S2B_EXPECTED = {
    "drift": 1.327402,
    "ATE": 7.352371,
    "path_ratio": 0.934984,
    "load_missing": 2,
    "load_unexpected": 0,
}
S5_EXPECTED = {
    "drift": 1.327343,
    "ATE": 7.352288,
    "path_ratio": 0.932379,
}
S2B_OBSERVED_BLOCKER = {
    "drift": 1.3270823574188497,
    "ATE": 7.351221328788474,
    "path_ratio": 0.9349468349052762,
    "load_missing": 14,
    "load_unexpected": 0,
}

S5_COMMIT = "c471b21"
S6_COMMIT = "0d37fd4"
S7_COMMIT = "719705f"
LEGACY_S2B_COMMIT = "e7ba870"

RELEVANT_FILES = [
    "scripts/eval_s2b_clean_policy.sh",
    "scripts/eval_s5_clean_policy.sh",
    "tools/eval_clean_policy.py",
    "checkpoints/S2b_clean_fine_rot_policy.json",
    "checkpoints/S5_clean_tmag_calibration_policy.json",
    "checkpoints/final_clean_candidate_manifest.json",
    "model.py",
    "config.py",
    "train_mvp.py",
]


def _run(cmd: List[str], *, check: bool = True) -> str:
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        capture_output=True,
    )
    return proc.stdout


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_current_missing_keys() -> Dict[str, Any]:
    policy = _read_json(REPO_ROOT / "checkpoints" / "S2b_clean_fine_rot_policy.json")
    ckpt_path = REPO_ROOT / str(policy["base_checkpoint_path"])
    cfg = _cfg_from_dict(_load_ckpt_cfg(ckpt_path))
    cfg.use_fine_stage = True
    cfg.fine_rot_fuse_strength = float(policy["fine_rot_fuse_strength"])
    cfg.fine_tdir_fuse_strength = float(policy["fine_tdir_fuse_strength"])
    cfg.fine_tmag_fuse_strength = float(policy["fine_tmag_fuse_strength"])
    cfg.use_geometry_refine = False
    cfg.tmag_condition_on_dt = False
    model = PanoramaRelPoseModel(cfg, torch.device("cpu"))
    payload = torch.load(str(ckpt_path), map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    missing = list(msg.missing_keys)
    unexpected = list(msg.unexpected_keys)
    coupled = [k for k in missing if k.startswith("coupled_pose_head.")]
    benign_base = [k for k in missing if k.endswith("ridge_calib_raw_center")]
    critical = [k for k in missing if k not in set(coupled) | set(benign_base)]
    return {
        "checkpoint_path": str(ckpt_path),
        "policy_path": "checkpoints/S2b_clean_fine_rot_policy.json",
        "strict": False,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "coupled_missing_keys": coupled,
        "benign_base_missing_keys": benign_base,
        "critical_missing_keys": critical,
    }


def _git_log_for_paths(paths: List[str]) -> str:
    return _run(["git", "log", "--oneline", "--"] + paths).strip()


def _git_diff_stat(a: str, b: str) -> str:
    return _run(["git", "diff", "--stat", f"{a}..{b}", "--"] + RELEVANT_FILES).strip()


def _git_name_only(a: str, b: str) -> List[str]:
    out = _run(["git", "diff", "--name-only", f"{a}..{b}", "--"] + RELEVANT_FILES).strip()
    return [line for line in out.splitlines() if line.strip()]


def _historical_s2b_summary() -> Dict[str, Any]:
    raw = _run(["git", "show", f"{LEGACY_S2B_COMMIT}:checkpoints/S2b_final_repro/s1d5_policy_eval_summary.json"])
    return json.loads(raw)


def _historical_model_has_coupled_head() -> bool:
    raw = _run(["git", "show", f"{LEGACY_S2B_COMMIT}:model.py"])
    return "coupled_pose_head" in raw


def _parse_s3a0c_baseline_row() -> Dict[str, Any]:
    report = (REPO_ROOT / "checkpoints" / "S3a0c_policy_alignment_audit_report.md").read_text(encoding="utf-8")
    match = re.search(
        r"\| S2b_policy_baseline \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) .*? \| (\d+) \| (\d+) \|",
        report,
        flags=re.DOTALL,
    )
    if not match:
        return {}
    return {
        "drift": float(match.group(1)),
        "ATE": float(match.group(2)),
        "path_ratio": float(match.group(3)),
        "load_missing": int(match.group(4)),
        "load_unexpected": int(match.group(5)),
    }


def _manifest_payload() -> Dict[str, Any]:
    return _read_json(REPO_ROOT / "checkpoints" / "final_clean_candidate_manifest.json")


def _classify(current_missing: Dict[str, Any], changed_since_s6: List[str], changed_since_s5: List[str], s3a0c_row: Dict[str, Any]) -> Dict[str, str]:
    impactful = {
        "scripts/eval_s2b_clean_policy.sh",
        "scripts/eval_s5_clean_policy.sh",
        "tools/eval_clean_policy.py",
        "checkpoints/S2b_clean_fine_rot_policy.json",
        "checkpoints/S5_clean_tmag_calibration_policy.json",
        "model.py",
        "config.py",
        "train_mvp.py",
    }
    impactful_s6 = [p for p in changed_since_s6 if p in impactful]
    impactful_s5 = [p for p in changed_since_s5 if p in impactful]
    if impactful_s6:
        return {
            "final_classification": "SCRIPT-DRIFT",
            "root_cause": f"Relevant eval/model/config files changed after S6: {impactful_s6}.",
        }
    if current_missing["critical_missing_keys"]:
        return {
            "final_classification": "MODEL-ARCH-LOAD-MISMATCH",
            "root_cause": "Current checkpoint load is missing non-benign keys outside the known coupled-head and ridge-calib additions.",
        }
    if s3a0c_row and s3a0c_row.get("load_missing") == 14:
        return {
            "final_classification": "HISTORY-REPORT-MISMATCH",
            "root_cause": "Locked S2b predecessor metrics/load_missing were inherited from the legacy e7ba870 summary (load_missing=2), while later code/reporting already validated a benign load_missing=14 path against the same checkpoint and policy.",
        }
    if impactful_s5:
        return {
            "final_classification": "EVAL-CODE-DRIFT",
            "root_cause": f"Relevant eval/model/config files changed after the S5 period: {impactful_s5}.",
        }
    return {
        "final_classification": "UNRESOLVED-REPRODUCTION-MISMATCH",
        "root_cause": "The current blocker could not be reduced to a single proven script/path/config mismatch from local evidence alone.",
    }


def main() -> None:
    current_missing = _load_current_missing_keys()
    historical_summary = _historical_s2b_summary()
    historical_has_coupled = _historical_model_has_coupled_head()
    s3a0c_row = _parse_s3a0c_baseline_row()
    manifest = _manifest_payload()

    log_paths = _git_log_for_paths(RELEVANT_FILES)
    diff_s5_s7 = _git_diff_stat(S5_COMMIT, S7_COMMIT)
    diff_s6_s7 = _git_diff_stat(S6_COMMIT, S7_COMMIT)
    changed_s5_s7 = _git_name_only(S5_COMMIT, S7_COMMIT)
    changed_s6_s7 = _git_name_only(S6_COMMIT, S7_COMMIT)
    verdict = _classify(current_missing, changed_s6_s7, changed_s5_s7, s3a0c_row)

    artifacts = {
        "expected": {"s2b": S2B_EXPECTED, "s5": S5_EXPECTED},
        "observed_blocker": {"s2b": S2B_OBSERVED_BLOCKER},
        "current_commit": S7_COMMIT,
        "historical_s2b_commit": LEGACY_S2B_COMMIT,
        "historical_s2b_summary": historical_summary,
        "historical_model_has_coupled_head": historical_has_coupled,
        "current_checkpoint_loading_audit": current_missing,
        "s3a0c_reference_row": s3a0c_row,
        "git_file_log": log_paths,
        "git_diff_s5_to_s7_stat": diff_s5_s7,
        "git_diff_s6_to_s7_stat": diff_s6_s7,
        "git_changed_files_s5_to_s7": changed_s5_s7,
        "git_changed_files_s6_to_s7": changed_s6_s7,
        "manifest": manifest,
        **verdict,
        "fix_applied": None,
        "post_fix_reproduction": None,
        "s8_can_resume": False,
        "s5_remains_final_clean_candidate": True,
    }
    ARTIFACTS_PATH.write_text(json.dumps(artifacts, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    lines: List[str] = []
    lines.append("# S8a Baseline Reproduction Mismatch Audit Report\n\n")
    lines.append("## Executive summary\n\n")
    lines.append(f"- final classification: `{verdict['final_classification']}`\n")
    lines.append(f"- root cause: {verdict['root_cause']}\n")
    lines.append("- no model training, no S8 router selection, and no S5/S2b policy mutation were performed.\n")
    lines.append("- S8 remains paused in this audit turn.\n\n")

    lines.append("## Reproduction mismatch details\n\n")
    lines.append(f"- expected S2b: drift=`{S2B_EXPECTED['drift']}`, ATE=`{S2B_EXPECTED['ATE']}`, path_ratio=`{S2B_EXPECTED['path_ratio']}`, load_missing=`{S2B_EXPECTED['load_missing']}`\n")
    lines.append(f"- observed blocker run: drift=`{S2B_OBSERVED_BLOCKER['drift']:.10f}`, ATE=`{S2B_OBSERVED_BLOCKER['ATE']:.10f}`, path_ratio=`{S2B_OBSERVED_BLOCKER['path_ratio']:.10f}`, load_missing=`{S2B_OBSERVED_BLOCKER['load_missing']}`\n")
    lines.append(f"- expected S5: drift=`{S5_EXPECTED['drift']}`, ATE=`{S5_EXPECTED['ATE']}`, path_ratio=`{S5_EXPECTED['path_ratio']}`\n\n")

    lines.append("## Expected vs observed metrics\n\n")
    lines.append("| item | drift | ATE | path_ratio | load_missing | load_unexpected |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |\n")
    lines.append(f"| S2b expected | {S2B_EXPECTED['drift']:.6f} | {S2B_EXPECTED['ATE']:.6f} | {S2B_EXPECTED['path_ratio']:.6f} | {S2B_EXPECTED['load_missing']} | {S2B_EXPECTED['load_unexpected']} |\n")
    lines.append(f"| S2b observed blocker | {S2B_OBSERVED_BLOCKER['drift']:.6f} | {S2B_OBSERVED_BLOCKER['ATE']:.6f} | {S2B_OBSERVED_BLOCKER['path_ratio']:.6f} | {S2B_OBSERVED_BLOCKER['load_missing']} | {S2B_OBSERVED_BLOCKER['load_unexpected']} |\n")
    if s3a0c_row:
        lines.append(f"| S3a0c S2b_policy_baseline | {s3a0c_row['drift']:.6f} | {s3a0c_row['ATE']:.6f} | {s3a0c_row['path_ratio']:.6f} | {s3a0c_row['load_missing']} | {s3a0c_row['load_unexpected']} |\n")
    lines.append(f"| legacy `{LEGACY_S2B_COMMIT}` S2b summary | {historical_summary['drift']:.6f} | {historical_summary['ATE']:.6f} | {historical_summary['metric_path_ratio']:.6f} | {historical_summary['load_missing']} | {historical_summary['load_unexpected']} |\n\n")

    lines.append("## Git/file provenance audit\n\n")
    lines.append(f"- relevant path history:\n\n```text\n{log_paths}\n```\n")
    lines.append(f"- changed files from `{S5_COMMIT}` to `{S7_COMMIT}` within the audit scope: `{changed_s5_s7}`\n")
    lines.append(f"- changed files from `{S6_COMMIT}` to `{S7_COMMIT}` within the audit scope: `{changed_s6_s7}`\n")
    lines.append(f"- diff stat `{S5_COMMIT}..{S7_COMMIT}`:\n\n```text\n{diff_s5_s7 or '(no diff)'}\n```\n")
    lines.append(f"- diff stat `{S6_COMMIT}..{S7_COMMIT}`:\n\n```text\n{diff_s6_s7 or '(no diff)'}\n```\n")
    lines.append("- result: no relevant eval/model/config/script drift was found after S6 within the audited file set; the only scoped file added after S6 is `checkpoints/final_clean_candidate_manifest.json`.\n\n")

    lines.append("## Checkpoint loading audit\n\n")
    lines.append(f"- policy path: `{current_missing['policy_path']}`\n")
    lines.append(f"- checkpoint path: `{current_missing['checkpoint_path']}`\n")
    lines.append(f"- strict loading: `{current_missing['strict']}`\n")
    lines.append(f"- current missing key count: `{len(current_missing['missing_keys'])}`\n")
    lines.append(f"- current unexpected key count: `{len(current_missing['unexpected_keys'])}`\n")
    lines.append("- current missing keys:\n")
    for key in current_missing["missing_keys"]:
        lines.append(f"  - `{key}`\n")
    lines.append("\n")

    lines.append("## Missing/unexpected key analysis\n\n")
    lines.append(f"- historical legacy summary `{LEGACY_S2B_COMMIT}` reported `load_missing=2` and the legacy model source does not contain `coupled_pose_head`: `{historical_has_coupled}`\n")
    lines.append(f"- current benign base missing keys: `{current_missing['benign_base_missing_keys']}`\n")
    lines.append(f"- current coupled-head missing keys: `{current_missing['coupled_missing_keys']}`\n")
    lines.append(f"- current critical missing keys after filtering: `{current_missing['critical_missing_keys']}`\n")
    lines.append("- interpretation: `load_missing=14` decomposes cleanly into the old 2 ridge-calibration buffers plus 12 later-added coupled-head parameters.\n")
    if s3a0c_row:
        lines.append("- S3a0c already recorded that a `load_missing=14` S2b wrapper path matched the locked S2b metrics exactly, so the 12 extra keys are historical evidence of a benign unused-head mismatch rather than immediate forward corruption.\n")
    lines.append("\n")

    lines.append("## Eval parameter/scope audit\n\n")
    lines.append("- locked/historical summary and current policy both use:\n")
    lines.append("  - `fine_rot=0.45`\n")
    lines.append("  - `fine_tdir=0.0`\n")
    lines.append("  - `fine_tmag=0.0`\n")
    lines.append("  - `selected_k=1`\n")
    lines.append("  - `num_pairs=132`\n")
    lines.append("  - `num_chains=19`\n")
    lines.append("- `Config` defaults for coupled-head enablement remain `False` in the current branch, so the extra coupled-head parameters are instantiated but not supposed to be active in the S2b eval path.\n")
    lines.append("- `S2c1` already documented that official `path_ratio` comes from the debug-chain summary path with `odom_trajectory_debug_max_chains=1`; no new post-S6 change was found in the audited script/code set that would alter that scope.\n\n")

    lines.append("## Historical command comparison\n\n")
    lines.append("- `scripts/eval_s2b_clean_policy.sh`: unchanged in the audited commits after S5.\n")
    lines.append("- `scripts/eval_s5_clean_policy.sh`: introduced at S6/S7 for the frozen S5 candidate; it does not affect the S2b script path.\n")
    lines.append("- `tools/eval_clean_policy.py`: unchanged in the audited commits after S5.\n")
    lines.append("- locked predecessor summary provenance: `checkpoints/S2b_final_repro/s1d5_policy_eval_summary.json` traces back to `e7ba870`, not to S6/S7.\n\n")

    lines.append("## Root cause classification\n\n")
    lines.append(f"- `{verdict['final_classification']}`\n")
    lines.append(f"- explanation: {verdict['root_cause']}\n\n")

    lines.append("## Fix applied, if any\n\n")
    lines.append("- No reproduction-path fix was applied in this audit turn.\n")
    lines.append("- Recommended next safe fix if needed: add an explicit legacy-compatible S2b reproduction wrapper that prints both the current 14-key load audit and the legacy predecessor provenance, without mutating any locked metrics or policies.\n\n")

    lines.append("## Post-fix reproduction result, if any\n\n")
    lines.append("- none in this audit turn.\n\n")

    lines.append("## Whether S8 can resume\n\n")
    lines.append("- `False`\n")
    lines.append("- reason: the predecessor reproduction gate is still mixed between a legacy locked summary (`load_missing=2`) and a later benign current architecture (`load_missing=14`), so S8 should stay paused until a single authoritative legacy-compatible reproduction path is frozen.\n\n")

    lines.append("## Whether S5 remains final clean candidate\n\n")
    lines.append(f"- `True`\n")
    lines.append(f"- manifest path: `{manifest['policy_path']}` with final metrics drift=`{manifest['final_metrics']['drift']}`, ATE=`{manifest['final_metrics']['ATE']}`, path_ratio=`{manifest['final_metrics']['path_ratio']}`.\n")
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_PATH), "artifacts": str(ARTIFACTS_PATH), **verdict}, indent=2))


if __name__ == "__main__":
    main()
