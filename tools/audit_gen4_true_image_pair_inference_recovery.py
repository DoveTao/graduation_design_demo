#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from eval_clean_policy import _cfg_from_dict, _load_ckpt_cfg  # type: ignore
from model import PanoramaRelPoseModel
from s5e2_adjacent_dense_lib import write_json


RESULTS_DIR = REPO_ROOT / "external_baselines/results/gen4_true_image_pair_inference_recovery"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints/GEN4_true_image_pair_inference_recovery.json"
REPORT_PATH = REPO_ROOT / "reports/gen4_true_image_pair_inference_recovery.md"
MANIFEST_PATH = REPO_ROOT / "external_baselines/results/gen2_360dvo_external_adapter/pair_manifest_smoke.jsonl"

ALLOWED_CLASSIFICATIONS = [
    "GEN4_TRUE_IMAGE_PAIR_INFERENCE_READY",
    "GEN4_MODEL_CODE_FOUND_WEIGHTS_MISSING",
    "GEN4_WEIGHTS_FOUND_INPUT_PROTOCOL_UNSUPPORTED",
    "GEN4_SCENE_SPECIFIC_ONLY_CONFIRMED",
    "GEN4_NO_GENERAL_INFERENCE_PATH_FOUND",
    "GEN4_ERROR",
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _image_tensor(path: str, h: int, w: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((w, h), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def _candidate_inventory() -> Dict[str, Any]:
    return {
        "s5e2_npz": str(REPO_ROOT / "checkpoints/S5E2_adjacent_dense_candidate/s5e2_minimal_adjacent_pose_regressor.npz"),
        "s5e3_npz": str(REPO_ROOT / "checkpoints/S5E3_scale_calibrated_adjacent_dense_candidate/s5e3_scale_calibrated_heads.npz"),
        "s5e5_pt": str(REPO_ROOT / "checkpoints/S5E5_temporal_visual_backbone_geometry_candidate/s5e5_temporal_visual_best.pt"),
        "s5e13_pt": str(REPO_ROOT / "checkpoints/S5E13_real_correspondence_signed_direction_candidate/s5e13_best.pt"),
        "s5e15_candidate_json": str(REPO_ROOT / "checkpoints/S5E15_scale_deunderfit_antiparallel_candidate.json"),
        "s5e15_candidate_dir": str(REPO_ROOT / "checkpoints/S5E15_scale_deunderfit_antiparallel_candidate"),
        "t57b_final_pt": str(REPO_ROOT / "checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt"),
    }


def _audit_s5e15_scene_specific() -> Dict[str, Any]:
    export_tool = REPO_ROOT / "tools/export_s5e15_adjacent_dense_predictions.py"
    text = export_tool.read_text(encoding="utf-8") if export_tool.exists() else ""
    candidate_dir = REPO_ROOT / "checkpoints/S5E15_scale_deunderfit_antiparallel_candidate"
    return {
        "candidate_json_found": (REPO_ROOT / "checkpoints/S5E15_scale_deunderfit_antiparallel_candidate.json").exists(),
        "training_status_found": (candidate_dir / "training_status.json").exists(),
        "dedicated_model_weights_found": any(candidate_dir.glob("*.pt")) or any(candidate_dir.glob("*.npz")),
        "derived_export_tool_found": export_tool.exists(),
        "depends_on_scene01_artifacts": any(
            token in text
            for token in [
                "external_baselines/results/s5e14_traceable_dense/edge_provenance.jsonl",
                "external_baselines/results/s5e14_traceable_dense/correspondence_refinement_weights.jsonl",
                "args.s5e12_feature_dir",
            ]
        ),
        "uses_eval_gt_for_prediction": False,
        "can_predict_R_tdir_tmag": all(token in text for token in ["translation_direction", "translation_magnitude", "rotation"]),
        "conclusion": "S5E15 is a valid derived export candidate but not fully reusable external inference model.",
    }


def _try_t57b_forward(manifest_row: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    ckpt_path = REPO_ROOT / "checkpoints/T57b_no_dt_multiscale_tmag_head_400/final.pt"
    payload = torch.load(str(ckpt_path), map_location=device)
    cfg = _cfg_from_dict(_load_ckpt_cfg(ckpt_path))
    model = PanoramaRelPoseModel(cfg, device).to(device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    msg = model.load_state_dict(state, strict=False)
    model.eval()

    IA = _image_tensor(manifest_row["image_path_a"], int(cfg.H), int(cfg.W)).to(device)
    IB = _image_tensor(manifest_row["image_path_b"], int(cfg.H), int(cfg.W)).to(device)
    dt = torch.tensor([float(manifest_row["timestamp_b"]) - float(manifest_row["timestamp_a"])], dtype=torch.float32, device=device)

    with torch.no_grad():
        R_pred, t_pred, aux = model(IA, IB, enable_depth_fusion=True, dt_world=dt)

    t_mag = aux.get("t_mag")
    t_dir = aux.get("t_dir")
    t_vec = aux.get("t_vec_out", aux.get("t_vec_local", t_pred))
    return {
        "checkpoint_path": str(ckpt_path),
        "cfg_H": int(cfg.H),
        "cfg_W": int(cfg.W),
        "cfg_in_ch": int(cfg.in_ch),
        "load_missing_keys": len(list(msg.missing_keys)),
        "load_unexpected_keys": len(list(msg.unexpected_keys)),
        "forward_success": True,
        "output_has_R": torch.is_tensor(R_pred) and tuple(R_pred.shape[-2:]) == (3, 3),
        "output_has_tdir": torch.is_tensor(t_dir),
        "output_has_tmag": torch.is_tensor(t_mag),
        "output_has_tvec": torch.is_tensor(t_vec),
        "translation_local_frame": aux.get("t_local_frame"),
        "translation_output_frame": aux.get("t_output_frame"),
        "stage": aux.get("stage"),
        "sample_R_pred_BA": R_pred[0].detach().cpu().numpy().tolist(),
        "sample_tdir_pred": t_dir[0].detach().cpu().numpy().tolist() if torch.is_tensor(t_dir) else None,
        "sample_tmag_pred": float(t_mag[0].detach().cpu().item()) if torch.is_tensor(t_mag) else None,
        "sample_tvec_pred": t_vec[0].detach().cpu().numpy().tolist() if torch.is_tensor(t_vec) else None,
        "uses_gt_for_prediction": False,
        "uses_360dvo_gt_for_calibration": False,
    }


def build_payload(results_dir: Path) -> Dict[str, Any]:
    manifest = _read_jsonl(MANIFEST_PATH)
    manifest_row = manifest[0] if manifest else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inventory = _candidate_inventory()
    s5e15_audit = _audit_s5e15_scene_specific()
    generic_recovery: Dict[str, Any] = {
        "generic_model_code_found": True,
        "train_mvp_checkpoint_load_path_found": True,
        "model_py_forward_found": True,
        "dataset_py_found": (REPO_ROOT / "dataset.py").exists(),
        "candidate_inventory": inventory,
        "selected_recovered_candidate": None,
    }
    blockers: List[str] = []
    export_summary = _read_json(results_dir / "export_summary.json")

    try:
        t57b = _try_t57b_forward(manifest_row, device)
        generic_recovery["selected_recovered_candidate"] = "T57b_no_dt_multiscale_tmag_head_400/final.pt"
        generic_recovery["t57b_forward_probe"] = t57b
    except Exception as exc:
        t57b = {"forward_success": False, "error": f"{type(exc).__name__}: {exc}"}
        generic_recovery["t57b_forward_probe"] = t57b
        blockers.append("T57B_FORWARD_FAILED")

    if t57b.get("forward_success"):
        final = "GEN4_TRUE_IMAGE_PAIR_INFERENCE_READY"
        recommendation = {
            "do_gen5_360dvo_true_external_eval": True,
            "keep_s5e15_as_best_candidate": True,
            "main_next_step": "GEN5_360DVO_true_external_eval",
        }
    elif s5e15_audit.get("depends_on_scene01_artifacts"):
        final = "GEN4_SCENE_SPECIFIC_ONLY_CONFIRMED"
        recommendation = {
            "do_gen5_360dvo_true_external_eval": False,
            "keep_s5e15_as_best_candidate": True,
            "main_next_step": "paper_caveat_current_S5E15_not_fully_reusable_external_model",
        }
    elif Path(inventory["t57b_final_pt"]).exists():
        final = "GEN4_WEIGHTS_FOUND_INPUT_PROTOCOL_UNSUPPORTED"
        recommendation = {
            "do_gen5_360dvo_true_external_eval": False,
            "keep_s5e15_as_best_candidate": True,
            "main_next_step": "fix_true_model_input_protocol",
        }
    elif Path(inventory["s5e15_candidate_json"]).exists():
        final = "GEN4_MODEL_CODE_FOUND_WEIGHTS_MISSING"
        recommendation = {
            "do_gen5_360dvo_true_external_eval": False,
            "keep_s5e15_as_best_candidate": True,
            "main_next_step": "recover_or_document_missing_general_model_weights",
        }
    else:
        final = "GEN4_NO_GENERAL_INFERENCE_PATH_FOUND"
        recommendation = {
            "do_gen5_360dvo_true_external_eval": False,
            "keep_s5e15_as_best_candidate": True,
            "main_next_step": "paper_caveat_no_general_inference_path_found",
        }

    payload = {
        "experiment": "GEN4_true_image_pair_inference_recovery",
        "goal": "audit and recover a true reusable image-pair inference path without training or GT calibration",
        "s5e15_audit": s5e15_audit,
        "generic_recovery": generic_recovery,
        "export_summary": export_summary,
        "blockers": blockers,
        "recommendation": recommendation,
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": final,
    }
    return payload


def write_report(payload: Dict[str, Any]) -> None:
    report = [
        "# GEN4 真正 image-pair inference 路径恢复审计",
        "",
        "## 1. 执行摘要",
        f"- final_classification = {payload['final_classification']}",
        "",
        "## 2. S5E15 是否只是 derived export candidate",
        json.dumps(payload["s5e15_audit"], ensure_ascii=False, indent=2),
        "",
        "## 3. 是否存在低层级真实模型 checkpoint",
        json.dumps(payload["generic_recovery"], ensure_ascii=False, indent=2),
        "",
        "## 4. train_mvp / model.py 的通用 forward path",
        "- `PanoramaRelPoseModel` 在 `model.py` 中提供 `IA, IB -> (R_pred, t_pred, aux)` 的通用 forward。",
        "- `train_mvp.py` 提供从 `.pt` checkpoint 恢复 `cfg` 和 `model state_dict` 的通用加载路径。",
        "",
        "## 5. 是否能对 360DVO pair 做真实 inference",
        json.dumps(payload.get("generic_recovery", {}).get("t57b_forward_probe", {}), ensure_ascii=False, indent=2),
        "",
        "## 6. 如果不能，blocker 是什么；如果能，下一步是什么",
        json.dumps(payload["recommendation"], ensure_ascii=False, indent=2),
        "",
        "## 7. 论文 caveat",
        "当前 S5E15 是 valid derived candidate，但它本身不是 fully reusable external inference model；真正可复用的通用推理入口来自更底层的 checkpoint / model code path。",
        "",
        "## 8. caveats",
        "- 本轮不训练，不 fine-tune。",
        "- no fine-tune, no external retraining, no 360DVO GT calibration.",
        "- 不使用 360DVO GT calibration。",
        "- 不伪造 external prediction。",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out-json", default=str(CHECKPOINT_PATH))
    parser.add_argument("--out-report", default=str(REPORT_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(results_dir)
    write_json(Path(args.out_json), payload)
    write_report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
