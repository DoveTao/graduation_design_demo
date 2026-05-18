#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from datasets.dset2c_manifest_dataset import Dset2CCanonicalPairDataset
from models.struct360b_match_free_coarse_to_fine import STRUCT360BMatchFreeCoarseToFineModel
from train360.core.config import Config


CONFIG_PATH = REPO_ROOT / "configs" / "final360i_struct360b_final.yaml"
VAL_MANIFEST = REPO_ROOT / "external_baselines" / "results" / "dset2c_360dvo_canonical" / "pair_manifest_val.jsonl"
TEST_MANIFEST = REPO_ROOT / "external_baselines" / "results" / "dset2c_360dvo_canonical" / "pair_manifest_test.jsonl"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "FINAL360I_struct360b_final" / "seed0" / "best_val.pt"
OUT_DIR = REPO_ROOT / "thesis" / "final_assets" / "figures" / "matching_geometry"
REPORT_PATH = REPO_ROOT / "reports" / "THESIS360_eval_only_matching_geometry_debug_export.md"
MANIFEST_PATH = REPO_ROOT / "reports" / "THESIS360_matching_geometry_debug_manifest.json"
ZH_CAPTION_PATH = REPO_ROOT / "thesis" / "final_assets" / "captions" / "final_figure_captions_zh.md"
EN_CAPTION_PATH = REPO_ROOT / "thesis" / "final_assets" / "captions" / "final_figure_captions_en.md"
FIG_MANIFEST_PATH = REPO_ROOT / "reports" / "THESIS362_figure_manifest.json"
VISUAL_MANIFEST_PATH = REPO_ROOT / "reports" / "THESIS362_visual_source_manifest.json"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, lines: Sequence[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_yaml_like(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _cfg_from_dict(cfg_dict: Dict[str, Any]) -> Config:
    cfg = Config()
    for key, value in cfg_dict.items():
        setattr(cfg, key, value)
    return cfg


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> STRUCT360BMatchFreeCoarseToFineModel:
    payload = torch.load(str(checkpoint_path), map_location=device)
    model_cfg = _cfg_from_dict(payload["cfg"])
    model = STRUCT360BMatchFreeCoarseToFineModel(model_cfg, device).to(device)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _safe_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _bearing_to_erp_uv(bearing: np.ndarray) -> np.ndarray:
    bearing = bearing / np.clip(np.linalg.norm(bearing, axis=-1, keepdims=True), 1.0e-8, None)
    x = bearing[..., 0]
    y = np.clip(bearing[..., 1], -1.0, 1.0)
    z = bearing[..., 2]
    lon = np.arctan2(x, z)
    lat = np.arcsin(y)
    u = (lon / (2.0 * np.pi)) + 0.5
    v = 0.5 - (lat / np.pi)
    return np.stack([u, v], axis=-1)


def _rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    delta = R_pred @ R_gt.T
    trace = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def _vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / max(float(np.linalg.norm(a)), 1.0e-8)
    b = b / max(float(np.linalg.norm(b)), 1.0e-8)
    cosine = np.clip(float(np.dot(a, b)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _relation_heatmap(a_tokens: np.ndarray, b_tokens: np.ndarray) -> np.ndarray:
    a = a_tokens / np.clip(np.linalg.norm(a_tokens, axis=-1, keepdims=True), 1.0e-8, None)
    b = b_tokens / np.clip(np.linalg.norm(b_tokens, axis=-1, keepdims=True), 1.0e-8, None)
    return a @ b.T


def _choose_sample_indices(dataset: Dset2CCanonicalPairDataset) -> List[Tuple[str, int]]:
    candidates: List[Tuple[str, int]] = []
    rules = [
        ("sample01_normal_mountains", lambda s: s["sequence_id"] == "mountains" and s["pair_type"] == "adjacent" and int(s["k"]) == 1),
        ("sample02_normal_downhill", lambda s: s["sequence_id"] == "downhill_biking" and s["pair_type"] == "adjacent" and int(s["k"]) == 1),
        ("sample03_scale_path_k5", lambda s: s["sequence_id"] == "mountains" and s["pair_type"] == "kstep" and int(s["k"]) == 5),
    ]
    for name, rule in rules:
        for idx, sample in enumerate(dataset.samples):
            if rule(sample):
                candidates.append((name, idx))
                break
    if not candidates:
        candidates.append(("sample01_fallback", 0))
    return candidates[:3]


def _sample_debug(model: STRUCT360BMatchFreeCoarseToFineModel, sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    IA = sample["IA"].unsqueeze(0).to(device)
    IB = sample["IB"].unsqueeze(0).to(device)
    R_gt = sample["R_gt"].unsqueeze(0).to(device)
    t_gt_vec = sample["t_gt_vec"].unsqueeze(0).to(device)
    dt_world = torch.tensor([float(sample["meta"]["dt_world"])], device=device, dtype=torch.float32)
    with torch.no_grad():
        R_pred, _t_local, aux = model(IA, IB, dt_world=dt_world, return_debug=True)
    debug = aux["debug_visuals"]
    fine_debug = debug["fine_refiner"]
    gt_tdir = F.normalize(t_gt_vec.float(), dim=-1, eps=1.0e-6)
    gt_tmag = torch.linalg.norm(t_gt_vec.float(), dim=-1)
    return {
        "sequence_id": sample["meta"]["sequence_id"],
        "pair_index": int(sample["meta"]["pair_index"]),
        "pair_type": sample["meta"]["pair_type"],
        "k": int(sample["meta"]["k"]),
        "timestamp_a": float(sample["meta"]["timestamp_a"]),
        "timestamp_b": float(sample["meta"]["timestamp_b"]),
        "dt_world": float(sample["meta"]["dt_world"]),
        "tokens_a_before_cross": _to_numpy(fine_debug["tokens_a_before_cross"][0]),
        "tokens_b_before_cross": _to_numpy(fine_debug["tokens_b_before_cross"][0]),
        "tokens_a_after_refine": _to_numpy(fine_debug["tokens_a_after_refine"][0]),
        "tokens_b_after_refine": _to_numpy(fine_debug["tokens_b_after_refine"][0]),
        "cross_context_ab": _to_numpy(fine_debug["cross_context_ab"][0]),
        "cross_context_ba": _to_numpy(fine_debug["cross_context_ba"][0]),
        "fine_gate": float(np.asarray(_to_numpy(fine_debug["fine_gate"])).reshape(-1)[0]),
        "residual_gate": float(np.asarray(_to_numpy(debug["residual_gate"])).reshape(-1)[0]),
        "delta_rot_vec": _to_numpy(debug["delta_rot_vec"][0]),
        "delta_tdir_vec": _to_numpy(debug["delta_tdir_vec"][0]),
        "delta_log_tmag": float(np.asarray(_to_numpy(debug["delta_log_tmag"])).reshape(-1)[0]),
        "bearingA_f": _to_numpy(debug["bearingA_f"][0]),
        "bearingB_f": _to_numpy(debug["bearingB_f"][0]),
        "bearingA_c": _to_numpy(debug["bearingA_c"][0]),
        "bearingB_c": _to_numpy(debug["bearingB_c"][0]),
        "R_pred": _to_numpy(R_pred[0]),
        "R_coarse": _to_numpy(debug["coarse_rotation"][0]),
        "R_gt": _to_numpy(R_gt[0]),
        "tdir_pred": _to_numpy(aux["tdir_after_residual"][0]),
        "tdir_coarse": _to_numpy(debug["coarse_tdir_out"][0]),
        "tdir_gt": _to_numpy(gt_tdir[0]),
        "tmag_final": float(np.asarray(_to_numpy(aux["t_mag"][0])).reshape(-1)[0]),
        "tmag_coarse": float(np.asarray(_to_numpy(aux["coarse_t_mag"][0])).reshape(-1)[0]),
        "tmag_gt": float(np.asarray(_to_numpy(gt_tmag[0])).reshape(-1)[0]),
        "rot_err_final_deg": _rotation_error_deg(_to_numpy(R_pred[0]), _to_numpy(R_gt[0])),
        "rot_err_coarse_deg": _rotation_error_deg(_to_numpy(debug["coarse_rotation"][0]), _to_numpy(R_gt[0])),
        "tdir_err_final_deg": _vector_angle_deg(_to_numpy(aux["tdir_after_residual"][0]), _to_numpy(gt_tdir[0])),
        "tdir_err_coarse_deg": _vector_angle_deg(_to_numpy(debug["coarse_tdir_out"][0]), _to_numpy(gt_tdir[0])),
    }


def _plot_cross_image_interaction(sample_name: str, payload: Dict[str, Any], out_path: Path) -> None:
    raw = _relation_heatmap(payload["tokens_a_before_cross"], payload["tokens_b_before_cross"])
    refined = _relation_heatmap(payload["tokens_a_after_refine"], payload["tokens_b_after_refine"])
    diff = refined - raw
    ctx = payload["cross_context_ab"]
    ctx_norm = np.linalg.norm(ctx, axis=-1, keepdims=True)
    ctx_map = ctx_norm @ ctx_norm.T
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    items = [
        (raw, "Pre-refine token relation", "viridis"),
        (refined, "Post-refine token relation", "viridis"),
        (diff, "Post - pre relation", "coolwarm"),
        (ctx_map, "Cross-context response energy", "magma"),
    ]
    for ax, (data, title, cmap) in zip(axes.reshape(-1), items):
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Token B")
        ax.set_ylabel("Token A")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{sample_name} | {payload['sequence_id']} | pair={payload['pair_index']} | k={payload['k']}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_residual_gate(samples: Sequence[Tuple[str, Dict[str, Any]]], out_path: Path) -> None:
    names = [name for name, _ in samples]
    residual_gate = [payload["residual_gate"] for _, payload in samples]
    fine_gate = [payload["fine_gate"] for _, payload in samples]
    delta_rot = [float(np.linalg.norm(payload["delta_rot_vec"]) * (180.0 / np.pi)) for _, payload in samples]
    delta_tdir = [float(np.linalg.norm(payload["delta_tdir_vec"])) for _, payload in samples]
    delta_log = [payload["delta_log_tmag"] for _, payload in samples]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].bar(names, residual_gate, color="#1d3557", label="residual_gate")
    axes[0].bar(names, fine_gate, color="#e76f51", alpha=0.65, label="fine_gate")
    axes[0].set_title("Gate responses")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[1].bar(names, delta_rot, color="#2a9d8f")
    axes[1].set_title("Applied delta rotation (deg)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[2].plot(names, delta_tdir, marker="o", linewidth=2.0, color="#8d99ae", label="delta_tdir_norm")
    axes[2].plot(names, delta_log, marker="s", linewidth=2.0, color="#f4a261", label="delta_log_tmag")
    axes[2].set_title("Applied translation residuals")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Residual refinement gate and update diagnostics", fontsize=12)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_coarse_final_pose(samples: Sequence[Tuple[str, Dict[str, Any]]], out_path: Path) -> None:
    names = [name for name, _ in samples]
    rot_coarse = [payload["rot_err_coarse_deg"] for _, payload in samples]
    rot_final = [payload["rot_err_final_deg"] for _, payload in samples]
    tdir_coarse = [payload["tdir_err_coarse_deg"] for _, payload in samples]
    tdir_final = [payload["tdir_err_final_deg"] for _, payload in samples]
    tmag_coarse = [payload["tmag_coarse"] / max(payload["tmag_gt"], 1.0e-8) for _, payload in samples]
    tmag_final = [payload["tmag_final"] / max(payload["tmag_gt"], 1.0e-8) for _, payload in samples]
    x = np.arange(len(names))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    axes[0].bar(x - width / 2, rot_coarse, width=width, label="coarse", color="#8d99ae")
    axes[0].bar(x + width / 2, rot_final, width=width, label="final", color="#1d3557")
    axes[0].set_title("Rotation error vs GT")
    axes[1].bar(x - width / 2, tdir_coarse, width=width, label="coarse", color="#8d99ae")
    axes[1].bar(x + width / 2, tdir_final, width=width, label="final", color="#e76f51")
    axes[1].set_title("Translation direction error vs GT")
    axes[2].bar(x - width / 2, tmag_coarse, width=width, label="coarse / gt", color="#8d99ae")
    axes[2].bar(x + width / 2, tmag_final, width=width, label="final / gt", color="#2a9d8f")
    axes[2].axhline(1.0, color="#444444", linestyle="--", linewidth=1.0)
    axes[2].set_title("Translation magnitude ratio")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle("Coarse vs final pose diagnostic on fixed eval-only samples", fontsize=12)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_token_layout(sample_name: str, payload: Dict[str, Any], out_path: Path) -> None:
    uv_c = _bearing_to_erp_uv(payload["bearingA_c"])
    uv_f = _bearing_to_erp_uv(payload["bearingA_f"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    axes[0].scatter(uv_c[:, 0], uv_c[:, 1], s=10, c=np.arange(len(uv_c)), cmap="viridis")
    axes[0].set_title("Coarse token ERP layout")
    axes[1].scatter(uv_f[:, 0], uv_f[:, 1], s=7, c=np.arange(len(uv_f)), cmap="plasma")
    axes[1].set_title("Fine token ERP layout")
    for ax in axes:
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1.0, 0.0)
        ax.grid(alpha=0.2)
        ax.set_aspect("equal")
    fig.suptitle(f"{sample_name} token layout visualization ({payload['sequence_id']})", fontsize=12)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _append_caption_block(path: Path, heading: str, title_line: str, caption_line: str) -> None:
    text = path.read_text(encoding="utf-8").rstrip()
    block = f"\n\n## {heading}\n- {title_line}\n- {caption_line}\n"
    if f"## {heading}" not in text:
        path.write_text(text + block, encoding="utf-8")


def main() -> None:
    _ensure_dir(OUT_DIR)
    cfg = _load_yaml_like(CONFIG_PATH)
    base_cfg = _load_yaml_like(REPO_ROOT / cfg["inputs"]["base_config"])
    dataset = Dset2CCanonicalPairDataset(
        str(VAL_MANIFEST if VAL_MANIFEST.exists() else TEST_MANIFEST),
        expected_split="val" if VAL_MANIFEST.exists() else "test",
        image_hw=tuple(int(x) for x in base_cfg["data"]["image_hw"]),
        require_paths=bool(base_cfg["data"]["require_paths"]),
        skip_invalid=bool(base_cfg["data"]["skip_invalid"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_from_checkpoint(CHECKPOINT_PATH, device)

    selected_samples = _choose_sample_indices(dataset)
    sample_payloads: List[Tuple[str, Dict[str, Any]]] = []
    figure_manifest_entries: List[Dict[str, Any]] = []
    for sample_name, idx in selected_samples:
        payload = _sample_debug(model, dataset[idx], device)
        sample_payloads.append((sample_name, payload))
        fig_name = f"thesis360_cross_image_interaction_{sample_name}.png"
        out_path = OUT_DIR / fig_name
        _plot_cross_image_interaction(sample_name, payload, out_path)
        figure_manifest_entries.append(
            {
                "file_path": _safe_rel(out_path),
                "title_zh": f"跨图像特征交互可视化 {sample_name}",
                "title_en": f"Cross-image interaction visualization {sample_name}",
                "source_files": [_safe_rel(CHECKPOINT_PATH), _safe_rel(VAL_MANIFEST if VAL_MANIFEST.exists() else TEST_MANIFEST)],
                "recommended_location": "thesis_appendix" if sample_name != "sample01_normal_mountains" else "defense_ppt",
                "notes": "Eval-only token-level relation heatmaps derived from real fine-stage tokens; this is not explicit keypoint matching.",
                "kind": "cross_image_interaction",
            }
        )

    gate_path = OUT_DIR / "thesis360_residual_gate_diagnostic.png"
    pose_path = OUT_DIR / "thesis360_coarse_final_pose_diagnostic.png"
    layout_path = OUT_DIR / "thesis360_token_layout_visualization.png"
    _plot_residual_gate(sample_payloads, gate_path)
    _plot_coarse_final_pose(sample_payloads, pose_path)
    _plot_token_layout(sample_payloads[0][0], sample_payloads[0][1], layout_path)
    figure_manifest_entries.extend(
        [
            {
                "file_path": _safe_rel(gate_path),
                "title_zh": "粗到细残差精化门控诊断图",
                "title_en": "Residual-gate diagnostic",
                "source_files": [_safe_rel(CHECKPOINT_PATH), _safe_rel(VAL_MANIFEST if VAL_MANIFEST.exists() else TEST_MANIFEST)],
                "recommended_location": "thesis_appendix",
                "notes": "Shows real fine-gate and residual-gate responses plus applied residual update magnitudes.",
                "kind": "residual_gate",
            },
            {
                "file_path": _safe_rel(pose_path),
                "title_zh": "coarse pose 与 final pose 诊断图",
                "title_en": "Coarse-to-final pose diagnostic",
                "source_files": [_safe_rel(CHECKPOINT_PATH), _safe_rel(VAL_MANIFEST if VAL_MANIFEST.exists() else TEST_MANIFEST)],
                "recommended_location": "thesis_appendix",
                "notes": "Compares coarse and final pose errors against GT on fixed eval-only samples.",
                "kind": "coarse_final_diagnostic",
            },
            {
                "file_path": _safe_rel(layout_path),
                "title_zh": "球面 token 布局可视化",
                "title_en": "Token layout visualization",
                "source_files": [_safe_rel(CHECKPOINT_PATH), _safe_rel(VAL_MANIFEST if VAL_MANIFEST.exists() else TEST_MANIFEST)],
                "recommended_location": "defense_ppt",
                "notes": "Visualizes actual coarse and fine spherical token locations in ERP coordinates.",
                "kind": "token_layout",
            },
        ]
    )

    unavailable_targets = [
        {
            "target": "explicit matching / keypoint correspondence",
            "reason": "The retained FINAL360I / STRUCT360B mainline is match-free and does not emit explicit keypoint matches.",
        },
        {
            "target": "epipolar allowed mask / epipolar residual / epipolar bias",
            "reason": "These tensors are not part of the retained STRUCT360B forward path and were not fabricated.",
        },
    ]

    manifest_payload = {
        "task_name": "THESIS360_eval_only_matching_geometry_debug_export",
        "eval_only_forward_executed": True,
        "training_executed": False,
        "checkpoints_modified": False,
        "metrics_modified": False,
        "debug_hooks_added": True,
        "default_model_behavior_changed": False,
        "sample_split": "val" if VAL_MANIFEST.exists() else "test",
        "samples": [
            {
                "sample_name": name,
                "sequence_id": payload["sequence_id"],
                "pair_index": payload["pair_index"],
                "pair_type": payload["pair_type"],
                "k": payload["k"],
                "dt_world": payload["dt_world"],
            }
            for name, payload in sample_payloads
        ],
        "figures": figure_manifest_entries,
        "unavailable_targets": unavailable_targets,
        "explicit_matching_visualized": False,
        "cross_image_interaction_visualized": True,
        "residual_gate_visualized": True,
        "coarse_final_diagnostic_visualized": True,
        "real_exported_quantities": [
            "fine token relation heatmap (pre-refine / post-refine cosine relation)",
            "cross-attention context response energy",
            "fine_gate and residual_gate",
            "delta rotation / delta tdir / delta log_tmag",
            "coarse pose vs final pose diagnostics",
            "coarse/fine token ERP layout",
        ],
    }
    _write_json(MANIFEST_PATH, manifest_payload)

    report_lines = [
        "# THESIS360 Eval-only Matching Geometry Debug Export",
        "",
        "- task name: `THESIS360_eval_only_matching_geometry_debug_export`",
        "- debug hook added: `true`",
        "- default model behavior changed: `false`",
        "- training executed: `false`",
        "- checkpoint modified: `false`",
        "- metrics modified: `false`",
        "- eval-only forward executed: `true`",
        "- sample split: `" + ("val" if VAL_MANIFEST.exists() else "test") + "`",
        "",
        "## Real Exported Quantities",
        "- cross-image token relation heatmaps before and after fine residual refinement",
        "- cross-attention context response energy",
        "- fine gate and residual gate scalar responses",
        "- applied delta rotation / delta tdir / delta log_tmag",
        "- coarse pose vs final pose diagnostics against GT",
        "- spherical token layout in ERP coordinates",
        "",
        "## Unavailable Targets",
    ]
    for item in unavailable_targets:
        report_lines.append(f"- {item['target']}: {item['reason']}")
    report_lines.extend(
        [
            "",
            "## Figure Usage",
        ]
    )
    for item in figure_manifest_entries:
        report_lines.append(f"- `{item['file_path']}` -> `{item['recommended_location']}`: {item['notes']}")
    report_lines.extend(
        [
            "",
            "## Naming Caveat",
            "- these figures visualize cross-image interaction / token relation / residual refinement",
            "- they are not explicit keypoint matching figures",
        ]
    )
    _write_md(REPORT_PATH, report_lines)

    _append_caption_block(
        ZH_CAPTION_PATH,
        "thesis/final_assets/figures/matching_geometry/thesis360_cross_image_interaction_sample01_normal_mountains.png",
        "标题：跨图像特征交互可视化",
        "图注：图中展示了两帧全景图像之间的 token-level relation heatmap。颜色越亮表示跨图像 token 关系响应越强。该图说明模型在相对位姿预测前进行了真实的跨图像关系建模，而不是简单的全局特征拼接。这里展示的是 cross-image interaction，不是显式 keypoint matching。",
    )
    _append_caption_block(
        EN_CAPTION_PATH,
        "thesis/final_assets/figures/matching_geometry/thesis360_cross_image_interaction_sample01_normal_mountains.png",
        "Title: Cross-image interaction visualization",
        "Caption: The heatmap shows token-level relation responses between two panoramic frames. Brighter values indicate stronger cross-image interactions. The figure illustrates that the model estimates relative pose through inter-frame feature relations rather than simple global feature concatenation. This is a cross-image interaction view, not explicit keypoint matching.",
    )
    _append_caption_block(
        ZH_CAPTION_PATH,
        "thesis/final_assets/figures/matching_geometry/thesis360_cross_image_interaction_sample02_normal_downhill.png",
        "标题：跨图像特征交互可视化",
        "图注：该图展示另一组验证样本上的 token-level cross-image relation heatmap，用于说明跨图像关系建模在不同场景下稳定存在。这里展示的是跨图像特征交互，而不是显式 keypoint correspondence。",
    )
    _append_caption_block(
        EN_CAPTION_PATH,
        "thesis/final_assets/figures/matching_geometry/thesis360_cross_image_interaction_sample02_normal_downhill.png",
        "Title: Cross-image interaction visualization",
        "Caption: This figure shows a second validation sample and visualizes the token-level cross-image relation heatmap. It indicates that the model consistently performs inter-frame relation modeling across scenes. This is a cross-image interaction view rather than explicit keypoint correspondence.",
    )
    _append_caption_block(
        ZH_CAPTION_PATH,
        "thesis/final_assets/figures/matching_geometry/thesis360_cross_image_interaction_sample03_scale_path_k5.png",
        "标题：跨图像特征交互可视化（k-step 样本）",
        "图注：该图展示较大时间间隔样本上的跨图像 token relation heatmap，用于辅助观察 scale/path 相关样本中关系建模的变化。该图不是显式匹配图，而是 relation-level 可视化。",
    )
    _append_caption_block(
        EN_CAPTION_PATH,
        "thesis/final_assets/figures/matching_geometry/thesis360_cross_image_interaction_sample03_scale_path_k5.png",
        "Title: Cross-image interaction visualization on a k-step sample",
        "Caption: This figure visualizes token-level cross-image relations on a larger temporal-gap sample, which is useful for inspecting scale/path-related behavior. The plot is a relation-level visualization, not an explicit matching figure.",
    )
    _append_caption_block(
        ZH_CAPTION_PATH,
        "thesis/final_assets/figures/matching_geometry/thesis360_residual_gate_diagnostic.png",
        "标题：粗到细残差精化中的门控响应分布",
        "图注：该图展示 fine residual 分支中的 fine gate 与 residual gate 以及对应的残差修正幅度。结果表明模型倾向于在保持 coarse pose 稳定的前提下进行保守的小幅精化。",
    )
    _append_caption_block(
        EN_CAPTION_PATH,
        "thesis/final_assets/figures/matching_geometry/thesis360_residual_gate_diagnostic.png",
        "Title: Residual-gate diagnostic",
        "Caption: The figure shows fine-gate and residual-gate responses together with the magnitude of the applied residual pose updates. The results suggest that the fine residual branch performs conservative refinements while preserving the stability of the coarse pose estimate.",
    )
    _append_caption_block(
        ZH_CAPTION_PATH,
        "thesis/final_assets/figures/matching_geometry/thesis360_coarse_final_pose_diagnostic.png",
        "标题：coarse pose 与 final pose 诊断图",
        "图注：该图比较 fixed eval-only 样本上 coarse pose 与 final pose 相对 GT 的误差，展示残差精化分支对旋转、平移方向和尺度的影响。该图用于说明模型主要是在 coarse 估计基础上做受控修正。",
    )
    _append_caption_block(
        EN_CAPTION_PATH,
        "thesis/final_assets/figures/matching_geometry/thesis360_coarse_final_pose_diagnostic.png",
        "Title: Coarse-to-final pose diagnostic",
        "Caption: This figure compares coarse-pose and final-pose errors against GT on fixed eval-only samples, showing how the residual refinement branch changes rotation, translation direction, and scale. It illustrates that the model mainly performs controlled corrections on top of the coarse estimate.",
    )
    _append_caption_block(
        ZH_CAPTION_PATH,
        "thesis/final_assets/figures/matching_geometry/thesis360_token_layout_visualization.png",
        "标题：球面 token 布局可视化",
        "图注：该图展示 coarse 与 fine 球面 token 在 ERP 坐标中的布局，用于说明模型在全景图像上采用的球面 tokenization 结构，而不是传统关键点检测与匹配流程。",
    )
    _append_caption_block(
        EN_CAPTION_PATH,
        "thesis/final_assets/figures/matching_geometry/thesis360_token_layout_visualization.png",
        "Title: Token layout visualization",
        "Caption: This figure shows the ERP-coordinate layout of the coarse and fine spherical tokens, illustrating the panoramic tokenization structure used by the model instead of a conventional keypoint detection-and-matching pipeline.",
    )

    if FIG_MANIFEST_PATH.exists():
        figure_manifest = _read_json(FIG_MANIFEST_PATH)
    else:
        figure_manifest = []
    existing_paths = {item["file_path"] for item in figure_manifest}
    for item in figure_manifest_entries:
        if item["file_path"] not in existing_paths:
            figure_manifest.append(
                {
                    "file_path": item["file_path"],
                    "title_zh": item["title_zh"],
                    "title_en": item["title_en"],
                    "source_files": item["source_files"],
                    "recommended_location": item["recommended_location"],
                    "notes": item["notes"] + " THESIS360 supplement added after the THESIS362 final asset pack.",
                }
            )
    _write_json(FIG_MANIFEST_PATH, figure_manifest)

    if VISUAL_MANIFEST_PATH.exists():
        visual_manifest = _read_json(VISUAL_MANIFEST_PATH)
    else:
        visual_manifest = []
    existing_outputs = {item["output_file"] for item in visual_manifest}
    for item in figure_manifest_entries:
        if item["file_path"] not in existing_outputs:
            visual_manifest.append(
                {
                    "output_file": item["file_path"],
                    "source_files": item["source_files"],
                    "notes": item["notes"] + " THESIS360 eval-only supplement.",
                }
            )
    _write_json(VISUAL_MANIFEST_PATH, visual_manifest)

    print(json.dumps({"written": True, "figures": len(figure_manifest_entries), "manifest": _safe_rel(MANIFEST_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
