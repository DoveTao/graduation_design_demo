#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from s5e2_adjacent_dense_lib import write_json


ALLOWED_CLASSIFICATIONS = [
    "GEN1_360DVO_PRIMARY_CANDIDATE",
    "GEN1_2D3DS_PRIMARY_CANDIDATE",
    "GEN1_EXTERNAL_DATASET_FEASIBLE",
    "GEN1_EXTERNAL_DATASET_DIFFICULT_BUT_POSSIBLE",
    "GEN1_NO_SUITABLE_EXTERNAL_DATASET_FOUND",
    "GEN1_AUDIT_ERROR",
]

REPORT_PATH = "reports/gen1_external_pano_dataset_feasibility_audit.md"
CHECKPOINT_PATH = "checkpoints/GEN1_external_pano_dataset_feasibility_audit.json"
DATA2_MOTIVATION_CLASSIFICATION = "DATA2_TRAIN_EVAL_MOTION_DISTRIBUTION_SHIFT"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        data2 = _read_json(Path(args.data2_checkpoint))
    except Exception as exc:
        payload = {
            "experiment": "GEN1_external_pano_dataset_feasibility_audit",
            "final_classification": "GEN1_AUDIT_ERROR",
            "error": str(exc),
            "s5_locked_metrics_policy_unchanged": True,
        }
        write_json(Path(args.out_json), payload)
        Path(args.out_report).write_text("# GEN1 审计失败\n", encoding="utf-8")
        return payload

    candidate_datasets: List[Dict[str, Any]] = [
        {
            "name": "360DVO Dataset",
            "task_type": "VO / OVO benchmark",
            "has_erp_images": True,
            "has_sequences": True,
            "has_pose_gt": True,
            "pose_type": "pseudo-GT (SfM / Agisoft Metashape)",
            "has_timestamps": True,
            "download_available": True,
            "license_or_access": "public project page + Hugging Face dataset card; exact dataset license should be rechecked before redistribution",
            "estimated_integration_difficulty": "low",
            "recommended_use": "external_eval",
            "source_urls": [
                "https://360dvo.hkustvgd.com/",
                "https://huggingface.co/datasets/chris1004336379/360DVO",
                "https://arxiv.org/abs/2601.02309",
            ],
            "notes": {
                "resolution": "3840x1920 ERP",
                "fps": 10,
                "num_sequences": 20,
                "size": "about 11.3 GB",
                "why_fit": "closest to omnidirectional VO with continuous sequences and trajectory supervision",
            },
        },
        {
            "name": "Stanford 2D-3D-S",
            "task_type": "indoor panorama / scene understanding / pair construction",
            "has_erp_images": True,
            "has_sequences": "partial",
            "has_pose_gt": True,
            "pose_type": "GT / camera information",
            "has_timestamps": False,
            "download_available": True,
            "license_or_access": "public Stanford dataset access; exact license terms should be checked on download page",
            "estimated_integration_difficulty": "medium",
            "recommended_use": "compatibility / pretrain / pair construction",
            "source_urls": [
                "https://cvgl.stanford.edu/resources.html",
                "https://arxiv.org/abs/1702.01105",
            ],
            "notes": {
                "modalities": "regular RGB + 360 equirectangular + depth + normals + semantic + camera info",
                "scale": "covers over 6000m2 and over 70000 RGB images according to the paper",
                "why_not_primary": "not presented as a standard VO sequence benchmark with native timestamps",
            },
        },
        {
            "name": "360VO Dataset",
            "task_type": "VO / related omnidirectional baseline dataset",
            "has_erp_images": True,
            "has_sequences": True,
            "has_pose_gt": True,
            "pose_type": "benchmark sequence GT / synthetic+real mix in paper context",
            "has_timestamps": "unknown",
            "download_available": True,
            "license_or_access": "project page points to SharePoint sequence links; access persistence should be verified",
            "estimated_integration_difficulty": "medium",
            "recommended_use": "diagnostic / secondary external_eval",
            "source_urls": [
                "https://researchportal.hkust.edu.hk/en/publications/360vo-visual-odometry-using-a-single-360-camera",
                "https://www.saikit.org/static/projects/360vo_2022/360VO_ICRA2022.pdf",
            ],
            "notes": {
                "why_not_primary": "access path is less stable than 360DVO and metadata packaging is less explicit",
            },
        },
        {
            "name": "360-Indoor",
            "task_type": "indoor panorama / detection / recognition",
            "has_erp_images": True,
            "has_sequences": False,
            "has_pose_gt": False,
            "pose_type": "none for VO",
            "has_timestamps": False,
            "download_available": "unknown",
            "license_or_access": "paper describes release intent, but not a clear VO-style packaged benchmark",
            "estimated_integration_difficulty": "high",
            "recommended_use": "pretrain / not_main_pose_eval",
            "source_urls": [
                "https://openaccess.thecvf.com/content_WACV_2020/html/Chou_360-Indoor_Towards_Learning_Real-World_Objects_in_360deg_Indoor_Equirectangular_Images_WACV_2020_paper.html",
            ],
            "notes": {
                "why_not_primary": "object detection dataset rather than VO sequence dataset",
            },
        },
    ]

    adapter_schema = {
        "primary_dataset": "360DVO Dataset",
        "sample_record": {
            "image_path_a": "...",
            "image_path_b": "...",
            "timestamp_a": 0.0,
            "timestamp_b": 0.1,
            "T_w_a": "...",
            "T_w_b": "...",
            "R_BA": "...",
            "t_BA_B": "...",
            "tmag": 0.0,
            "seq_id": "easy_00_bridge_night",
            "pair_type": "adjacent",
        },
        "convention_notes": {
            "T_wc_vs_T_cw": "adapter should normalize raw dataset poses into T_w_c first, then derive R_BA and t_BA_B using the same relative_pose_A_to_B_in_B convention already used in this repo",
            "quaternion_order": "store explicit adapter field for dataset quaternion order and convert to repo standard before pair generation",
            "tdir_frame": "B frame, matching current external evaluator assumptions",
            "scale_unit": "meters if provided; otherwise document pseudo-GT scale source",
            "timestamp_alignment": "for 360DVO use native 10 FPS frame index / timestamps; for datasets without timestamps synthesize monotonically increasing timestamps but mark them as synthetic",
            "erp_resolution_resize_policy": "keep native ERP metadata, then resize to current pipeline resolution in adapter config, not by overwriting raw files",
            "train_eval_split_design": "split by sequence id, not by random adjacent pairs, to avoid another DATA2-style motion leakage",
            "tum_export": "adapter should emit TUM-compatible absolute trajectory so current external evaluator can be reused",
        },
    }

    one_week_plan = {
        "Day 1": "inspect download instructions, metadata structure, sample file tree, and one sequence worth of pose files",
        "Day 2": "write dataset adapter that converts raw poses and ERP frames into repo pair manifest and TUM trajectory format",
        "Day 3": "build a 100-pair smoke split with adjacent and k-step pairs, then run dataset statistics audit",
        "Day 4": "run current S5E15-like export/eval path if feasible, or at minimum verify rot/tdir/tmag/path metrics can be computed",
        "Day 5": "write GEN2 external adapter report with failure cases, split policy, and data caveats",
    }

    final = "GEN1_360DVO_PRIMARY_CANDIDATE"
    recommendation = {
        "do_gen2_external_adapter": True,
        "preferred_dataset": "360DVO Dataset",
        "keep_s5e15_as_best_candidate": True,
        "main_next_step": "GEN2_external_adapter_for_360DVO",
    }
    payload = {
        "experiment": "GEN1_external_pano_dataset_feasibility_audit",
        "data2_motivation": {
            "final_classification": data2.get("final_classification"),
            "do_gen1_external_dataset": bool(data2.get("recommendation", {}).get("do_gen1_external_dataset")),
        },
        "candidate_datasets": candidate_datasets,
        "primary_candidate": "360DVO Dataset",
        "secondary_candidate": "Stanford 2D-3D-S",
        "not_main_pose_eval": ["360-Indoor"],
        "adapter_schema": adapter_schema,
        "one_week_plan": one_week_plan,
        "recommendation": recommendation,
        "s5_locked_metrics_policy_unchanged": True,
        "final_classification": final,
    }
    write_json(Path(args.out_json), payload)

    report = [
        "# GEN1 外部全景数据集泛化可行性审计",
        "",
        "## 1. 为什么需要外部泛化",
        "DATA2 已经显示当前 scene01 split 存在明显 train/eval motion distribution shift，因此需要外部全景 VO 数据来验证当前结论是否只是单数据集分布问题。",
        "",
        "## 2. DATA2 对单数据 split 风险的证据",
        json.dumps(payload["data2_motivation"], ensure_ascii=False, indent=2),
        "",
        "## 3. 候选数据集表",
        json.dumps(candidate_datasets, ensure_ascii=False, indent=2),
        "",
        "## 4. primary 推荐数据集",
        "360DVO Dataset。它最接近当前目标：ERP 全景、连续序列、10 FPS、轨迹监督、明确作为 OVO benchmark 发布，最适合做 GEN2 external eval。",
        "",
        "## 5. secondary 推荐数据集",
        "Stanford 2D-3D-S。它有 360 equirectangular 图像和 camera information，更适合做 compatibility / pair construction / pretraining，而不是直接替代 main VO benchmark。",
        "",
        "## 6. 不推荐作为 main pose eval 的数据集",
        "360-Indoor 不适合作为 main pose eval，因为它是 360 室内检测/识别数据集，不是连续 VO sequence benchmark。",
        "",
        "## 7. adapter schema",
        json.dumps(adapter_schema, ensure_ascii=False, indent=2),
        "",
        "## 8. 一周 GEN2 实施计划",
        json.dumps(one_week_plan, ensure_ascii=False, indent=2),
        "",
        "## 9. caveats",
        "- 本轮是不训练模型、不生成新 candidate 的外部数据接入可行性审计。",
        "- 本审计不下载大型原始数据。",
        "- 360VO / 360DVO 的访问链接与授权条款在真正接入前需要再次人工确认。",
        "- Stanford 2D-3D-S 更像 compatibility / pretraining / pair construction 资源，不一定天然具备 VO timestamp protocol。",
        "",
        "## 附：输出路径",
        f"- checkpoint: {CHECKPOINT_PATH}",
        f"- report: {REPORT_PATH}",
    ]
    Path(args.out_report).write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--data2-checkpoint", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-report", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
