#!/usr/bin/env python3
"""check_triplet_transform_composition.py

只读单元测试：验证 dataset triplet 的 transform composition 是否正确。

验证：
  T_BA = (R_gt, t_gt_vec)：B frame 下 A→B 变换
  T_CB = (R_gt_bc, t_gt_bc_vec)：C frame 下 B→C 变换
  组合 T_CA = T_CB * T_BA：
    R_CA_comp = R_CB @ R_BA
    t_CA_comp = R_CB @ t_BA + t_CB

  直接 T_CA（从 world pose）：
    R_CA_direct = R_wC.T @ R_wA
    t_CA_direct = R_wC.T @ (t_wA - t_wC)

不训练，不写 checkpoints。
"""

import argparse
import sys
import os
import numpy as np
import torch

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_pano_only import RflyPanoPanoramaPairsMixedK


def build_dataset(data_dir):
    """构造最小化 dataset，仅启用 triplet。"""
    return RflyPanoPanoramaPairsMixedK(
        data_root=data_dir,
        hw=(1024, 2048),
        k_choices=[1],
        k_probs=[1.0],
        pair_step=1,
        min_dt=0.02,
        max_tries=10,
        seed=1234,
        split='train',
        split_by='scene_seq',
        train_ratio=0.999,
        return_seq_turn_triplet=True,
        seq_turn_only_k=1,
    )


def rotation_error(R1, R2):
    """Frobenius norm of R1 - R2, 归一化到[0,2]。"""
    return float(np.linalg.norm(R1 - R2, ord='fro'))


def translation_error(t1, t2):
    """L2 norm of t1 - t2。"""
    return float(np.linalg.norm(t1 - t2))


def cosine_error(t1, t2):
    """1 - cosine similarity。"""
    n1 = np.linalg.norm(t1)
    n2 = np.linalg.norm(t2)
    if n1 < 1e-12 and n2 < 1e-12:
        return 0.0
    if n1 < 1e-12 or n2 < 1e-12:
        return 1.0
    return float(1.0 - np.dot(t1, t2) / (n1 * n2))


def main():
    parser = argparse.ArgumentParser(description="Verify triplet transform composition")
    parser.add_argument("--num", type=int, default=50,
                        help="Number of valid triplet samples to check (default: 50)")
    parser.add_argument("--data-dir", type=str, default="data/PanoramaView",
                        help="Panorama dataset root")
    args = parser.parse_args()

    cfg = build_dataset(args.data_dir)  # cfg is now the dataset
    dataset = cfg

    print(f"[Init] dataset size = {len(dataset)}")

    # 误差收集
    rot_errs = []
    trans_errs = []
    cos_errs = []
    # camera-center 解释误差
    cc_CA_direct_errs = []   # t_CA vs R_wC.T @ (t_wA - t_wC)
    cc_CA_negRT_errs = []    # -R_CA.T @ t_CA vs R_wA.T @ (t_wC - t_wA)
    # t_BA 解释：camera center of A in B
    cc_BA_direct_errs = []  # t_BA vs R_wB.T @ (t_wA - t_wB)
    cc_BA_negRT_errs = []   # -R_BA.T @ t_BA vs R_wA.T @ (t_wB - t_wA)

    checked = 0
    idx = 0
    first_fail = None

    while checked < args.num and idx < len(dataset):
        sample = dataset[idx]
        meta = sample["meta"]

        if not meta.get("has_seq_turn_triplet", False):
            idx += 1
            continue

        # 从 meta 查找对应的 seq_data
        scene = meta.get("scene", "")
        seq = meta.get("seq", "")
        sd = None
        for s in dataset.seqs_data:
            if s.get("scene") == scene and s.get("seq") == seq:
                sd = s
                break
        if sd is None:
            print(f"[WARN] cannot find seq_data for scene={scene} seq={seq}, skip")
            idx += 1
            continue

        R_w = sd["R_w"]   # [n, 3, 3]
        t_w = sd["t_w"]   # [n, 3]
        i = int(meta["i"])
        j = int(meta["j"])

        R_wA = R_w[i].astype(np.float64)
        t_wA = t_w[i].astype(np.float64)
        R_wB = R_w[j].astype(np.float64)
        t_wB = t_w[j].astype(np.float64)
        c_idx = j + 1  # triplet C is always j+1 when k == seq_turn_only_k
        R_wC = R_w[c_idx].astype(np.float64)
        t_wC = t_w[c_idx].astype(np.float64)

        # 从 sample 读取 T_BA 和 T_CB
        R_BA = sample["R_gt"].numpy().astype(np.float64)       # A→B in B frame
        t_BA = sample["t_gt_vec"].numpy().astype(np.float64)
        R_CB = sample["R_gt_bc"].numpy().astype(np.float64)   # B→C in C frame
        t_CB = sample["t_gt_bc_vec"].numpy().astype(np.float64)

        # ----- 验证 1: composition T_CA = T_CB * T_BA -----
        R_CA_comp = R_CB @ R_BA
        t_CA_comp = R_CB @ t_BA + t_CB

        # 直接 T_CA = world→C @ A→world
        R_CA_direct = R_wC.T @ R_wA
        t_CA_direct = R_wC.T @ (t_wA - t_wC)

        r_err = rotation_error(R_CA_comp, R_CA_direct)
        t_err = translation_error(t_CA_comp, t_CA_direct)
        c_err = cosine_error(t_CA_comp, t_CA_direct)

        rot_errs.append(r_err)
        trans_errs.append(t_err)
        cos_errs.append(c_err)

        if first_fail is None and (r_err > 1e-4 or t_err > 1e-4):
            first_fail = {
                "idx": idx, "i": int(i), "j": int(j), "c": c_idx,
                "rot_err": r_err, "trans_err": t_err,
                "R_comp": R_CA_comp, "R_direct": R_CA_direct,
                "t_comp": t_CA_comp, "t_direct": t_CA_direct,
            }

        # ----- 验证 2: camera center / displacement -----
        # t_BA 作为 A 的 camera center 在 B frame
        # direct: R_wB.T @ (t_wA - t_wB)
        cc_BA_direct = R_wB.T @ (t_wA - t_wB)
        cc_BA_direct_errs.append(translation_error(t_BA, cc_BA_direct))

        # -R_BA.T @ t_BA 作为 B 原点在 A frame
        # direct: R_wA.T @ (t_wB - t_wA)
        cc_BA_negRT = -R_BA.T @ t_BA
        cc_BA_negRT_target = R_wA.T @ (t_wB - t_wA)
        cc_BA_negRT_errs.append(translation_error(cc_BA_negRT, cc_BA_negRT_target))

        # t_CA_comp 作为 A 的 camera center 在 C frame
        cc_CA_direct_errs.append(translation_error(t_CA_comp, t_CA_direct))

        # -R_CA_comp.T @ t_CA_comp 作为 C 原点在 A frame
        cc_CA_negRT = -R_CA_comp.T @ t_CA_comp
        cc_CA_negRT_target = R_wA.T @ (t_wC - t_wA)
        cc_CA_negRT_errs.append(translation_error(cc_CA_negRT, cc_CA_negRT_target))

        checked += 1
        idx += 1

    # ----- 输出统计 -----
    print(f"\n{'='*70}")
    print(f"Checked {checked} valid triplet samples")
    print(f"{'='*70}")

    rot_errs = np.array(rot_errs)
    trans_errs = np.array(trans_errs)
    cos_errs = np.array(cos_errs)

    print(f"\n--- Composition Verification: T_CA = T_CB * T_BA ---")
    print(f"  Rotation Frobenius error:")
    print(f"    max={rot_errs.max():.2e}  mean={rot_errs.mean():.2e}  median={np.median(rot_errs):.2e}")
    print(f"  Translation L2 error:")
    print(f"    max={trans_errs.max():.2e}  mean={trans_errs.mean():.2e}  median={np.median(trans_errs):.2e}")
    print(f"  Cosine error (1 - cos):")
    print(f"    max={cos_errs.max():.2e}  mean={cos_errs.mean():.2e}  median={np.median(cos_errs):.2e}")

    if rot_errs.max() < 1e-10 and trans_errs.max() < 1e-10:
        print(f"\n  ✅ T_CA = T_CB * T_BA 组合公式精确成立（机器精度内）")
    elif rot_errs.max() < 1e-4 and trans_errs.max() < 1e-4:
        print(f"\n  ✅ T_CA = T_CB * T_BA 组合公式近似成立（误差 < 1e-4）")
    else:
        print(f"\n  ❌ 存在不可忽略的误差！")
        if first_fail:
            print(f"  First failed sample: idx={first_fail['idx']} i={first_fail['i']} j={first_fail['j']} c={first_fail['c']}")
            print(f"    rot_err={first_fail['rot_err']:.6e}  trans_err={first_fail['trans_err']:.6e}")

    # ----- Camera center 验证 -----
    cc_BA_direct_errs = np.array(cc_BA_direct_errs)
    cc_BA_negRT_errs = np.array(cc_BA_negRT_errs)
    cc_CA_direct_errs = np.array(cc_CA_direct_errs)
    cc_CA_negRT_errs = np.array(cc_CA_negRT_errs)

    print(f"\n--- Camera Center / Displacement Interpretation ---")
    print(f"  t_BA as camera center of A in B frame (t_BA vs R_wB.T @ (t_wA - t_wB)):")
    print(f"    max={cc_BA_direct_errs.max():.2e}  mean={cc_BA_direct_errs.mean():.2e}  median={np.median(cc_BA_direct_errs):.2e}")
    print(f"  -R_BA.T @ t_BA as B origin in A frame (vs R_wA.T @ (t_wB - t_wA)):")
    print(f"    max={cc_BA_negRT_errs.max():.2e}  mean={cc_BA_negRT_errs.mean():.2e}  median={np.median(cc_BA_negRT_errs):.2e}")

    print(f"  t_CA_comp as camera center of A in C frame:")
    print(f"    max={cc_CA_direct_errs.max():.2e}  mean={cc_CA_direct_errs.mean():.2e}  median={np.median(cc_CA_direct_errs):.2e}")
    print(f"  -R_CA.T @ t_CA_comp as C origin in A frame:")
    print(f"    max={cc_CA_negRT_errs.max():.2e}  mean={cc_CA_negRT_errs.mean():.2e}  median={np.median(cc_CA_negRT_errs):.2e}")

    # 结论
    print(f"\n--- Conclusion ---")
    if cc_BA_direct_errs.max() < 1e-4:
        print(f"  ✅ t_BA = camera center of A in B frame（R_wB.T @ (t_wA - t_wB)）")
    else:
        print(f"  ❌ t_BA 不等于 camera center，检查 convention")
    if cc_BA_negRT_errs.max() < 1e-4:
        print(f"  ✅ -R_BA.T @ t_BA = B origin in A frame（R_wA.T @ (t_wB - t_wA)）")
    else:
        print(f"  ❌ -R_BA.T @ t_BA 不等于 B origin，检查 convention")

    print(f"\n  Camera center interpretation confirmed: t in (R,t) IS the source's origin")
    print(f"  expressed in the target frame (x_dst = R @ x_src + t, so t = x_dst when x_src=0).")
    print(f"  -R.T @ t gives the TARGET's origin expressed in the SOURCE frame.")

    # 最终判决
    all_close = rot_errs.max() < 1e-4 and trans_errs.max() < 1e-4
    if all_close:
        print(f"\n{'='*70}")
        print(f"  ✅ T_CA = T_CB * T_BA composition verified — ready for T53b.")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"  ❌ Composition verification FAILED — DO NOT proceed with T53b.")
        print(f"{'='*70}")
        sys.exit(1)


if __name__ == "__main__":
    main()
