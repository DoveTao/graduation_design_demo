import numpy as np
from dataset_pano_only import RflyPanoPanoramaPairs, _parse_label_13, FrameRec

def rot_angle_deg(R):
    # R: [3,3]
    tr = np.trace(R)
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def rel_R_t(frameA: FrameRec, frameB: FrameRec):
    R_wA, t_wA = _parse_label_13(frameA.label_path)  # R_wb, t_w
    R_wB, t_wB = _parse_label_13(frameB.label_path)
    R_BA = R_wB.T @ R_wA
    dt = np.linalg.norm(t_wB - t_wA)   # world displacement magnitude (m)
    return R_BA, dt

def summarize(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "n": int(arr.size),
    }

def main():
    data_root = "./data"   # 你的 config 相对路径逻辑下，这里一般就是 ./data
    scene = ["scene01"]
    seqs = ["seq01", "seq02", "seq03"]

    # 我们先“只用来收集 frame 列表”，k_stride/pair_step 这里无所谓
    ds = RflyPanoPanoramaPairs(data_root=data_root, hw=(1024,2048), k_stride=1, pair_step=1, scenes=scene, seqs=seqs)

    # 直接从 ds 内部拿 (FrameRec, FrameRec) 列表也可以，但它当前只按 k_stride 构 pairs。
    # 所以我们重新扫描每个 seq 的 frame 列表：最简办法是复用 ds.pairs 中的 FrameRec 去反推 frames。
    # 这里给一个直接做法：再次用 ds 的内部构造逻辑会更复杂；建议你把 dataset_pano_only.py 里 seq_frames 暴露。
    # 为了不改 dataset 文件，这里用一个折中：只统计每个 seq 下真实文件排序后的 frames（你也可以自己实现scan）。
    import os, glob
    from pathlib import Path

    pano_root = Path(data_root) / "PanoramaView" / "scene01"
    seq_dirs = [p for p in pano_root.iterdir() if p.is_dir() and p.name.startswith("seq")]
    seq_dirs = [p for p in seq_dirs if p.name in seqs]

    K = [1, 2, 5, 10]
    all_stats = {}

    for k in K:
        angs = []
        dts = []
        for sd in seq_dirs:
            panos = sorted(sd.glob("panorama_*.*"))
            # build frames list (match label)
            frames = []
            for pano_path in panos:
                stem = pano_path.stem  # panorama_<ts>
                if stem.startswith("panorama_"):
                    ts_str = stem[len("panorama_"):]
                else:
                    continue
                label_path = sd / f"label_{ts_str}.txt"
                if not label_path.exists():
                    continue
                frames.append(FrameRec("scene01", sd.name, ts_str, float(ts_str), str(pano_path), str(label_path)))
            frames.sort(key=lambda r: r.ts_val)
            n = len(frames)
            for i in range(0, n - k):
                R_BA, dt = rel_R_t(frames[i], frames[i+k])
                angs.append(rot_angle_deg(R_BA))
                dts.append(dt)

        all_stats[k] = {"rot_deg": summarize(angs), "dt_m": summarize(dts)}

    for k in K:
        s = all_stats[k]
        print(f"\n=== k={k} ===")
        print("rot_deg:", s["rot_deg"])
        print("dt_m   :", s["dt_m"])

if __name__ == "__main__":
    main()
