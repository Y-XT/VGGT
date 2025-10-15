#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uav_frame_metrics.py

针对uavid2020
若不做过滤，阶段一会对 data/ 与 data_1280/ 分别产出 frames.jsonl；阶段二也会分别构建三元组，数据量翻倍

阶段一（离线逐帧统计）：
  - 递归遍历 root_dir 下的所有“叶子图像目录”（含 >=min_images 的图像目录）
  - 为每帧计算轻量图像级指标（不依赖 VGGT）：
      * 尺寸 H,W
      * 亮度均值/方差（HSV 的 V 通道）
      * HSV 的 S(饱和度)均值
      * 拉普拉斯方差（模糊度 proxy）
      * Sobel 边缘密度（纹理/细节 proxy）
      * 灰度直方图熵（信息量 proxy）
      * clip_dark / clip_bright（欠曝/过曝像素比例）
  - 每个序列输出： out_dir/<rel_seq>/frames.jsonl
  - 汇总输出：     out_dir/global_stats.json（全库分位数/均值，用于归一化，比如 blur 的 P50）

用法：
  python uav_frame_metrics.py \
    --root_dir /path/to/train \
    --out_dir  /path/to/out_metrics \
    --min_images 10 --short_side 640
"""
import argparse
import json
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

def find_image_dirs(root_dir: str, min_images: int = 10) -> List[Path]:
    root = Path(root_dir)
    seq_dirs = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        try:
            files = [f for f in os.listdir(p) if f.lower().endswith(IMG_EXTS)]
        except PermissionError:
            continue
        if len(files) >= min_images:
            seq_dirs.append(p)
    seq_dirs.sort()
    return seq_dirs

def list_images(seq_dir: Path) -> List[str]:
    return [str(seq_dir / f) for f in sorted(os.listdir(seq_dir)) if f.lower().endswith(IMG_EXTS)]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def lap_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def sobel_edge_density(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    thr = float(0.1 * (mag.mean() + 1e-6))
    ed = float((mag > thr).mean())
    return ed

def entropy_gray(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    p = hist / (hist.sum() + 1e-12)
    p = p[p>0]
    return float(-(p*np.log2(p)).sum())

def brightness_stats(img_bgr: np.ndarray):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    v_mean = float(v.mean()); v_std = float(v.std())
    s_mean = float(s.mean())
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray, s_mean, v_mean, v_std

def clip_ratios(gray: np.ndarray, low_thr=5, high_thr=250) -> tuple:
    total = gray.size
    dark = float((gray <= low_thr).sum()) / total
    bright = float((gray >= high_thr).sum()) / total
    return dark, bright

def resize_short_side(img: np.ndarray, short_side: int) -> np.ndarray:
    if short_side <= 0:
        return img
    h, w = img.shape[:2]
    if h <= w:
        new_h = short_side
        new_w = int(round(w * short_side / h))
    else:
        new_w = short_side
        new_h = int(round(h * short_side / w))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str,
                    default="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany_tri/All",
                    #default="/mnt/data_nvme3n1p1/dataset/UAV_ula/tri_images",
                    help="根目录（包含多个子目录/序列）")
    ap.add_argument("--out_dir",  type=str,
                    default="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany_tri/tri_metrics",
                    #default="/mnt/data_nvme3n1p1/dataset/UAV_ula/tri_metric",
                    help="输出根目录（每序列一个 frames.jsonl + 全局 global_stats.json）")
    ap.add_argument("--min_images", type=int, default=10, help="识别为序列目录的最少图像数")
    ap.add_argument("--short_side", type=int, default=0, help="指标计算时的短边（0 表示不缩放）")
    args = ap.parse_args()

    seq_dirs = find_image_dirs(args.root_dir, args.min_images)
    if not seq_dirs:
        raise FileNotFoundError(f"在 {args.root_dir} 未找到含 >= {args.min_images} 张图片的目录")

    out_root = Path(args.out_dir); ensure_dir(out_root)

    all_lap = []; all_vmean = []; all_entropy = []

    for seq in tqdm(seq_dirs, desc="逐序列统计"):
        imgs = list_images(seq)
        if not imgs: 
            continue
        rel = seq.relative_to(args.root_dir)
        out_dir = out_root / rel
        ensure_dir(out_dir)
        with (out_dir / "frames.jsonl").open("w", encoding="utf-8") as jsonl:
            for idx, p in enumerate(tqdm(imgs, desc=f"{rel}", leave=False)):
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None: 
                    continue
                img = resize_short_side(img, args.short_side)

                gray, s_mean, v_mean, v_std = brightness_stats(img)
                lv   = lap_var(gray)
                ent  = entropy_gray(gray)
                ed   = sobel_edge_density(gray)
                cdk, cbk = clip_ratios(gray)

                H, W = gray.shape
                rec = {
                    "idx": idx,
                    "file": Path(p).name,
                    "H": int(H), "W": int(W),
                    "lap_var": lv,
                    "entropy": ent,
                    "edge_density": ed,
                    "sat_mean": s_mean,
                    "val_mean": v_mean,
                    "val_std": v_std,
                    "clip_dark": cdk,
                    "clip_bright": cbk
                }
                jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

                all_lap.append(lv); all_vmean.append(v_mean); all_entropy.append(ent)

    def quant(arr, q):
        if not arr: return None
        return float(np.quantile(np.array(arr, dtype=np.float64), q))

    stats = {
        "lap_var_p50": quant(all_lap, 0.5),
        "lap_var_p25": quant(all_lap, 0.25),
        "lap_var_p75": quant(all_lap, 0.75),
        "val_mean_p50": quant(all_vmean, 0.5),
        "entropy_p50": quant(all_entropy, 0.5),
        "count_frames": int(len(all_lap))
    }
    (out_root / "global_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print("✅ 完成：离线逐帧指标已生成。")

if __name__ == "__main__":
    main()
