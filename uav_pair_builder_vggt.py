#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uav_triplet_builder_vggt.py  —  全程三帧（三元组构建，无 short_side）

功能：
  - 读取阶段一输出：<metrics_root>/<seq>/frames.jsonl 与 <metrics_root>/global_stats.json
  - 对每个中心帧 t，在窗口 [t-k, t+k] 内分别取“过去侧/未来侧”候选，先做逐帧质量过滤
  - 对每个候选组合 (prev, t, next) 以【t, prev, next】顺序送入 VGGT 一次性推理（保留原始分辨率）
  - 用中帧深度与三帧 K/E 计算两侧几何指标（O/U/f/r），叠加画质分 Q，得到两侧分与三元组综合分
  - 为每个 t 仅保留综合分最高的一个 (prev*, t, next*)，写入 triplets.jsonl（一行一三元组）
  - 额外：仅记录“两类拒绝样本”到 jsonl：
        1) frame_rejects.jsonl：逐帧质检淘汰的帧（中心/候选）
        2) geom_rejects.jsonl：几何评估淘汰的三元组候选，及“无有效三元组”的中心帧

输出（每行）字段要点：
  - 文件名与相对路径
  - K_t/K_prev/K_next；T_prev_to_t 与 T_next_to_t（源→目标）
  - 侧向 metrics 与分数、综合分 score_triplet
  - H/W（推理分辨率），保存中帧深度 depth_t.npy

用法示例：
  python uav_triplet_builder_vggt.py \
    --image_root   /mnt/data/.../dataset/Train \
    --metrics_root /mnt/data/.../uav_metrics \
    --out_root     /mnt/data/.../uav_triplets \
    --window 2
"""
import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from contextlib import nullcontext

# ===== VGGT =====
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


# ---------------- 基础工具 ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _normalize_conf_map(conf: np.ndarray) -> np.ndarray:
    """把置信度大致归一到 [0,1]。
    - 若已在 [0,1] 则原样返回
    - 否则做 1~99 百分位的 min-max，避免极端值影响
    """
    c = np.asarray(conf, np.float32)
    if (c.min() >= 0.0) and (c.max() <= 1.0):
        return c
    lo, hi = np.percentile(c, 1), np.percentile(c, 99)
    c = (c - lo) / max(hi - lo, 1e-6)
    return np.clip(c, 0.0, 1.0).astype(np.float32)


def _append_jsonl(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_frames_jsonl(path: Path) -> list:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _as_hw_depth(depth: np.ndarray) -> np.ndarray:
    """统一深度到 (H,W)；兼容 (H,W)/(1,H,W)/(H,W,1)。"""
    d = np.asarray(depth)
    if d.ndim == 2:
        return d
    if d.ndim == 3:
        if d.shape[0] == 1:     # (1,H,W)->(H,W)
            return d[0]
        if d.shape[-1] == 1:    # (H,W,1)->(H,W)
            return d[..., 0]
    raise ValueError(f"Expect depth (H,W)/(1,H,W)/(H,W,1), got {d.shape}")


def to44(E34: np.ndarray) -> np.ndarray:
    G = np.eye(4, dtype=np.float64)
    G[:3, :4] = E34
    return G


def rotation_angle_from_R(R: np.ndarray) -> float:
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(tr))


def forward_vector_world(E: np.ndarray) -> np.ndarray:
    # E = [R|t] (world->cam)
    R = E[:, :3]
    z_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z_world = np.linalg.inv(R) @ z_cam
    n = np.linalg.norm(z_world) + 1e-12
    return z_world / n


def camera_center_from_extrinsic(E: np.ndarray) -> np.ndarray:
    R = E[:, :3]
    t = E[:, 3:4]
    C = -np.linalg.inv(R) @ t
    return C[:, 0]


# ---------------- 几何：反投/投影与指标 ----------------
def unproject_depth_to_world(depth_hw: np.ndarray, E: np.ndarray, K: np.ndarray,
                             sample_stride: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 t 帧深度反投到世界坐标（稀疏采样）。
    返回：world_pts(N,3), pix_uv(N,2), valid(N,)
    """
    depth = _as_hw_depth(depth_hw)
    H, W = depth.shape

    ys = np.arange(0, H, sample_stride, dtype=np.intp)
    xs = np.arange(0, W, sample_stride, dtype=np.intp)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    vs_i = grid_y.reshape(-1).astype(np.intp)   # int 索引
    us_i = grid_x.reshape(-1).astype(np.intp)

    d = depth[vs_i, us_i].astype(np.float64)
    valid = np.isfinite(d) & (d > 0)

    Kinv = np.linalg.inv(K.astype(np.float64))
    ones = np.ones_like(us_i, dtype=np.float64)
    pix = np.stack([us_i.astype(np.float64), vs_i.astype(np.float64), ones], axis=-1)  # (N,3)
    rays = (Kinv @ pix.T).T
    Xc = rays * d[:, None]

    R = E[:, :3].astype(np.float64)
    t = E[:, 3:4].astype(np.float64)
    Rinv = np.linalg.inv(R)
    Xw = (Rinv @ Xc.T).T - (Rinv @ t).T

    return Xw[valid], np.stack([us_i, vs_i], axis=-1)[valid], valid


def project_world_to_image(world_pts: np.ndarray, E: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = E[:, :3]
    t = E[:, 3:4]
    Pc = (R @ world_pts.T + t).T
    z = Pc[:, 2:3]
    z_safe = np.where(np.abs(z) < 1e-12, 1e-12, z)
    uvw = (K @ Pc.T).T
    u = uvw[:, 0] / z_safe[:, 0]
    v = uvw[:, 1] / z_safe[:, 0]
    return np.stack([u, v], axis=-1), z[:, 0]


def compute_geom_metrics(depth_t: np.ndarray, E_t: np.ndarray, K_t: np.ndarray,
                         E_src: np.ndarray, K_src: np.ndarray,
                         eps_px=1.0, delta_px=40.0, sample_stride=4) -> Optional[dict]:
    """以 t 为目标，评估某个源帧的几何可用性（O/U/f/r）。"""
    depth_t = _as_hw_depth(depth_t)
    world_pts, pix_uv, _ = unproject_depth_to_world(depth_t, E_t, K_t, sample_stride)
    if world_pts.shape[0] == 0:
        return None

    uv_src, z_src = project_world_to_image(world_pts, E_src, K_src)
    H, W = depth_t.shape
    in_bounds = (uv_src[:, 0] >= 0) & (uv_src[:, 0] < W) & (uv_src[:, 1] >= 0) & (uv_src[:, 1] < H) & (z_src > 0)
    if in_bounds.sum() == 0:
        return None

    O = in_bounds.sum() / float(world_pts.shape[0])

    disp = np.linalg.norm(uv_src - pix_uv, axis=1)
    usable = (disp >= eps_px) & (disp <= delta_px) & in_bounds
    denom = max(in_bounds.sum(), 1)
    U = usable.sum() / float(denom)

    C_t = camera_center_from_extrinsic(E_t)
    C_s = camera_center_from_extrinsic(E_src)
    d_world = C_s - C_t
    d_norm = np.linalg.norm(d_world) + 1e-12
    fwd = forward_vector_world(E_t)
    f = abs(float(np.dot(d_world, fwd))) / d_norm

    R_t = E_t[:, :3]; R_s = E_src[:, :3]
    R_rel = R_s @ np.linalg.inv(R_t)
    r = rotation_angle_from_R(R_rel)

    return dict(O=float(O), U=float(U), f=float(f), r=float(r))


# ---------------- 评分 ----------------
def pair_score(metrics: dict, Q_pair: float,
               gamma_O=2.0, gamma_U=3.0, w_Q=1.0,
               lam_f=2.0, lam_r=1.0) -> float:
    O, U, f, r = metrics["O"], metrics["U"], metrics["f"], metrics["r"]
    return float((O ** gamma_O) * (U ** gamma_U) * (Q_pair ** w_Q) * np.exp(-lam_f * f) * np.exp(-lam_r * r))


# ---------------- 辅助：初始化与通用子流程 ----------------
def load_global_stats(metrics_root: Path) -> float:
    gstats_path = metrics_root / "global_stats.json"
    if not gstats_path.exists():
        raise FileNotFoundError(f"缺少 {gstats_path}，请先运行阶段一（逐帧指标统计）")
    gstats = json.loads(gstats_path.read_text())
    return float(gstats.get("lap_var_p50", 1.0))


def setup_device_and_model() -> Tuple[str, torch.dtype, VGGT]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    return device, dtype, model


def clear_sequence_outputs(out_dir: Path):
    ensure_dir(out_dir)
    for name in ["triplets.jsonl", "frame_rejects.jsonl", "geom_rejects.jsonl"]:
        p = out_dir / name
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def compute_seq_lap_thresh(frames: list, min_lap_p: float) -> float:
    seq_laps = np.array([r["lap_var"] for r in frames], dtype=np.float64)
    return float(np.quantile(seq_laps, min_lap_p)) if len(seq_laps) else 0.0


def compute_Q_frame(frames: list, lap_p50: float) -> List[float]:
    Q = []
    for rec in frames:
        q = rec["lap_var"] / max(lap_p50, 1e-6)
        q = float(np.clip(q, 0.5, 1.5))
        Q.append(q)
    return Q


def frame_bad(rec: dict, lap_thresh: float, max_clip: float, min_val: float) -> bool:
    return (rec["lap_var"] < lap_thresh) or (rec["clip_dark"] > max_clip) or \
           (rec["clip_bright"] > max_clip) or (rec["val_mean"] < min_val)


def log_frame_reject(frame_rejects_path: Path, seq_rel: Path, kind: str, idx: int, rec: dict, args):
    _append_jsonl(frame_rejects_path, {
        "seq": str(seq_rel), "type": kind, "idx": int(idx), "file": rec["file"],
        "reasons": {
            "lap_below_quantile": bool(rec["lap_var"] < 0),  # 具体阈值在调用处展开
            "clip_dark_gt_max":    bool(rec["clip_dark"] > args.max_clip),
            "clip_bright_gt_max":  bool(rec["clip_bright"] > args.max_clip),
            "val_mean_lt_min":     bool(rec["val_mean"] < args.min_val)
        },
        "metrics": {k: rec[k] for k in ["lap_var","entropy","edge_density","val_mean","val_std","clip_dark","clip_bright"]}
    })


def collect_candidates(frames: list, t: int, window: int, lap_thresh: float, args,
                       frame_rejects_path: Path, seq_rel: Path) -> Tuple[List[int], List[int]]:
    prev_cands, next_cands = [], []
    for dt in range(1, window + 1):
        tp = t - dt
        if tp >= 0:
            rp = frames[tp]
            if frame_bad(rp, lap_thresh, args.max_clip, args.min_val):
                _append_jsonl(frame_rejects_path, {
                    "seq": str(seq_rel), "type": "prev_cand", "center_idx": int(t),
                    "idx": int(tp), "file": rp["file"],
                    "reasons": {
                        "lap_below_quantile": bool(rp["lap_var"] < lap_thresh),
                        "clip_dark_gt_max":    bool(rp["clip_dark"] > args.max_clip),
                        "clip_bright_gt_max":  bool(rp["clip_bright"] > args.max_clip),
                        "val_mean_lt_min":     bool(rp["val_mean"] < args.min_val)
                    }
                })
            else:
                prev_cands.append(tp)
        tn = t + dt
        if tn < len(frames):
            rn = frames[tn]
            if frame_bad(rn, lap_thresh, args.max_clip, args.min_val):
                _append_jsonl(frame_rejects_path, {
                    "seq": str(seq_rel), "type": "next_cand", "center_idx": int(t),
                    "idx": int(tn), "file": rn["file"],
                    "reasons": {
                        "lap_below_quantile": bool(rn["lap_var"] < lap_thresh),
                        "clip_dark_gt_max":    bool(rn["clip_dark"] > args.max_clip),
                        "clip_bright_gt_max":  bool(rn["clip_bright"] > args.max_clip),
                        "val_mean_lt_min":     bool(rn["val_mean"] < args.min_val)
                    }
                })
            else:
                next_cands.append(tn)
    return prev_cands, next_cands


def vggt_forward_triplet(model: VGGT, device: str, dtype: torch.dtype, paths: List[str]):
    images = load_and_preprocess_images(paths).to(device)
    amp = torch.cuda.amp.autocast(dtype=dtype) if device == "cuda" else nullcontext()
    with torch.no_grad():
        with amp:
            preds = model(images)
    E, K = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
    E = E[0].detach().cpu().numpy()
    K = K[0].detach().cpu().numpy()
    D = preds["depth"][0].detach().cpu().numpy()
    if D.ndim == 4 and D.shape[1] == 1:
        D = D[:, 0]
    D_t = _as_hw_depth(D[0])
    H, W = D_t.shape
    # 置信度（若存在；同样只取中心帧）
    conf_t = None
    if "depth_conf" in preds:
        C = preds["depth_conf"][0].detach().cpu().numpy()  # (3,H,W) 或 (3,1,H,W)
        if C.ndim == 4 and C.shape[1] == 1:
            C = C[:, 0]
        conf_t = _as_hw_depth(C[0]).astype(np.float32)
        conf_t = _normalize_conf_map(conf_t)

    return E, K, D_t, conf_t, H, W


def evaluate_triplet(frames: list, image_paths: List[str], t: int, p: int, n: int,
                     model: VGGT, device: str, dtype: torch.dtype, args,
                     Q_frame: List[float], seq_rel: Path, geom_rejects_path: Path):
    paths = [image_paths[t], image_paths[p], image_paths[n]]
    E, K, D_t, conf_t, H, W = vggt_forward_triplet(model, device, dtype, paths)

    metrics_prev = compute_geom_metrics(D_t, E[0], K[0], E[1], K[1],
                                        eps_px=args.eps_px, delta_px=args.delta_px, sample_stride=4)
    metrics_next = compute_geom_metrics(D_t, E[0], K[0], E[2], K[2],
                                        eps_px=args.eps_px, delta_px=args.delta_px, sample_stride=4)
    if (metrics_prev is None) or (metrics_next is None):
        _append_jsonl(geom_rejects_path, {
            "seq": str(seq_rel),
            "center": {"idx": int(t), "file": frames[t]["file"]},
            "prev":   {"idx": int(p), "file": frames[p]["file"]},
            "next":   {"idx": int(n), "file": frames[n]["file"]},
            "reason": "metrics_none",
            "which":  "prev_none" if (metrics_prev is None) else "next_none"
        })
        return None

    # 硬阈
    fail_flags = []
    if metrics_prev["r"] > math.radians(args.rmax_deg): fail_flags.append("prev_r")
    if metrics_prev["f"] > args.fmax:                   fail_flags.append("prev_f")
    if metrics_prev["O"] < args.Omin:                   fail_flags.append("prev_O")
    if metrics_prev["U"] < args.Umin:                   fail_flags.append("prev_U")
    if metrics_next["r"] > math.radians(args.rmax_deg): fail_flags.append("next_r")
    if metrics_next["f"] > args.fmax:                   fail_flags.append("next_f")
    if metrics_next["O"] < args.Omin:                   fail_flags.append("next_O")
    if metrics_next["U"] < args.Umin:                   fail_flags.append("next_U")

    if fail_flags:
        _append_jsonl(geom_rejects_path, {
            "seq": str(seq_rel),
            "center": {"idx": int(t), "file": frames[t]["file"]},
            "prev":   {"idx": int(p), "file": frames[p]["file"], "metrics": metrics_prev},
            "next":   {"idx": int(n), "file": frames[n]["file"], "metrics": metrics_next},
            "reason": "hard_threshold",
            "fail_flags": fail_flags,
            "thresholds": {
                "rmax_deg": args.rmax_deg, "fmax": args.fmax,
                "Omin": args.Omin, "Umin": args.Umin
            }
        })
        return None

    # 打分
    Qp = min(Q_frame[t], Q_frame[p])
    Qn = min(Q_frame[t], Q_frame[n])
    sp = pair_score(metrics_prev, Qp, args.gamma_O, args.gamma_U, args.w_Q, args.lam_f, args.lam_r)
    sn = pair_score(metrics_next, Qn, args.gamma_O, args.gamma_U, args.w_Q, args.lam_f, args.lam_r)
    s_trip = float(np.sqrt(max(sp, 1e-12) * max(sn, 1e-12)))

    # T 源→目标
    Gt    = to44(E[0]); Gprev = to44(E[1]); Gnext = to44(E[2])
    T_prev_to_t = (Gt @ np.linalg.inv(Gprev)).astype(np.float32)
    T_next_to_t = (Gt @ np.linalg.inv(Gnext)).astype(np.float32)

    return {
        "center": {"idx": int(t), "file": frames[t]["file"]},
        "prev": {
            "idx": int(p), "file": frames[p]["file"], "dt": int(p - t),
            "metrics": {k: float(v) for k, v in metrics_prev.items()},
            "Q_pair": float(Qp), "score_side": float(sp)
        },
        "next": {
            "idx": int(n), "file": frames[n]["file"], "dt": int(n - t),
            "metrics": {k: float(v) for k, v in metrics_next.items()},
            "Q_pair": float(Qn), "score_side": float(sn)
        },
        "score_triplet": float(s_trip),
        "K_t":   K[0].astype(np.float32),
        "K_prev":K[1].astype(np.float32),
        "K_next":K[2].astype(np.float32),
        "T_prev_to_t": T_prev_to_t,
        "T_next_to_t": T_next_to_t,
        "H": int(H), "W": int(W),
        "depth_t": D_t.astype(np.float32),
        "depth_conf_t": (conf_t.astype(np.float32) if conf_t is not None else None)  # ★ 新增
    }


def save_best_triplet(out_dir: Path, seq_rel: Path, best: dict):
    npy_dir = out_dir / "npy"; ensure_dir(npy_dir)
    center_stem = Path(best["center"]["file"]).stem
    depth_path = npy_dir / f"{center_stem}_depth.npy"
    np.save(depth_path, best.pop("depth_t"))

    conf_dir = out_dir / "conf";
    ensure_dir(conf_dir)
    # 深度置信度（中心帧）——改为保存到 conf/
    conf_path = None
    conf_arr = best.pop("depth_conf_t", None)
    if conf_arr is not None:
        conf_path = conf_dir / f"{center_stem}_depthconf.npy"
        np.save(conf_path, conf_arr)


    rec = {
        "seq": str(seq_rel),
        "center": best["center"],
        "prev": best["prev"],
        "next": best["next"],
        "score_triplet": best["score_triplet"],
        "K_t":   best["K_t"].tolist(),
        "K_prev":best["K_prev"].tolist(),
        "K_next":best["K_next"].tolist(),
        "T_prev_to_t": best["T_prev_to_t"].tolist(),
        "T_next_to_t": best["T_next_to_t"].tolist(),
        "H": best["H"], "W": best["W"],
        "depth_t_npy": str(depth_path),
        "depth_conf_t_npy": (str(conf_path) if conf_path is not None else None)
    }
    _append_jsonl(out_dir / "triplets.jsonl", rec)


# ---------------- 主流程（拆分为可复用函数） ----------------
def process_sequence(seq_rel: Path, jpath: Path, image_root: Path, metrics_root: Path,
                     out_root: Path, model: VGGT, device: str, dtype: torch.dtype,
                     args, lap_p50: float):
    img_dir = image_root / seq_rel
    if not img_dir.exists():
        print(f"[跳过] 找不到图像目录：{img_dir}")
        return

    frames = read_frames_jsonl(jpath)
    image_paths = [str(img_dir / rec["file"]) for rec in frames]
    N = len(image_paths)
    if N < 3:
        return

    out_dir = out_root / seq_rel
    clear_sequence_outputs(out_dir)

    frame_rejects_path = out_dir / "frame_rejects.jsonl"
    geom_rejects_path  = out_dir / "geom_rejects.jsonl"

    lap_thresh = compute_seq_lap_thresh(frames, args.min_lap_p)
    Q_frame    = compute_Q_frame(frames, lap_p50)

    any_valid = False
    for t in tqdm(range(N), desc=str(seq_rel), leave=False):
        rt = frames[t]
        if frame_bad(rt, lap_thresh, args.max_clip, args.min_val):
            _append_jsonl(frame_rejects_path, {
                "seq": str(seq_rel),
                "type": "center",
                "idx": int(t),
                "file": frames[t]["file"],
                "reasons": {
                    "lap_below_quantile": bool(rt["lap_var"] < lap_thresh),
                    "clip_dark_gt_max":    bool(rt["clip_dark"] > args.max_clip),
                    "clip_bright_gt_max":  bool(rt["clip_bright"] > args.max_clip),
                    "val_mean_lt_min":     bool(rt["val_mean"] < args.min_val)
                },
                "metrics": {k: rt[k] for k in ["lap_var","entropy","edge_density","val_mean","val_std","clip_dark","clip_bright"]}
            })
            continue

        prev_cands, next_cands = collect_candidates(frames, t, args.window, lap_thresh, args,
                                                    frame_rejects_path, seq_rel)
        if (not prev_cands) or (not next_cands):
            _append_jsonl(geom_rejects_path, {
                "seq": str(seq_rel),
                "center": {"idx": int(t), "file": frames[t]["file"]},
                "reason": "no_candidates_on_one_side"
            })
            continue

        best = None
        for p in prev_cands:
            for n in next_cands:
                cand = evaluate_triplet(frames, image_paths, t, p, n, model, device, dtype,
                                        args, Q_frame, seq_rel, geom_rejects_path)
                if cand is None:
                    continue
                if (best is None) or (cand["score_triplet"] > best["score_triplet"]):
                    best = cand

        if best is None:
            _append_jsonl(geom_rejects_path, {
                "seq": str(seq_rel),
                "center": {"idx": int(t), "file": frames[t]["file"]},
                "reason": "no_valid_triplet"
            })
            continue

        save_best_triplet(out_dir, seq_rel, best)
        any_valid = True

    if not any_valid:
        # 可选：在无任何有效三元组时记录汇总
        _append_jsonl(geom_rejects_path, {
            "seq": str(seq_rel), "summary": "no_valid_triplet_in_sequence"
        })


def run_all_sequences(args):
    image_root   = Path(args.image_root)
    metrics_root = Path(args.metrics_root)
    out_root     = Path(args.out_root)
    ensure_dir(out_root)

    lap_p50 = load_global_stats(metrics_root)
    device, dtype, model = setup_device_and_model()

    seq_jsonls = sorted(metrics_root.rglob("frames.jsonl"))
    for jpath in tqdm(seq_jsonls, desc="构建三元组"):
        seq_rel = jpath.parent.relative_to(metrics_root)
        process_sequence(seq_rel, jpath, image_root, metrics_root, out_root,
                         model, device, dtype, args, lap_p50)


# ---------------- 入口 ----------------
def build_argparser():
    ap = argparse.ArgumentParser()
    # 输入：图像与阶段一指标
    ap.add_argument("--image_root",   type=str, default="/mnt/data_nvme3n1p1/dataset/UAV_ula/dataset/Validation",
                    help="原始图像根目录（与阶段一一致）")
    ap.add_argument("--metrics_root", type=str, default="/mnt/data_nvme3n1p1/dataset/UAV_ula/uav_triplets",
                    help="阶段一输出根目录（包含 <seq>/frames.jsonl 与 global_stats.json）")
    # 输出
    ap.add_argument("--out_root",     type=str, default="/mnt/data_nvme3n1p1/dataset/UAV_ula/uav_triplets",
                    help="输出根目录（每序列一个 triplets.jsonl；另含 frame/geom 拒绝记录）")
    # 滑窗
    ap.add_argument("--window",       type=int, default=2, help="候选前后帧窗口半径 k（全程三帧将做 k×k 笛卡尔积）")
    # 逐帧质量过滤阈值
    ap.add_argument("--min_lap_p",    type=float, default=0.10, help="拉普拉斯方差低于本序列分位数则过滤（0~1）")
    ap.add_argument("--max_clip",     type=float, default=0.10, help="欠/过曝比例任何一项超过则过滤")
    ap.add_argument("--min_val",      type=float, default=20.0, help="V 均值过低过滤 [0,255]")
    # 几何/指标/阈值
    ap.add_argument("--eps_px",       type=float, default=0.5, help="可用视差下限（像素）")
    ap.add_argument("--delta_px",     type=float, default=40.0, help="可用视差上限（像素）")
    ap.add_argument("--rmax_deg",     type=float, default=30.0, help="旋转上限（度）")
    ap.add_argument("--fmax",         type=float, default=0.85, help="前向占比上限")
    ap.add_argument("--Omin",         type=float, default=0.50, help="重叠率下限（单侧）")
    ap.add_argument("--Umin",         type=float, default=0.10, help="可用视差率下限（单侧）")
    # 打分权重（单侧）
    ap.add_argument("--gamma_O",      type=float, default=2.0, help="O 幂")
    ap.add_argument("--gamma_U",      type=float, default=3.0, help="U 幂")
    ap.add_argument("--w_Q",          type=float, default=1.0, help="Q 幂（画质权重）")
    ap.add_argument("--lam_f",        type=float, default=2.0, help="前向占比软惩罚系数")
    ap.add_argument("--lam_r",        type=float, default=1.0, help="旋转软惩罚系数")
    return ap


def main():
    args = build_argparser().parse_args()
    run_all_sequences(args)


if __name__ == "__main__":
    main()

