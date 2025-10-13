#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三帧三元组构建（VGGT）— 极简重写版：仅使用「画质(Q) × 可用视差率(U) × 学习有效视差奖励(R_target)」打分

去除项：
  - 不计算/不使用：光度项(SSIM/L1)、重叠率O、前向占比f、相对旋转r、任何幂指数/软惩罚；
  - 不使用 pair_score，不设 rmax/lam 等阈值；
  - 不读取图像像素，仅用几何与深度。

核心：
  - 单侧分：score_side = Q_norm(Q_pair) * U * R_target(d_med; d_star)
      Q_pair = min(Q_t, Q_s)，Q_t = clip(lap_var_t/lap_p50, 0.5, 1.5)，Q_norm: [0.5,1.5]→[0,1]
      U      = 在 in-bounds 且 z>0 的像素中，满足 eps<=|disp|<=delta 的加权比例（权=conf_t，若提供）
      d_med  = 可用掩码内视差中位数；  d_star = target_disp_px 或 target_disp_factor*min(H,W)
      R_target(d_med,d_star) = exp(-0.5*((d_med-d_star)/(0.35*d_star))^2)（若 d_star<=0 或无 d_med 则 0）
  - 两侧各自独立选最优（prev_best, next_best）。
  - 三元组分（日志/横向比较用）：sqrt(score_prev*score_next)。
  - 若任一侧非相邻(±1)，可选地与相邻基线(t-1,t+1)比较，写 triplets_nonadj_summary.jsonl。

输入：
  - <seq>/frames.jsonl（含 file, lap_var, clip_dark, clip_bright, val_mean ...）
  - metrics_root/global_stats.json（lap_var_p50）
产物：
  - <seq>/triplets.jsonl（每个中心帧一条最佳三元组记录）
  - <seq>/npy/<center>_depth.npy 与（可选）conf/<center>_depthconf.npy
  - <seq>/frame_rejects.jsonl, <seq>/geom_rejects.jsonl
  - <seq>/triplets_nonadj_summary.jsonl（当最优非相邻时，含与相邻基线对比）
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from contextlib import nullcontext

# ===== VGGT =====
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# ---------------- 基础工具 ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _append_jsonl(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_frames_jsonl(path: Path) -> list:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _as_hw_depth(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth)
    if d.ndim == 2:
        return d
    if d.ndim == 3:
        if d.shape[0] == 1:
            return d[0]
        if d.shape[-1] == 1:
            return d[..., 0]
    raise ValueError(f"Expect depth (H,W)/(1,H,W)/(H,W,1), got {d.shape}")


def _normalize_conf_map(conf: np.ndarray) -> np.ndarray:
    c = np.asarray(conf, np.float32)
    if (c.min() >= 0.0) and (c.max() <= 1.0):
        return c
    lo, hi = np.percentile(c, 1), np.percentile(c, 99)
    c = (c - lo) / max(hi - lo, 1e-6)
    return np.clip(c, 0.0, 1.0).astype(np.float32)


def to44(E34: np.ndarray) -> np.ndarray:
    G = np.eye(4, dtype=np.float64)
    G[:3, :4] = E34
    return G


# ---------- 日志辅助 ----------
import re as _re


def get_numeric_idx(file_or_name: str) -> Optional[int]:
    stem = Path(file_or_name).stem
    m = _re.search(r"(\d+)$", stem) or _re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else None


def idx_for_log(frames: list, i: int) -> int:
    v = get_numeric_idx(frames[i]["file"])
    return i if v is None else v


def frame_ref(frames: list, i: int) -> dict:
    return {"idx": idx_for_log(frames, i), "file": frames[i]["file"]}


# ---------------- 几何：反投/投影/重映射 ----------------
def warp_src_to_t_via_depth(depth_t: np.ndarray,
                            K_t: np.ndarray, E_t: np.ndarray,
                            K_s: np.ndarray, E_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用中心帧深度 D_t 将源帧 s 重投影到 t，返回源图采样坐标与源相机深度。
    返回：map_x(H,W), map_y(H,W), z_s_map(H,W)
    """
    D = _as_hw_depth(depth_t).astype(np.float64)
    H, W = D.shape

    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="xy")

    Kt_inv = np.linalg.inv(K_t.astype(np.float64))
    pix = np.stack([uu, vv, np.ones_like(uu)], axis=-1).reshape(-1, 3)
    rays = (Kt_inv @ pix.T).T
    Xc_t = rays * D.reshape(-1, 1)

    R_t = E_t[:, :3].astype(np.float64);
    t_t = E_t[:, 3:4].astype(np.float64)
    R_t_inv = np.linalg.inv(R_t)
    Xw = (R_t_inv @ Xc_t.T).T - (R_t_inv @ t_t).T

    R_s = E_s[:, :3].astype(np.float64);
    t_s = E_s[:, 3:4].astype(np.float64)
    Xc_s = (R_s @ Xw.T + t_s).T

    z_s = Xc_s[:, 2:3]
    z_safe = np.where(np.abs(z_s) < 1e-12, 1e-12, z_s)

    uvw = (K_s.astype(np.float64) @ Xc_s.T).T
    u_s = (uvw[:, 0:1] / z_safe).reshape(H, W).astype(np.float32)
    v_s = (uvw[:, 1:2] / z_safe).reshape(H, W).astype(np.float32)
    z_s_map = z_s.reshape(H, W).astype(np.float32)

    return u_s, v_s, z_s_map


def rot_only_map_uv(K_t: np.ndarray, E_t: np.ndarray,
                    K_s: np.ndarray, E_s: np.ndarray,
                    H: int, W: int,
                    precomp: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    计算“仅旋转”的像素映射：把 t 帧的每个像素方向（不依赖深度）
    通过 R_rel = R_s @ R_t^{-1} 旋转到 s 相机，再用 K_s 投影到 s 的像素平面。
    返回：u_rot(H,W), v_rot(H,W)，以及可复用的 precomp（网格与 Kt_inv）
    """
    # 提取旋转（E: world->cam, x_c = R X_w + t）
    R_t = E_t[:, :3].astype(np.float64)
    R_s = E_s[:, :3].astype(np.float64)

    # 预计算像素网格和 K_t^{-1}
    if precomp is None:
        u = np.arange(W, dtype=np.float64)
        v = np.arange(H, dtype=np.float64)
        uu, vv = np.meshgrid(u, v, indexing="xy")
        pix = np.stack([uu, vv, np.ones_like(uu)], axis=-1).reshape(-1, 3)  # (HW,3)
        Kt_inv = np.linalg.inv(K_t.astype(np.float64))
        precomp = {"uu": uu, "vv": vv, "pix": pix, "Kt_inv": Kt_inv}
    else:
        uu, vv, pix, Kt_inv = precomp["uu"], precomp["vv"], precomp["pix"], precomp["Kt_inv"]

    # 方向向量（中心相机坐标，未归一深度）
    rays_t = (Kt_inv @ precomp["pix"].T).T  # (HW,3)

    # 仅旋转：rays_s = R_rel @ rays_t
    R_rel = (R_s @ np.linalg.inv(R_t)).astype(np.float64)
    rays_s = (R_rel @ rays_t.T).T  # (HW,3)

    # 投影到源相机像素坐标（不考虑平移）
    uvw = (K_s.astype(np.float64) @ rays_s.T).T
    z = np.where(np.abs(uvw[:, 2]) < 1e-12, 1e-12, uvw[:, 2])
    u_rot = (uvw[:, 0] / z).reshape(H, W).astype(np.float32)
    v_rot = (uvw[:, 1] / z).reshape(H, W).astype(np.float32)

    return u_rot, v_rot, precomp


def compute_U_and_disp_median(depth_t: np.ndarray,
                              K_t: np.ndarray, E_t: np.ndarray,
                              K_s: np.ndarray, E_s: np.ndarray,
                              conf_t: Optional[np.ndarray],
                              eps_px: float, delta_px: float, precomp=None) -> Tuple[Optional[float], Optional[float]]:
    """使用『去旋转视差(parallax)』：
       - 先用中心深度把像素从 t 映射到 s，得 (map_x, map_y)、z_s；
       - 再计算纯旋转映射 (u_rot, v_rot)（与深度无关）；
       - parallax = || [map_x,map_y] - [u_rot,v_rot] ||_2 仅保留平移诱导的位移；
       - U：在 in-bounds & z>0 且 eps<=parallax<=delta 的像素比例（可按 conf_t 加权）；
       - disp_med：上述可用区域的 parallax 中位数。
    """
    D = _as_hw_depth(depth_t).astype(np.float64)
    H, W = D.shape
    N = float(H * W)

    # --- 含平移的重映射（已有逻辑） ---
    map_x, map_y, z_s = warp_src_to_t_via_depth(D, K_t, E_t, K_s, E_s)

    # --- 仅旋转映射（新） ---
    u_rot, v_rot, _ = rot_only_map_uv(K_t, E_t, K_s, E_s, H, W, precomp=precomp)

    # --- translation-only parallax ---
    parallax = np.sqrt((map_x - u_rot) ** 2 + (map_y - v_rot) ** 2).astype(np.float32)

    # 可见性：在源图像范围内且到源相机的深度为正
    in_bounds = (map_x >= 0) & (map_x < W - 1) & (map_y >= 0) & (map_y < H - 1) & (z_s > 0)

    # 有效视差带
    usable = in_bounds & (parallax >= float(eps_px)) & (parallax <= float(delta_px))

    if usable.sum() == 0:
        return None, None

    # 计算 U（整幅归一化；有 conf 则加权）
    if conf_t is not None:
        conf = _as_hw_depth(conf_t).astype(np.float32)
        # 若原图不是 [0,1]，做一次稳健线性拉伸
        if not ((conf.min() >= 0.0) and (conf.max() <= 1.0)):
            lo, hi = np.percentile(conf, 1), np.percentile(conf, 99)
            if hi - lo > 1e-6:
                conf = np.clip((conf - lo) / (hi - lo), 0.0, 1.0)
            else:
                conf = None  # 退化，回退为均匀权
        if conf is not None:
            denom_full = float(conf.sum())
            if denom_full <= 1e-12:
                U = float(usable.mean())  # 退化回退
            else:
                U = float((conf * usable.astype(np.float32)).sum() / denom_full)
        else:
            U = float(usable.mean())
    else:
        U = float(usable.sum() / N)

    # 可用区域的中位“去旋转视差”
    disp_med = float(np.median(parallax[usable]))
    return U, disp_med


# ---------------- 打分（Q × U × R_target） ----------------
def compute_Q_frame(frames: list, lap_p50: float) -> List[float]:
    Q = []
    for rec in frames:
        q = rec["lap_var"] / max(lap_p50, 1e-6)
        q = float(np.clip(q, 0.5, 1.5))
        Q.append(q)
    return Q


def Q_pair_norm(Q_t: float, Q_s: float) -> float:
    q_pair = min(Q_t, Q_s)  # 短板优先
    q01 = (q_pair - 0.5) / 1.0  # [0.5,1.5] → [0,1]
    return float(np.clip(q01, 0.0, 1.0))


def R_target(d_med: Optional[float], d_star: float, kappa: float = 0.35) -> float:
    if (d_med is None) or (d_star <= 0):
        return 0.0
    sigma = max(kappa * d_star, 1e-6)
    return float(np.exp(-0.5 * ((d_med - d_star) / sigma) ** 2))  # ∈(0,1]


# ---------------- 保存 ----------------
def save_best_triplet(out_dir: Path, seq_rel: Path, best: dict):
    npy_dir = out_dir / "npy";
    ensure_dir(npy_dir)
    center_stem = Path(best["center"]["file"]).stem

    depth_path = npy_dir / f"{center_stem}_depth.npy"
    np.save(depth_path, best.pop("depth_t"))

    conf_dir = out_dir / "conf";
    ensure_dir(conf_dir)
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
        "K_t": best["K_t"].tolist(),
        "K_prev": best["K_prev"].tolist(),
        "K_next": best["K_next"].tolist(),
        "T_prev_to_t": best["T_prev_to_t"].tolist(),
        "T_next_to_t": best["T_next_to_t"].tolist(),
        "H": best["H"], "W": best["W"],
        "depth_t_npy": str(depth_path),
        "depth_conf_t_npy": (str(conf_path) if conf_path is not None else None)
    }
    _append_jsonl(out_dir / "triplets.jsonl", rec)


# ---------------- 质量过滤与候选收集 ----------------
def compute_seq_lap_thresh(frames: list, min_lap_p: float) -> float:
    seq_laps = np.array([r["lap_var"] for r in frames], dtype=np.float64)
    return float(np.quantile(seq_laps, min_lap_p)) if len(seq_laps) else 0.0


def frame_bad(rec: dict, lap_thresh: float, max_clip: float, min_val: float) -> bool:
    return (rec["lap_var"] < lap_thresh) or (rec["clip_dark"] > max_clip) or \
        (rec["clip_bright"] > max_clip) or (rec["val_mean"] < min_val)


def collect_candidates(frames: list, t: int, window: int, lap_thresh: float, args,
                       frame_rejects_path: Path, seq_rel: Path) -> Tuple[List[int], List[int]]:
    prev_cands, next_cands = [], []
    for dt in range(1, window + 1):
        tp = t - dt
        if tp >= 0:
            rp = frames[tp]
            if frame_bad(rp, lap_thresh, args.max_clip, args.min_val):
                _append_jsonl(frame_rejects_path, {
                    "seq": str(seq_rel), "type": "prev_cand",
                    "center_idx": idx_for_log(frames, t),
                    "idx": idx_for_log(frames, tp), "file": rp["file"],
                    "reasons": {
                        "lap_below_quantile": bool(rp["lap_var"] < lap_thresh),
                        "clip_dark_gt_max": bool(rp["clip_dark"] > args.max_clip),
                        "clip_bright_gt_max": bool(rp["clip_bright"] > args.max_clip),
                        "val_mean_lt_min": bool(rp["val_mean"] < args.min_val)
                    }
                })
            else:
                prev_cands.append(tp)
        tn = t + dt
        if tn < len(frames):
            rn = frames[tn]
            if frame_bad(rn, lap_thresh, args.max_clip, args.min_val):
                _append_jsonl(frame_rejects_path, {
                    "seq": str(seq_rel), "type": "next_cand",
                    "center_idx": idx_for_log(frames, t),
                    "idx": idx_for_log(frames, tn), "file": rn["file"],
                    "reasons": {
                        "lap_below_quantile": bool(rn["lap_var"] < lap_thresh),
                        "clip_dark_gt_max": bool(rn["clip_dark"] > args.max_clip),
                        "clip_bright_gt_max": bool(rn["clip_bright"] > args.max_clip),
                        "val_mean_lt_min": bool(rn["val_mean"] < args.min_val)
                    }
                })
            else:
                next_cands.append(tn)
    return prev_cands, next_cands


# ---------------- 前向与评估 ----------------
def build_window_order(t: int, N: int, window: int) -> List[int]:
    prevs = [i for i in range(t - 1, max(-1, t - window - 1), -1) if 0 <= i < N]
    nexts = [i for i in range(t + 1, min(N, t + window + 1)) if 0 <= i < N]
    return [t] + prevs + nexts


def vggt_forward_window(model: VGGT, device: str, dtype: torch.dtype,
                        paths: List[str], center_pos: int = 0):
    images = load_and_preprocess_images(paths).to(device)
    amp = torch.cuda.amp.autocast(dtype=dtype) if device == "cuda" else nullcontext()
    with torch.no_grad():
        with amp:
            preds = model(images)

    E_all, K_all = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
    E_all = E_all[0].detach().cpu().numpy()
    K_all = K_all[0].detach().cpu().numpy()

    D = preds["depth"][0].detach().cpu().numpy()
    if D.ndim == 4 and D.shape[1] == 1:
        D = D[:, 0]
    D_t = _as_hw_depth(D[center_pos])
    H, W = D_t.shape

    conf_t = None
    if "depth_conf" in preds:
        C = preds["depth_conf"][0].detach().cpu().numpy()
        if C.ndim == 4 and C.shape[1] == 1:
            C = C[:, 0]
        conf_t = _normalize_conf_map(_as_hw_depth(C[center_pos]).astype(np.float32))

    return E_all, K_all, D_t, conf_t, H, W


def evaluate_side(frames: list,
                  t: int, s: int,
                  pos_of: dict,
                  E_all: np.ndarray, K_all: np.ndarray,
                  D_t: np.ndarray, conf_t: Optional[np.ndarray],
                  H: int, W: int,
                  Q_frame: List[float],
                  args, seq_rel: Path,
                  geom_rejects_path: Path,
                  side_label: str, precomp=None) -> Optional[dict]:
    ct = pos_of[t]
    cs = pos_of.get(s, None)
    if cs is None:
        _append_jsonl(geom_rejects_path, {
            "seq": str(seq_rel),
            "center": frame_ref(frames, t),
            side_label: frame_ref(frames, s),
            "reason": "index_not_in_window_pack"
        })
        return None

    delta_eff = max(args.delta_px, args.delta_px_frac * float(min(H, W)))
    eps_eff = float(args.eps_px)

    U, d_med = compute_U_and_disp_median(D_t, K_all[ct], E_all[ct], K_all[cs], E_all[cs], conf_t,
                                         eps_px=eps_eff, delta_px=delta_eff, precomp=precomp)
    if (U is None):
        _append_jsonl(geom_rejects_path, {
            "seq": str(seq_rel),
            "center": frame_ref(frames, t),
            side_label: frame_ref(frames, s),
            "reason": "U_none_or_no_usable_pixels"
        })
        return None

    # 目标视差 d_star
    d_star = (args.target_disp_px if args.target_disp_px > 0
              else args.target_disp_factor * float(min(H, W)))

    Qn = Q_pair_norm(Q_frame[t], Q_frame[s])
    R = R_target(d_med, d_star)
    score = float(Qn * float(np.clip(U, 0.0, 1.0)) * R)

    diag = {
        "idx": idx_for_log(frames, s), "file": frames[s]["file"],
        "dt": int(idx_for_log(frames, s) - idx_for_log(frames, t)),
        "U": float(U), "disp_med": (float(d_med) if d_med is not None else None),
        "Q_pair": float(min(Q_frame[t], Q_frame[s])),
        "Q_norm": float(Qn),
        "R_target": float(R),
        "score_side": float(score)
    }
    return {"diag": diag, "score": float(score)}


# ---------------- 主流程 ----------------
def clear_sequence_outputs(out_dir: Path):
    ensure_dir(out_dir)
    for name in ["triplets.jsonl", "frame_rejects.jsonl", "geom_rejects.jsonl", "triplets_nonadj_summary.jsonl"]:
        p = out_dir / name
        try:
            p.unlink()
        except FileNotFoundError:
            pass


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
    geom_rejects_path = out_dir / "geom_rejects.jsonl"
    summary_path = out_dir / "triplets_nonadj_summary.jsonl"

    lap_thresh = compute_seq_lap_thresh(frames, args.min_lap_p)
    Q_frame = compute_Q_frame(frames, lap_p50)

    any_valid = False
    for t in tqdm(range(N), desc=str(seq_rel), leave=False):
        rt = frames[t]
        if frame_bad(rt, lap_thresh, args.max_clip, args.min_val):
            _append_jsonl(frame_rejects_path, {
                "seq": str(seq_rel), "type": "center",
                "idx": idx_for_log(frames, t), "file": frames[t]["file"],
                "reasons": {
                    "lap_below_quantile": bool(rt["lap_var"] < lap_thresh),
                    "clip_dark_gt_max": bool(rt["clip_dark"] > args.max_clip),
                    "clip_bright_gt_max": bool(rt["clip_bright"] > args.max_clip),
                    "val_mean_lt_min": bool(rt["val_mean"] < args.min_val)
                },
                "metrics": {k: rt[k] for k in
                            ["lap_var", "entropy", "edge_density", "val_mean", "val_std", "clip_dark", "clip_bright"]}
            })
            continue

        prev_cands, next_cands = collect_candidates(frames, t, args.window, lap_thresh, args,
                                                    frame_rejects_path, seq_rel)
        if (not prev_cands) or (not next_cands):
            _append_jsonl(geom_rejects_path, {
                "seq": str(seq_rel), "center": frame_ref(frames, t),
                "reason": "no_candidates_on_one_side"
            })
            continue

        order = build_window_order(t, N, args.window)
        paths = [image_paths[i] for i in order]
        pos_of = {idx: k for k, idx in enumerate(order)}
        E_all, K_all, D_t, conf_t, H, W = vggt_forward_window(model, device, dtype, paths, center_pos=0)

        # === 每个中心帧仅构建一次 precomp（复用于所有候选） ===
        ct = pos_of[t]
        u_g = np.arange(W, dtype=np.float64)
        v_g = np.arange(H, dtype=np.float64)
        uu, vv = np.meshgrid(u_g, v_g, indexing="xy")
        pix = np.stack([uu, vv, np.ones_like(uu)], axis=-1).reshape(-1, 3)
        precomp_rot = {
            "uu": uu,
            "vv": vv,
            "pix": pix,
            "Kt_inv": np.linalg.inv(K_all[ct].astype(np.float64)),
        }

        # === prev 侧：仅从通过门槛的候选里挑分数最高者 ===
        best_prev, best_prev_idx = None, None
        for p in prev_cands:
            if p not in pos_of:
                continue
            res_p = evaluate_side(frames, t, p, pos_of, E_all, K_all, D_t, conf_t, H, W,
                                  Q_frame, args, seq_rel, geom_rejects_path, "prev",
                                  precomp=precomp_rot)
            if res_p is None:
                continue
            if not side_pass(res_p["diag"], args):
                _append_jsonl(geom_rejects_path, {
                    "seq": str(seq_rel), "center": frame_ref(frames, t),
                    "prev": frame_ref(frames, p), "reason": "side_gate_fail_prev",
                    "gate": {"min_U": args.min_U, "min_R": args.min_R,
                             "min_Qn": args.min_Qn, "min_side_score": args.min_side_score},
                    "diag": res_p["diag"]
                })
                continue
            if (best_prev is None) or (res_p["score"] > best_prev["score"]):
                best_prev, best_prev_idx = res_p, p

        # === next 侧：仅从通过门槛的候选里挑分数最高者 ===
        best_next, best_next_idx = None, None
        for n in next_cands:
            if n not in pos_of:
                continue
            res_n = evaluate_side(frames, t, n, pos_of, E_all, K_all, D_t, conf_t, H, W,
                                  Q_frame, args, seq_rel, geom_rejects_path, "next",
                                  precomp=precomp_rot)
            if res_n is None:
                continue
            if not side_pass(res_n["diag"], args):
                _append_jsonl(geom_rejects_path, {
                    "seq": str(seq_rel), "center": frame_ref(frames, t),
                    "next": frame_ref(frames, n), "reason": "side_gate_fail_next",
                    "gate": {"min_U": args.min_U, "min_R": args.min_R,
                             "min_Qn": args.min_Qn, "min_side_score": args.min_side_score},
                    "diag": res_n["diag"]
                })
                continue
            if (best_next is None) or (res_n["score"] > best_next["score"]):
                best_next, best_next_idx = res_n, n

        # === 若任一侧无通过门槛的候选，直接拒绝该中心帧 ===
        if (best_prev is None) or (best_next is None):
            _append_jsonl(geom_rejects_path, {
                "seq": str(seq_rel), "center": frame_ref(frames, t),
                "reason": "no_valid_side_after_gates",
                "detail": {"prev_ok": best_prev is not None, "next_ok": best_next is not None}
            })
            continue

        # 齐次变换与三元组分
        ct, cp, cn = pos_of[t], pos_of[best_prev_idx], pos_of[best_next_idx]
        Gt, Gprev, Gnext = to44(E_all[ct]), to44(E_all[cp]), to44(E_all[cn])
        T_prev_to_t = (Gt @ np.linalg.inv(Gprev)).astype(np.float32)
        T_next_to_t = (Gt @ np.linalg.inv(Gnext)).astype(np.float32)

        sp, sn = best_prev["score"], best_next["score"]
        s_trip = float(np.sqrt(max(sp, 1e-12) * max(sn, 1e-12)))

        # 非相邻比较（可选摘要）：仅按单一偏移 ±k（由 --adjacent_offset 控制）
        k = int(args.adjacent_offset)
        is_nonadj = (best_prev_idx != max(0, t - k)) or (best_next_idx != min(N - 1, t + k))
        if is_nonadj:
            cand_adj_prev = None
            cand_adj_next = None

            p_adj = t - k
            n_adj = t + k

            if (0 <= p_adj < N) and (p_adj in pos_of):
                cand_adj_prev = evaluate_side(
                    frames, t, p_adj, pos_of, E_all, K_all, D_t, conf_t, H, W,
                    Q_frame, args, seq_rel, geom_rejects_path, "prev",
                    precomp=precomp_rot
                )

            if (0 <= n_adj < N) and (n_adj in pos_of):
                cand_adj_next = evaluate_side(
                    frames, t, n_adj, pos_of, E_all, K_all, D_t, conf_t, H, W,
                    Q_frame, args, seq_rel, geom_rejects_path, "next",
                    precomp=precomp_rot
                )

            if (cand_adj_prev is not None) and (cand_adj_next is not None):
                s_adj = float(np.sqrt(
                    max(cand_adj_prev["score"], 1e-12) * max(cand_adj_next["score"], 1e-12)
                ))
                summary_obj = {
                    "seq": str(seq_rel),
                    "center": frame_ref(frames, t),
                    "best": {
                        "prev": best_prev["diag"],
                        "next": best_next["diag"],
                        "score_triplet": float(s_trip)
                    },
                    "adjacent_offset": k,
                    "adjacent_baseline": {
                        "prev": cand_adj_prev["diag"],
                        "next": cand_adj_next["diag"],
                        "score_triplet": float(s_adj)
                    },
                    "better_than_adj": bool(s_trip > s_adj)
                }
            else:
                summary_obj = {
                    "seq": str(seq_rel),
                    "center": frame_ref(frames, t),
                    "best": {
                        "prev": best_prev["diag"],
                        "next": best_next["diag"],
                        "score_triplet": float(s_trip)
                    },
                    "adjacent_offset": k,
                    "adjacent_baseline": None,
                    "better_than_adj": None
                }

            _append_jsonl(summary_path, summary_obj)

        # 保存
        rec = {
            "center": {"idx": idx_for_log(frames, t), "file": frames[t]["file"]},
            "prev": best_prev["diag"],
            "next": best_next["diag"],
            "score_triplet": s_trip,
            "K_t": K_all[ct].astype(np.float32),
            "K_prev": K_all[cp].astype(np.float32),
            "K_next": K_all[cn].astype(np.float32),
            "T_prev_to_t": T_prev_to_t,
            "T_next_to_t": T_next_to_t,
            "H": int(H), "W": int(W),
            "depth_t": D_t.astype(np.float32),
            "depth_conf_t": (conf_t.astype(np.float32) if conf_t is not None else None)
        }
        save_best_triplet(out_dir, seq_rel, rec)
        any_valid = True

    if not any_valid:
        _append_jsonl(geom_rejects_path, {"seq": str(seq_rel), "summary": "no_valid_triplet_in_sequence"})


# ---------------- 批量入口 ----------------
def load_global_stats(metrics_root: Path) -> float:
    gstats_path = metrics_root / "global_stats.json"
    if not gstats_path.exists():
        raise FileNotFoundError(f"缺少 {gstats_path}，请先运行阶段一（逐帧指标统计）")
    gstats = json.loads(gstats_path.read_text())
    return float(gstats.get("lap_var_p50", 1.0))


def run_all_sequences(args):
    image_root = Path(args.image_root)
    metrics_root = Path(args.metrics_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    lap_p50 = load_global_stats(metrics_root)
    device, dtype, model = setup_device_and_model()

    seq_jsonls = sorted(metrics_root.rglob("frames.jsonl"))
    for jpath in tqdm(seq_jsonls, desc="构建三元组"):
        seq_rel = jpath.parent.relative_to(metrics_root)
        process_sequence(seq_rel, jpath, image_root, metrics_root, out_root,
                         model, device, dtype, args, lap_p50)


def setup_device_and_model() -> Tuple[str, torch.dtype, VGGT]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (
                torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    return device, dtype, model


def side_pass(diag: dict, args) -> bool:
    """单侧候选是否通过硬门槛。diag 为 evaluate_side 返回的 diag 字段。"""
    try:
        return (float(diag.get("U", 0.0)) >= float(args.min_U) and
                float(diag.get("R_target", 0.0)) >= float(args.min_R) and
                float(diag.get("Q_norm", 0.0)) >= float(args.min_Qn) and
                float(diag.get("score_side", 0.0)) >= float(args.min_side_score))
    except Exception:
        return False


# ---------------- 参数 ----------------
def build_argparser():
    ap = argparse.ArgumentParser()
    # 输入与输出
    ap.add_argument("--image_root", type=str, required=False,
                    #default="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany_tri/All",
                    default="/mnt/data_nvme3n1p1/dataset/UAV_ula/tri_images",
                    help="原始图像根目录（与阶段一一致）")
    ap.add_argument("--metrics_root", type=str, required=False,
                    #default="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany_tri/tri_win10_new",
                    default="/mnt/data_nvme3n1p1/dataset/UAV_ula/tri_metric",
                    help="阶段一输出根目录（包含 <seq>/frames.jsonl 与 global_stats.json）")
    ap.add_argument("--out_root", type=str, required=False,
                    #default="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany_tri/tri_win10_new",
                    default="/mnt/data_nvme3n1p1/dataset/UAV_ula/tri_win10_0.06_lap0.05",
                    help="输出根目录（每序列一个 triplets.jsonl；另含 frame/geom 拒绝记录）")

    # 滑窗（候选范围）
    ap.add_argument("--window", type=int, default=10,
                    help="候选前后帧窗口半径")

    # 稀疏几何评估采样（用于 U 的稳定性/速度折衷）
    ap.add_argument("--sample_stride", type=int, default=4,
                    help="（保留，仅用于速度。若使用 compute_U_and_disp_median 的稠密实现，可忽略）")

    # 逐帧质量硬阈（过滤极差样本）
    ap.add_argument("--min_lap_p", type=float, default=0.05,
                    help="拉普拉斯方差低于本序列分位数则过滤（0~1）")
    ap.add_argument("--max_clip", type=float, default=0.10,
                    help="欠/过曝比例任何一项超过则过滤")
    ap.add_argument("--min_val", type=float, default=20.0,
                    help="V 均值过低过滤 [0,255]")

    # 视差带（U 与 d_med 的定义域）
    ap.add_argument("--eps_px", type=float, default=2.0,
                    help="可用视差下限（像素）")
    ap.add_argument("--delta_px", type=float, default=80.0,
                    help="可用视差上限（像素）")
    ap.add_argument("--delta_px_frac", type=float, default=0.05,
                    help="自适应上限：max(delta_px, frac*min(H,W))")

    # 目标视差（训练有效性）
    ap.add_argument("--target_disp_px", type=float, default=0.0,
                    help="目标视差（像素）；<=0 时按 factor 推断")
    ap.add_argument("--target_disp_factor", type=float, default=0.06,
                    help="目标视差系数（0.02*min(H,W)）")

    # === 新增：邻接偏移（单一整数输入） ===
    ap.add_argument("--adjacent_offset", type=int, default=1,
                    help="定义邻接基线的偏移 k（如 1 或 5），仅比较 ±k 基线与最佳方案")

    # === 侧级硬门槛（single-side gates） ===
    ap.add_argument("--min_U", type=float, default=0.07,
                    help="单侧覆盖率 U 的硬阈")
    ap.add_argument("--min_R", type=float, default=0.1,
                    help="单侧 R_target 的硬阈")
    ap.add_argument("--min_Qn", type=float, default=0.15,
                    help="单侧 Q_norm 的硬阈")
    ap.add_argument("--min_side_score", type=float, default=0.010,
                    help="单侧 score_side 的硬阈")

    return ap


# ---------------- 主入口 ----------------
def main():
    args = build_argparser().parse_args()
    run_all_sequences(args)


if __name__ == "__main__":
    main()