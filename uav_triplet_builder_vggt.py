#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uav_triplet_builder_vggt.py — 三帧三元组构建（VGGT；置信度加权几何 + 目标视差奖励 + 基于投影的 SSIM+L1 光度一致性）

要点：
  - 以【t,p,n】顺序送入 VGGT，拿到 (K_t,E_t),(K_p,E_p),(K_n,E_n) 与 D_t（可含 conf）
  - 以 t 为目标，用 D_t 与 (K/E) 将源帧重投影到 t，得到重建图 I_s->t
  - 光度损失使用 Monodepth2 风格：L = α*(1-SSIM)/2 + (1-α)*L1
  - 仅在“可用视差”掩码（z>0，边界内，位移∈[eps,delta]）内统计，并可用 conf_t 做权重
  - 单侧分 = 几何基线 * 目标视差奖励 * exp(-λ_photo * L_photo)
  - 为每个 t 选综合分最高的 (p*,t,n*)

用法示例：
  python uav_triplet_builder_vggt.py \
    --image_root   /mnt/data/.../dataset/Train \
    --metrics_root /mnt/data/.../uav_metrics \
    --out_root     /mnt/data/.../uav_triplets \
    --window 6
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


# ---------------- 基础工具 ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _normalize_conf_map(conf: np.ndarray) -> np.ndarray:
    """把置信度归一到 [0,1]（若已在范围内则保持；否则按1~99分位做min-max）。"""
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
            s = line.strip()
            if s:
                rows.append(json.loads(s))
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
    R = E[:, :3]; t = E[:, 3:4]
    C = -np.linalg.inv(R) @ t
    return C[:, 0]


# ---------------- 几何：反投/投影（稀疏用于 O/U，稠密用于光度） ----------------
def unproject_depth_to_world(depth_hw: np.ndarray, E: np.ndarray, K: np.ndarray,
                             sample_stride: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    稀疏反投：从 t 帧深度反投到世界坐标。返回：world_pts(N,3), pix_uv(N,2 int)
    """
    depth = _as_hw_depth(depth_hw)
    H, W = depth.shape
    ys = np.arange(0, H, sample_stride, dtype=np.intp)
    xs = np.arange(0, W, sample_stride, dtype=np.intp)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    vs_i = grid_y.reshape(-1).astype(np.intp)
    us_i = grid_x.reshape(-1).astype(np.intp)

    d = depth[vs_i, us_i].astype(np.float64)
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.intp)

    Kinv = np.linalg.inv(K.astype(np.float64))
    pix = np.stack([us_i[valid].astype(np.float64),
                    vs_i[valid].astype(np.float64),
                    np.ones(valid.sum(), dtype=np.float64)], axis=-1)  # (N,3)
    rays = (Kinv @ pix.T).T
    Xc = rays * d[valid][:, None]

    R = E[:, :3].astype(np.float64); t = E[:, 3:4].astype(np.float64)
    Rinv = np.linalg.inv(R)
    Xw = (Rinv @ Xc.T).T - (Rinv @ t).T

    pix_uv = np.stack([us_i[valid], vs_i[valid]], axis=-1)
    return Xw, pix_uv

def project_world_to_image(world_pts: np.ndarray, E: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """投影世界点到图像，返回 uv(float)、以及相机坐标深度 z。"""
    R = E[:, :3]; t = E[:, 3:4]
    Pc = (R @ world_pts.T + t).T  # (N,3)
    z = Pc[:, 2:3]
    z_safe = np.where(np.abs(z) < 1e-12, 1e-12, z)
    uvw = (K @ Pc.T).T
    u = uvw[:, 0] / z_safe[:, 0]
    v = uvw[:, 1] / z_safe[:, 0]
    return np.stack([u, v], axis=-1), z[:, 0]

def warp_src_to_t_via_depth(depth_t: np.ndarray,
                            K_t: np.ndarray, E_t: np.ndarray,
                            K_s: np.ndarray, E_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    D = _as_hw_depth(depth_t).astype(np.float64)
    H, W = D.shape
    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="xy")  # (H,W)

    # t 像素 -> t 相机坐标
    Kt_inv = np.linalg.inv(K_t.astype(np.float64))
    ones = np.ones_like(uu)
    pix = np.stack([uu, vv, ones], axis=-1).reshape(-1, 3)           # (N,3)
    rays = (Kt_inv @ pix.T).T                                         # (N,3)
    d = D.reshape(-1, 1)                                              # (N,1)
    Xc_t = rays * d                                                   # (N,3)

    # t 相机 -> 世界
    R_t = E_t[:, :3].astype(np.float64); t_t = E_t[:, 3:4].astype(np.float64)
    R_t_inv = np.linalg.inv(R_t)
    Xw = (R_t_inv @ Xc_t.T).T - (R_t_inv @ t_t).T                     # (N,3)

    # 世界 -> 源相机
    R_s = E_s[:, :3].astype(np.float64); t_s = E_s[:, 3:4].astype(np.float64)
    Xc_s = (R_s @ Xw.T + t_s).T                                       # (N,3)

    # ！！！这里改为 2:3（而不是 2:1）
    z_s = Xc_s[:, 2:3]                                                # (N,1)
    z_safe = np.where(np.abs(z_s) < 1e-12, 1e-12, z_s)

    uvw = (K_s.astype(np.float64) @ Xc_s.T).T                         # (N,3)
    u_s = (uvw[:, 0:1] / z_safe).reshape(H, W).astype(np.float32)
    v_s = (uvw[:, 1:2] / z_safe).reshape(H, W).astype(np.float32)
    z_s_map = z_s.reshape(H, W).astype(np.float32)

    return u_s, v_s, z_s_map



# ---------------- 置信度加权的几何指标（稀疏） ----------------
def compute_geom_metrics(depth_t: np.ndarray, E_t: np.ndarray, K_t: np.ndarray,
                         E_src: np.ndarray, K_src: np.ndarray,
                         conf_t: Optional[np.ndarray] = None,
                         eps_px: float = 1.0, delta_px: float = 40.0,
                         sample_stride: int = 4) -> Optional[dict]:
    """
    返回加权几何指标：O（重叠率）、U（可用视差率）、f（前向占比）、r（相对旋转弧度）
    """
    depth_t = _as_hw_depth(depth_t)
    H, W = depth_t.shape
    world_pts, pix_uv = unproject_depth_to_world(depth_t, E_t, K_t, sample_stride)
    if world_pts.shape[0] == 0:
        return None

    uv_src, z_src = project_world_to_image(world_pts, E_src, K_src)
    in_bounds = (uv_src[:, 0] >= 0) & (uv_src[:, 0] < W) & (uv_src[:, 1] >= 0) & (uv_src[:, 1] < H) & (z_src > 0)
    if in_bounds.sum() == 0:
        return None

    # 置信度权重
    if conf_t is not None:
        conf = _as_hw_depth(conf_t)
        w_all = conf[pix_uv[:, 1], pix_uv[:, 0]].astype(np.float64)
    else:
        w_all = np.ones(pix_uv.shape[0], dtype=np.float64)

    disp = np.linalg.norm(uv_src - pix_uv.astype(np.float64), axis=1)
    usable = (disp >= eps_px) & (disp <= delta_px) & in_bounds

    denom_all = max(float(w_all.sum()), 1e-12)
    O_w = float(np.sum(w_all * in_bounds) / denom_all)

    denom_ib = max(float(np.sum(w_all * in_bounds)), 1e-12)
    U_w = float(np.sum(w_all * usable) / denom_ib)

    # 前向/旋转
    C_t = camera_center_from_extrinsic(E_t)
    C_s = camera_center_from_extrinsic(E_src)
    d_world = C_s - C_t
    d_norm = np.linalg.norm(d_world) + 1e-12
    fwd = forward_vector_world(E_t)
    f = abs(float(np.dot(d_world, fwd))) / d_norm

    R_t = E_t[:, :3]; R_s = E_src[:, :3]
    R_rel = R_s @ np.linalg.inv(R_t)
    r = rotation_angle_from_R(R_rel)

    return dict(O=float(O_w), U=float(U_w), f=float(f), r=float(r))


# ---------------- SSIM + L1 光度一致性（投影重建后计算） ----------------
def ssim_map_rgb(I0: np.ndarray, I1: np.ndarray, ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    基于高斯平滑的近似 SSIM（对 3 通道分别计算后取均值）。
    输入 I0,I1: (H,W,3) float32 in [0,1]
    返回：ssim_map (H,W) float32 in [-1,1]
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    def _gauss(x): return cv2.GaussianBlur(x, (ksize, ksize), sigma)

    ssim_ch = []
    for c in range(3):
        x = I0[:, :, c]
        y = I1[:, :, c]
        mu_x = _gauss(x); mu_y = _gauss(y)
        mu_x2 = mu_x * mu_x; mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = _gauss(x * x) - mu_x2
        sigma_y2 = _gauss(y * y) - mu_y2
        sigma_xy = _gauss(x * y) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim = np.where(den > 0, num / den, 1.0)
        ssim_ch.append(ssim.astype(np.float32))
    return np.mean(np.stack(ssim_ch, axis=0), axis=0)

def photometric_loss_dense(img_t_bgr: np.ndarray, img_s_bgr: np.ndarray,
                           depth_t: np.ndarray,
                           K_t: np.ndarray, E_t: np.ndarray,
                           K_s: np.ndarray, E_s: np.ndarray,
                           conf_t: Optional[np.ndarray],
                           eps_px: float, delta_px: float,
                           alpha_ssim: float = 0.85) -> Tuple[Optional[float], Optional[float]]:
    """
    基于坐标变换与投影的 SSIM+L1 光度一致性：
      1) 用 D_t 与 (K/E) 计算源帧到 t 的重映射 (map_x,map_y)
      2) 用 cv2.remap 把源帧重建到 t：I_s->t
      3) 在有效几何掩码上，计算 photo = α*(1-SSIM)/2 + (1-α)*L1（置信度加权平均）
    返回：(L_photo, disp_med)。若无有效像素，返回 (None, None)。
    """
    D = _as_hw_depth(depth_t)
    H, W = D.shape

    # 读入并缩放到 (W,H)，归一到 [0,1] 且转 RGB 顺序无关（对称计算）
    It = cv2.resize(img_t_bgr, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    Is = cv2.resize(img_s_bgr, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    # BGR -> RGB（可选；对 SSIM/L1 无本质影响）
    It = It[:, :, ::-1]
    Is = Is[:, :, ::-1]

    # 计算重映射坐标
    map_x, map_y, z_s = warp_src_to_t_via_depth(D, K_t, E_t, K_s, E_s)

    # 几何掩码：在前方 + 边界内
    in_bounds = (map_x >= 0) & (map_x < W - 1) & (map_y >= 0) & (map_y < H - 1) & (z_s > 0)

    # 视差（像素位移）
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
    disp = np.sqrt((map_x - uu) ** 2 + (map_y - vv) ** 2)

    usable = in_bounds & (disp >= eps_px) & (disp <= delta_px)
    if usable.sum() == 0:
        return None, None

    # 重建图：源 -> t
    Iw = cv2.remap(Is, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # L1（对 3 通道取均值）
    l1 = np.mean(np.abs(It - Iw), axis=2).astype(np.float32)

    # SSIM（对 3 通道分别计算再均值）
    ssim = ssim_map_rgb(It, Iw, ksize=3, sigma=1.0)
    photo = alpha_ssim * (1.0 - ssim) * 0.5 + (1.0 - alpha_ssim) * l1

    # 置信度权重
    if conf_t is not None:
        conf = _normalize_conf_map(_as_hw_depth(conf_t))
    else:
        conf = np.ones((H, W), dtype=np.float32)

    w = conf * usable.astype(np.float32)
    w_sum = float(w.sum())
    if w_sum <= 1e-12:
        return None, None

    L_photo = float(np.sum(w * photo) / w_sum)
    disp_med = float(np.median(disp[usable]))

    return L_photo, disp_med


# ---------------- 评分 ----------------
def pair_score(metrics: dict, Q_pair: float,
               gamma_O=1.5, gamma_U=2.0, w_Q=1.0,
               lam_f=2.0, lam_r=1.0,
               beta_disp=0.12, target_disp_px=8.0,
               lambda_photo=3.0, L_photo: Optional[float] = None) -> float:
    """
    单侧分 = 几何基线 * 目标视差奖励 * 光度一致性惩罚
    - 几何基线： O^γ_O * U^γ_U * Q * exp(-λ_f f) * exp(-λ_r r)
    - 目标视差奖励：以 disp_med 靠近 target_disp_px 为佳（高斯窗）
    - 光度一致性：exp(-λ_photo * L_photo)，L_photo 为空时不惩罚
    """
    O, U, f, r = metrics["O"], metrics["U"], metrics["f"], metrics["r"]
    base = (O ** gamma_O) * (U ** gamma_U) * (Q_pair ** w_Q) * np.exp(-lam_f * f) * np.exp(-lam_r * r)

    # 目标视差奖励
    if target_disp_px > 0 and np.isfinite(target_disp_px):
        # 由调用方传入 disp_med 时再乘奖励（这里留在外层以减少重复计算）
        pass

    # 光度一致性惩罚
    if (L_photo is not None) and np.isfinite(L_photo):
        base *= float(np.exp(-lambda_photo * L_photo))

    return float(base)


# ---------------- VGGT 前向 ----------------
def vggt_forward_triplet(model: VGGT, device: str, dtype: torch.dtype, paths: List[str]):
    images = load_and_preprocess_images(paths).to(device)
    amp = torch.cuda.amp.autocast(dtype=dtype) if device == "cuda" else nullcontext()
    with torch.no_grad():
        with amp:
            preds = model(images)
    E, K = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
    E = E[0].detach().cpu().numpy()
    K = K[0].detach().cpu().numpy()
    D = preds["depth"][0].detach().cpu().numpy()  # (3,H,W) 或 (3,1,H,W)
    if D.ndim == 4 and D.shape[1] == 1:
        D = D[:, 0]
    D_t = _as_hw_depth(D[0])
    H, W = D_t.shape

    conf_t = None
    if "depth_conf" in preds:
        C = preds["depth_conf"][0].detach().cpu().numpy()
        if C.ndim == 4 and C.shape[1] == 1:
            C = C[:, 0]
        conf_t = _normalize_conf_map(_as_hw_depth(C[0]).astype(np.float32))

    return E, K, D_t, conf_t, H, W


# ---------------- 主评估：单个三元组 ----------------
def evaluate_triplet(frames: list, image_paths: List[str], t: int, p: int, n: int,
                     model: VGGT, device: str, dtype: torch.dtype, args,
                     Q_frame: List[float], seq_rel: Path, geom_rejects_path: Path):
    paths = [image_paths[t], image_paths[p], image_paths[n]]
    E, K, D_t, conf_t, H, W = vggt_forward_triplet(model, device, dtype, paths)

    # 分辨率自适应的位移上限
    delta_eff = max(args.delta_px, args.delta_px_frac * float(min(H, W)))
    eps_eff   = float(args.eps_px)

    # 加权几何指标（稀疏）
    m_prev = compute_geom_metrics(D_t, E[0], K[0], E[1], K[1], conf_t,
                                  eps_px=eps_eff, delta_px=delta_eff, sample_stride=args.sample_stride)
    m_next = compute_geom_metrics(D_t, E[0], K[0], E[2], K[2], conf_t,
                                  eps_px=eps_eff, delta_px=delta_eff, sample_stride=args.sample_stride)
    if (m_prev is None) or (m_next is None):
        _append_jsonl(geom_rejects_path, {
            "seq": str(seq_rel),
            "center": {"idx": int(t), "file": frames[t]["file"]},
            "prev":   {"idx": int(p), "file": frames[p]["file"]},
            "next":   {"idx": int(n), "file": frames[n]["file"]},
            "reason": "metrics_none",
            "which":  "prev_none" if (m_prev is None) else "next_none"
        })
        return None

    # 硬阈
    def basic(md: dict):
        return {"O": float(md["O"]), "U": float(md["U"]), "f": float(md["f"]), "r": float(md["r"])}
    m_prev_basic, m_next_basic = basic(m_prev), basic(m_next)

    fail_flags = []
    if m_prev_basic["r"] > math.radians(args.rmax_deg): fail_flags.append("prev_r")
    if m_prev_basic["f"] > args.fmax:                   fail_flags.append("prev_f")
    if m_prev_basic["O"] < args.Omin:                   fail_flags.append("prev_O")
    if m_prev_basic["U"] < args.Umin:                   fail_flags.append("prev_U")
    if m_next_basic["r"] > math.radians(args.rmax_deg): fail_flags.append("next_r")
    if m_next_basic["f"] > args.fmax:                   fail_flags.append("next_f")
    if m_next_basic["O"] < args.Omin:                   fail_flags.append("next_O")
    if m_next_basic["U"] < args.Umin:                   fail_flags.append("next_U")
    if fail_flags:
        _append_jsonl(geom_rejects_path, {
            "seq": str(seq_rel),
            "center": {"idx": int(t), "file": frames[t]["file"]},
            "prev":   {"idx": int(p), "file": frames[p]["file"], "metrics": m_prev_basic},
            "next":   {"idx": int(n), "file": frames[n]["file"], "metrics": m_next_basic},
            "reason": "hard_threshold",
            "fail_flags": fail_flags,
            "thresholds": {
                "rmax_deg": args.rmax_deg, "fmax": args.fmax,
                "Omin": args.Omin, "Umin": args.Umin
            }
        })
        return None

    # 读取原图
    img_t = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    img_p = cv2.imread(paths[1], cv2.IMREAD_COLOR)
    img_n = cv2.imread(paths[2], cv2.IMREAD_COLOR)
    if (img_t is None) or (img_p is None) or (img_n is None):
        _append_jsonl(geom_rejects_path, {
            "seq": str(seq_rel),
            "center": {"idx": int(t), "file": frames[t]["file"]},
            "reason": "image_read_fail"
        })
        return None

    # 基于投影的 SSIM+L1 光度项（源->t）
    Lp, disp_med_p = photometric_loss_dense(
        img_t, img_p, D_t, K[0], E[0], K[1], E[1], conf_t,
        eps_px=eps_eff, delta_px=delta_eff, alpha_ssim=args.photo_alpha_ssIM
    )
    Ln, disp_med_n = photometric_loss_dense(
        img_t, img_n, D_t, K[0], E[0], K[2], E[2], conf_t,
        eps_px=eps_eff, delta_px=delta_eff, alpha_ssim=args.photo_alpha_ssIM
    )

    if (Lp is None) or (Ln is None):
        _append_jsonl(geom_rejects_path, {
            "seq": str(seq_rel),
            "center": {"idx": int(t), "file": frames[t]["file"]},
            "prev": {"idx": int(p), "file": frames[p]["file"]},
            "next": {"idx": int(n), "file": frames[n]["file"]},
            "reason": "photo_none"
        })
        return None

    # 单侧打分
    Qp = min(Q_frame[t], Q_frame[p])
    Qn = min(Q_frame[t], Q_frame[n])

    target_disp_px = (args.target_disp_px
                      if args.target_disp_px > 0
                      else args.target_disp_factor * float(min(H, W)))

    sp = pair_score(m_prev_basic, Qp,
                    gamma_O=args.gamma_O, gamma_U=args.gamma_U, w_Q=args.w_Q,
                    lam_f=args.lam_f, lam_r=args.lam_r,
                    beta_disp=args.beta_disp_reward, target_disp_px=target_disp_px,
                    lambda_photo=args.lambda_photo, L_photo=Lp)
    sn = pair_score(m_next_basic, Qn,
                    gamma_O=args.gamma_O, gamma_U=args.gamma_U, w_Q=args.w_Q,
                    lam_f=args.lam_f, lam_r=args.lam_r,
                    beta_disp=args.beta_disp_reward, target_disp_px=target_disp_px,
                    lambda_photo=args.lambda_photo, L_photo=Ln)

    # 将“目标视差奖励”乘到两侧（使用 disp_med）
    if target_disp_px > 0:
        sigma = 0.35 * target_disp_px
        if disp_med_p is not None:
            sp *= (1.0 + args.beta_disp_reward *
                   float(np.exp(-0.5 * ((disp_med_p - target_disp_px) / max(sigma, 1e-6)) ** 2)))
        if disp_med_n is not None:
            sn *= (1.0 + args.beta_disp_reward *
                   float(np.exp(-0.5 * ((disp_med_n - target_disp_px) / max(sigma, 1e-6)) ** 2)))

    s_trip = float(np.sqrt(max(sp, 1e-12) * max(sn, 1e-12)))

    # T 源→目标
    Gt, Gprev, Gnext = to44(E[0]), to44(E[1]), to44(E[2])
    T_prev_to_t = (Gt @ np.linalg.inv(Gprev)).astype(np.float32)
    T_next_to_t = (Gt @ np.linalg.inv(Gnext)).astype(np.float32)

    # 诊断统计
    prev_diag = {
        "idx": int(p), "file": frames[p]["file"], "dt": int(p - t),
        "metrics": m_prev_basic, "Q_pair": float(Qp), "score_side": float(sp),
        "disp_med": (float(disp_med_p) if disp_med_p is not None else None),
        "L_photo": float(Lp)
    }
    next_diag = {
        "idx": int(n), "file": frames[n]["file"], "dt": int(n - t),
        "metrics": m_next_basic, "Q_pair": float(Qn), "score_side": float(sn),
        "disp_med": (float(disp_med_n) if disp_med_n is not None else None),
        "L_photo": float(Ln)
    }

    return {
        "center": {"idx": int(t), "file": frames[t]["file"]},
        "prev": prev_diag,
        "next": next_diag,
        "score_triplet": float(s_trip),
        "K_t":   K[0].astype(np.float32),
        "K_prev":K[1].astype(np.float32),
        "K_next":K[2].astype(np.float32),
        "T_prev_to_t": T_prev_to_t,
        "T_next_to_t": T_next_to_t,
        "H": int(H), "W": int(W),
        "depth_t": D_t.astype(np.float32),
        "depth_conf_t": (conf_t.astype(np.float32) if conf_t is not None else None)
    }


# ---------------- 保存最佳三元组 ----------------
def save_best_triplet(out_dir: Path, seq_rel: Path, best: dict):
    npy_dir = out_dir / "npy"; ensure_dir(npy_dir)
    center_stem = Path(best["center"]["file"]).stem

    depth_path = npy_dir / f"{center_stem}_depth.npy"
    np.save(depth_path, best.pop("depth_t"))

    conf_dir = out_dir / "conf"; ensure_dir(conf_dir)
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


# ---------------- 主流程 ----------------
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
        _append_jsonl(geom_rejects_path, {"seq": str(seq_rel), "summary": "no_valid_triplet_in_sequence"})

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
    # 输入与输出
    ap.add_argument("--image_root",   type=str, required=False,
                    default="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany_tri/All",
                    help="原始图像根目录（与阶段一一致）")
    ap.add_argument("--metrics_root", type=str, required=False,
                    default="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany_tri/tri",
                    help="阶段一输出根目录（包含 <seq>/frames.jsonl 与 global_stats.json）")
    ap.add_argument("--out_root",     type=str, required=False,
                    default="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany_tri/tri_new",
                    help="输出根目录（每序列一个 triplets.jsonl；另含 frame/geom 拒绝记录）")
    # 滑窗
    ap.add_argument("--window",       type=int, default=6,
                    help="候选前后帧窗口半径 k（全程三帧将做 k×k 笛卡尔积）")
    # 稀疏采样（几何指标）
    ap.add_argument("--sample_stride", type=int, default=4,
                    help="几何 O/U 评估的稀疏采样步长（像素）")
    # 逐帧质量过滤阈值
    ap.add_argument("--min_lap_p",    type=float, default=0.10, help="拉普拉斯方差低于本序列分位数则过滤（0~1）")
    ap.add_argument("--max_clip",     type=float, default=0.10, help="欠/过曝比例任何一项超过则过滤")
    ap.add_argument("--min_val",      type=float, default=20.0, help="V 均值过低过滤 [0,255]")
    # 几何阈值
    ap.add_argument("--eps_px",       type=float, default=2.0,  help="可用视差下限（像素）")
    ap.add_argument("--delta_px",     type=float, default=80.0, help="可用视差上限（像素）下限值")
    ap.add_argument("--delta_px_frac",type=float, default=0.03, help="自适应上限：max(delta_px, frac*min(H,W))")
    ap.add_argument("--rmax_deg",     type=float, default=30.0, help="旋转上限（度）")
    ap.add_argument("--fmax",         type=float, default=0.85, help="前向占比上限")
    ap.add_argument("--Omin",         type=float, default=0.50, help="重叠率下限（单侧）")
    ap.add_argument("--Umin",         type=float, default=0.10, help="可用视差率下限（单侧）")
    # 打分权重
    ap.add_argument("--gamma_O",      type=float, default=1.5, help="O 幂")
    ap.add_argument("--gamma_U",      type=float, default=2.0, help="U 幂")
    ap.add_argument("--w_Q",          type=float, default=1.0, help="Q 幂（画质权重）")
    ap.add_argument("--lam_f",        type=float, default=2.0, help="前向占比软惩罚系数")
    ap.add_argument("--lam_r",        type=float, default=1.0, help="旋转软惩罚系数")
    # 目标视差奖励 & 光度一致性
    ap.add_argument("--target_disp_px",     type=float, default=0.0,  help="目标视差（像素）；<=0 时按 factor 推断")
    ap.add_argument("--target_disp_factor", type=float, default=0.02, help="目标视差系数（0.02*min(H,W)）")
    ap.add_argument("--beta_disp_reward",   type=float, default=0.12, help="目标视差奖励强度（乘性）")
    ap.add_argument("--lambda_photo",       type=float, default=3.0,  help="光度一致性惩罚权重")
    ap.add_argument("--photo_alpha_ssIM",   type=float, default=0.85, help="光度项中 SSIM 的权重 α")
    return ap

def main():
    args = build_argparser().parse_args()
    run_all_sequences(args)

if __name__ == "__main__":
    main()
