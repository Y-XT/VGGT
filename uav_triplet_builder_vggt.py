#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uav_triplet_builder_vggt.py — 三帧三元组构建（VGGT；置信度加权几何 + 目标视差奖励 + 基于投影的 SSIM+L1 光度一致性）

核心思路（高层）：
  1) 以顺序 [t, p, n]（中心、过去侧、未来侧）将三帧输入 VGGT，一次性得到三帧的外参/内参 (E,K) 与中心帧深度 D_t（可带置信度 conf_t）。
  2) 以中心帧 t 为“目标视角”做度量：
     - 几何：用 D_t 反投为世界点，再投到源帧（p 或 n），得到重叠率 O、可用视差率 U、前向占比 f、相对旋转 r。
       （本实现可选用中心帧深度置信度对 O/U 做加权，增强鲁棒性）
     - 光度：把源帧通过 D_t、(K/E) 几何地重投影回目标 t，获得 I_s->t，与 I_t 计算 Monodepth2 风格的
             L_photo = α*(1-SSIM)/2 + (1-α)*L1，并在“可用视差”掩码内做（可选）置信度加权平均。
     - 目标视差奖励：鼓励中位视差（disp_med）接近预期 target_disp_px（或按分辨率自适应比例推断）。
  3) 单侧分 = 几何基线 * 目标视差奖励 * exp(-λ_photo * L_photo)；
     三元组分 = sqrt(单侧分_prev * 单侧分_next)（几何平均，鼓励两侧均衡）。
  4) 对每个中心帧 t，穷举窗口内 (p,n) 组合，保留分数最高的 (p*,t,n*)；将细节写入 triplets.jsonl；
     并记录各种拒绝原因（逐帧质量、几何阈值、光度无效、等）到对应 jsonl，便于调参与复盘。

输入/产物：
  - 输入：原始图像、阶段一逐帧统计（frames.jsonl）与全局统计（global_stats.json: lap_var_p50）；
  - 产物：<seq>/triplets.jsonl、npy/中心深度.npy、conf/中心深度置信度.npy（如有）、frame_rejects.jsonl、geom_rejects.jsonl。

注意的约定：
  - 外参 E 采用 world->cam 约定：X_cam = R * X_world + t；
  - K 为标准 pinhole 内参；
  - 所有度量默认在中心帧分辨率 (H,W) 上进行；delta 上限可随分辨率自适应；
  - 置信度（若模型提供）用于 O/U 加权与光度项像素加权，提升噪声/遮挡鲁棒性。
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
    """确保目录存在（递归创建）。"""
    p.mkdir(parents=True, exist_ok=True)


def _normalize_conf_map(conf: np.ndarray) -> np.ndarray:
    """把置信度归一到 [0,1]。
    - 若已在范围内则直接返回；
    - 否则用 1~99 分位做 min-max，抑制极端值影响。
    """
    c = np.asarray(conf, np.float32)
    if (c.min() >= 0.0) and (c.max() <= 1.0):
        return c
    lo, hi = np.percentile(c, 1), np.percentile(c, 99)
    c = (c - lo) / max(hi - lo, 1e-6)
    return np.clip(c, 0.0, 1.0).astype(np.float32)


def _append_jsonl(p: Path, obj: dict):
    """以 JSONL（逐行 JSON）方式附加写入。"""
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_frames_jsonl(path: Path) -> list:
    """读取阶段一逐帧统计（frames.jsonl）为 list[dict]。"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _as_hw_depth(depth: np.ndarray) -> np.ndarray:
    """统一深度为 (H,W)；兼容 (H,W)/(1,H,W)/(H,W,1)。"""
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
    """把 3×4 外参 [R|t] 嵌入到 4×4 齐次矩阵。"""
    G = np.eye(4, dtype=np.float64)
    G[:3, :4] = E34
    return G


def rotation_angle_from_R(R: np.ndarray) -> float:
    """由旋转矩阵取旋转角（弧度）。基于 trace 的稳定实现。"""
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(tr))


def forward_vector_world(E: np.ndarray) -> np.ndarray:
    """返回相机前向（z_cam）在世界系的单位向量。
    约定：E 为 world->cam，因此 z_world = R^{-1} * z_cam。
    """
    R = E[:, :3]
    z_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z_world = np.linalg.inv(R) @ z_cam
    n = np.linalg.norm(z_world) + 1e-12
    return z_world / n


def camera_center_from_extrinsic(E: np.ndarray) -> np.ndarray:
    """计算相机中心 C（世界系）。
    对 world->cam：X_cam = R X_world + t → C = -R^{-1} t。
    """
    R = E[:, :3]; t = E[:, 3:4]
    C = -np.linalg.inv(R) @ t
    return C[:, 0]


# ---------------- 几何：反投/投影（稀疏用于 O/U，稠密用于光度） ----------------
def unproject_depth_to_world(depth_hw: np.ndarray, E: np.ndarray, K: np.ndarray,
                             sample_stride: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    稀疏反投：从 t 帧的深度图反投到世界坐标（按 stride 采样）。
    返回：
      - world_pts (N,3)：世界点
      - pix_uv    (N,2,int)：对应的像素 (u,v)（整数索引）
    """
    depth = _as_hw_depth(depth_hw)
    H, W = depth.shape
    # 生成规则网格并按 stride 采样
    ys = np.arange(0, H, sample_stride, dtype=np.intp)
    xs = np.arange(0, W, sample_stride, dtype=np.intp)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    vs_i = grid_y.reshape(-1).astype(np.intp)
    us_i = grid_x.reshape(-1).astype(np.intp)

    # 拉取深度并做有效性过滤
    d = depth[vs_i, us_i].astype(np.float64)
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.intp)

    # 像素 → 归一化相机坐标 → 乘深度
    Kinv = np.linalg.inv(K.astype(np.float64))
    pix = np.stack([us_i[valid].astype(np.float64),
                    vs_i[valid].astype(np.float64),
                    np.ones(valid.sum(), dtype=np.float64)], axis=-1)  # (N,3)
    rays = (Kinv @ pix.T).T
    Xc = rays * d[valid][:, None]

    # 相机系 → 世界系（E：world->cam）
    R = E[:, :3].astype(np.float64); t = E[:, 3:4].astype(np.float64)
    Rinv = np.linalg.inv(R)
    Xw = (Rinv @ Xc.T).T - (Rinv @ t).T

    pix_uv = np.stack([us_i[valid], vs_i[valid]], axis=-1)
    return Xw, pix_uv


def project_world_to_image(world_pts: np.ndarray, E: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """世界点投影到图像平面，返回：
       - uv (N,2,float)：像素坐标
       - z  (N, )：相机坐标系下深度（>0 表示位于相机前方）
    """
    R = E[:, :3]; t = E[:, 3:4]
    Pc = (R @ world_pts.T + t).T  # (N,3)
    z = Pc[:, 2:3]
    z_safe = np.where(np.abs(z) < 1e-12, 1e-12, z)  # 防止除零
    uvw = (K @ Pc.T).T
    u = uvw[:, 0] / z_safe[:, 0]
    v = uvw[:, 1] / z_safe[:, 0]
    return np.stack([u, v], axis=-1), z[:, 0]


def warp_src_to_t_via_depth(depth_t: np.ndarray,
                            K_t: np.ndarray, E_t: np.ndarray,
                            K_s: np.ndarray, E_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用中心帧深度 D_t 将源帧 s（p 或 n）重投影到 t，计算源图中的采样坐标 (map_x,map_y) 以及源相机深度 z_s。
    返回：
      - map_x (H,W) float32：源图的 u 坐标
      - map_y (H,W) float32：源图的 v 坐标
      - z_s_map (H,W) float32：源相机前向深度（>0 为在前方）
    """
    D = _as_hw_depth(depth_t).astype(np.float64)
    H, W = D.shape
    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="xy")  # (H,W)

    # t 像素 → t 相机坐标
    Kt_inv = np.linalg.inv(K_t.astype(np.float64))
    ones = np.ones_like(uu)
    pix = np.stack([uu, vv, ones], axis=-1).reshape(-1, 3)           # (N,3)
    rays = (Kt_inv @ pix.T).T                                         # (N,3)
    d = D.reshape(-1, 1)                                              # (N,1)
    Xc_t = rays * d                                                   # (N,3)

    # t 相机 → 世界
    R_t = E_t[:, :3].astype(np.float64); t_t = E_t[:, 3:4].astype(np.float64)
    R_t_inv = np.linalg.inv(R_t)
    Xw = (R_t_inv @ Xc_t.T).T - (R_t_inv @ t_t).T                     # (N,3)

    # 世界 → 源相机
    R_s = E_s[:, :3].astype(np.float64); t_s = E_s[:, 3:4].astype(np.float64)
    Xc_s = (R_s @ Xw.T + t_s).T                                       # (N,3)

    # 源相机深度（前向）
    z_s = Xc_s[:, 2:3]                                                # (N,1)
    z_safe = np.where(np.abs(z_s) < 1e-12, 1e-12, z_s)

    # 源相机像素坐标
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
    以中心帧 t 为目标，评估源帧 s 的几何可用性，返回：
      - O：重叠率（边界内 & z>0 的比例，置信度可加权）
      - U：可用视差率（位移 ∈ [eps,delta] 的比例，置信度可加权）
      - f：前向占比（两相机光心连线在 t 前向上的绝对比值，0~1）
      - r：相对旋转（弧度）
    无有效采样点时返回 None。
    """
    depth_t = _as_hw_depth(depth_t)
    H, W = depth_t.shape

    # 稀疏反投到世界、记录对应像素坐标（整数）
    world_pts, pix_uv = unproject_depth_to_world(depth_t, E_t, K_t, sample_stride)
    if world_pts.shape[0] == 0:
        return None

    # 投回源帧，检查是否在画幅内且位于相机前方
    uv_src, z_src = project_world_to_image(world_pts, E_src, K_src)
    in_bounds = (uv_src[:, 0] >= 0) & (uv_src[:, 0] < W) & (uv_src[:, 1] >= 0) & (uv_src[:, 1] < H) & (z_src > 0)
    if in_bounds.sum() == 0:
        return None

    # 置信度像素权（若提供 conf_t，则按采样像素位置取权重；否则权重全 1）
    if conf_t is not None:
        conf = _as_hw_depth(conf_t)
        w_all = conf[pix_uv[:, 1], pix_uv[:, 0]].astype(np.float64)
    else:
        w_all = np.ones(pix_uv.shape[0], dtype=np.float64)

    # 视差位移与可用掩码
    disp = np.linalg.norm(uv_src - pix_uv.astype(np.float64), axis=1)
    usable = (disp >= eps_px) & (disp <= delta_px) & in_bounds

    # 重叠率（加权）
    denom_all = max(float(w_all.sum()), 1e-12)
    O_w = float(np.sum(w_all * in_bounds) / denom_all)

    # 可用视差率（加权；分母只统计 in_bounds 权重）
    denom_ib = max(float(np.sum(w_all * in_bounds)), 1e-12)
    U_w = float(np.sum(w_all * usable) / denom_ib)

    # 前向占比与相对旋转（标量，全局描述）
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
    高斯窗近似 SSIM（对 3 通道分别算后均值）。
    输入：
      - I0, I1: (H,W,3) float32 in [0,1]
    返回：
      - ssim_map: (H,W) float32 in [-1,1]
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
    基于重投影的密集光度一致性（Monodepth2 风格）：
      1) 用 D_t、(K/E) 计算源图到 t 的重映射坐标 (map_x,map_y)；
      2) remap 获取 I_s->t；
      3) 在“可用视差”（位移范围 + 在前方 + 画幅内）掩码上计算：
         photo = α*(1-SSIM)/2 + (1-α)*L1，并以 conf_t（若提供）做像素加权平均。
    返回：
      - L_photo：光度损失（标量，越小越好；用于 exp(-λ_photo * L_photo)）；
      - disp_med：可用区域的像素位移中位数（用于目标视差奖励）。
    若无有效像素，返回 (None, None)。
    """
    D = _as_hw_depth(depth_t)
    H, W = D.shape

    # 统一到中心深度的分辨率，并归一到 [0,1]
    It = cv2.resize(img_t_bgr, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    Is = cv2.resize(img_s_bgr, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    # BGR -> RGB（对称性；非必须）
    It = It[:, :, ::-1]
    Is = Is[:, :, ::-1]

    # 源→目标 t 的采样网格与深度
    map_x, map_y, z_s = warp_src_to_t_via_depth(D, K_t, E_t, K_s, E_s)

    # 基本几何掩码：在前方 + 画幅内（注意边界减 1 以保证插值安全）
    in_bounds = (map_x >= 0) & (map_x < W - 1) & (map_y >= 0) & (map_y < H - 1) & (z_s > 0)

    # 像素视差（二维位移幅值）
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
    disp = np.sqrt((map_x - uu) ** 2 + (map_y - vv) ** 2)

    # 可用视差约束
    usable = in_bounds & (disp >= eps_px) & (disp <= delta_px)
    if usable.sum() == 0:
        return None, None

    # 重建图：把源图按 map_x/map_y 采样到 t
    Iw = cv2.remap(Is, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # L1：三通道平均
    l1 = np.mean(np.abs(It - Iw), axis=2).astype(np.float32)

    # SSIM：三通道分别计算后取均值；映射到 [0,1] 的 (1-SSIM)/2
    ssim = ssim_map_rgb(It, Iw, ksize=3, sigma=1.0)
    photo = alpha_ssim * (1.0 - ssim) * 0.5 + (1.0 - alpha_ssim) * l1

    # 置信度像素权
    if conf_t is not None:
        conf = _normalize_conf_map(_as_hw_depth(conf_t))
    else:
        conf = np.ones((H, W), dtype=np.float32)

    w = conf * usable.astype(np.float32)
    w_sum = float(w.sum())
    if w_sum <= 1e-12:
        return None, None

    # 加权平均的光度损失与位移中位数
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
    单侧分（不含“目标视差奖励”的乘法项，由调用方另乘）：
      base = O^γ_O * U^γ_U * Q^w_Q * exp(-λ_f * f) * exp(-λ_r * r)
      若 L_photo 给定，则再乘 exp(-λ_photo * L_photo)。

    参数说明：
      - metrics: {"O","U","f","r"}；
      - Q_pair: 画质短板（中心与侧帧的 min）；
      - gamma_O/gamma_U: O/U 的幂指数（强调程度）；
      - lam_f/lam_r: 前向/旋转的软惩罚强度；
      - lambda_photo: 光度一致性的软惩罚强度；L_photo 越大，惩罚越强；
      - 目标视差奖励（beta_disp/target_disp_px）在外部基于 disp_med 乘到 sp/sn。
    """
    O, U, f, r = metrics["O"], metrics["U"], metrics["f"], metrics["r"]
    base = (O ** gamma_O) * (U ** gamma_U) * (Q_pair ** w_Q) * np.exp(-lam_f * f) * np.exp(-lam_r * r)

    # 光度一致性惩罚：L_photo 大 → 惩罚大（分值降低）
    if (L_photo is not None) and np.isfinite(L_photo):
        base *= float(np.exp(-lambda_photo * L_photo))

    return float(base)


# ---------------- VGGT 前向 ----------------
def vggt_forward_triplet(model: VGGT, device: str, dtype: torch.dtype, paths: List[str]):
    """以顺序 [t,p,n] 载入图像并前向一次：
       返回三帧的 E/K（numpy）、中心帧深度 D_t、中心帧深度置信度 conf_t（如有）、以及 H/W。
    """
    images = load_and_preprocess_images(paths).to(device)
    amp = torch.cuda.amp.autocast(dtype=dtype) if device == "cuda" else nullcontext()
    with torch.no_grad():
        with amp:
            preds = model(images)

    # 从 pose encoding 解码 E/K（注意 E 为 world->cam）
    E, K = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
    E = E[0].detach().cpu().numpy()
    K = K[0].detach().cpu().numpy()

    # 深度：只取中心帧（索引 0）；处理可能的 (3,1,H,W) 格式
    D = preds["depth"][0].detach().cpu().numpy()  # (3,H,W) 或 (3,1,H,W)
    if D.ndim == 4 and D.shape[1] == 1:
        D = D[:, 0]
    D_t = _as_hw_depth(D[0])
    H, W = D_t.shape

    # 置信度（若模型提供 depth_conf）
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
    """评估 (p,t,n)：
       - VGGT 前向拿 E/K/D_t/conf_t/H/W；
       - 稀疏几何 O/U/f/r（可选置信度加权），并做几何硬阈；
       - 基于投影的光度项 L_photo（SSIM+L1），并计算中位视差 disp_med；
       - 单侧分：几何基线 * exp(-λ_photo * L_photo)，再乘“目标视差奖励”；
       - 三元组分：sqrt(sp * sn)；
       - 构造返回结构（K、T、H/W、深度/置信度、诊断信息）或写拒绝原因并返回 None。
    """
    # [center, prev, next]
    paths = [image_paths[t], image_paths[p], image_paths[n]]
    E, K, D_t, conf_t, H, W = vggt_forward_triplet(model, device, dtype, paths)

    # 自适应位移上限（像素）：固定下限与分辨率比例的较大者
    delta_eff = max(args.delta_px, args.delta_px_frac * float(min(H, W)))
    eps_eff   = float(args.eps_px)

    # 加权几何（稀疏，含 O/U/f/r）
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

    # 提取基本标量，便于阈值判断与记录
    def basic(md: dict):
        return {"O": float(md["O"]), "U": float(md["U"]), "f": float(md["f"]), "r": float(md["r"])}
    m_prev_basic, m_next_basic = basic(m_prev), basic(m_next)

    # ---------------- 几何硬阈（单侧底线） ----------------
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

    # 读取原图（光度项需要）
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

    # ---------------- 基于投影的 SSIM+L1 光度项（源->t） ----------------
    Lp, disp_med_p = photometric_loss_dense(
        img_t, img_p, D_t, K[0], E[0], K[1], E[1], conf_t,
        eps_px=eps_eff, delta_px=delta_eff, alpha_ssim=args.photo_alpha_ssIM  # 注意：参数名大小写按现有代码
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

    # ---------------- 单侧打分（含光度惩罚；目标视差奖励在外层乘） ----------------
    Qp = min(Q_frame[t], Q_frame[p])  # 画质短板
    Qn = min(Q_frame[t], Q_frame[n])

    # 目标视差（像素）：显式值优先，否则按短边比例推断
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

    # 目标视差奖励（基于 disp_med 的高斯窗，乘到单侧分）
    if target_disp_px > 0:
        sigma = 0.35 * target_disp_px  # 窗宽，经验值，可调
        if disp_med_p is not None:
            sp *= (1.0 + args.beta_disp_reward *
                   float(np.exp(-0.5 * ((disp_med_p - target_disp_px) / max(sigma, 1e-6)) ** 2)))
        if disp_med_n is not None:
            sn *= (1.0 + args.beta_disp_reward *
                   float(np.exp(-0.5 * ((disp_med_n - target_disp_px) / max(sigma, 1e-6)) ** 2)))

    # 三元组综合分（几何平均）
    s_trip = float(np.sqrt(max(sp, 1e-12) * max(sn, 1e-12)))

    # 源→目标（t）的 4×4 齐次变换 T
    Gt, Gprev, Gnext = to44(E[0]), to44(E[1]), to44(E[2])
    T_prev_to_t = (Gt @ np.linalg.inv(Gprev)).astype(np.float32)
    T_next_to_t = (Gt @ np.linalg.inv(Gnext)).astype(np.float32)

    # 组织诊断信息（便于分析）
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
    """保存中心帧深度（及置信度，若有）为 .npy，并把三元组记录写入 triplets.jsonl。"""
    npy_dir = out_dir / "npy"; ensure_dir(npy_dir)
    center_stem = Path(best["center"]["file"]).stem

    # 深度 .npy
    depth_path = npy_dir / f"{center_stem}_depth.npy"
    np.save(depth_path, best.pop("depth_t"))

    # 置信度 .npy（可选）
    conf_dir = out_dir / "conf"; ensure_dir(conf_dir)
    conf_path = None
    conf_arr = best.pop("depth_conf_t", None)
    if conf_arr is not None:
        conf_path = conf_dir / f"{center_stem}_depthconf.npy"
        np.save(conf_path, conf_arr)

    # JSON 行（把矩阵转 list，便于跨语言读取）
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
    """序列内清晰度硬阈：按 lap_var 的分位数（min_lap_p ∈ [0,1]）。"""
    seq_laps = np.array([r["lap_var"] for r in frames], dtype=np.float64)
    return float(np.quantile(seq_laps, min_lap_p)) if len(seq_laps) else 0.0


def compute_Q_frame(frames: list, lap_p50: float) -> List[float]:
    """逐帧画质软权 Q：clip(lap_var / lap_p50, 0.5, 1.5)，用于打分（与几何项乘）。"""
    Q = []
    for rec in frames:
        q = rec["lap_var"] / max(lap_p50, 1e-6)
        q = float(np.clip(q, 0.5, 1.5))
        Q.append(q)
    return Q


def frame_bad(rec: dict, lap_thresh: float, max_clip: float, min_val: float) -> bool:
    """单帧质量硬阈：
       - lap_var 低于序列分位阈；
       - 欠/过曝比例任一 > max_clip；
       - 亮度均值 < min_val。
    """
    return (rec["lap_var"] < lap_thresh) or (rec["clip_dark"] > max_clip) or \
           (rec["clip_bright"] > max_clip) or (rec["val_mean"] < min_val)


def collect_candidates(frames: list, t: int, window: int, lap_thresh: float, args,
                       frame_rejects_path: Path, seq_rel: Path) -> Tuple[List[int], List[int]]:
    """为中心帧 t 在 [t-window, t+window] 内收集两侧候选；不合格者写入 frame_rejects.jsonl。"""
    prev_cands, next_cands = [], []
    for dt in range(1, window + 1):
        # 过去侧
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

        # 未来侧
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
    """读取全局统计（global_stats.json），取 lap_var_p50 作为 Q 的对齐基准。"""
    gstats_path = metrics_root / "global_stats.json"
    if not gstats_path.exists():
        raise FileNotFoundError(f"缺少 {gstats_path}，请先运行阶段一（逐帧指标统计）")
    gstats = json.loads(gstats_path.read_text())
    return float(gstats.get("lap_var_p50", 1.0))


def setup_device_and_model() -> Tuple[str, torch.dtype, VGGT]:
    """配置设备与精度并加载预训练 VGGT。
       CUDA 可用时：算力 >= 8 优先 bfloat16，否则 float16；CPU 回退 nullcontext。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    return device, dtype, model


def clear_sequence_outputs(out_dir: Path):
    """清理本序列输出（triplets/frame_rejects/geom_rejects）。如需断点续跑，可改附加模式。"""
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
    """处理单个序列：收集候选、评估全部 (p,n)、保存最佳、记录拒绝原因。"""
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

    # 硬阈（序列内）与软权（跨序列对齐）
    lap_thresh = compute_seq_lap_thresh(frames, args.min_lap_p)
    Q_frame    = compute_Q_frame(frames, lap_p50)

    any_valid = False
    for t in tqdm(range(N), desc=str(seq_rel), leave=False):
        rt = frames[t]
        # 中心帧质量不过线：记录并跳过
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

        # 收集两侧候选；若某侧为空，记录后跳过
        prev_cands, next_cands = collect_candidates(frames, t, args.window, lap_thresh, args,
                                                    frame_rejects_path, seq_rel)
        if (not prev_cands) or (not next_cands):
            _append_jsonl(geom_rejects_path, {
                "seq": str(seq_rel),
                "center": {"idx": int(t), "file": frames[t]["file"]},
                "reason": "no_candidates_on_one_side"
            })
            continue

        # 穷举 (p,n)，保留分数最高者
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
    """遍历 metrics_root 下的所有序列（以 frames.jsonl 为存在标识），逐一构建三元组。"""
    image_root   = Path(args.image_root)
    metrics_root = Path(args.metrics_root)
    out_root     = Path(args.out_root)
    ensure_dir(out_root)

    lap_p50 = load_global_stats(metrics_root)         # 跨序列画质对齐基准
    device, dtype, model = setup_device_and_model()   # 模型/设备/精度

    seq_jsonls = sorted(metrics_root.rglob("frames.jsonl"))
    for jpath in tqdm(seq_jsonls, desc="构建三元组"):
        seq_rel = jpath.parent.relative_to(metrics_root)
        process_sequence(seq_rel, jpath, image_root, metrics_root, out_root,
                         model, device, dtype, args, lap_p50)


# ---------------- 入口 ----------------
def build_argparser():
    """命令行参数：默认值偏保守，建议结合数据规模/视差范围适当调节。"""
    ap = argparse.ArgumentParser()
    # 输入与输出
    ap.add_argument("--image_root",   type=str, required=False,
                    default="/mnt/data_nvme3n1p1/dataset/UAV_ula/tri_images",
                    help="原始图像根目录（与阶段一一致）")
    ap.add_argument("--metrics_root", type=str, required=False,
                    default="/mnt/data_nvme3n1p1/dataset/UAV_ula/tri_win5",
                    help="阶段一输出根目录（包含 <seq>/frames.jsonl 与 global_stats.json）")
    ap.add_argument("--out_root",     type=str, required=False,
                    default="/mnt/data_nvme3n1p1/dataset/UAV_ula/tri_win5",
                    help="输出根目录（每序列一个 triplets.jsonl；另含 frame/geom 拒绝记录）")

    # 滑窗（组合复杂度 ~ k^2）
    ap.add_argument("--window",       type=int, default=5,
                    help="候选前后帧窗口半径 k（全程三帧将做 k×k 笛卡尔积）")

    # 稀疏采样（几何指标）
    ap.add_argument("--sample_stride", type=int, default=4,
                    help="几何 O/U 评估的稀疏采样步长（像素）")

    # 逐帧质量过滤阈值（硬阈）
    ap.add_argument("--min_lap_p",    type=float, default=0.10, help="拉普拉斯方差低于本序列分位数则过滤（0~1）")
    ap.add_argument("--max_clip",     type=float, default=0.10, help="欠/过曝比例任何一项超过则过滤")
    ap.add_argument("--min_val",      type=float, default=20.0, help="V 均值过低过滤 [0,255]")

    # 几何阈值（硬阈与视差带）
    ap.add_argument("--eps_px",       type=float, default=2.0,  help="可用视差下限（像素）")
    ap.add_argument("--delta_px",     type=float, default=80.0, help="可用视差上限（像素）下限值")
    ap.add_argument("--delta_px_frac",type=float, default=0.03, help="自适应上限：max(delta_px, frac*min(H,W))")
    ap.add_argument("--rmax_deg",     type=float, default=30.0, help="旋转上限（度）")
    ap.add_argument("--fmax",         type=float, default=0.85, help="前向占比上限")
    ap.add_argument("--Omin",         type=float, default=0.50, help="重叠率下限（单侧）")
    ap.add_argument("--Umin",         type=float, default=0.10, help="可用视差率下限（单侧）")

    # 打分权重（几何软项）
    ap.add_argument("--gamma_O",      type=float, default=1.5, help="O 幂")
    ap.add_argument("--gamma_U",      type=float, default=2.0, help="U 幂")
    ap.add_argument("--w_Q",          type=float, default=1.0, help="Q 幂（画质权重）")
    ap.add_argument("--lam_f",        type=float, default=2.0, help="前向占比软惩罚系数")
    ap.add_argument("--lam_r",        type=float, default=1.0, help="旋转软惩罚系数")

    # 目标视差奖励 & 光度一致性（建议与视差带/场景尺度联动调参）
    ap.add_argument("--target_disp_px",     type=float, default=0.0,  help="目标视差（像素）；<=0 时按 factor 推断")
    ap.add_argument("--target_disp_factor", type=float, default=0.02, help="目标视差系数（0.02*min(H,W)）")
    ap.add_argument("--beta_disp_reward",   type=float, default=0.12, help="目标视差奖励强度（乘性增益系数）")
    ap.add_argument("--lambda_photo",       type=float, default=3.0,  help="光度一致性惩罚权重（越大越苛刻）")
    ap.add_argument("--photo_alpha_ssIM",   type=float, default=0.85, help="光度项中 SSIM 的权重 α（其余为 L1）")
    return ap


def main():
    """脚本入口：解析参数并逐序列构建三元组。"""
    args = build_argparser().parse_args()
    run_all_sequences(args)


if __name__ == "__main__":
    main()
