#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGGT比例三元组构建器：按3:5:2比例构建小、中、大视差三元组

文件名：vggt_proportional_triplet_builder.py
功能：基于VGGT模型的智能三元组构建系统

=== 整体功能说明 ===
这个脚本是一个用于计算机视觉中立体匹配和深度估计的数据预处理工具。
它从无人机视频序列中构建高质量的三帧三元组（前帧-中心帧-后帧），
用于训练立体匹配、深度估计或SLAM算法。

=== 核心改进 ===
  - 每个中心帧构建3个三元组：小视差(30%)、中视差(50%)、大视差(20%)
  - 小视差：target_disp_factor = 0.02 (小视差)
  - 中视差：target_disp_factor = 0.03 (中视差) 
  - 大视差：target_disp_factor = 0.04 (大视差)
  - 每个中心帧只构建一次，不重复构建

=== 评分机制 ===
  - 单侧分：score_side = Q_norm(Q_pair) * U * R_target(d_med; d_star)
  - 根据目标视差范围选择最佳候选

=== 输入输出 ===
输入：
  - <seq>/frames.jsonl（含 file, lap_var, clip_dark, clip_bright, val_mean ...）
  - metrics_root/global_stats.json（lap_var_p50）
产物：
  - <seq>/triplets.jsonl（包含所有视差类型的三元组，每条记录含disparity_type字段）
  - <seq>/npy/<center>_depth.npy 与（可选）conf/<center>_depthconf.npy
  - <seq>/frame_rejects.jsonl, <seq>/geom_rejects.jsonl
  - <seq>/triplets_proportional_summary.jsonl（比例构建摘要）

=== 工作流程 ===
1. 质量过滤：过滤掉质量差的帧（低拉普拉斯方差、过曝等）
2. 候选收集：在时间窗口内收集前后帧候选
3. VGGT推理：使用预训练的VGGT模型进行深度估计和姿态预测
4. 几何评估：计算视差、覆盖率等几何指标
5. 评分选择：根据评分选择最佳的前后帧组合
6. 比例构建：按3:5:2比例随机选择视差类型
7. 保存结果：保存三元组数据和统计信息
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import random

import numpy as np
import torch
from tqdm import tqdm
from contextlib import nullcontext

# ===== VGGT =====
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# ==================== 基础工具函数 ====================
def ensure_dir(p: Path):
    """确保目录存在，如果不存在则创建"""
    p.mkdir(parents=True, exist_ok=True)


def _append_jsonl(p: Path, obj: dict):
    """向JSONL文件追加一条记录"""
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_frames_jsonl(path: Path) -> list:
    """读取frames.jsonl文件，返回帧信息列表"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _as_hw_depth(depth: np.ndarray) -> np.ndarray:
    """将深度图转换为(H,W)格式，处理不同的输入维度"""
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
    """归一化置信度图到[0,1]范围"""
    c = np.asarray(conf, np.float32)
    if (c.min() >= 0.0) and (c.max() <= 1.0):
        return c
    # 使用1%和99%分位数进行稳健归一化
    lo, hi = np.percentile(c, 1), np.percentile(c, 99)
    c = (c - lo) / max(hi - lo, 1e-6)
    return np.clip(c, 0.0, 1.0).astype(np.float32)


def to44(E34: np.ndarray) -> np.ndarray:
    """将3x4外参矩阵转换为4x4齐次变换矩阵"""
    G = np.eye(4, dtype=np.float64)
    G[:3, :4] = E34
    return G


# ==================== 日志辅助函数 ====================
import re as _re


def get_numeric_idx(file_or_name: str) -> Optional[int]:
    """从文件名中提取数字索引，用于日志记录"""
    stem = Path(file_or_name).stem
    m = _re.search(r"(\d+)$", stem) or _re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else None


def idx_for_log(frames: list, i: int) -> int:
    """获取用于日志记录的帧索引"""
    v = get_numeric_idx(frames[i]["file"])
    return i if v is None else v


def frame_ref(frames: list, i: int) -> dict:
    """创建帧引用字典，包含索引和文件名"""
    return {"idx": idx_for_log(frames, i), "file": frames[i]["file"]}


# ==================== 几何变换核心函数 ====================
def warp_src_to_t_via_depth(depth_t: np.ndarray,
                            K_t: np.ndarray, E_t: np.ndarray,
                            K_s: np.ndarray, E_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    用中心帧深度 D_t 将源帧 s 重投影到 t，返回源图采样坐标与源相机深度。
    
    这是立体匹配中的核心几何变换：
    1. 从中心帧的每个像素出发，利用深度信息反投影到3D世界坐标
    2. 将3D点投影到源帧的像素坐标
    3. 返回源帧的采样坐标和深度信息
    
    参数:
        depth_t: 中心帧深度图 (H,W)
        K_t, E_t: 中心帧的内参和外参矩阵
        K_s, E_s: 源帧的内参和外参矩阵
    
    返回:
        map_x, map_y: 源帧采样坐标 (H,W)
        z_s_map: 源帧深度图 (H,W)
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
    计算"仅旋转"的像素映射：把 t 帧的每个像素方向（不依赖深度）
    通过 R_rel = R_s @ R_t^{-1} 旋转到 s 相机，再用 K_s 投影到 s 的像素平面。
    
    这个方法用于计算纯旋转引起的像素位移，不包含平移分量。
    在立体匹配中，这有助于分离旋转和平移对像素位移的贡献。
    
    参数:
        K_t, E_t: 中心帧的内参和外参
        K_s, E_s: 源帧的内参和外参
        H, W: 图像尺寸
        precomp: 预计算的网格和逆矩阵（用于加速）
    
    返回:
        u_rot, v_rot: 仅旋转的像素映射坐标 (H,W)
        precomp: 可复用的预计算数据
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
    """
    计算去旋转视差(parallax)的覆盖率和中位数视差。
    
    这是立体匹配中的关键几何评估函数：
    1. 使用中心帧深度将像素从t映射到s，得到(map_x, map_y)和z_s
    2. 计算纯旋转映射(u_rot, v_rot)（与深度无关）
    3. 计算parallax = ||[map_x,map_y] - [u_rot,v_rot]||_2，仅保留平移诱导的位移
    4. 计算U：在有效范围内且视差在[eps, delta]之间的像素比例
    5. 计算disp_med：可用区域的视差中位数
    
    参数:
        depth_t: 中心帧深度图
        K_t, E_t: 中心帧内参和外参
        K_s, E_s: 源帧内参和外参
        conf_t: 深度置信度图（可选）
        eps_px, delta_px: 视差范围阈值
        precomp: 预计算数据（用于加速）
    
    返回:
        U: 有效像素覆盖率 [0,1]
        disp_med: 中位数视差值
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

    # 可用区域的中位"去旋转视差"
    disp_med = float(np.median(parallax[usable]))
    return U, disp_med


# ==================== 评分系统 ====================
def compute_Q_frame(frames: list, lap_p50: float) -> List[float]:
    """
    计算每帧的质量分数Q，基于拉普拉斯方差。
    
    拉普拉斯方差是图像清晰度的指标，值越高表示图像越清晰。
    这里将每帧的拉普拉斯方差与全局中位数进行比较，得到归一化的质量分数。
    
    参数:
        frames: 帧信息列表
        lap_p50: 全局拉普拉斯方差中位数
    
    返回:
        Q: 每帧的质量分数列表
    """
    Q = []
    for rec in frames:
        q = rec["lap_var"] / max(lap_p50, 1e-6)
        q = float(np.clip(q, 0.5, 1.5))  # 限制在[0.5, 1.5]范围内
        Q.append(q)
    return Q


def Q_pair_norm(Q_t: float, Q_s: float) -> float:
    """
    计算帧对的质量分数，采用短板原则。
    
    三元组的质量由质量较差的帧决定，这确保了整体质量。
    
    参数:
        Q_t: 中心帧质量分数
        Q_s: 源帧质量分数
    
    返回:
        归一化的帧对质量分数 [0,1]
    """
    q_pair = min(Q_t, Q_s)  # 短板优先
    q01 = (q_pair - 0.5) / 1.0  # [0.5,1.5] → [0,1]
    return float(np.clip(q01, 0.0, 1.0))


def R_target(d_med: Optional[float], d_star: float, kappa: float = 0.35) -> float:
    """
    计算目标视差匹配度R_target。
    
    使用高斯函数评估实际视差与目标视差的匹配程度。
    当d_med接近d_star时，R_target接近1；偏离越大，R_target越小。
    
    参数:
        d_med: 实际中位数视差
        d_star: 目标视差
        kappa: 高斯函数的标准差系数
    
    返回:
        目标匹配度 [0,1]
    """
    if (d_med is None) or (d_star <= 0):
        return 0.0
    sigma = max(kappa * d_star, 1e-6)
    return float(np.exp(-0.5 * ((d_med - d_star) / sigma) ** 2))  # ∈(0,1]


# ==================== 比例构建核心逻辑 ====================
def get_disparity_type_proportionally(small_ratio: float, medium_ratio: float, large_ratio: float) -> str:
    """
    按指定比例随机选择视差类型。
    
    这是比例构建的核心函数，确保生成的三元组按照预设比例分布：
    - 小视差：用于近距离物体的精细匹配
    - 中视差：用于中等距离物体的平衡匹配
    - 大视差：用于远距离物体的粗粒度匹配
    
    参数:
        small_ratio: 小视差比例
        medium_ratio: 中视差比例  
        large_ratio: 大视差比例
    
    返回:
        视差类型字符串 ("small", "medium", "large")
    """
    rand = random.random()
    if rand < small_ratio:
        return "small"
    elif rand < small_ratio + medium_ratio:
        return "medium"
    else:
        return "large"


def get_target_disp_factor(disp_type: str, small_factor: float, medium_factor: float, large_factor: float) -> float:
    """
    根据视差类型返回目标视差因子。
    
    视差因子决定了目标视差的大小，用于评分系统中的R_target计算。
    不同的视差因子对应不同的匹配难度和应用场景。
    
    参数:
        disp_type: 视差类型
        small_factor, medium_factor, large_factor: 对应的视差因子
    
    返回:
        目标视差因子
    """
    if disp_type == "small":
        return small_factor
    elif disp_type == "medium":
        return medium_factor
    elif disp_type == "large":
        return large_factor
    else:
        raise ValueError(f"Unknown disparity type: {disp_type}")


def evaluate_side_with_target(frames: list,
                              t: int, s: int,
                              pos_of: dict,
                              E_all: np.ndarray, K_all: np.ndarray,
                              D_t: np.ndarray, conf_t: Optional[np.ndarray],
                              H: int, W: int,
                              Q_frame: List[float],
                              target_disp_factor: float,
                              args, seq_rel: Path,
                              geom_rejects_path: Path,
                              side_label: str, precomp=None) -> Optional[dict]:
    """评估单侧候选，使用指定的目标视差因子"""
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

    # 使用指定的目标视差
    d_star = target_disp_factor * float(min(H, W))

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
        "score_side": float(score),
        "target_disp_factor": target_disp_factor,
        "d_star": float(d_star)
    }
    return {"diag": diag, "score": float(score)}


# ---------------- 保存 ----------------
def save_proportional_triplet(out_dir: Path, seq_rel: Path, triplet_data: dict, disp_type: str):
    """保存比例构建的三元组到统一的triplets.jsonl"""
    npy_dir = out_dir / "npy"
    ensure_dir(npy_dir)
    center_stem = Path(triplet_data["center"]["file"]).stem

    depth_path = npy_dir / f"{center_stem}_depth.npy"
    np.save(depth_path, triplet_data.pop("depth_t"))

    conf_dir = out_dir / "conf"
    ensure_dir(conf_dir)
    conf_path = None
    conf_arr = triplet_data.pop("depth_conf_t", None)
    if conf_arr is not None:
        conf_path = conf_dir / f"{center_stem}_depthconf.npy"
        np.save(conf_path, conf_arr)

    rec = {
        "seq": str(seq_rel),
        "center": triplet_data["center"],
        "prev": triplet_data["prev"],
        "next": triplet_data["next"],
        "score_triplet": triplet_data["score_triplet"],
        "disparity_type": disp_type,
        "target_disp_factor": triplet_data["target_disp_factor"],
        "K_t": triplet_data["K_t"].tolist(),
        "K_prev": triplet_data["K_prev"].tolist(),
        "K_next": triplet_data["K_next"].tolist(),
        "T_prev_to_t": triplet_data["T_prev_to_t"].tolist(),
        "T_next_to_t": triplet_data["T_next_to_t"].tolist(),
        "H": triplet_data["H"], "W": triplet_data["W"],
        "depth_t_npy": str(depth_path),
        "depth_conf_t_npy": (str(conf_path) if conf_path is not None else None)
    }
    
    # 保存到统一的triplets.jsonl文件
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
                        "clip_dark_gt_max": bool(rn["clip_bright"] > args.max_clip),
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


def side_pass(diag: dict, args) -> bool:
    """单侧候选是否通过硬门槛。diag 为 evaluate_side 返回的 diag 字段。"""
    try:
        return (float(diag.get("U", 0.0)) >= float(args.min_U) and
                float(diag.get("R_target", 0.0)) >= float(args.min_R) and
                float(diag.get("Q_norm", 0.0)) >= float(args.min_Qn) and
                float(diag.get("score_side", 0.0)) >= float(args.min_side_score))
    except Exception:
        return False


# ---------------- 主流程 ----------------
def clear_sequence_outputs(out_dir: Path):
    ensure_dir(out_dir)
    for name in ["triplets.jsonl", "frame_rejects.jsonl", "geom_rejects.jsonl", "triplets_proportional_summary.jsonl"]:
        p = out_dir / name
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def process_sequence_proportional(seq_rel: Path, jpath: Path, image_root: Path, metrics_root: Path,
                                 out_root: Path, model: VGGT, device: str, dtype: torch.dtype,
                                 args, lap_p50: float):
    """
    比例构建版本的主处理函数。
    
    这是整个系统的核心函数，负责处理单个视频序列的三元组构建：
    1. 读取帧信息和质量指标
    2. 对每帧进行质量过滤
    3. 收集前后帧候选
    4. 使用VGGT模型进行深度估计和姿态预测
    5. 按比例选择视差类型
    6. 评估几何质量并选择最佳候选
    7. 保存三元组数据和统计信息
    
    参数:
        seq_rel: 序列相对路径
        jpath: frames.jsonl文件路径
        image_root: 图像根目录
        metrics_root: 指标根目录
        out_root: 输出根目录
        model: VGGT模型
        device: 计算设备
        dtype: 数据类型
        args: 命令行参数
        lap_p50: 全局拉普拉斯方差中位数
    """
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
    summary_path = out_dir / "triplets_proportional_summary.jsonl"

    lap_thresh = compute_seq_lap_thresh(frames, args.min_lap_p)
    Q_frame = compute_Q_frame(frames, lap_p50)

    # 统计信息：记录不同视差类型的数量
    stats = {"small": 0, "medium": 0, "large": 0, "total_processed": 0, "total_rejected": 0}
    
    any_valid = False
    # 遍历序列中的每一帧，构建三元组
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
            stats["total_rejected"] += 1
            continue

        prev_cands, next_cands = collect_candidates(frames, t, args.window, lap_thresh, args,
                                                    frame_rejects_path, seq_rel)
        if (not prev_cands) or (not next_cands):
            _append_jsonl(geom_rejects_path, {
                "seq": str(seq_rel), "center": frame_ref(frames, t),
                "reason": "no_candidates_on_one_side"
            })
            stats["total_rejected"] += 1
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

        # === 按比例选择视差类型 ===
        # 这是比例构建的核心：随机选择视差类型，确保数据平衡
        disp_type = get_disparity_type_proportionally(args.small_ratio, args.medium_ratio, args.large_ratio)
        target_disp_factor = get_target_disp_factor(disp_type, args.small_factor, args.medium_factor, args.large_factor)

        # === 构建该视差类型的最优三元组 ===
        # 在选定的视差类型下，寻找最佳的前后帧组合
        best_prev, best_prev_idx = None, None
        for p in prev_cands:
            if p not in pos_of:
                continue
            res_p = evaluate_side_with_target(frames, t, p, pos_of, E_all, K_all, D_t, conf_t, H, W,
                                              Q_frame, target_disp_factor, args, seq_rel, geom_rejects_path, "prev",
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

        best_next, best_next_idx = None, None
        for n in next_cands:
            if n not in pos_of:
                continue
            res_n = evaluate_side_with_target(frames, t, n, pos_of, E_all, K_all, D_t, conf_t, H, W,
                                              Q_frame, target_disp_factor, args, seq_rel, geom_rejects_path, "next",
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
            stats["total_rejected"] += 1
            continue

        # === 计算齐次变换矩阵 ===
        # 计算从前后帧到中心帧的变换矩阵，用于后续的几何变换
        ct, cp, cn = pos_of[t], pos_of[best_prev_idx], pos_of[best_next_idx]
        Gt, Gprev, Gnext = to44(E_all[ct]), to44(E_all[cp]), to44(E_all[cn])
        T_prev_to_t = (Gt @ np.linalg.inv(Gprev)).astype(np.float32)
        T_next_to_t = (Gt @ np.linalg.inv(Gnext)).astype(np.float32)

        # === 计算三元组总分 ===
        # 使用几何平均计算三元组总分，确保两侧质量平衡
        sp, sn = best_prev["score"], best_next["score"]
        s_trip = float(np.sqrt(max(sp, 1e-12) * max(sn, 1e-12)))

        # === 保存比例构建的三元组 ===
        # 构建包含所有必要信息的三元组记录
        rec = {
            "center": {"idx": idx_for_log(frames, t), "file": frames[t]["file"]},
            "prev": best_prev["diag"],
            "next": best_next["diag"],
            "score_triplet": s_trip,
            "target_disp_factor": target_disp_factor,
            "K_t": K_all[ct].astype(np.float32),
            "K_prev": K_all[cp].astype(np.float32),
            "K_next": K_all[cn].astype(np.float32),
            "T_prev_to_t": T_prev_to_t,
            "T_next_to_t": T_next_to_t,
            "H": int(H), "W": int(W),
            "depth_t": D_t.astype(np.float32),
            "depth_conf_t": (conf_t.astype(np.float32) if conf_t is not None else None)
        }
        save_proportional_triplet(out_dir, seq_rel, rec, disp_type)
        
        # === 更新统计信息 ===
        stats[disp_type] += 1
        stats["total_processed"] += 1
        any_valid = True

    # === 保存比例构建摘要 ===
    # 记录实际的比例分布，用于验证构建效果
    if any_valid:
        summary_obj = {
            "seq": str(seq_rel),
            "disparity_distribution": {
                "small": stats["small"],
                "medium": stats["medium"], 
                "large": stats["large"],
                "total_processed": stats["total_processed"],
                "total_rejected": stats["total_rejected"]
            },
            "proportions": {
                "small_ratio": stats["small"] / max(stats["total_processed"], 1),
                "medium_ratio": stats["medium"] / max(stats["total_processed"], 1),
                "large_ratio": stats["large"] / max(stats["total_processed"], 1)
            }
        }
        _append_jsonl(summary_path, summary_obj)

    if not any_valid:
        _append_jsonl(geom_rejects_path, {"seq": str(seq_rel), "summary": "no_valid_triplet_in_sequence"})


# ==================== 批量处理入口 ====================
def load_global_stats(metrics_root: Path) -> float:
    """
    加载全局统计信息，获取拉普拉斯方差中位数。
    
    这个值用于归一化每帧的质量分数，确保不同序列间的质量评估一致性。
    """
    gstats_path = metrics_root / "global_stats.json"
    if not gstats_path.exists():
        raise FileNotFoundError(f"缺少 {gstats_path}，请先运行阶段一（逐帧指标统计）")
    gstats = json.loads(gstats_path.read_text())
    return float(gstats.get("lap_var_p50", 1.0))


def run_all_sequences_proportional(args):
    """
    批量处理所有序列的比例构建。
    
    这是整个系统的入口函数，负责：
    1. 加载全局统计信息
    2. 初始化VGGT模型
    3. 遍历所有序列进行三元组构建
    4. 生成统计报告
    """
    image_root = Path(args.image_root)
    metrics_root = Path(args.metrics_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    # 加载全局统计信息，用于质量归一化
    lap_p50 = load_global_stats(metrics_root)
    device, dtype, model = setup_device_and_model()

    # 遍历所有序列的frames.jsonl文件
    seq_jsonls = sorted(metrics_root.rglob("frames.jsonl"))
    for jpath in tqdm(seq_jsonls, desc="比例构建三元组"):
        seq_rel = jpath.parent.relative_to(metrics_root)
        process_sequence_proportional(seq_rel, jpath, image_root, metrics_root, out_root,
                                     model, device, dtype, args, lap_p50)


def setup_device_and_model() -> Tuple[str, torch.dtype, VGGT]:
    """
    初始化计算设备和VGGT模型。
    
    根据硬件能力选择合适的数据类型：
    - 支持bfloat16的GPU使用bfloat16（更稳定）
    - 其他情况使用float16（更兼容）
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (
                torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    return device, dtype, model


# ==================== 命令行参数配置 ====================
def build_argparser():
    """
    构建命令行参数解析器。
    
    包含所有必要的参数配置，从输入输出路径到质量阈值和比例参数。
    """
    ap = argparse.ArgumentParser()
    
    # === 输入与输出路径 ===
    ap.add_argument("--image_root", type=str, required=False,
                    default="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany_tri/All",
                    help="原始图像根目录（与阶段一一致）")
    ap.add_argument("--metrics_root", type=str, required=False,
                    default="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany_tri/tri_metrics",
                    help="阶段一输出根目录（包含 <seq>/frames.jsonl 与 global_stats.json）")
    ap.add_argument("--out_root", type=str, required=False,
                    default="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany_tri/tri_rw20_ratio352_disp234_lap0.05",
                    help="输出根目录（每序列一个 triplets.jsonl，含所有视差类型；另含 frame/geom 拒绝记录）")

    # === 滑窗参数 ===
    ap.add_argument("--window", type=int, default=20,
                    help="候选前后帧窗口半径")

    # === 几何评估参数 ===
    ap.add_argument("--sample_stride", type=int, default=4,
                    help="（保留，仅用于速度。若使用 compute_U_and_disp_median 的稠密实现，可忽略）")

    # === 质量过滤阈值 ===
    ap.add_argument("--min_lap_p", type=float, default=0.05,
                    help="拉普拉斯方差低于本序列分位数则过滤（0~1）")
    ap.add_argument("--max_clip", type=float, default=0.10,
                    help="欠/过曝比例任何一项超过则过滤")
    ap.add_argument("--min_val", type=float, default=20.0,
                    help="V 均值过低过滤 [0,255]")

    # === 视差范围参数 ===
    ap.add_argument("--eps_px", type=float, default=2.0,
                    help="可用视差下限（像素）")
    ap.add_argument("--delta_px", type=float, default=80.0,
                    help="可用视差上限（像素）")
    ap.add_argument("--delta_px_frac", type=float, default=0.05,
                    help="自适应上限：max(delta_px, frac*min(H,W))")

    # === 侧级硬门槛（single-side gates） ===
    ap.add_argument("--min_U", type=float, default=0.07,
                    help="单侧覆盖率 U 的硬阈")
    ap.add_argument("--min_R", type=float, default=0,
                    help="单侧 R_target 的硬阈")
    ap.add_argument("--min_Qn", type=float, default=0.15,
                    help="单侧 Q_norm 的硬阈")
    ap.add_argument("--min_side_score", type=float, default=0.010,
                    help="单侧 score_side 的硬阈")

    # === 比例构建参数 ===
    ap.add_argument("--small_ratio", type=float, default=0.3,
                    help="小视差三元组比例")
    ap.add_argument("--medium_ratio", type=float, default=0.5,
                    help="中视差三元组比例")
    ap.add_argument("--large_ratio", type=float, default=0.2,
                    help="大视差三元组比例")
    ap.add_argument("--small_factor", type=float, default=0.02,
                    help="小视差目标因子")
    ap.add_argument("--medium_factor", type=float, default=0.03,
                    help="中视差目标因子")
    ap.add_argument("--large_factor", type=float, default=0.04,
                    help="大视差目标因子")

    return ap


# ==================== 主入口函数 ====================
def main():
    """
    主入口函数，负责参数解析和程序启动。
    
    包含比例参数的验证和自动归一化功能。
    """
    args = build_argparser().parse_args()
    
    # === 验证比例参数 ===
    # 确保比例参数总和为1，否则自动归一化
    total_ratio = args.small_ratio + args.medium_ratio + args.large_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"警告：比例参数总和为 {total_ratio:.3f}，不等于1.0")
        print("将自动归一化比例参数...")
        args.small_ratio /= total_ratio
        args.medium_ratio /= total_ratio
        args.large_ratio /= total_ratio
        print(f"归一化后：小={args.small_ratio:.3f}, 中={args.medium_ratio:.3f}, 大={args.large_ratio:.3f}")
    
    # 启动批量处理
    run_all_sequences_proportional(args)


if __name__ == "__main__":
    main()
