"""Numpy quaternion helpers (wxyz convention)."""
from __future__ import annotations

import numpy as np


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qw = q[..., 0:1]
    qxyz = q[..., 1:4]
    t = 2.0 * np.cross(qxyz, v)
    return v + qw * t + np.cross(qxyz, t)


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX-extrinsic: q = q_yaw * q_pitch * q_roll. Returns (w, x, y, z)."""
    hr, hp, hy = 0.5 * roll, 0.5 * pitch, 0.5 * yaw
    cr, sr = np.cos(hr), np.sin(hr)
    cp, sp = np.cos(hp), np.sin(hp)
    cy, sy = np.cos(hy), np.sin(hy)
    return np.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ], dtype=np.float64)


def yaw_from_quat(q) -> float:
    """Yaw of x-axis heading (calc_heading convention)."""
    qw, qx, qy, qz = (float(v) for v in q)
    dir_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    dir_y = 2.0 * (qw * qz + qx * qy)
    return float(np.arctan2(dir_y, dir_x))


def planar_rot_matrix(yaw: float) -> np.ndarray:
    """Rotation matrix for planar (yaw-only) frame: planar_base -> world."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
