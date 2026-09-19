from __future__ import annotations

import numpy as np

from utils.math import quat_apply, quat_conj, quat_mul  # deploy_real (see _paths.setup)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.asarray(q)[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.asarray(q)[..., [1, 2, 3, 0]]


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_unique(q: np.ndarray) -> np.ndarray:
    """Flip sign so w >= 0."""
    q = np.asarray(q)
    return np.where(q[..., :1] < 0.0, -q, q)


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_apply(quat_conj(np.asarray(q, dtype=np.float64)), np.asarray(v, dtype=np.float64))


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t) -> np.ndarray:
    """Spherical interpolation, shortest arc. ``t`` scalar or (..., 1)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if t.ndim < q0.ndim:
        t = t[..., None]
    cos_half = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(cos_half < 0.0, -q1, q1)
    cos_half = np.abs(cos_half)
    small = cos_half > 0.9995
    half = np.arccos(np.clip(cos_half, -1.0, 1.0))
    sin_half = np.sin(half)
    with np.errstate(divide="ignore", invalid="ignore"):
        r0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * half) / sin_half)
        r1 = np.where(small, t, np.sin(t * half) / sin_half)
    return quat_normalize(r0 * q0 + r1 * q1)


def quat_to_exp_map(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of wxyz quat(s), angle in [0, pi]."""
    q = quat_unique(np.asarray(q, dtype=np.float64))
    w = q[..., 0]
    v = q[..., 1:]
    sin_half = np.linalg.norm(v, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        axis = np.where(sin_half[..., None] > 1e-8, v / np.maximum(sin_half, 1e-12)[..., None], 0.0)
    return axis * angle[..., None]


def quat_error_magnitude(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angle of q1 * conj(q2) -- IsaacLab ``quat_error_magnitude``."""
    rel = quat_mul(np.asarray(q1, dtype=np.float64), quat_conj(np.asarray(q2, dtype=np.float64)))
    rel = quat_unique(rel)
    return 2.0 * np.arctan2(np.linalg.norm(rel[..., 1:], axis=-1), rel[..., 0])


def so3_angular_velocity(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity [T,3] of a quaternion sequence, central differences
    (forward / backward at the ends) -- MotionLib._compute_so3_derivative."""
    q = np.asarray(quats_wxyz, dtype=np.float64)
    T = q.shape[0]
    if T < 3:
        rel = quat_mul(q[1:], quat_conj(q[:-1]))
        w = quat_to_exp_map(rel) / dt
        return np.concatenate([w, w[-1:]], axis=0) if T == 2 else np.zeros((T, 3))
    rel = quat_mul(q[2:], quat_conj(q[:-2]))
    w_int = quat_to_exp_map(rel) / (2.0 * dt)
    w0 = quat_to_exp_map(quat_mul(q[1], quat_conj(q[0])))[None] / dt
    w1 = quat_to_exp_map(quat_mul(q[-1], quat_conj(q[-2])))[None] / dt
    return np.concatenate([w0, w_int, w1], axis=0)


def projected_gravity_b(q_wxyz: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse(q_wxyz, np.array([0.0, 0.0, -1.0]))


def quat_to_rpy(q_wxyz: np.ndarray) -> np.ndarray:
    """Vectorised deploy ``quatToEuler`` (roll, pitch, yaw)."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return np.stack([roll, pitch, yaw], axis=-1)
