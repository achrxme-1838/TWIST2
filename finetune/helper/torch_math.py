from __future__ import annotations

import torch


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qw, qxyz = q[..., 0:1], q[..., 1:4]
    t = 2.0 * torch.cross(qxyz, v, dim=-1)
    return v + qw * t + torch.cross(qxyz, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conj(q), v)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quat_unique(q: torch.Tensor) -> torch.Tensor:
    """Flip sign so w >= 0."""
    return torch.where(q[..., :1] < 0.0, -q, q)


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Angle of q1 * conj(q2) -- IsaacLab quat_error_magnitude."""
    rel = quat_unique(quat_mul(q1, quat_conj(q2)))
    return 2.0 * torch.atan2(rel[..., 1:].norm(dim=-1), rel[..., 0])


def quat_to_exp_map(q: torch.Tensor) -> torch.Tensor:
    """Rotation vector (axis * angle) of wxyz quat(s), angle in [0, pi]."""
    q = quat_unique(q)
    w, v = q[..., 0], q[..., 1:]
    sin_half = v.norm(dim=-1)
    angle = 2.0 * torch.atan2(sin_half, w)
    axis = torch.where(sin_half[..., None] > 1e-8, v / sin_half.clamp_min(1e-12)[..., None],
                       torch.zeros_like(v))
    return axis * angle[..., None]


def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical interpolation, shortest arc. ``t`` broadcasts against the leading dims."""
    if t.shape[-1:] != (1,):
        t = t[..., None]
    cos_half = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(cos_half < 0.0, -q1, q1)
    cos_half = cos_half.abs()
    small = cos_half > 0.9995
    half = torch.acos(cos_half.clamp(-1.0, 1.0))
    sin_half = torch.sin(half).clamp_min(1e-12)
    r0 = torch.where(small, 1.0 - t, torch.sin((1.0 - t) * half) / sin_half)
    r1 = torch.where(small, t, torch.sin(t * half) / sin_half)
    return quat_normalize(r0 * q0 + r1 * q1)


def quat_to_rpy(q: torch.Tensor) -> torch.Tensor:
    """Batched deploy ``quatToEuler`` (roll, pitch, yaw)."""
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = torch.asin((2 * (qw * qy - qz * qx)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return torch.stack([roll, pitch, yaw], dim=-1)


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return q[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return q[..., [3, 0, 1, 2]]


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Yaw of the x-axis heading (calc_heading convention)."""
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def heading_quat_from_quat(q: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Yaw-only (heading) quat; matches DEX calc_heading_quat / calc_heading_quat_inv."""
    half = 0.5 * yaw_from_quat(q)
    if inverse:
        half = -half
    zero = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zero, zero, torch.sin(half)], dim=-1)


def projected_gravity_b(q: torch.Tensor) -> torch.Tensor:
    """Gravity direction in the body frame (IsaacLab projected_gravity)."""
    g = torch.zeros(q.shape[:-1] + (3,), dtype=q.dtype, device=q.device)
    g[..., 2] = -1.0
    return quat_rotate_inverse(q, g)


def rpy_to_quat(rpy: torch.Tensor) -> torch.Tensor:
    """ZYX-extrinsic: q = q_yaw * q_pitch * q_roll. [..., 3] -> [..., 4] wxyz."""
    h = 0.5 * rpy
    cr, sr = torch.cos(h[..., 0]), torch.sin(h[..., 0])
    cp, sp = torch.cos(h[..., 1]), torch.sin(h[..., 1])
    cy, sy = torch.cos(h[..., 2]), torch.sin(h[..., 2])
    return torch.stack([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ], dim=-1)


def quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    """wxyz quat(s) -> 6D rotation rep (first two rotation-matrix columns). [...,4] -> [...,6]."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    col0 = torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], dim=-1)
    col1 = torch.stack([2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)], dim=-1)
    return torch.cat([col0, col1], dim=-1)


def yaw_quat(yaw: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Yaw-only quat from a yaw angle [...] -> [..., 4] wxyz."""
    half = -0.5 * yaw if inverse else 0.5 * yaw
    zero = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zero, zero, torch.sin(half)], dim=-1)
