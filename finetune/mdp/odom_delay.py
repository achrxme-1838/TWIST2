"""Odometry delay + drift model"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch

from finetune.helper.torch_math import quat_apply, quat_conj, quat_to_rpy, rpy_to_quat

# OdomDelayCfg fields of the drift model (tasks/ft_cfg.py)
DRIFT_CFG_KEYS = ("drift_scale", "drift_xy_dist", "drift_xy_time", "drift_z_dist", "drift_yaw_dist",
                  "drift_yaw_rot", "drift_yaw_time", "drift_scale_bias_std", "drift_yaw_bias_std",
                  "max_xy_dev", "max_z_dev", "max_yaw_dev", "init_xy_dev", "init_yaw_dev")


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def _rotz2(yaw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate planar vectors v [N, 2] by yaw [N]."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]), dim=-1)


class OdomDelayBuffer:
    def __init__(self, num_envs: int, update_period_steps: Sequence[int], latency_steps: Sequence[int],
                 device: torch.device, seed: Optional[int] = None, drift: Optional[dict] = None,
                 dt: float = 0.02):
        """``drift``: dict with the ``DRIFT_CFG_KEYS`` of OdomDelayCfg (cfg units: m, deg, %), None = off.
        ``dt``: control (policy) step [s]."""
        p_lo, p_hi = int(update_period_steps[0]), int(update_period_steps[1])
        l_lo, l_hi = int(latency_steps[0]), int(latency_steps[1])
        if not (1 <= p_lo <= p_hi) or not (0 <= l_lo <= l_hi):
            raise ValueError(f"invalid odom_delay ranges: update_period_steps={list(update_period_steps)}, "
                             f"latency_steps={list(latency_steps)}")
        if l_hi >= p_lo:
            print(f"[WARN] odom delay: latency max ({l_hi}) >= period min ({p_lo}); "
                  f"the held odometry may occasionally step back in time.")
        self.period_range = (p_lo, p_hi)
        self.latency_range = (l_lo, l_hi)
        self.num_envs = n = int(num_envs)
        self.device = torch.device(device)
        self.H = l_hi + 1  # history needed for the max latency

        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(int(seed))

        self.hist_pos = torch.zeros(n, self.H, 3, device=self.device)
        self.hist_yaw = torch.zeros(n, self.H, device=self.device)
        self.held_pos = torch.zeros(n, 3, device=self.device)
        self.held_yaw = torch.zeros(n, device=self.device)
        self.countdown = torch.zeros(n, dtype=torch.long, device=self.device)
        self.latency = torch.zeros(n, dtype=torch.long, device=self.device)
        self.slot = 0  # ring buffer write index (shared by all envs)

        # ---- drift (all in SI units: m, rad, s)
        self.drift = drift is not None
        self.dt = float(dt)
        if self.drift:
            missing = [k for k in DRIFT_CFG_KEYS if k not in drift]
            if missing:
                raise ValueError(f"odom drift: missing keys {missing}")
            d = {k: float(drift[k]) for k in DRIFT_CFG_KEYS}
            k, d2r = d["drift_scale"], math.pi / 180.0
            self.xy_dist, self.xy_time = k * d["drift_xy_dist"], k * d["drift_xy_time"]
            self.z_dist = k * d["drift_z_dist"]
            self.yaw_dist = k * d["drift_yaw_dist"] * d2r
            self.yaw_rot = k * d["drift_yaw_rot"] * math.sqrt(d2r)        # deg/sqrt(deg) -> rad/sqrt(rad)
            self.yaw_time = k * d["drift_yaw_time"] * d2r
            self.scale_bias_std = k * d["drift_scale_bias_std"] / 100.0
            self.yaw_bias_std = k * d["drift_yaw_bias_std"] * d2r
            self.max_xy, self.max_z, self.max_yaw = d["max_xy_dev"], d["max_z_dev"], d["max_yaw_dev"] * d2r
            self.init_xy, self.init_yaw = d["init_xy_dev"], d["init_yaw_dev"] * d2r
        self.est_pos = torch.zeros(n, 3, device=self.device)     # drifted estimate
        self.est_yaw = torch.zeros(n, device=self.device)
        self.prev_pos = torch.zeros(n, 3, device=self.device)    # true pose of the previous step
        self.prev_yaw = torch.zeros(n, device=self.device)
        self.b_scale = torch.zeros(n, device=self.device)        # per-episode biases
        self.b_yaw = torch.zeros(n, device=self.device)

    def describe(self) -> str:
        s = (f"odom delay on actor obs: update period {list(self.period_range)} steps, "
             f"latency {list(self.latency_range)} steps")
        if not self.drift:
            return s + ", drift off"
        r2d = 180.0 / math.pi
        return (s + f", drift[xy={self.xy_dist * 1e3:.1f}mm/sqrt(m)+{self.xy_time * 1e3:.1f}mm/sqrt(s) "
                f"z={self.z_dist * 1e3:.1f}mm/sqrt(m) yaw={self.yaw_dist * r2d:.3f}deg/sqrt(m)"
                f"+{self.yaw_rot / math.sqrt(math.pi / 180.0):.3f}deg/sqrt(deg)+{self.yaw_time * r2d:.3f}deg/sqrt(s) "
                f"bias_std=({self.scale_bias_std * 100:.2f}%, {self.yaw_bias_std * r2d:.3f}deg/m) "
                f"max=({self.max_xy:.2f}m, {self.max_z:.2f}m, {self.max_yaw * r2d:.1f}deg) "
                f"init=({self.init_xy:.2f}m, {self.init_yaw * r2d:.1f}deg)]")

    # ---------------------------------------------------------------- internals
    def _randint(self, lo_hi, size: int) -> torch.Tensor:
        return torch.randint(lo_hi[0], lo_hi[1] + 1, (size,), device=self.device, generator=self.gen)

    def _rand(self, *size) -> torch.Tensor:
        return torch.rand(*size, device=self.device, generator=self.gen)

    def _randn(self, *size) -> torch.Tensor:
        return torch.randn(*size, device=self.device, generator=self.gen)

    def _drift_step(self, pos: torch.Tensor, yaw: torch.Tensor):
        """Integrate one noisy step increment into the estimate and return the (clamped) estimate.

        Envs reset this step get a meaningless increment (teleport) here; the env re-initialises them
        with ``init`` right after, so it is overwritten."""
        n = self.num_envs
        dyaw = _wrap(yaw - self.prev_yaw)
        dxy = _rotz2(-(self.prev_yaw + 0.5 * dyaw), pos[:, :2] - self.prev_pos[:, :2])  # heading frame
        dz = pos[:, 2] - self.prev_pos[:, 2]
        ds = torch.linalg.norm(dxy, dim=-1)

        std_xy = torch.sqrt(self.xy_dist ** 2 * ds + self.xy_time ** 2 * self.dt)
        dxy_n = (1.0 + self.b_scale).unsqueeze(-1) * dxy + self._randn(n, 2) * std_xy.unsqueeze(-1)
        dz_n = dz + self._randn(n) * self.z_dist * torch.sqrt(ds)
        std_yaw = torch.sqrt(self.yaw_dist ** 2 * ds + self.yaw_rot ** 2 * dyaw.abs() + self.yaw_time ** 2 * self.dt)
        dyaw_n = dyaw + self.b_yaw * ds + self._randn(n) * std_yaw

        self.est_pos[:, :2] += _rotz2(self.est_yaw + 0.5 * dyaw_n, dxy_n)   # yaw error bends the path
        self.est_pos[:, 2] += dz_n
        self.est_yaw = _wrap(self.est_yaw + dyaw_n)
        self.prev_pos.copy_(pos)
        self.prev_yaw.copy_(yaw)

        # bound the error (estimate - truth)
        e_xy = self.est_pos[:, :2] - pos[:, :2]
        if self.max_xy > 0:
            norm = torch.linalg.norm(e_xy, dim=-1, keepdim=True).clamp_min(1e-9)
            e_xy = e_xy * (self.max_xy / norm).clamp(max=1.0)
        e_z = self.est_pos[:, 2] - pos[:, 2]
        if self.max_z > 0:
            e_z = e_z.clamp(-self.max_z, self.max_z)
        e_yaw = _wrap(self.est_yaw - yaw)
        if self.max_yaw > 0:
            e_yaw = e_yaw.clamp(-self.max_yaw, self.max_yaw)
        self.est_pos[:, :2] = pos[:, :2] + e_xy
        self.est_pos[:, 2] = pos[:, 2] + e_z
        self.est_yaw = _wrap(yaw + e_yaw)
        return self.est_pos, self.est_yaw

    # ---------------------------------------------------------------- api
    def init(self, ids: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor):
        """(Re)start ``ids`` with fresh odometry. root_pos [n, 3] / root_quat [n, 4] wxyz of those ids."""
        n = int(ids.numel())
        if n == 0:
            return
        pos = root_pos.to(self.hist_pos.dtype)
        yaw = quat_to_rpy(root_quat)[:, 2].to(self.hist_yaw.dtype)
        est_pos, est_yaw = pos.clone(), yaw.clone()
        if self.drift:
            self.prev_pos[ids] = pos
            self.prev_yaw[ids] = yaw
            self.b_scale[ids] = self._randn(n) * self.scale_bias_std
            self.b_yaw[ids] = self._randn(n) * self.yaw_bias_std
            if self.init_xy > 0:     # uniform in a disc
                r = self.init_xy * torch.sqrt(self._rand(n))
                a = 2.0 * math.pi * self._rand(n)
                est_pos[:, 0] += r * torch.cos(a)
                est_pos[:, 1] += r * torch.sin(a)
            if self.init_yaw > 0:
                est_yaw = _wrap(est_yaw + (2.0 * self._rand(n) - 1.0) * self.init_yaw)
            self.est_pos[ids] = est_pos
            self.est_yaw[ids] = est_yaw
        self.hist_pos[ids] = est_pos.unsqueeze(1)
        self.hist_yaw[ids] = est_yaw.unsqueeze(1)
        self.held_pos[ids] = est_pos
        self.held_yaw[ids] = est_yaw
        # random phase so that envs do not receive odometry on the same step
        self.countdown[ids] = self._randint((1, self.period_range[1]), n)
        self.latency[ids] = self._randint(self.latency_range, n)

    def step(self, root_pos: torch.Tensor, root_quat: torch.Tensor):
        """Advance one control step with the true pelvis pose of every env ([N, 3], [N, 4] wxyz)."""
        pos = root_pos.to(self.hist_pos.dtype)
        yaw = quat_to_rpy(root_quat)[:, 2].to(self.hist_yaw.dtype)
        if self.drift:
            pos, yaw = self._drift_step(pos, yaw)
        self.slot = (self.slot + 1) % self.H
        self.hist_pos[:, self.slot] = pos
        self.hist_yaw[:, self.slot] = yaw

        self.countdown -= 1
        ids = (self.countdown <= 0).nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            read = (self.slot - self.latency[ids]) % self.H
            self.held_pos[ids] = self.hist_pos[ids, read]
            self.held_yaw[ids] = self.hist_yaw[ids, read]
            self.countdown[ids] = self._randint(self.period_range, ids.numel())
            self.latency[ids] = self._randint(self.latency_range, ids.numel())

    def pelvis_pose(self, root_quat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimated pelvis pose: odom xyz + rpy(current roll/pitch, odom yaw). -> ([N, 3], [N, 4] wxyz)."""
        rpy = quat_to_rpy(root_quat)
        quat = rpy_to_quat(torch.stack((rpy[:, 0], rpy[:, 1], self.held_yaw.to(rpy.dtype)), dim=-1))
        return self.held_pos.clone(), quat


def drift_kwargs(odom_cfg) -> Optional[dict]:
    """OdomDelayCfg -> ``drift`` argument of OdomDelayBuffer (None when drift is off)."""
    if not bool(getattr(odom_cfg, "drift_enabled", False)):
        return None
    return {k: getattr(odom_cfg, k) for k in DRIFT_CFG_KEYS}


def reattach_body_pos(body_pos: torch.Tensor, true_pos: torch.Tensor, true_quat: torch.Tensor,
                      est_pos: torch.Tensor, est_quat: torch.Tensor) -> torch.Tensor:
    """FK with the estimated pelvis pose and the current joints: move body positions [N, B, 3]
    rigidly from the true pelvis pose to the estimated one."""
    B = body_pos.shape[1]
    tq = true_quat[:, None, :].expand(-1, B, 4)
    eq = est_quat[:, None, :].expand(-1, B, 4)
    local = quat_apply(quat_conj(tq), body_pos - true_pos[:, None, :])
    return quat_apply(eq, local) + est_pos[:, None, :]
