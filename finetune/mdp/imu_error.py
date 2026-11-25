"""On-board IMU / AHRS error model for the actor observations.

Port of DEX_RL_LAB ``dex_rl_lab/mdp/observations/imu_error.py`` (same equations / defaults), advanced
once per control step:

    gyro bias        b      <- b + N(0, rw * sqrt(dt))                          (turn-on bias ~ N(0, gyro_bias))
    attitude error   dth    <- dth + R_true (b dt + N(0, gyro_noise sqrt(dt / imu_rate)))   (world-frame rotvec)
    accel tilt ref   ref    =  [atan2(a_y, g + a_z), -atan2(a_x, g + a_z)],     a = accel_tilt * d(v_w)/dt
    AHRS correction  dth_rp <- dth_rp + (1 - exp(-dt / tau)) (ref - dth_rp)     (roll/pitch only, yaw drifts)

    IMU orientation  q_imu  =  q(dth) * q_true * q_misalign(roll, pitch)
    gyro             w_imu  =  w_true + b + N(0, gyro_noise)

Only roll / pitch of the IMU orientation are used by the obs (the yaw comes from the odometry).
Per episode: a DR scale k ~ U(scale_range) multiplies gyro_noise / gyro_bias / gyro_bias_rw / accel_tilt /
misalign, and tau ~ U(ahrs_tau_range).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from finetune.helper.torch_math import quat_apply, quat_from_rotvec, quat_mul, rpy_to_quat

GRAVITY = 9.81

# ImuErrorCfg fields of the error model (tasks/ft_cfg.py), same names as DEX_RL_LAB IMU_ERROR_CFG_KEYS
IMU_ERROR_CFG_KEYS = ("scale_range", "gyro_noise", "gyro_bias", "gyro_bias_rw", "ahrs_tau_range",
                      "accel_tilt", "misalign_deg", "imu_rate", "init_err_deg", "max_err_deg")


class ImuErrorBuffer:
    def __init__(self, num_envs: int, cfg: dict, device: torch.device, dt: float = 0.02,
                 seed: Optional[int] = None):
        """``cfg``: dict with the ``IMU_ERROR_CFG_KEYS`` of ImuErrorCfg. ``dt``: control (policy) step [s]."""
        missing = [k for k in IMU_ERROR_CFG_KEYS if k not in cfg]
        if missing:
            raise ValueError(f"imu error: missing keys {missing}")
        self.scale_range = (float(cfg["scale_range"][0]), float(cfg["scale_range"][1]))
        self.tau_range = (float(cfg["ahrs_tau_range"][0]), float(cfg["ahrs_tau_range"][1]))
        self.gyro_noise = float(cfg["gyro_noise"])
        self.gyro_bias = float(cfg["gyro_bias"])
        self.gyro_bias_rw = float(cfg["gyro_bias_rw"])
        self.accel_tilt = float(cfg["accel_tilt"])
        self.misalign = math.radians(float(cfg["misalign_deg"]))
        self.imu_rate = float(cfg["imu_rate"])
        self.init_err = math.radians(float(cfg["init_err_deg"]))
        self.max_err = math.radians(float(cfg["max_err_deg"]))
        self.dt = float(dt)
        self.num_envs = n = int(num_envs)
        self.device = torch.device(device)

        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(int(seed))

        z = lambda *s: torch.zeros(n, *s, device=self.device)  # noqa: E731
        self.k = z()                 # per-episode DR scale
        self.tau = z()               # per-episode AHRS time constant [s]
        self.bias = z(3)             # gyro bias [rad/s] (body frame)
        self.noise = z(3)            # gyro white noise of the current step [rad/s]
        self.dth = z(3)              # AHRS attitude error, world-frame rotvec [rad]
        self.q_mis = z(4)            # mounting misalignment (body frame)
        self.q_mis[:, 0] = 1.0
        self.prev_vel = z(3)

    def describe(self) -> str:
        r2d = 180.0 / math.pi
        return (f"imu error on actor obs: scale U{self.scale_range} x [gyro noise {self.gyro_noise} rad/s, "
                f"bias {self.gyro_bias} rad/s, bias rw {self.gyro_bias_rw} rad/s/sqrt(s), "
                f"accel tilt {self.accel_tilt}, misalign {self.misalign * r2d:.2f} deg], "
                f"tau U{self.tau_range} s, init +-{self.init_err * r2d:.1f} deg, max {self.max_err * r2d:.1f} deg")

    # ---------------------------------------------------------------- internals
    def _uniform(self, lo_hi, m: int) -> torch.Tensor:
        lo, hi = lo_hi
        return lo + (hi - lo) * torch.rand(m, device=self.device, generator=self.gen)

    def _randn(self, *size) -> torch.Tensor:
        return torch.randn(*size, device=self.device, generator=self.gen)

    # ---------------------------------------------------------------- api
    def init(self, ids: torch.Tensor, root_lin_vel_w: torch.Tensor):
        """(Re)start ``ids`` with a fresh IMU error. root_lin_vel_w [n, 3]: world velocity of those ids."""
        m = int(ids.numel())
        if m == 0:
            return
        k = self._uniform(self.scale_range, m)
        self.k[ids] = k
        self.tau[ids] = self._uniform(self.tau_range, m)
        self.bias[ids] = self._randn(m, 3) * (self.gyro_bias * k).unsqueeze(-1)
        self.noise[ids] = 0.0
        dth = torch.zeros(m, 3, device=self.device)
        if self.init_err > 0:
            dth[:, :2] = (2.0 * torch.rand(m, 2, device=self.device, generator=self.gen) - 1.0) * self.init_err
        self.dth[ids] = dth
        mis = self._randn(m, 2) * (self.misalign * k).unsqueeze(-1)
        self.q_mis[ids] = rpy_to_quat(torch.cat((mis, torch.zeros(m, 1, device=self.device)), dim=-1))
        self.prev_vel[ids] = root_lin_vel_w.to(self.prev_vel.dtype)

    def step(self, root_quat: torch.Tensor, root_lin_vel_w: torch.Tensor):
        """Advance one control step with the true pelvis orientation [N, 4] wxyz / world velocity [N, 3].
        Envs reset this step get a meaningless update here; ``init`` overwrites them right after."""
        n, dt = self.num_envs, self.dt
        k = self.k.unsqueeze(-1)
        if self.gyro_bias_rw > 0:
            self.bias += self._randn(n, 3) * (self.gyro_bias_rw * math.sqrt(dt)) * k
        # integrated gyro error (bias + angle random walk of the internal integration), world frame
        w_err = self.bias * dt
        if self.gyro_noise > 0:
            w_err = w_err + self._randn(n, 3) * (self.gyro_noise * math.sqrt(dt / self.imu_rate)) * k
            self.noise = self._randn(n, 3) * self.gyro_noise * k
        self.dth += quat_apply(root_quat.to(w_err.dtype), w_err)
        # accelerometer tilt reference: the AHRS aligns the specific force with "up"
        vel = root_lin_vel_w.to(self.prev_vel.dtype)
        acc = (vel - self.prev_vel) / dt * (self.accel_tilt * k)
        self.prev_vel = vel.clone()
        fz = GRAVITY + acc[:, 2]
        ref = torch.stack((torch.atan2(acc[:, 1], fz), -torch.atan2(acc[:, 0], fz)), dim=-1)
        alpha = (1.0 - torch.exp(-dt / self.tau.clamp_min(1e-6))).unsqueeze(-1)
        self.dth[:, :2] += alpha * (ref - self.dth[:, :2])
        if self.max_err > 0:
            self.dth[:, :2] = self.dth[:, :2].clamp(-self.max_err, self.max_err)

    def pelvis_quat(self, root_quat: torch.Tensor) -> torch.Tensor:
        """AHRS orientation of the pelvis (IMU rigidly attached): q(dth) * q_true * q_misalign. [N, 4] wxyz."""
        q = root_quat.to(self.dth.dtype)
        return quat_mul(quat_mul(quat_from_rotvec(self.dth), q), self.q_mis)

    def gyro_error(self) -> torch.Tensor:
        """Additive gyro error [N, 3] (bias + white noise), body frame."""
        return self.bias + self.noise


def imu_error_kwargs(imu_cfg) -> dict:
    """ImuErrorCfg -> ``cfg`` argument of ImuErrorBuffer."""
    return {k: getattr(imu_cfg, k) for k in IMU_ERROR_CFG_KEYS}
