"""FAST-LIO emulator for the MuJoCo stand-in (pose level, no point clouds)."""
from __future__ import annotations

from collections import deque

import numpy as np

from odom_estimator import pose_inv, pose_mul, wrap, yaw_quat
from ros_odom import OdomMsg
from utils.math import quat_mul, yaw_from_quat


def _rotvec_quat(v):
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = np.asarray(v) / a
    return np.concatenate([[np.cos(0.5 * a)], np.sin(0.5 * a) * ax])


def _rotz2(yaw, v):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


DRIFT_DEFAULTS = dict(
    xy_dist=0.01,      # [m/sqrt(m)]     xy random walk per travelled distance  (100 m -> 10 cm, 1 sigma)
    xy_time=0.001,     # [m/sqrt(s)]     xy random walk while standing          (60 s  -> 0.8 cm)
    z_dist=0.005,      # [m/sqrt(m)]     z random walk per travelled distance   (100 m -> 5 cm)
    yaw_dist=0.05,     # [deg/sqrt(m)]   yaw random walk per travelled distance (100 m -> 0.5 deg)
    yaw_rot=0.03,      # [deg/sqrt(deg)] yaw random walk per rotated angle      (360 deg -> 0.57 deg)
    yaw_time=0.01,     # [deg/sqrt(s)]   yaw random walk while standing         (60 s  -> 0.08 deg)
    scale_bias=0.5,    # [%]             std of the distance scale bias (sampled once per launch)
    yaw_bias=0.01,     # [deg/m]         std of the heading bias        (sampled once per launch)
)


def parse_dropouts(spec):
    out = []
    for item in filter(None, (s.strip() for s in (spec or "").split(","))):
        t0, dur = item.split(":")
        out.append((float(t0), float(t0) + float(dur)))
    return out


class FastLioEmulator:
    def __init__(self, env, site="livox_imu", rate=10.0, grav=True, latency=0.0, latency_std=0.0,
                 pos_std=0.0, rot_std=0.0, drift_xy_dist=0.0, drift_xy_time=0.0, drift_z_dist=0.0,
                 drift_yaw_dist=0.0, drift_yaw_rot=0.0, drift_yaw_time=0.0, scale_bias_std=0.0,
                 yaw_bias_std=0.0, jump_prob=0.0, jump_mag=0.0, dropouts="", seed=0):
        self.env = env
        self.site_id = env.model.site(site).id
        self.period = 1.0 / float(rate)
        self.grav = bool(grav)
        self.latency, self.latency_std = float(latency), float(latency_std)
        self.pos_std, self.rot_std = float(pos_std), float(rot_std)
        self.xy_dist, self.xy_time, self.z_dist = float(drift_xy_dist), float(drift_xy_time), float(drift_z_dist)
        self.yaw_dist, self.yaw_rot, self.yaw_time = float(drift_yaw_dist), float(drift_yaw_rot), float(drift_yaw_time)
        self.scale_bias_std, self.yaw_bias_std = float(scale_bias_std), float(yaw_bias_std)
        self.jump_prob, self.jump_mag = float(jump_prob), float(jump_mag)
        self.dropouts = parse_dropouts(dropouts)
        self.rng = np.random.default_rng(seed)
        self.world_from_W = None
        self._next_scan = None
        self._queue = deque()          # in flight: (arrival, OdomMsg)
        self._latest = None
        self._seq = 0
        self._prev = None              # last true (p, yaw) in W
        self._est_p = np.zeros(3)      # drifted estimate (p, yaw) in W
        self._est_yaw = 0.0
        self._b_scale = self._b_yaw = 0.0
        self.n_scans = self.n_dropped = self.n_jumps = 0

    def describe(self):
        return (f"[lio] site={self.env.model.site(self.site_id).name} rate={1 / self.period:.0f}Hz grav={self.grav} "
                f"latency={self.latency * 1e3:.0f}ms(+-{self.latency_std * 1e3:.0f}) pos_std={self.pos_std * 1e3:.1f}mm "
                f"rot_std={np.rad2deg(self.rot_std):.2f}deg "
                f"drift[xy={self.xy_dist * 1e3:.1f}mm/sqrt(m)+{self.xy_time * 1e3:.1f}mm/sqrt(s) "
                f"z={self.z_dist * 1e3:.1f}mm/sqrt(m) yaw={np.rad2deg(self.yaw_dist):.3f}deg/sqrt(m)"
                f"+{self.yaw_rot / np.sqrt(np.deg2rad(1.0)):.3f}deg/sqrt(deg)"
                f"+{np.rad2deg(self.yaw_time):.3f}deg/sqrt(s) "
                f"bias_std=({self.scale_bias_std * 100:.2f}%, {np.rad2deg(self.yaw_bias_std):.3f}deg/m)] "
                f"jump=({self.jump_prob}, {self.jump_mag}m) "
                f"dropouts={self.dropouts}")

    def _site_pose(self):
        d = self.env.data
        q = np.empty(4)
        import mujoco
        mujoco.mju_mat2Quat(q, d.site_xmat[self.site_id])
        return d.site_xpos[self.site_id].copy(), q

    def launch(self):
        """Start FAST-LIO: the current sensor pose becomes the odometry frame W."""
        p, q = self._site_pose()
        self.world_from_W = (p, yaw_quat(yaw_from_quat(q)) if self.grav else q)
        self._next_scan = self.env.now() + self.period
        self._prev = None
        self._b_scale = self.rng.normal(0.0, self.scale_bias_std) if self.scale_bias_std > 0 else 0.0
        self._b_yaw = self.rng.normal(0.0, self.yaw_bias_std) if self.yaw_bias_std > 0 else 0.0
        print(f"[lio] drift bias: scale={self._b_scale * 100:+.2f}% yaw={np.rad2deg(self._b_yaw):+.4f}deg/m")
        print(f"[lio] launched at t={self.env.now():.2f}s; W origin {np.round(p, 3)}")

    def update(self):
        """Called after every physics advance (control tick)."""
        if self.world_from_W is None:
            return
        now = self.env.now()
        while self._next_scan is not None and self._next_scan <= now + 1e-9:
            self._scan(self._next_scan)
            self._next_scan += self.period
        while self._queue and self._queue[0][0] <= now + 1e-9:
            self._latest = self._queue.popleft()[1]

    def _drift_step(self, p_t, yaw_t):
        """Integrate one noisy scan-to-scan increment (x, y, z, yaw) into the drifted estimate."""
        if self._prev is None:
            self._est_p, self._est_yaw = p_t.copy(), yaw_t
            self._prev = (p_t.copy(), yaw_t)
            return
        p0, yaw0 = self._prev
        dyaw = wrap(yaw_t - yaw0)
        dp = _rotz2(-(yaw0 + 0.5 * dyaw), p_t[:2] - p0[:2])      # true increment, heading frame
        dz = p_t[2] - p0[2]
        ds, dt, r = float(np.linalg.norm(dp)), self.period, self.rng
        dp_n = (1.0 + self._b_scale) * dp + r.normal(0.0, np.sqrt(self.xy_dist ** 2 * ds + self.xy_time ** 2 * dt), 2)
        dz_n = dz + r.normal(0.0, self.z_dist * np.sqrt(ds))
        dyaw_n = dyaw + self._b_yaw * ds + r.normal(
            0.0, np.sqrt(self.yaw_dist ** 2 * ds + self.yaw_rot ** 2 * abs(dyaw) + self.yaw_time ** 2 * dt))
        self._est_p[:2] += _rotz2(self._est_yaw + 0.5 * dyaw_n, dp_n)   # yaw error bends the path
        self._est_p[2] += dz_n
        self._est_yaw += dyaw_n
        self._prev = (p_t.copy(), yaw_t)

    def _scan(self, stamp):
        p_t, q_t = pose_mul(pose_inv(self.world_from_W), self._site_pose())
        yaw_t = yaw_from_quat(q_t)
        self._drift_step(p_t, yaw_t)   # keeps integrating through dropouts
        if self.jump_prob > 0 and self.rng.random() < self.jump_prob:
            a = self.rng.uniform(0, 2 * np.pi)
            self._est_p[:2] += self.jump_mag * np.array([np.cos(a), np.sin(a)])
            self.n_jumps += 1
        if any(t0 <= stamp < t1 for t0, t1 in self.dropouts):
            self.n_dropped += 1
            return
        # drifted position; heading replaced by the drifted yaw, roll/pitch kept (gravity-observable)
        p = self._est_p.copy()
        q = quat_mul(yaw_quat(self._est_yaw - yaw_t), q_t)
        if self.pos_std > 0:
            p = p + self.rng.normal(0.0, self.pos_std, 3)
        if self.rot_std > 0:
            q = quat_mul(q, _rotvec_quat(self.rng.normal(0.0, self.rot_std, 3)))
        lat = max(0.0, self.latency + (self.rng.normal(0.0, self.latency_std) if self.latency_std > 0 else 0.0))
        self._seq += 1
        arrival = stamp + lat
        msg = OdomMsg(self._seq, float(stamp), p, q / np.linalg.norm(q), float(arrival))
        # keep arrival order monotonic (a late message never overtakes a newer one)
        if self._queue and arrival < self._queue[-1][0]:
            arrival = self._queue[-1][0]
            msg = msg._replace(recv_time=arrival)
        self._queue.append((arrival, msg))
        self.n_scans += 1

    def get_latest(self):
        return self._latest

    def close(self):
        pass
