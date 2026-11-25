"""LIO odometry (livox IMU frame) -> robot root in the global frame G.

livox IMU -> IMU in pelvis:
    T^W_I = T^W_LI * (T^P_T(theta) * T^T_L * T^L_LI)^-1 * T^P_I

Origin reset(A):
    T^W_G = (x_X, y_X, z_X - h_X ; yaw(R^W_P)),  X = pelvis or IMU (--origin_ref)
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import mujoco
import numpy as np

from utils.math import quat_apply, quat_conj, quat_mul, rpy_to_quat, yaw_from_quat

MID360_IN_TORSO = (0.0002835, 0.00003, 0.41618, 0.0, 0.04014257279586953, 0.0)
LIVOX_IMU_IN_MID360 = (0.011, 0.02329, -0.04412, 0.0, 0.0, 0.0)
IMU_IN_PELVIS = (0.04525, 0.0, -0.08339, 0.0, 0.0, 0.0)
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
ODOM_FRAMES = ("livox_imu", "lidar", "pelvis")
ORIGIN_REFS = ("pelvis", "imu")


# --------------------------------------------------------------------------- pose helpers
def pose(xyzrpy):
    v = [float(x) for x in xyzrpy]
    return np.array(v[:3], dtype=np.float64), rpy_to_quat(*v[3:6])


def pose_mul(a, b):
    return a[0] + quat_apply(a[1], b[0]), quat_mul(a[1], b[1])


def pose_inv(a):
    qi = quat_conj(np.asarray(a[1], dtype=np.float64))
    return -quat_apply(qi, np.asarray(a[0], dtype=np.float64)), qi


def yaw_quat(yaw):
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)


def wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def roll_pitch(q):
    """ZYX roll / pitch of a wxyz quat (independent of its yaw)."""
    w, x, y, z = (float(v) for v in q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


def gravity_error(q1, q2):
    """Angle between world-up vectors expressed in the pelvis (yaw independent)."""
    up = np.array([0.0, 0.0, 1.0])
    directions = []
    for q in (q1, q2):
        q = np.asarray(q, dtype=np.float64)
        norm = np.linalg.norm(q)
        if not np.isfinite(norm) or norm < 1e-8:
            return np.inf
        directions.append(quat_apply(quat_conj(q / norm), up))
    return float(np.arccos(np.clip(np.dot(*directions), -1.0, 1.0)))


IDENTITY = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))


# --------------------------------------------------------------------------- kinematics
class FrameChain:
    """Fixed transforms + FK (own MjData) of the robot for the odometry chain."""

    def __init__(self, model, odom_frame="livox_imu", mid360_in_torso=MID360_IN_TORSO,
                 livox_imu_in_mid360=LIVOX_IMU_IN_MID360, imu_in_pelvis=IMU_IN_PELVIS):
        if odom_frame not in ODOM_FRAMES:
            raise ValueError(f"odom_frame must be one of {ODOM_FRAMES}")
        self.model = model
        self.data = mujoco.MjData(model)
        self.num_actions = model.nq - 7
        self.odom_frame = odom_frame
        self.torso_id = model.body("torso_link").id
        T_T_L = pose(mid360_in_torso)
        T_L_LI = pose(livox_imu_in_mid360)
        self.T_T_S = {"livox_imu": pose_mul(T_T_L, T_L_LI), "lidar": T_T_L, "pelvis": None}[odom_frame]
        self.T_P_I = pose(imu_in_pelvis)
        self.foot_geoms = []
        for name in FOOT_BODIES:
            bid = model.body(name).id
            g = [i for i in range(model.ngeom)
                 if model.geom_bodyid[i] == bid and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE]
            if not g:
                raise ValueError(f"no sphere contact geoms on {name}")
            self.foot_geoms.append(np.array(g, dtype=np.int64))

    def _fk(self, dof_pos, root_quat=(1.0, 0.0, 0.0, 0.0)):
        d = self.data
        d.qpos[:3] = 0.0
        d.qpos[3:7] = root_quat
        d.qpos[7:7 + self.num_actions] = dof_pos
        mujoco.mj_kinematics(self.model, d)

    def sensor_in_pelvis(self, dof_pos):
        """T^P_S(theta)."""
        if self.T_T_S is None:
            return IDENTITY
        self._fk(dof_pos)
        T_P_T = (self.data.xpos[self.torso_id].copy(), self.data.xquat[self.torso_id].copy())
        return pose_mul(T_P_T, self.T_T_S)

    def pelvis_from_sensor(self, T_W_S, dof_pos):
        """T^W_P = T^W_S * (T^P_S)^-1."""
        return pose_mul(T_W_S, pose_inv(self.sensor_in_pelvis(dof_pos)))

    def point(self, T_W_P, which):
        """Reference point X (pelvis or IMU) of a pelvis pose."""
        return T_W_P if which == "pelvis" else pose_mul(T_W_P, self.T_P_I)

    def point_offset_xy(self, root_quat, which):
        """xy of (X - pelvis) in the parent frame, for a pelvis orientation."""
        if which == "pelvis":
            return np.zeros(2)
        return quat_apply(np.asarray(root_quat, dtype=np.float64), self.T_P_I[0])[:2]

    def height_above_feet(self, dof_pos, roll, pitch, which):
        """(h, foot_dz): height of X above the lowest foot contact along gravity (pelvis
        tilted by the IMU roll/pitch), and the height difference of the two feet."""
        q = rpy_to_quat(roll, pitch, 0.0)
        self._fk(dof_pos, q)
        d = self.data
        lows = [float(np.min(d.geom_xpos[g, 2] - self.model.geom_size[g, 0])) for g in self.foot_geoms]
        x_z = 0.0 if which == "pelvis" else float(quat_apply(q, self.T_P_I[0])[2])
        return x_z - min(lows), abs(lows[0] - lows[1])


# --------------------------------------------------------------------------- estimator
class OdomEstimator:
    def __init__(self, chain: FrameChain, origin_ref="pelvis", timeout=0.3, ma_len=1,
                 sync="stamp", sync_tol=0.1, v_max=4.0, yaw_rate_max=8.0,
                 origin_window=0.5, origin_min_samples=3, origin_pos_std=0.01,
                 origin_yaw_std=np.deg2rad(1.0), feet_dz_max=0.02, grav_warn=np.deg2rad(2.0),
                 origin_strict=False, grav_reject=np.deg2rad(30.0)):
        if origin_ref not in ORIGIN_REFS:
            raise ValueError(f"origin_ref must be one of {ORIGIN_REFS}")
        self.chain = chain
        self.origin_ref = origin_ref
        self.timeout = float(timeout)
        self.sync = sync
        self.sync_tol = float(sync_tol)
        self.v_max = float(v_max)
        self.yaw_rate_max = float(yaw_rate_max)
        self.origin_window = float(origin_window)
        self.origin_min_samples = int(origin_min_samples)
        self.origin_pos_std = float(origin_pos_std)
        self.origin_yaw_std = float(origin_yaw_std)
        self.feet_dz_max = float(feet_dz_max)
        self.grav_warn = float(grav_warn)
        self.grav_reject = float(grav_reject)
        self.origin_strict = bool(origin_strict)

        self._joints = deque(maxlen=200)              # (t, dof_pos)
        self._pos_ma = deque(maxlen=max(1, int(ma_len)))
        self._window = deque(maxlen=200)              # (recv_time, T^W_P raw)
        self.last_seq = None
        self.stamp = None
        self.recv_time = None
        self.T_W_S = None
        self.T_W_P_raw = None
        self.T_W_P = None            # filtered (MA position, raw rotation)
        self.T_W_G = None
        self.origin_count = 0
        self.n_msgs = 0
        self.sync_miss = 0
        self.jump = None             # reason string of the last implausible step

    # ----- inputs -----
    def push_joints(self, t, dof_pos):
        self._joints.append((float(t), np.asarray(dof_pos, dtype=np.float64).copy()))

    def _joints_at(self, stamp):
        if not self._joints:
            return None
        if self.sync == "stamp":
            ts = np.fromiter((j[0] for j in self._joints), dtype=np.float64)
            k = int(np.argmin(np.abs(ts - stamp)))
            if abs(ts[k] - stamp) <= self.sync_tol:
                return self._joints[k][1]
            self.sync_miss += 1
        return self._joints[-1][1]

    def update(self, msg) -> bool:
        """Consume the latest odometry message (seq/stamp/pos/quat/recv_time). True if new."""
        if msg is None or msg.seq == self.last_seq:
            return False
        dof = self._joints_at(msg.stamp)
        if dof is None:
            return False
        self.last_seq = msg.seq
        T_W_S = (np.asarray(msg.pos, dtype=np.float64), np.asarray(msg.quat, dtype=np.float64))
        T_W_P = self.chain.pelvis_from_sensor(T_W_S, dof)
        prev, prev_stamp = self.T_W_P_raw, self.stamp
        self.jump = None
        if prev is not None and prev_stamp is not None and msg.stamp > prev_stamp:
            dt = msg.stamp - prev_stamp
            v = np.linalg.norm(T_W_P[0] - prev[0]) / dt
            w = abs(wrap(yaw_from_quat(T_W_P[1]) - yaw_from_quat(prev[1]))) / dt
            if v > self.v_max:
                self.jump = f"odometry jump {v:.1f} m/s > {self.v_max:.1f}"
            elif w > self.yaw_rate_max:
                self.jump = f"odometry yaw jump {w:.1f} rad/s > {self.yaw_rate_max:.1f}"
        self.T_W_S, self.T_W_P_raw = T_W_S, T_W_P
        self.stamp, self.recv_time = float(msg.stamp), float(msg.recv_time)
        self._pos_ma.append(T_W_P[0].copy())
        self.T_W_P = (np.mean(self._pos_ma, axis=0), T_W_P[1].copy())
        self._window.append((self.recv_time, T_W_P))
        self.n_msgs += 1
        return True

    # ----- status -----
    def age(self, now):
        return np.inf if self.recv_time is None else float(now) - self.recv_time

    def fresh(self, now):
        return self.age(now) <= self.timeout

    @property
    def has_origin(self):
        return self.T_W_G is not None

    # ----- global origin (A button) -----
    def reset_origin(self, now, dof_pos, imu_quat):
        """Latch T^W_G from the last origin_window seconds. Returns (ok, message).
        Refused on stale odometry or a large pelvis/IMU gravity mismatch, before
        changing the origin. origin_strict also refuses too few samples, a moving
        robot or uneven feet -- otherwise those are warnings."""
        if not self.fresh(now):
            return False, f"odometry not fresh (age {self.age(now) * 1e3:.0f} ms)"
        warns = []
        samples = [s for s in self._window if s[0] >= now - self.origin_window]
        if len(samples) < self.origin_min_samples:
            if self.origin_strict or not self._window:
                return False, f"only {len(samples)} odometry samples in {self.origin_window:.2f}s"
            samples = samples or [self._window[-1]]
            warns.append(f"{len(samples)} samples")
        which = self.origin_ref
        pts = [self.chain.point(T, which) for _, T in samples]
        P = np.stack([p[0] for p in pts])
        yaws = np.array([yaw_from_quat(T[1]) for _, T in samples])
        yaw_mean = float(np.arctan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws))))
        yaw_dev = np.array([wrap(y - yaw_mean) for y in yaws])
        pos_std, yaw_std = float(np.max(P.std(axis=0))), float(yaw_dev.std())
        if pos_std > self.origin_pos_std or yaw_std > self.origin_yaw_std:
            w = f"robot not static (pos std {pos_std * 1e3:.1f} mm, yaw std {np.rad2deg(yaw_std):.2f} deg)"
            if self.origin_strict:
                return False, w
            warns.append(w)
        imu_roll, imu_pitch = roll_pitch(imu_quat)
        # Test the converted PELVIS orientation, not the raw sensor orientation:
        # a correctly configured upside-down sensor must still yield upright pelvis.
        odom_quat = samples[-1][1][1]
        o_roll, o_pitch = roll_pitch(odom_quat)
        g_err = gravity_error(odom_quat, imu_quat)
        gravity_msg = (f"odom pelvis vs IMU gravity differ by {np.rad2deg(g_err):.1f} deg "
                       f"(odom r/p {np.rad2deg(o_roll):+.1f}/{np.rad2deg(o_pitch):+.1f}, "
                       f"IMU r/p {np.rad2deg(imu_roll):+.1f}/{np.rad2deg(imu_pitch):+.1f} deg)")
        if not np.isfinite(g_err) or g_err > self.grav_reject:
            return False, (f"{gravity_msg}; check the odometry child pose (--odom_frame), "
                           "sensor mounting extrinsics and world gravity alignment before starting the policy")
        h, foot_dz = self.chain.height_above_feet(dof_pos, imu_roll, imu_pitch, which)
        if foot_dz > self.feet_dz_max:
            w = f"feet not level ({foot_dz * 1e3:.0f} mm > {self.feet_dz_max * 1e3:.0f} mm)"
            if self.origin_strict:
                return False, w
            warns.append(w)
        p = P.mean(axis=0)
        self.T_W_G = (np.array([p[0], p[1], p[2] - h]), yaw_quat(yaw_mean))
        self.origin_count += 1
        msg = (f"origin #{self.origin_count} ({which}): W xyz [{p[0]:+.3f} {p[1]:+.3f} {p[2] - h:+.3f}] "
               f"yaw {np.rad2deg(yaw_mean):+.1f} deg, {which} height {h:.3f} m, {len(samples)} samples")
        if g_err > self.grav_warn:
            msg += f"  [WARN] {gravity_msg}; check the odometry frame and mounting extrinsics"
        if warns:
            msg += "  [WARN] " + "; ".join(warns)
        self._pos_ma.clear()
        if self.T_W_P_raw is not None:
            self._pos_ma.append(self.T_W_P_raw[0].copy())
        return True, msg

    # ----- outputs in G -----
    def pelvis_in_G(self):
        if self.T_W_G is None or self.T_W_P is None:
            return None
        return pose_mul(pose_inv(self.T_W_G), self.T_W_P)

    def root_in_G(self, imu_quat):
        """Root pose for the obs: position + yaw from odometry, roll/pitch from the IMU."""
        T = self.pelvis_in_G()
        if T is None:
            return None
        r, p = roll_pitch(imu_quat)
        return T[0].copy(), rpy_to_quat(r, p, yaw_from_quat(T[1]))

    def point_in_G(self):
        """(xy, yaw) of the reference point X in G (B-button deviation check)."""
        T = self.pelvis_in_G()
        if T is None:
            return None
        X = self.chain.point(T, self.origin_ref)
        return X[0][:2].copy(), yaw_from_quat(X[1])
