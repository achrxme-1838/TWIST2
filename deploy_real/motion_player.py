"""In-process reference motion playback (replaces server_motion_lib.py + Redis).

Steps through a motion precomputed by export_motion_ref.py with the controller's own
policy-step counter, reproducing the motion server's IDLE -> BLEND -> MOTION -> RETURN
sequence and its Redis payloads (action_body / future_motion_* / ref_root_world /
motion_epoch / motion_phase) without any IPC or wall-clock pacing. numpy only, so no
torch / CUDA is needed on the robot.
"""
import json
from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np

IDLE, BLEND, MOTION, RETURN = "IDLE", "BLEND", "MOTION", "RETURN"


class MotionSample(NamedTuple):
    action_mimic: np.ndarray        # mimic target (action_body_*), float32
    future_raw: Optional[Tuple]     # parse_future_raw() layout, or None (stationary fallback)
    ref_root: Optional[np.ndarray]  # reference root xyz (ref_root_world_*), or None
    epoch: Optional[int]            # per-motion epoch (motion_epoch_*), or None
    phase: str                      # IDLE / BLEND / MOTION / RETURN


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def rotate_z_xyzw(q, delta):
    """Rotate xyzw quaternions about world z by delta (as server_motion_lib.build_mimic_obs)."""
    cw, sz = np.cos(0.5 * delta), np.sin(0.5 * delta)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([cw * x - sz * y, sz * x + cw * y, cw * z + sz * w, cw * w - sz * z], axis=-1)


def rotate_z_xy(v, delta):
    """Rotate the xy components of [..., 3] vectors about world z by delta."""
    c, s = np.cos(delta), np.sin(delta)
    out = v.copy()
    out[..., 0] = c * v[..., 0] - s * v[..., 1]
    out[..., 1] = s * v[..., 0] + c * v[..., 1]
    return out


class LocalMotionPlayer:
    """Plays one precomputed motion. Call ``start(robot_yaw)`` once (may be repeated
    after ``done``), then ``step()`` exactly once per policy step."""

    _FUTURE_KEYS = ("future_root_pos", "future_root_rot", "future_dof_pos",
                    "future_root_lin_vel", "future_root_ang_vel")

    def __init__(self, npz_path: str, default_mimic_obs, control_dt: Optional[float] = None,
                 future_motion_steps: Optional[Sequence[int]] = None,
                 blend_in_time: Optional[float] = None, return_time: Optional[float] = None):
        with np.load(npz_path, allow_pickle=False) as d:
            self.meta = json.loads(str(d["meta"]))
            self._mimic = d["mimic_obs"].astype(np.float64)
            self._ref_root = d["ref_root"].astype(np.float64)
            self._future = [d[k].astype(np.float64) for k in self._FUTURE_KEYS]
        m = self.meta
        self.control_dt = float(m["control_dt"])
        if control_dt is not None and abs(control_dt - self.control_dt) > 1e-6:
            raise ValueError(f"{npz_path}: exported at control_dt={self.control_dt}, "
                             f"controller runs at {control_dt}")
        if future_motion_steps is not None and list(future_motion_steps) != list(m["future_motion_steps"]):
            raise ValueError(f"{npz_path}: future steps {m['future_motion_steps']} != policy spec "
                             f"{list(future_motion_steps)} -- re-export with this checkpoint's --obs_cfg")

        self.num_steps = len(self._mimic)
        bt = m["blend_in_time"] if blend_in_time is None else blend_in_time
        self.n_blend = max(1, int(bt / self.control_dt)) if bt > 0 else 0
        rt = m["return_time"] if return_time is None else return_time
        self.n_return = int(rt / self.control_dt)
        self._default = np.asarray(default_mimic_obs, dtype=np.float64)
        self._yaw_cols = [j * m["frame_dim"] + m["yaw_index"] for j in range(len(m["steps"]))]

        self.started = False
        self.done = False
        self.delta = None
        self._phase = IDLE
        self._i = 0
        self._epoch = 0
        self._idle_obs = self._default
        # Like Redis keys: none before the first motion, last value kept afterwards.
        self._ref_pub = None
        self._epoch_pub = None

    def start(self, robot_yaw: float):
        """Anchor motion frame 0's yaw to the robot heading and begin blend-in."""
        m = self.meta
        self.delta = float(robot_yaw - m["motion_yaw_0"]) if m["align_heading"] else 0.0
        d = self.delta

        mimic = self._mimic.copy()
        mimic[:, self._yaw_cols] = _wrap(mimic[:, self._yaw_cols] + d)
        self._mimic_rot = mimic.astype(np.float32)
        pos, rot, dof, lin, ang = self._future
        if m["fix_root_pos"]:   # xy pinned to the frame-0 reference after the rotation
            self._ref_rot, fut_pos = self._ref_root, pos
        else:
            self._ref_rot, fut_pos = rotate_z_xy(self._ref_root, d), rotate_z_xy(pos, d)
        self._future_rot = (fut_pos, rotate_z_xyzw(rot, d), dof, rotate_z_xy(lin, d), rotate_z_xy(ang, d))

        target0 = mimic[0]
        end_obs = target0 if m["send_start_frame_as_end_frame"] else self._default
        self._blend_start = end_obs
        self._blend_target = target0
        self._return_target = end_obs
        self._last_obs = self._blend_start

        self._epoch += 1
        self.started = True
        self.done = False
        self._phase = BLEND if self.n_blend > 0 else MOTION
        self._i = 0
        print(f"[motion_player] start: robot_yaw={robot_yaw:.4f} motion_yaw_0={m['motion_yaw_0']:.4f} "
              f"delta={d:.4f}; blend {self.n_blend} + motion {self.num_steps} + return {self.n_return} steps")

    def _sample(self, obs, phase, future=None):
        return MotionSample(np.asarray(obs, dtype=np.float32), future, self._ref_pub, self._epoch_pub, phase)

    def step(self) -> MotionSample:
        if self._phase == BLEND:
            obs = self._blend_start + (self._blend_target - self._blend_start) * (self._i / self.n_blend)
            s = self._sample(obs, BLEND)
            self._i += 1
            if self._i >= self.n_blend:
                self._phase, self._i = MOTION, 0
            return s

        if self._phase == MOTION:
            k = self._i
            self._ref_pub = self._ref_rot[k].copy()
            self._epoch_pub = self._epoch
            self._last_obs = self._mimic_rot[k].astype(np.float64)
            s = self._sample(self._mimic_rot[k], MOTION, tuple(a[k] for a in self._future_rot))
            self._i += 1
            if self._i >= self.num_steps:
                self._phase, self._i = RETURN, 0
            return s

        if self._phase == RETURN:
            if self._i < self.n_return:
                obs = self._last_obs + (self._return_target - self._last_obs) * (self._i / self.n_return)
                self._i += 1
                return self._sample(obs, RETURN)
            self._phase, self._i = IDLE, 0
            self._idle_obs = self._return_target
            self.done = True

        return self._sample(self._idle_obs, IDLE)
