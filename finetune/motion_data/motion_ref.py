"""In-process reference motion source for the fine-tuning env."""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import List, Optional, Sequence

import mujoco
import numpy as np

from finetune.tasks.ft_cfg import MotionCfg
from finetune.helper.math_utils import (
    quat_rotate_inverse, quat_slerp, quat_to_exp_map, quat_to_rpy, so3_angular_velocity,
    wxyz_to_xyzw, xyzw_to_wxyz,
)
from observations import _drive_ref_data_pose, compute_extended_body_pos_w, compute_extended_body_quat_w
from utils.math import quat_conj, quat_mul


@dataclass
class MotionFrame:
    root_pos: np.ndarray        # [3] world
    root_quat: np.ndarray       # [4] wxyz
    root_lin_vel: np.ndarray    # [3] world
    root_ang_vel: np.ndarray    # [3] world
    dof_pos: np.ndarray         # [J] SDK order
    dof_vel: np.ndarray         # [J] SDK order


@dataclass
class RefBodyState:
    pos: np.ndarray             # [B_ext, 3] tracked + extended bodies, world
    quat: np.ndarray            # [B_ext, 4] wxyz
    lin_vel: np.ndarray         # [B, 3] tracked bodies only, world
    ang_vel: np.ndarray         # [B, 3] tracked bodies only, world
    lin_vel_ext: np.ndarray = None   # [B_ext, 3] tracked + extended (DEX body_lin_vel_extend)
    ang_vel_ext: np.ndarray = None   # [B_ext, 3]


class RefFK:
    """FK on a private MjData: reference frame -> extended body poses (same body order
    as the diff_body_* observation terms)."""

    def __init__(self, model, tracked_body_ids, extended_parent_ids, extended_local_offsets, num_actions):
        self.model = model
        self.data = mujoco.MjData(model)
        self.tracked_body_ids = np.asarray(tracked_body_ids)
        self.extended_parent_ids = np.asarray(extended_parent_ids)
        self.extended_local_offsets = np.asarray(extended_local_offsets, dtype=np.float64)
        self.num_actions = num_actions
        self.num_tracked = len(self.tracked_body_ids)

    def body_pose(self, root_pos, root_quat_wxyz, dof_pos):
        _drive_ref_data_pose(self.model, self.data, root_pos, root_quat_wxyz, dof_pos, self.num_actions)
        pos = compute_extended_body_pos_w(self.data, self.tracked_body_ids, self.extended_parent_ids,
                                          self.extended_local_offsets)
        quat = compute_extended_body_quat_w(self.data, self.tracked_body_ids, self.extended_parent_ids)
        return pos.copy(), quat.copy()


def _load_pickle(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        import joblib
        return joblib.load(path)


def _detect_format(entry) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    if "root_pos" in entry and "root_rot" in entry and "dof_pos" in entry:
        return "GMR"
    if "root_trans" in entry and "root_ori" in entry and "dof_pos" in entry:
        return "PHUMA"
    return None


class _Motion:
    __slots__ = ("name", "fps", "dt", "num_frames", "length", "root_pos", "root_quat",
                 "root_lin_vel", "root_ang_vel", "dof_pos", "dof_vel")

    def __init__(self, name: str, entry: dict, fmt: str):
        self.name = name
        self.fps = float(entry["fps"])
        self.dt = 1.0 / self.fps
        pos_key, rot_key = ("root_trans", "root_ori") if fmt == "PHUMA" else ("root_pos", "root_rot")
        root_pos = np.asarray(entry[pos_key], dtype=np.float64).copy()
        root_rot_xyzw = np.asarray(entry[rot_key], dtype=np.float64)
        dof = np.asarray(entry["dof_pos"], dtype=np.float64)
        if dof.ndim == 3 and dof.shape[-1] == 1:
            dof = dof[..., 0]
        # MotionLib / IsaacLab loaders: trajectory starts at the world xy origin.
        root_pos[:, :2] -= root_pos[0:1, :2]
        self.num_frames = root_pos.shape[0]
        self.length = self.dt * (self.num_frames - 1)
        self.root_pos = root_pos
        self.root_quat = xyzw_to_wxyz(root_rot_xyzw)
        self.root_quat /= np.linalg.norm(self.root_quat, axis=-1, keepdims=True)
        self.dof_pos = dof
        if self.num_frames > 1:
            self.root_lin_vel = np.gradient(root_pos, self.dt, axis=0)
            self.dof_vel = np.gradient(dof, self.dt, axis=0)
            self.root_ang_vel = so3_angular_velocity(self.root_quat, self.dt)
        else:
            self.root_lin_vel = np.zeros_like(root_pos)
            self.dof_vel = np.zeros_like(dof)
            self.root_ang_vel = np.zeros((1, 3))


class MotionSet:
    def __init__(self, path: str, cfg: MotionCfg, fk: RefFK, num_actions: int, verbose: bool = True):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        self.cfg = cfg
        self.fk = fk
        self.num_actions = num_actions
        data = _load_pickle(path)
        fmt = _detect_format(data)
        entries = {os.path.splitext(os.path.basename(path))[0]: data} if fmt else dict(data)

        self.motions: List[_Motion] = []
        for name, entry in entries.items():
            if cfg.name_filter and not any(s in str(name) for s in cfg.name_filter):
                continue
            fmt = _detect_format(entry)
            if fmt is None:
                if verbose:
                    print(f"[MotionSet] skip {name}: unknown format (keys={list(entry)[:6] if isinstance(entry, dict) else type(entry)})")
                continue
            m = _Motion(str(name), entry, fmt)
            if m.dof_pos.shape[1] != num_actions:
                raise ValueError(f"{name}: dof_pos has {m.dof_pos.shape[1]} joints, expected {num_actions}")
            if m.num_frames < 2:
                continue
            self.motions.append(m)
            if cfg.max_motions and len(self.motions) >= cfg.max_motions:
                break
        if not self.motions:
            raise ValueError(f"no motion loaded from {path} (filter={cfg.name_filter})")
        if cfg.align_to_ground:
            self._align_to_ground()
        self.lengths = np.array([m.length for m in self.motions])
        if verbose:
            print(f"[MotionSet] {len(self.motions)} motions, total {self.lengths.sum():.1f}s "
                  f"(min {self.lengths.min():.1f}s, max {self.lengths.max():.1f}s) from {os.path.basename(path)}")

    # ----- loading helpers -----
    def _align_to_ground(self):
        """IsaacLab align_motion_to_ground: lowest tracked-body point -> z = -0.02."""
        for m in self.motions:
            min_z = np.inf
            for f in range(m.num_frames):
                pos, _ = self.fk.body_pose(m.root_pos[f], m.root_quat[f], m.dof_pos[f])
                min_z = min(min_z, float(pos[: self.fk.num_tracked, 2].min()))
            m.root_pos[:, 2] -= (min_z + 0.02)

    # ----- queries -----
    @property
    def num_motions(self) -> int:
        return len(self.motions)

    @property
    def names(self) -> List[str]:
        return [m.name for m in self.motions]

    def length(self, mid: int) -> float:
        return self.motions[mid].length

    def sample_motion(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.num_motions))

    def sample_start_time(self, mid: int, rng: np.random.Generator) -> float:
        if self.cfg.start_time_mode == "random":
            return float(rng.random() * self.motions[mid].length)
        if self.cfg.start_time_mode == "zero":
            return 0.0
        raise ValueError(f"unknown start_time_mode {self.cfg.start_time_mode}")

    def frame(self, mid: int, t: float) -> MotionFrame:
        """Reference frame at motion time ``t`` (clamped to the motion; no looping).
        Frame blending mirrors MotionLib._calc_frame_blend; velocities are blended too."""
        m = self.motions[mid]
        phase = float(np.clip(t / m.length, 0.0, 1.0)) if m.length > 0 else 0.0
        f0 = int(phase * (m.num_frames - 1))
        f1 = min(f0 + 1, m.num_frames - 1)
        b = phase * (m.num_frames - 1) - f0
        lerp = lambda a: (1.0 - b) * a[f0] + b * a[f1]
        return MotionFrame(
            root_pos=lerp(m.root_pos),
            root_quat=quat_slerp(m.root_quat[f0], m.root_quat[f1], b),
            root_lin_vel=lerp(m.root_lin_vel),
            root_ang_vel=lerp(m.root_ang_vel),
            dof_pos=lerp(m.dof_pos),
            dof_vel=lerp(m.dof_vel),
        )

    @staticmethod
    def mimic_obs(fr: MotionFrame) -> np.ndarray:
        """Deploy ``build_mimic_obs`` 38-dim layout:
        [xy_vel_local(2), z(1), roll, pitch, yaw, ang_vel_local(3), dof(J)]."""
        rpy = quat_to_rpy(fr.root_quat)
        v_l = quat_rotate_inverse(fr.root_quat, fr.root_lin_vel)
        w_l = quat_rotate_inverse(fr.root_quat, fr.root_ang_vel)
        return np.concatenate([v_l[:2], fr.root_pos[2:3], rpy, w_l, fr.dof_pos]).astype(np.float32)

    def future_raw(self, mid: int, t: float, step_offsets: Sequence[int], control_dt: float):
        """Future frames at ``t + k*control_dt`` for k in step_offsets, in the layout
        ``parse_future_raw`` returns: (pos [T,3], rot_xyzw [T,4], dof [T,J], lin_vel [T,3], ang_vel [T,3]).
        Vectorised version of ``frame`` (same blending)."""
        m = self.motions[mid]
        ts = t + np.asarray(step_offsets, dtype=np.float64) * control_dt
        phase = np.clip(ts / m.length, 0.0, 1.0) if m.length > 0 else np.zeros_like(ts)
        f0 = (phase * (m.num_frames - 1)).astype(np.int64)
        f1 = np.minimum(f0 + 1, m.num_frames - 1)
        b = (phase * (m.num_frames - 1) - f0)[:, None]
        lerp = lambda a: (1.0 - b) * a[f0] + b * a[f1]
        pos = lerp(m.root_pos)
        rot = wxyz_to_xyzw(quat_slerp(m.root_quat[f0], m.root_quat[f1], b))
        return pos, rot, lerp(m.dof_pos), lerp(m.root_lin_vel), lerp(m.root_ang_vel)

    def body_state(self, mid: int, t: float) -> RefBodyState:
        """Extended-body poses at ``t`` and tracked-body velocities by central FK
        differences over one motion frame (forward/backward at the motion ends)."""
        m = self.motions[mid]
        h = m.dt
        t = float(np.clip(t, 0.0, m.length))
        fr = self.frame(mid, t)
        pos, quat = self.fk.body_pose(fr.root_pos, fr.root_quat, fr.dof_pos)
        t0 = max(0.0, t - h)
        t1 = min(m.length, t + h)
        n = self.fk.num_tracked
        if t1 - t0 < 1e-9:
            z = np.zeros((pos.shape[0], 3))
            return RefBodyState(pos, quat, z[:n], z[:n], z, z)
        fa = self.frame(mid, t0)
        fb = self.frame(mid, t1)
        pa, qa = self.fk.body_pose(fa.root_pos, fa.root_quat, fa.dof_pos)
        pb, qb = self.fk.body_pose(fb.root_pos, fb.root_quat, fb.dof_pos)
        lin_vel_ext = (pb - pa) / (t1 - t0)
        ang_vel_ext = quat_to_exp_map(quat_mul(qb, quat_conj(qa))) / (t1 - t0)
        return RefBodyState(pos, quat, lin_vel_ext[:n], ang_vel_ext[:n], lin_vel_ext, ang_vel_ext)
