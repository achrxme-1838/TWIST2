from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from finetune.helper.torch_math import (
    quat_conj, quat_mul, quat_slerp, quat_to_exp_map, quat_to_rpy, quat_rotate_inverse, wxyz_to_xyzw,
)
from finetune.motion_data.motion_ref import MotionSet


@dataclass
class BatchedMotionFrame:
    root_pos: torch.Tensor        # [N, 3] world
    root_quat: torch.Tensor       # [N, 4] wxyz
    root_lin_vel: torch.Tensor    # [N, 3] world
    root_ang_vel: torch.Tensor    # [N, 3] world
    dof_pos: torch.Tensor         # [N, J] SDK order
    dof_vel: torch.Tensor         # [N, J] SDK order


@dataclass
class BatchedRefBodyState:
    pos: torch.Tensor             # [N, B_ext, 3] tracked + extended, world
    quat: torch.Tensor            # [N, B_ext, 4] wxyz
    lin_vel: torch.Tensor         # [N, B, 3] tracked only, world
    ang_vel: torch.Tensor         # [N, B, 3]
    lin_vel_ext: torch.Tensor     # [N, B_ext, 3]
    ang_vel_ext: torch.Tensor


class BatchedRefFK:
    """Reference-pose FK for ``num_envs`` poses at once, on its own collision-free warp Data."""

    def __init__(self, mjm: mujoco.MjModel, mw, num_envs: int, tracked_body_ids: torch.Tensor,
                 extended_parent_ids: torch.Tensor, extended_local_offsets: torch.Tensor,
                 num_actions: int, device: str = "cuda:0"):
        self.mw = mw
        self.wp_device = str(device)
        self.num_actions = num_actions
        self.tracked_body_ids = tracked_body_ids
        self.extended_parent_ids = extended_parent_ids
        self.extended_local_offsets = extended_local_offsets
        self.num_tracked = len(tracked_body_ids)
        mjd = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, mjd)
        # kinematics only: no contacts, no constraints
        with wp.ScopedDevice(self.wp_device):
            self.dw = mjwarp.put_data(mjm, mjd, nworld=num_envs, njmax=1, naconmax=1)
        self._qpos = wp.to_torch(self.dw.qpos)
        self._xpos = wp.to_torch(self.dw.xpos)
        self._xquat = wp.to_torch(self.dw.xquat)

    def body_pose(self, root_pos: torch.Tensor, root_quat_wxyz: torch.Tensor,
                  dof_pos_sdk: torch.Tensor):
        """([N,B_ext,3], [N,B_ext,4]) -- same body order as the diff_body_* observation terms."""
        from finetune.helper.torch_math import quat_apply
        J = self.num_actions
        self._qpos[:, 0:3] = root_pos
        self._qpos[:, 3:7] = root_quat_wxyz
        self._qpos[:, 7:7 + J] = dof_pos_sdk
        mjwarp.kinematics(self.mw, self.dw)
        parent_pos = self._xpos[:, self.extended_parent_ids]
        parent_quat = self._xquat[:, self.extended_parent_ids]
        ext_pos = parent_pos + quat_apply(parent_quat, self.extended_local_offsets.expand_as(parent_pos))
        pos = torch.cat([self._xpos[:, self.tracked_body_ids], ext_pos], dim=1)
        quat = torch.cat([self._xquat[:, self.tracked_body_ids], parent_quat], dim=1)
        return pos.clone(), quat.clone()


class BatchedMotionSet:
    """Flat GPU tables over all motions + batched queries mirroring MotionSet."""

    def __init__(self, cpu_set: MotionSet, fk: BatchedRefFK, device: torch.device):
        self.cpu = cpu_set
        self.fk = fk
        self.device = device
        self.num_actions = cpu_set.num_actions
        ms = cpu_set.motions
        self.num_motions = len(ms)
        self.names = [m.name for m in ms]

        f = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
        i = lambda x: torch.as_tensor(np.asarray(x, dtype=np.int64), device=device)
        self.offset = i(np.cumsum([0] + [m.num_frames for m in ms])[:-1])
        self.nframes = i([m.num_frames for m in ms])
        self.dt = f([m.dt for m in ms])
        self.lengths = f([m.length for m in ms])
        cat = lambda attr: f(np.concatenate([getattr(m, attr) for m in ms], axis=0))
        self.root_pos = cat("root_pos")
        self.root_quat = cat("root_quat")
        self.root_lin_vel = cat("root_lin_vel")
        self.root_ang_vel = cat("root_ang_vel")
        self.dof_pos = cat("dof_pos")
        self.dof_vel = cat("dof_vel")

    # ----- sampling (kept on the CPU rng so runs stay reproducible against the CPU env) -----
    def sample_motion(self, rng: np.random.Generator, n: int) -> torch.Tensor:
        return torch.as_tensor(rng.integers(self.num_motions, size=n), dtype=torch.long, device=self.device)

    def sample_start_time(self, mid: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        mode = self.cpu.cfg.start_time_mode
        if mode == "zero":
            return torch.zeros(mid.shape[0], device=self.device)
        if mode == "random":
            u = torch.as_tensor(rng.random(mid.shape[0]), dtype=torch.float32, device=self.device)
            return u * self.lengths[mid]
        raise ValueError(f"unknown start_time_mode {mode}")

    def length(self, mid: torch.Tensor) -> torch.Tensor:
        return self.lengths[mid]

    # ----- queries -----
    def _blend(self, mid: torch.Tensor, t: torch.Tensor):
        """MotionLib._calc_frame_blend: flat indices of the two frames and the blend weight."""
        L = self.lengths[mid]
        nf = self.nframes[mid]
        phase = torch.where(L > 0, (t / L.clamp_min(1e-12)).clamp(0.0, 1.0), torch.zeros_like(t))
        x = phase * (nf - 1).to(t.dtype)
        f0 = x.long()          # phase >= 0, so trunc == floor (matches int() on the CPU)
        f1 = torch.minimum(f0 + 1, nf - 1)
        b = x - f0.to(x.dtype)
        base = self.offset[mid]
        return base + f0, base + f1, b

    def frame(self, mid: torch.Tensor, t: torch.Tensor) -> BatchedMotionFrame:
        i0, i1, b = self._blend(mid, t)
        w = b[:, None]
        lerp = lambda T: (1.0 - w) * T[i0] + w * T[i1]
        return BatchedMotionFrame(
            root_pos=lerp(self.root_pos),
            root_quat=quat_slerp(self.root_quat[i0], self.root_quat[i1], b),
            root_lin_vel=lerp(self.root_lin_vel),
            root_ang_vel=lerp(self.root_ang_vel),
            dof_pos=lerp(self.dof_pos),
            dof_vel=lerp(self.dof_vel),
        )

    def mimic_obs(self, fr: BatchedMotionFrame) -> torch.Tensor:
        """Deploy build_mimic_obs layout: [xy_vel_local(2), z(1), rpy(3), ang_vel_local(3), dof(J)]."""
        rpy = quat_to_rpy(fr.root_quat)
        v_l = quat_rotate_inverse(fr.root_quat, fr.root_lin_vel)
        w_l = quat_rotate_inverse(fr.root_quat, fr.root_ang_vel)
        return torch.cat([v_l[:, :2], fr.root_pos[:, 2:3], rpy, w_l, fr.dof_pos], dim=-1)

    def future_raw(self, mid: torch.Tensor, t: torch.Tensor, step_offsets: Sequence[int],
                   control_dt: float):
        """(pos [N,T,3], rot_xyzw [N,T,4], dof [N,T,J], lin_vel [N,T,3], ang_vel [N,T,3])."""
        off = torch.as_tensor(np.asarray(step_offsets, dtype=np.float32), device=self.device)
        ts = t[:, None] + off[None, :] * control_dt                     # [N, T]
        T = ts.shape[1]
        mid_f = mid[:, None].expand(-1, T).reshape(-1)
        i0, i1, b = self._blend(mid_f, ts.reshape(-1))
        w = b[:, None]
        lerp = lambda M: ((1.0 - w) * M[i0] + w * M[i1]).view(ts.shape[0], T, -1)
        rot = quat_slerp(self.root_quat[i0], self.root_quat[i1], b).view(ts.shape[0], T, 4)
        return (lerp(self.root_pos), wxyz_to_xyzw(rot), lerp(self.dof_pos),
                lerp(self.root_lin_vel), lerp(self.root_ang_vel))

    def body_state(self, mid: torch.Tensor, t: torch.Tensor) -> BatchedRefBodyState:
        """Extended-body poses at ``t``; tracked-body velocities by central FK differences over
        one motion frame (forward/backward at the motion ends). Three FK calls, as on the CPU."""
        L = self.lengths[mid]
        h = self.dt[mid]
        t = t.clamp(min=torch.zeros_like(L), max=L)
        fr = self.frame(mid, t)
        pos, quat = self.fk.body_pose(fr.root_pos, fr.root_quat, fr.dof_pos)

        t0 = (t - h).clamp_min(0.0)
        t1 = torch.minimum(t + h, L)
        dt = (t1 - t0)[:, None, None]
        fa, fb = self.frame(mid, t0), self.frame(mid, t1)
        pa, qa = self.fk.body_pose(fa.root_pos, fa.root_quat, fa.dof_pos)
        pb, qb = self.fk.body_pose(fb.root_pos, fb.root_quat, fb.dof_pos)
        safe = dt.clamp_min(1e-9)
        lin_ext = torch.where(dt > 1e-9, (pb - pa) / safe, torch.zeros_like(pos))
        ang_ext = torch.where(dt > 1e-9, quat_to_exp_map(quat_mul(qb, quat_conj(qa))) / safe,
                              torch.zeros_like(pos))
        n = self.fk.num_tracked
        return BatchedRefBodyState(pos=pos, quat=quat, lin_vel=lin_ext[:, :n], ang_vel=ang_ext[:, :n],
                                   lin_vel_ext=lin_ext, ang_vel_ext=ang_ext)
