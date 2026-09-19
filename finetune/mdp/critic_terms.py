"""Privileged observation terms for the critic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from finetune.helper.math_utils import projected_gravity_b, quat_rotate_inverse
from finetune.motion_data.motion_ref import MotionFrame, RefBodyState
from obs_terms import ObsContext, term
from finetune.env.sim_core import RobotState
from utils.math import heading_quat_from_quat, quat_apply, quat_conj, quat_mul, quat_to_rot6d


@dataclass
class TrainObsContext(ObsContext):
    robot: Optional[RobotState] = None          # post-step simulator state
    ref: Optional[RefBodyState] = None          # reference bodies at the *current* motion time
    ref_frame: Optional[MotionFrame] = None     # reference root / dof at the current motion time
    motion_phase: float = 0.0                   # t / motion length
    time_left: float = 0.0                      # seconds until the motion ends
    feet_body_idx: tuple = (6, 12)              # TRACKED_BODY_NAMES indices of left/right ankle_roll_link
    _pcache: dict = field(default_factory=dict, repr=False)

    def _req(self):
        if self.robot is None or self.ref is None or self.ref_frame is None:
            raise RuntimeError("priv_* terms need TrainObsContext.robot / ref / ref_frame (critic spec only).")


def _n_tracked(s) -> int:
    return len(s.tracked_body_ids)


@term("priv_root_lin_vel_b", dim=lambda s, p: 3)
def priv_root_lin_vel_b(c: TrainObsContext, params: dict) -> np.ndarray:
    c._req()
    return quat_rotate_inverse(c.robot.root_quat, c.robot.root_lin_vel_w).astype(np.float32)


@term("priv_root_height", dim=lambda s, p: 1)
def priv_root_height(c: TrainObsContext, params: dict) -> np.ndarray:
    c._req()
    return np.asarray(c.robot.root_pos[2:3], dtype=np.float32)


@term("priv_projected_gravity", dim=lambda s, p: 3)
def priv_projected_gravity(c: TrainObsContext, params: dict) -> np.ndarray:
    c._req()
    return projected_gravity_b(c.robot.root_quat).astype(np.float32)


@term("priv_ref_root_pos_diff_b", dim=lambda s, p: 3)
def priv_ref_root_pos_diff_b(c: TrainObsContext, params: dict) -> np.ndarray:
    """(ref root - robot root) in the robot base frame, current reference frame."""
    c._req()
    return quat_rotate_inverse(c.robot.root_quat, c.ref_frame.root_pos - c.robot.root_pos).astype(np.float32)


@term("priv_ref_root_rot_diff", dim=lambda s, p: 6)
def priv_ref_root_rot_diff(c: TrainObsContext, params: dict) -> np.ndarray:
    """rot6d of heading-corrected base->ref-root rotation (future_motion_anchor style)."""
    c._req()
    base = c.robot.root_quat
    planar = heading_quat_from_quat(base)
    rel = quat_mul(c.ref_frame.root_quat, quat_conj(base))
    rel_hc = quat_mul(quat_mul(quat_conj(planar), rel), planar)
    return quat_to_rot6d(rel_hc).astype(np.float32)


@term("priv_ref_root_lin_vel_b", dim=lambda s, p: 3)
def priv_ref_root_lin_vel_b(c: TrainObsContext, params: dict) -> np.ndarray:
    c._req()
    return quat_rotate_inverse(c.robot.root_quat, c.ref_frame.root_lin_vel).astype(np.float32)


@term("priv_ref_root_ang_vel_b", dim=lambda s, p: 3)
def priv_ref_root_ang_vel_b(c: TrainObsContext, params: dict) -> np.ndarray:
    c._req()
    return quat_rotate_inverse(c.robot.root_quat, c.ref_frame.root_ang_vel).astype(np.float32)


@term("priv_body_lin_vel_diff_b", dim=lambda s, p: _n_tracked(s) * 3)
def priv_body_lin_vel_diff_b(c: TrainObsContext, params: dict) -> np.ndarray:
    c._req()
    n = c.ref.lin_vel.shape[0]
    diff = c.ref.lin_vel - c.robot.body_lin_vel[:n]
    return quat_apply(np.broadcast_to(quat_conj(c.robot.root_quat), (n, 4)), diff).astype(np.float32).reshape(-1)


@term("priv_body_ang_vel_diff_b", dim=lambda s, p: _n_tracked(s) * 3)
def priv_body_ang_vel_diff_b(c: TrainObsContext, params: dict) -> np.ndarray:
    c._req()
    n = c.ref.ang_vel.shape[0]
    diff = c.ref.ang_vel - c.robot.body_ang_vel[:n]
    return quat_apply(np.broadcast_to(quat_conj(c.robot.root_quat), (n, 4)), diff).astype(np.float32).reshape(-1)


@term("priv_body_pos_diff_b", dim=lambda s, p: s.num_bodies * 3)
def priv_body_pos_diff_b(c: TrainObsContext, params: dict) -> np.ndarray:
    """(ref - robot) extended-body positions at the *current* motion time (the actor's
    diff_body_pos_b is built from the mimic frame ``mimic_step_offset`` steps ahead)."""
    c._req()
    diff = c.ref.pos - c.robot.body_pos
    B = diff.shape[0]
    return quat_apply(np.broadcast_to(quat_conj(c.robot.root_quat), (B, 4)), diff).astype(np.float32).reshape(-1)


@term("priv_feet_contact", dim=lambda s, p: 2)
def priv_feet_contact(c: TrainObsContext, params: dict) -> np.ndarray:
    c._req()
    thr = float(params.get("threshold", 1.0))
    ids = c.static.tracked_body_ids[list(c.feet_body_idx)]
    return (c.robot.contact_force[ids] > thr).astype(np.float32)


@term("priv_joint_torque", dim=lambda s, p: s.num_actions)
def priv_joint_torque(c: TrainObsContext, params: dict) -> np.ndarray:
    c._req()
    scale = float(params.get("scale", 0.01))
    return (c.robot.torque_sdk * scale).astype(np.float32)


@term("priv_motion_phase", dim=lambda s, p: 2)
def priv_motion_phase(c: TrainObsContext, params: dict) -> np.ndarray:
    c._req()
    return np.array([c.motion_phase, min(c.time_left, 5.0) / 5.0], dtype=np.float32)
