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


# --------------------------------------------------------------------------- teacher critic terms
# Deploy implementations of the IsaacLab critic observation group of the DEX_RL_LAB teacher
# (g1_29dof_mapo: robot_observations.projected_gravity, mimic_observations.extended_body_*_h,
# diff_body_{lin,ang}_vel_pb). Needed when critic.source=teacher so the teacher's critic
# weights see the observation layout they were trained on. Conventions:
#   *_h  : heading frame  = calc_heading_quat_inv(root_quat)   (x-axis heading, yaw only)
#   *_pb : planar base    = planar_root_quat_w (euler-xyz yaw of the root, roll/pitch zeroed)

def _heading_inv(c: TrainObsContext) -> np.ndarray:
    return heading_quat_from_quat(c.robot.root_quat, inverse=True)


def _planar_inv(c: TrainObsContext) -> np.ndarray:
    yaw = float(c.rpy[2])
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, -np.sin(0.5 * yaw)], dtype=np.float64)


def _rotate_rows(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_apply(np.broadcast_to(q, (v.shape[0], 4)), v)


@term("projected_gravity", dim=lambda s, p: 3)
def projected_gravity(c: TrainObsContext, params: dict) -> np.ndarray:
    c._req()
    return projected_gravity_b(c.robot.root_quat).astype(np.float32)


@term("extended_body_pos_h", dim=lambda s, p: s.num_bodies * 3)
def extended_body_pos_h(c: TrainObsContext, params: dict) -> np.ndarray:
    c._req()
    rel = c.robot.body_pos - c.robot.root_pos[None, :]
    return _rotate_rows(_heading_inv(c), rel).astype(np.float32).reshape(-1)


@term("extended_body_quat_h", dim=lambda s, p: s.num_bodies * 4)
def extended_body_quat_h(c: TrainObsContext, params: dict) -> np.ndarray:
    c._req()
    q = c.robot.body_quat
    return quat_mul(np.broadcast_to(_heading_inv(c), q.shape), q).astype(np.float32).reshape(-1)


@term("extended_body_lin_vel_h", dim=lambda s, p: s.num_bodies * 3)
def extended_body_lin_vel_h(c: TrainObsContext, params: dict) -> np.ndarray:
    c._req()
    return _rotate_rows(_heading_inv(c), c.robot.body_lin_vel_ext).astype(np.float32).reshape(-1)


@term("extended_body_ang_vel_h", dim=lambda s, p: s.num_bodies * 3)
def extended_body_ang_vel_h(c: TrainObsContext, params: dict) -> np.ndarray:
    c._req()
    return _rotate_rows(_heading_inv(c), c.robot.body_ang_vel_ext).astype(np.float32).reshape(-1)


@term("diff_body_lin_vel_pb", dim=lambda s, p: s.num_bodies * 3)
def diff_body_lin_vel_pb(c: TrainObsContext, params: dict) -> np.ndarray:
    """(ref - robot) extended-body linear velocity in the planar base frame (current motion time)."""
    c._req()
    diff = c.ref.lin_vel_ext - c.robot.body_lin_vel_ext
    return _rotate_rows(_planar_inv(c), diff).astype(np.float32).reshape(-1)


@term("diff_body_ang_vel_pb", dim=lambda s, p: s.num_bodies * 3)
def diff_body_ang_vel_pb(c: TrainObsContext, params: dict) -> np.ndarray:
    c._req()
    diff = c.ref.ang_vel_ext - c.robot.body_ang_vel_ext
    return _rotate_rows(_planar_inv(c), diff).astype(np.float32).reshape(-1)
