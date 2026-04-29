"""Observation helpers for the deployed G1 policy.

Per-frame FK + ref/robot diff terms used to build PolicyCfg-style observations.
Mimic obs layout (38 dims): [xy_vel(2), z(1), roll(1), pitch(1), yaw(1),
ang_vel(3), dof_pos(29)].
"""
from __future__ import annotations

import numpy as np
import mujoco

from utils.math import (
    planar_rot_matrix,
    quat_apply,
    quat_conj,
    quat_mul,
    rpy_to_quat,
    yaw_from_quat,
)


def compute_extended_body_pos_w(data, tracked_body_ids, extended_parent_ids, extended_local_offsets):
    """[N+E, 3] world positions: tracked bodies + (parent_xpos + parent_R @ offset)."""
    tracked_pos = data.xpos[tracked_body_ids]
    parent_pos = data.xpos[extended_parent_ids]
    parent_mat = data.xmat[extended_parent_ids].reshape(-1, 3, 3)
    ext_pos = parent_pos + np.einsum("bij,bj->bi", parent_mat, extended_local_offsets)
    return np.concatenate([tracked_pos, ext_pos], axis=0)


def compute_extended_body_quat_w(data, tracked_body_ids, extended_parent_ids):
    """[N+E, 4] world quats (wxyz). Extended bodies inherit parent orientation."""
    tracked_quat = data.xquat[tracked_body_ids]
    parent_quat = data.xquat[extended_parent_ids]
    return np.concatenate([tracked_quat, parent_quat], axis=0)


def _drive_ref_data(model, ref_data, action_mimic, num_actions):
    """Fill ref MjData from mimic root pose + dof pos and run kinematics."""
    ref_z = float(action_mimic[2])
    ref_roll = float(action_mimic[3])
    ref_pitch = float(action_mimic[4])
    ref_yaw = float(action_mimic[5])
    ref_dof_pos = np.asarray(action_mimic[-num_actions:], dtype=np.float64)

    ref_data.qpos[:3] = (0.0, 0.0, ref_z)
    ref_data.qpos[3:7] = rpy_to_quat(ref_roll, ref_pitch, ref_yaw)
    ref_data.qpos[7:7 + num_actions] = ref_dof_pos
    mujoco.mj_kinematics(model, ref_data)


def compute_diff_body_pos_b(
    model, data, ref_data, action_mimic,
    tracked_body_ids, extended_parent_ids, extended_local_offsets,
    num_actions, use_pb=True,
):
    """Per-body world position diff (ref - robot) rotated into robot base frame.

    Each side's pelvis (index 0) is subtracted before diffing so root xy/z offsets
    cancel. use_pb=True -> planar (yaw-only) base frame (matches diff_body_pos_pb);
    use_pb=False -> full pelvis frame. Returns flat [(N+E)*3] float32.
    """
    _drive_ref_data(model, ref_data, action_mimic, num_actions)

    ref_pos_w = compute_extended_body_pos_w(
        ref_data, tracked_body_ids, extended_parent_ids, extended_local_offsets
    )
    robot_pos_w = compute_extended_body_pos_w(
        data, tracked_body_ids, extended_parent_ids, extended_local_offsets
    )
    diff_w = (ref_pos_w - ref_pos_w[0:1]) - (robot_pos_w - robot_pos_w[0:1])

    if use_pb:
        # v_pb = R_planar.T @ v_w == v_w @ R_planar.
        R = planar_rot_matrix(yaw_from_quat(data.qpos[3:7]))
    else:
        # xmat[pelvis] maps body->world; v_b = R.T @ v_w == v_w @ R.
        pelvis_id = int(tracked_body_ids[0])
        R = data.xmat[pelvis_id].reshape(3, 3)
    return (diff_w @ R).astype(np.float32).reshape(-1)


def compute_diff_body_tannorm_b(
    model, data, ref_data, action_mimic,
    tracked_body_ids, extended_parent_ids,
    num_actions, use_pb=True,
):
    """Per-body 6D tan-norm of ref/robot rotation diff in robot base frame.

        diff_q_w  = q_ref_w * conj(q_robot_w)
        diff_q_b  = conj(q_base) * diff_q_w * q_base
        tannorm   = [diff_q_b * (1,0,0), diff_q_b * (0,0,1)]   # 6D per body

    use_pb=True uses yaw-only base; use_pb=False uses full pelvis quat.
    Returns flat [(N+E)*6] float32.
    """
    _drive_ref_data(model, ref_data, action_mimic, num_actions)

    ref_quat_w = compute_extended_body_quat_w(ref_data, tracked_body_ids, extended_parent_ids)
    robot_quat_w = compute_extended_body_quat_w(data, tracked_body_ids, extended_parent_ids)
    diff_quat_w = quat_mul(ref_quat_w, quat_conj(robot_quat_w))

    if use_pb:
        yaw = yaw_from_quat(data.qpos[3:7])
        base_quat = np.array(
            [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64
        )
    else:
        base_quat = np.asarray(data.qpos[3:7], dtype=np.float64)
    inv_base_quat = quat_conj(base_quat)

    base_b = np.broadcast_to(base_quat, diff_quat_w.shape)
    inv_base_b = np.broadcast_to(inv_base_quat, diff_quat_w.shape)
    diff_quat_b = quat_mul(quat_mul(inv_base_b, diff_quat_w), base_b)

    ref_tan = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ref_norm = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    tan = quat_apply(diff_quat_b, ref_tan)
    norm = quat_apply(diff_quat_b, ref_norm)
    return np.concatenate([tan, norm], axis=-1).astype(np.float32).reshape(-1)
