"""Observation helpers for the deployed G1 policy.

Per-frame FK + ref/robot diff terms used to build PolicyCfg-style observations.
Mimic obs layout (38 dims): [xy_vel(2), z(1), roll(1), pitch(1), yaw(1),
ang_vel(3), dof_pos(29)].
"""
from __future__ import annotations

import json

import numpy as np
import mujoco

from utils.math import (
    heading_quat_from_quat,
    planar_rot_matrix,
    quat_apply,
    quat_conj,
    quat_mul,
    quat_to_rot6d,
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


def _drive_ref_data(model, ref_data, action_mimic, num_actions, ref_root_xy_w=None):
    """Fill ref MjData from mimic root pose + dof pos and run kinematics.

    ``ref_root_xy_w`` (odom mode) places the reference root at its world xy in the
    odom frame (origin anchored to the robot root at motion start). Default None
    keeps the legacy behaviour of pinning the ref root xy to the origin -- valid
    because the non-odom diff cancels root translation anyway.
    """
    ref_z = float(action_mimic[2])
    ref_roll = float(action_mimic[3])
    ref_pitch = float(action_mimic[4])
    ref_yaw = float(action_mimic[5])
    ref_dof_pos = np.asarray(action_mimic[-num_actions:], dtype=np.float64)

    if ref_root_xy_w is None:
        ref_data.qpos[:3] = (0.0, 0.0, ref_z)
    else:
        ref_data.qpos[0] = float(ref_root_xy_w[0])
        ref_data.qpos[1] = float(ref_root_xy_w[1])
        ref_data.qpos[2] = ref_z
    ref_data.qpos[3:7] = rpy_to_quat(ref_roll, ref_pitch, ref_yaw)
    ref_data.qpos[7:7 + num_actions] = ref_dof_pos
    mujoco.mj_kinematics(model, ref_data)


def compute_diff_body_pos_b(
    model, data, ref_data, action_mimic,
    tracked_body_ids, extended_parent_ids, extended_local_offsets,
    num_actions, use_pb=True,
    update_robot_w_odom=False, ref_root_xy_w=None,
):
    """Per-body world position diff (ref - robot) rotated into robot base frame.

    Default (update_robot_w_odom=False): each side's pelvis (index 0) is
    subtracted before diffing so root xy/z offsets cancel -- i.e. the motion is
    assumed glued to the robot root and root translation is ignored.

    update_robot_w_odom=True: the robot root pose comes from odometry (caller
    drives ``data`` with the odom root, sim GT or FAST-LIO) and the reference
    root is placed at ``ref_root_xy_w`` in the same (odom) world frame, anchored
    so motion frame 0 coincides with the robot root at playback start. The
    pelvis is NOT subtracted, so the diff captures the absolute root translation
    (and height) error.

    use_pb=True -> planar (yaw-only) base frame (matches diff_body_pos_pb);
    use_pb=False -> full pelvis frame. Returns flat [(N+E)*3] float32.
    """
    _drive_ref_data(
        model, ref_data, action_mimic, num_actions,
        ref_root_xy_w=ref_root_xy_w if update_robot_w_odom else None,
    )

    ref_pos_w = compute_extended_body_pos_w(
        ref_data, tracked_body_ids, extended_parent_ids, extended_local_offsets
    )
    robot_pos_w = compute_extended_body_pos_w(
        data, tracked_body_ids, extended_parent_ids, extended_local_offsets
    )
    if update_robot_w_odom:
        # World-frame diff incl. root translation (ref/robot share the odom frame).
        diff_w = ref_pos_w - robot_pos_w
    else:
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
    update_robot_w_odom=False, ref_root_xy_w=None,
):
    """Per-body 6D tan-norm of ref/robot rotation diff in robot base frame.

        diff_q_w  = q_ref_w * conj(q_robot_w)
        diff_q_b  = conj(q_base) * diff_q_w * q_base
        tannorm   = [diff_q_b * (1,0,0), diff_q_b * (0,0,1)]   # 6D per body

    use_pb=True uses yaw-only base; use_pb=False uses full pelvis quat.

    Rotations are translation-invariant, so update_robot_w_odom only matters here
    through the robot root *orientation*: in odom mode the caller drives ``data``
    with the odom yaw (roll/pitch still from IMU), so the base frame and robot
    body quats follow odom. ref_root_xy_w is accepted for signature symmetry; it
    does not affect body orientations. Returns flat [(N+E)*6] float32.
    """
    _drive_ref_data(
        model, ref_data, action_mimic, num_actions,
        ref_root_xy_w=ref_root_xy_w if update_robot_w_odom else None,
    )

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


def _drive_ref_data_pose(model, ref_data, root_pos_w, root_quat_wxyz, dof_pos, num_actions):
    """Place ref MjData at an explicit world root pose (+ dof) and run kinematics.

    Unlike ``_drive_ref_data`` (which takes the 38-dim mimic and builds the quat
    from rpy), this takes the root pose directly: ``root_pos_w`` (xyz, already in
    the robot world frame) and ``root_quat_wxyz`` (wxyz). Used for the future
    reference frames (future_motion_pos_h).
    """
    ref_data.qpos[:3] = np.asarray(root_pos_w, dtype=np.float64)
    ref_data.qpos[3:7] = np.asarray(root_quat_wxyz, dtype=np.float64)
    ref_data.qpos[7:7 + num_actions] = np.asarray(dof_pos, dtype=np.float64)
    mujoco.mj_kinematics(model, ref_data)


def compute_future_motion_pos_h(
    model, ref_data,
    future_root_pos_w, future_root_quat_w, future_dof_pos,
    tracked_body_ids, extended_parent_ids, extended_local_offsets,
    num_actions, robot_root_pos_w, robot_root_quat_w,
):
    """future_motion_pos_h: future reference extended-body positions expressed in
    the robot's heading frame, relative to the robot root. Flat [T*B*3] float32.

    Mirrors DEX_RL_LAB ``mdp.future_motion_pos_h``:
        pos_h = heading_inv(robot) ⊗ (future_body_pos_w - robot_root_pos_w)
    with ``heading_inv = calc_heading_quat_inv(robot_root_quat_w)`` (yaw-only).

    The extended body positions per future step are obtained by driving ``ref_data``
    with that step's (root pose, dof) and running FK -- the same machinery as the
    diff_body terms, so the 33-body order matches exactly.

    Args:
        future_root_pos_w:  [T, 3] future ref root xyz, in the robot world frame.
        future_root_quat_w: [T, 4] future ref root quat (wxyz), robot world frame.
        future_dof_pos:     [T, J] future ref joint positions (SDK order).
        robot_root_pos_w:   [3]    current robot root xyz (world).
        robot_root_quat_w:  [4]    current robot root quat (wxyz).
    """
    future_root_pos_w = np.asarray(future_root_pos_w, dtype=np.float64)
    T = future_root_pos_w.shape[0]
    num_bodies = len(tracked_body_ids) + len(extended_parent_ids)

    body_pos_w = np.empty((T, num_bodies, 3), dtype=np.float64)
    for t in range(T):
        _drive_ref_data_pose(
            model, ref_data, future_root_pos_w[t], future_root_quat_w[t],
            future_dof_pos[t], num_actions,
        )
        body_pos_w[t] = compute_extended_body_pos_w(
            ref_data, tracked_body_ids, extended_parent_ids, extended_local_offsets,
        )

    rel = body_pos_w - np.asarray(robot_root_pos_w, dtype=np.float64).reshape(1, 1, 3)
    heading_inv = heading_quat_from_quat(robot_root_quat_w, inverse=True)  # [4]
    heading_inv_b = np.broadcast_to(heading_inv, (T, num_bodies, 4))
    pos_h = quat_apply(heading_inv_b, rel)  # [T, B, 3]
    return pos_h.astype(np.float32).reshape(-1)


def compute_future_motion_anchor(future_root_quat_w, robot_root_quat_w):
    """future_motion_anchor: heading-corrected relative base->future-ref-root
    rotations as 6D. Flat [T*6] float32.

    Mirrors DEX_RL_LAB ``mdp.future_motion_anchor``:
        rel_w   = q_ref_w * conj(q_base_w)
        rel_hc  = conj(q_planar) * rel_w * q_planar
        anchor  = rot6d(rel_hc)
    with ``q_planar = planar_root_quat_w`` (yaw-only, calc_heading convention,
    matching the diff_body_tannorm planar base used elsewhere in this module).

    Args:
        future_root_quat_w: [T, 4] future ref root quat (wxyz), robot world frame.
        robot_root_quat_w:  [4]    current robot root quat (wxyz).
    """
    future_root_quat_w = np.asarray(future_root_quat_w, dtype=np.float64)
    T = future_root_quat_w.shape[0]

    base_quat = np.asarray(robot_root_quat_w, dtype=np.float64)
    planar = heading_quat_from_quat(base_quat, inverse=False)  # planar_root_quat_w
    inv_planar = quat_conj(planar)

    base_b = np.broadcast_to(base_quat, (T, 4))
    rel_quat_w = quat_mul(future_root_quat_w, quat_conj(base_b))
    planar_b = np.broadcast_to(planar, (T, 4))
    inv_planar_b = np.broadcast_to(inv_planar, (T, 4))
    rel_hc = quat_mul(quat_mul(inv_planar_b, rel_quat_w), planar_b)

    return quat_to_rot6d(rel_hc).astype(np.float32).reshape(-1)


def parse_future_raw(raw_pos, raw_rot, raw_dof, num_future_steps, num_actions):
    """Parse the motion server's future-frame Redis payloads into arrays.

    Each payload is a flattened JSON list. Returns
    ``(root_pos [T,3], root_rot [T,4] xyzw, dof [T,J])`` or ``None`` when any
    payload is missing or the wrong size (callers then use the stationary
    fallback)."""
    if raw_pos is None or raw_rot is None or raw_dof is None:
        return None
    T, J = num_future_steps, num_actions
    try:
        pos = np.asarray(json.loads(raw_pos), dtype=np.float64)
        rot = np.asarray(json.loads(raw_rot), dtype=np.float64)
        dof = np.asarray(json.loads(raw_dof), dtype=np.float64)
    except Exception:
        return None
    if pos.size != T * 3 or rot.size != T * 4 or dof.size != T * J:
        return None
    return pos.reshape(T, 3), rot.reshape(T, 4), dof.reshape(T, J)


def compute_future_motion_obs(
    model, ref_data, num_future_steps, num_actions,
    tracked_body_ids, extended_parent_ids, extended_local_offsets,
    robot_root_pos_w, robot_root_quat_w,
    future_raw=None,
    anchor_delta_xy=None,
    fallback_action_mimic=None,
    fallback_root_xy=None,
):
    """Assemble the (future_motion_pos_h, future_motion_anchor) obs terms.

    Resolves the T future reference frames into the robot world frame, then defers
    to ``compute_future_motion_pos_h`` / ``compute_future_motion_anchor``.

    Args:
        future_raw: (root_pos [T,3], root_rot [T,4] xyzw, dof [T,J]) as published by
            the motion server (in the published motion frame), or None to use the
            stationary fallback below.
        anchor_delta_xy: [2] xy translation mapping the published motion frame to the
            robot world frame (= odom_origin_xy - ref_origin_xy). None -> no shift.
        fallback_action_mimic: 38-dim mimic target; when future_raw is None the
            current frame is repeated T times (z/rpy/dof from here).
        fallback_root_xy: [2] robot-frame xy for the stationary fallback frames.

    Returns:
        (future_motion_pos_h [T*B*3], future_motion_anchor [T*6]) float32 arrays.
    """
    T = num_future_steps
    if future_raw is not None:
        f_pos, f_rot_xyzw, f_dof = future_raw
        f_pos = np.asarray(f_pos, dtype=np.float64).reshape(T, 3).copy()
        if anchor_delta_xy is not None:
            f_pos[:, :2] += np.asarray(anchor_delta_xy, dtype=np.float64).reshape(2)
        f_rot_xyzw = np.asarray(f_rot_xyzw, dtype=np.float64).reshape(T, 4)
        # motion-lib quats are xyzw; the FK/quat math here uses wxyz.
        f_quat_wxyz = f_rot_xyzw[:, [3, 0, 1, 2]]
        f_dof = np.asarray(f_dof, dtype=np.float64).reshape(T, num_actions)
    else:
        # Stationary fallback: repeat the current mimic target frame T times. Used
        # while idle / before the motion server publishes future frames; the robot
        # is holding pose so a non-progressing future is the right default.
        am = np.asarray(fallback_action_mimic, dtype=np.float64)
        z, roll, pitch, yaw = am[2], am[3], am[4], am[5]
        dof = am[-num_actions:]
        xy = (np.asarray(fallback_root_xy, dtype=np.float64).reshape(2)
              if fallback_root_xy is not None else np.zeros(2, dtype=np.float64))
        pose = np.array([xy[0], xy[1], z], dtype=np.float64)
        quat = rpy_to_quat(float(roll), float(pitch), float(yaw))
        f_pos = np.broadcast_to(pose, (T, 3)).copy()
        f_quat_wxyz = np.broadcast_to(quat, (T, 4)).copy()
        f_dof = np.broadcast_to(np.asarray(dof, dtype=np.float64), (T, num_actions)).copy()

    pos_h = compute_future_motion_pos_h(
        model, ref_data, f_pos, f_quat_wxyz, f_dof,
        tracked_body_ids, extended_parent_ids, extended_local_offsets,
        num_actions, robot_root_pos_w, robot_root_quat_w,
    )
    anchor = compute_future_motion_anchor(f_quat_wxyz, robot_root_quat_w)
    return pos_h, anchor
