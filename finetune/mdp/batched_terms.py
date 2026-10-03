from __future__ import annotations

import torch

from finetune.helper.torch_math import (
    heading_quat_from_quat, projected_gravity_b, quat_apply, quat_conj, quat_mul, quat_rotate_inverse,
    quat_to_rot6d, quat_to_rpy, rpy_to_quat, yaw_quat,
)
from finetune.mdp.batched_obs import BatchedObsContext, bterm
from obs_terms import future_num_steps


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def _rot_rows(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate [N, K, 3] rows by the per-env quat [N, 4]."""
    return quat_apply(q[:, None, :].expand(v.shape[0], v.shape[1], 4), v)


def _heading_inv(c: BatchedObsContext) -> torch.Tensor:
    return heading_quat_from_quat(c.robot.root_quat, inverse=True)


def _planar_inv(c: BatchedObsContext) -> torch.Tensor:
    """Inverse planar (yaw-only) base quat, euler-xyz yaw -- matches _planar_inv on the CPU."""
    return yaw_quat(c.rpy[:, 2], inverse=True)


# --------------------------------------------------------------------------- proprio
@bterm("base_ang_vel")
def base_ang_vel(c, p):
    return c.ang_vel


@bterm("base_roll_pitch")
def base_roll_pitch(c, p):
    return c.rpy[:, :2]


@bterm("joint_pos_rel")
def joint_pos_rel(c, p):
    default = torch.as_tensor(c.static.default_dof_pos_isaac, dtype=c.dof_pos_isaac.dtype,
                              device=c.device)
    return c.dof_pos_isaac - default


@bterm("joint_vel_rel")
def joint_vel_rel(c, p):
    v = c.dof_vel_isaac
    mask = p.get("mask_joint_names")
    if mask:
        idx = c.static.joint_indices(mask)
        if idx:
            v = v.clone()
            v[:, torch.as_tensor(idx, dtype=torch.long, device=c.device)] = 0.0
    return v


@bterm("last_action")
def last_action(c, p):
    return c.last_action


# --------------------------------------------------------------------------- diff_body_*
def _mimic_ref_body_pose(c: BatchedObsContext, odom: bool = True):
    """(pos, quat) of the extended reference bodies for the pose rebuilt from action_mimic
    (_drive_ref_data). ``odom=True`` places the reference root at ``ref_root_xy_w``; ``odom=False``
    pins its xy to the origin, the deploy convention that cancels root translation anyway.
    Cached per variant: several diff_body_* terms share one FK pass."""
    key = f"mimic_ref_body_pose_{'odom' if odom else 'origin'}"
    if key not in c._cache:
        J = c.static.num_actions
        am = c.action_mimic
        xy = c.ref_root_xy_w if odom else torch.zeros_like(am[:, :2])
        root_pos = torch.cat([xy, am[:, 2:3]], dim=-1)
        root_quat = rpy_to_quat(am[:, 3:6])
        c._cache[key] = c.ref_fk.body_pose(root_pos, root_quat, am[:, -J:])
    return c._cache[key]


def _planar_base_inv(c: BatchedObsContext) -> torch.Tensor:
    """Inverse planar base quat with the *heading* yaw (yaw_from_quat), i.e. the
    planar_rot_matrix(yaw_from_quat(...)) convention the deploy diff_body_*_pb helpers use.
    Not the same as _planar_inv, which takes the quatToEuler yaw."""
    return quat_conj(heading_quat_from_quat(c.robot.root_quat))


@bterm("diff_body_pos_b")
def diff_body_pos_b(c, p):
    """(ref - robot) extended-body positions in the robot base frame, from the mimic frame.
    The training env runs with odom on, so root translation is *not* cancelled."""
    diff_w = _mimic_ref_body_pose(c)[0] - c.robot.body_pos
    return _flat(_rot_rows(quat_conj(c.robot.root_quat), diff_w))


@bterm("diff_body_pos_b_deploy")
def diff_body_pos_b_deploy(c, p):
    """Deploy convention: each side's pelvis is subtracted first, so root translation cancels."""
    ref = _mimic_ref_body_pose(c, odom=False)[0]
    diff_w = (ref - ref[:, 0:1]) - (c.robot.body_pos - c.robot.body_pos[:, 0:1])
    return _flat(_rot_rows(quat_conj(c.robot.root_quat), diff_w))


@bterm("diff_body_pos_pb")
def diff_body_pos_pb(c, p):
    """Same diff in the planar (yaw-only) base frame. With odom on, ``z_align`` drops only the
    pelvis z mismatch; without odom the pelvis is subtracted on every axis."""
    diff_w = _mimic_ref_body_pose(c)[0] - c.robot.body_pos
    if bool(p.get("z_align", False)):
        diff_w = torch.cat([diff_w[..., :2], diff_w[..., 2:] - diff_w[:, 0:1, 2:]], dim=-1)
    return _flat(_rot_rows(_planar_base_inv(c), diff_w))


@bterm("diff_body_tannorm_pb")
def diff_body_tannorm_pb(c, p):
    """Per-body 6D tan-norm of the ref/robot rotation difference in the planar base frame."""
    ref_quat = _mimic_ref_body_pose(c)[1]
    diff_w = quat_mul(ref_quat, quat_conj(c.robot.body_quat))
    base = heading_quat_from_quat(c.robot.root_quat)[:, None, :].expand_as(diff_w)
    diff_b = quat_mul(quat_mul(quat_conj(base), diff_w), base)
    tan = quat_apply(diff_b, torch.tensor([1.0, 0.0, 0.0], device=c.device).expand_as(diff_b[..., :3]))
    norm = quat_apply(diff_b, torch.tensor([0.0, 0.0, 1.0], device=c.device).expand_as(diff_b[..., :3]))
    return _flat(torch.cat([tan, norm], dim=-1))


@bterm("upcoming_twist_mimic_target")
def upcoming_twist_mimic_target(c, p):
    return c.action_mimic


# --------------------------------------------------------------------------- future motion
def _future_frames(c: BatchedObsContext):
    """(pos [N,T,3], quat_wxyz [N,T,4], dof [N,T,J], lin [N,T,3], ang [N,T,3]) in the robot
    world frame -- resolve_future_frames' future_raw path."""
    return c.future_frames


def _future_body_pos_w(c: BatchedObsContext) -> torch.Tensor:
    """[N, n_fk, B_ext, 3] extended-body world positions of the first n_fk future frames."""
    if "future_body_pos_w" not in c._cache:
        f_pos, f_quat, f_dof, _, _ = _future_frames(c)
        out = [c.ref_fk.body_pose(f_pos[:, k], f_quat[:, k], f_dof[:, k])[0]
               for k in range(c.n_fk)]
        c._cache["future_body_pos_w"] = torch.stack(out, dim=1)
    return c._cache["future_body_pos_w"]


@bterm("future_motion_pos_h")
def future_motion_pos_h(c, p):
    n = future_num_steps(c.static, p)
    body = _future_body_pos_w(c)[:, :n]                     # [N, n, B, 3]
    rel = body - c.robot.root_pos[:, None, None, :]
    h_inv = _heading_inv(c)[:, None, None, :].expand(*rel.shape[:-1], 4)
    return _flat(quat_apply(h_inv, rel))


@bterm("future_motion_anchor")
def future_motion_anchor(c, p):
    n = future_num_steps(c.static, p)
    f_quat = _future_frames(c)[1][:, :n]                     # [N, n, 4]
    base = c.robot.root_quat[:, None, :].expand_as(f_quat)
    planar = heading_quat_from_quat(c.robot.root_quat)[:, None, :].expand_as(f_quat)
    rel_w = quat_mul(f_quat, quat_conj(base))
    rel_hc = quat_mul(quat_mul(quat_conj(planar), rel_w), planar)
    return _flat(quat_to_rot6d(rel_hc))


@bterm("future_motion_pos_ref_b")
def future_motion_pos_ref_b(c, p):
    """Future extended-body positions in each future reference root's own frame."""
    n = future_num_steps(c.static, p)
    f_pos, f_quat = _future_frames(c)[0][:, :n], _future_frames(c)[1][:, :n]
    body = _future_body_pos_w(c)[:, :n]                      # [N, n, B, 3]
    offset = body - f_pos[:, :, None, :]
    inv_q = quat_conj(f_quat)[:, :, None, :].expand(*offset.shape[:-1], 4)
    return _flat(quat_apply(inv_q, offset))


@bterm("future_motion_joint_pos")
def future_motion_joint_pos(c, p):
    n = future_num_steps(c.static, p)
    return _flat(_future_frames(c)[2][:, :n])


@bterm("future_motion_root_height")
def future_motion_root_height(c, p):
    n = future_num_steps(c.static, p)
    return _future_frames(c)[0][:, :n, 2]


@bterm("future_motion_root_roll_pitch")
def future_motion_root_roll_pitch(c, p):
    n = future_num_steps(c.static, p)
    return _flat(quat_to_rpy(_future_frames(c)[1][:, :n])[..., :2])


@bterm("future_motion_root_lin_vel_pb")
def future_motion_root_lin_vel_pb(c, p):
    n = future_num_steps(c.static, p)
    v_w = _future_frames(c)[3][:, :n]
    # CPU uses the quatToEuler yaw of the *robot* root (planar base, non-inverse sign convention)
    yaw = quat_to_rpy(c.robot.root_quat)[:, 2]
    return _flat(_rot_rows(quat_conj(yaw_quat(yaw)), v_w))


@bterm("future_motion_root_ang_vel_b")
def future_motion_root_ang_vel_b(c, p):
    n = future_num_steps(c.static, p)
    w_w = _future_frames(c)[4][:, :n]
    return _flat(_rot_rows(quat_conj(c.robot.root_quat), w_w))


# --------------------------------------------------------------------------- priv_* (critic)
@bterm("priv_root_lin_vel_b")
def priv_root_lin_vel_b(c, p):
    return quat_rotate_inverse(c.robot.root_quat, c.robot.root_lin_vel_w)


@bterm("priv_root_height")
def priv_root_height(c, p):
    return c.robot.root_pos[:, 2:3]


@bterm("priv_projected_gravity")
def priv_projected_gravity(c, p):
    return projected_gravity_b(c.robot.root_quat)


@bterm("priv_ref_root_pos_diff_b")
def priv_ref_root_pos_diff_b(c, p):
    return quat_rotate_inverse(c.robot.root_quat, c.ref_frame.root_pos - c.robot.root_pos)


@bterm("priv_ref_root_rot_diff")
def priv_ref_root_rot_diff(c, p):
    base = c.robot.root_quat
    planar = heading_quat_from_quat(base)
    rel = quat_mul(c.ref_frame.root_quat, quat_conj(base))
    rel_hc = quat_mul(quat_mul(quat_conj(planar), rel), planar)
    return quat_to_rot6d(rel_hc)


@bterm("priv_ref_root_lin_vel_b")
def priv_ref_root_lin_vel_b(c, p):
    return quat_rotate_inverse(c.robot.root_quat, c.ref_frame.root_lin_vel)


@bterm("priv_ref_root_ang_vel_b")
def priv_ref_root_ang_vel_b(c, p):
    return quat_rotate_inverse(c.robot.root_quat, c.ref_frame.root_ang_vel)


@bterm("priv_body_pos_diff_b")
def priv_body_pos_diff_b(c, p):
    return _flat(_rot_rows(quat_conj(c.robot.root_quat), c.ref.pos - c.robot.body_pos))


@bterm("priv_body_lin_vel_diff_b")
def priv_body_lin_vel_diff_b(c, p):
    n = c.ref.lin_vel.shape[1]
    diff = c.ref.lin_vel - c.robot.body_lin_vel[:, :n]
    return _flat(_rot_rows(quat_conj(c.robot.root_quat), diff))


@bterm("priv_body_ang_vel_diff_b")
def priv_body_ang_vel_diff_b(c, p):
    n = c.ref.ang_vel.shape[1]
    diff = c.ref.ang_vel - c.robot.body_ang_vel[:, :n]
    return _flat(_rot_rows(quat_conj(c.robot.root_quat), diff))


@bterm("priv_feet_contact")
def priv_feet_contact(c, p):
    thr = float(p.get("threshold", 1.0))
    ids = torch.as_tensor(c.static.tracked_body_ids, dtype=torch.long,
                          device=c.device)[list(c.feet_body_idx)]
    return (c.robot.contact_force[:, ids] > thr).to(c.robot.contact_force.dtype)


@bterm("priv_joint_torque")
def priv_joint_torque(c, p):
    return c.robot.torque_sdk * float(p.get("scale", 0.01))


@bterm("priv_motion_phase")
def priv_motion_phase(c, p):
    return torch.stack([c.motion_phase, c.time_left.clamp(max=5.0) / 5.0], dim=-1)


# --------------------------------------------------------------------------- teacher critic terms
@bterm("projected_gravity")
def projected_gravity(c, p):
    return projected_gravity_b(c.robot.root_quat)


@bterm("extended_body_pos_h")
def extended_body_pos_h(c, p):
    rel = c.robot.body_pos - c.robot.root_pos[:, None, :]
    return _flat(_rot_rows(_heading_inv(c), rel))


@bterm("extended_body_quat_h")
def extended_body_quat_h(c, p):
    q = c.robot.body_quat
    h = _heading_inv(c)[:, None, :].expand_as(q)
    return _flat(quat_mul(h, q))


@bterm("extended_body_lin_vel_h")
def extended_body_lin_vel_h(c, p):
    return _flat(_rot_rows(_heading_inv(c), c.robot.body_lin_vel_ext))


@bterm("extended_body_ang_vel_h")
def extended_body_ang_vel_h(c, p):
    return _flat(_rot_rows(_heading_inv(c), c.robot.body_ang_vel_ext))


@bterm("diff_body_lin_vel_pb")
def diff_body_lin_vel_pb(c, p):
    diff = c.ref.lin_vel_ext - c.robot.body_lin_vel_ext
    return _flat(_rot_rows(_planar_inv(c), diff))


@bterm("diff_body_ang_vel_pb")
def diff_body_ang_vel_pb(c, p):
    diff = c.ref.ang_vel_ext - c.robot.body_ang_vel_ext
    return _flat(_rot_rows(_planar_inv(c), diff))
