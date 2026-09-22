from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import mujoco
import numpy as np

from cfg import g1_29dof_cfg as cfg   # deploy_real/cfg
from finetune.helper.math_utils import quat_rotate_inverse
from observations import compute_extended_body_pos_w, compute_extended_body_quat_w
from obs_terms import StaticContext
from finetune.motion_data.motion_ref import RefFK


@dataclass
class RobotState:
    """Post-step robot state (Isaac joint order where noted)."""
    root_pos: np.ndarray        # [3]
    root_quat: np.ndarray       # [4] wxyz
    root_lin_vel_w: np.ndarray  # [3]
    root_ang_vel_b: np.ndarray  # [3] body frame (MuJoCo free-joint qvel[3:6])
    dof_pos_sdk: np.ndarray     # [J]
    dof_vel_sdk: np.ndarray     # [J]
    dof_pos_isaac: np.ndarray   # [J]
    dof_vel_isaac: np.ndarray   # [J]
    torque_sdk: np.ndarray      # [J] last applied torque
    body_pos: np.ndarray        # [B_ext, 3] tracked + extended, world
    body_quat: np.ndarray       # [B_ext, 4]
    body_lin_vel: np.ndarray    # [B, 3] tracked, world (link frame origin)
    body_ang_vel: np.ndarray    # [B, 3] tracked, world
    contact_force: np.ndarray   # [nbody] max net contact force norm over the substeps
    # [B_ext, 3] tracked + extended; an extended body moves with its parent
    # (DEX extending_body_lin_vel_w / extending_body_ang_vel_w, HOVER convention).
    body_lin_vel_ext: np.ndarray = None
    body_ang_vel_ext: np.ndarray = None


class G1SimCore:
    def __init__(self, xml_path: str, future_steps: int, future_fk_steps: int,
                 sim_dt: float = 0.001, control_hz: int = 50,
                 kp_scale: float = 1.0, kd_scale: float = 1.0,
                 action_scale: float = cfg.ACTION_SCALE, action_clip: float = 10.0,
                 contact_sensor_dt: float = 0.005, contact_history: int = 3,
                 model: Optional[mujoco.MjModel] = None):
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = sim_dt
        self.data = mujoco.MjData(self.model)
        self.sim_dt = sim_dt
        self.control_dt = 1.0 / control_hz
        self.decimation = int(round(self.control_dt / sim_dt))
        assert abs(self.decimation * sim_dt - self.control_dt) < 1e-9, (sim_dt, control_hz)
        # IsaacLab ContactSensor(update_period=sim.dt=0.005, history_length=3): net forces are
        # sampled every ``contact_sensor_dt`` and the reward takes the max over the last
        # ``contact_history`` samples -> evaluate contacts only on those substeps.
        period = max(int(round(contact_sensor_dt / sim_dt)), 1)
        self.contact_substeps = {self.decimation - k * period for k in range(contact_history)
                                 if self.decimation - k * period >= 1}

        self.num_actions = cfg.NUM_ACTIONS
        self.action_scale = np.full(self.num_actions, action_scale, dtype=np.float32)
        self.action_clip = action_clip
        self.default_dof_pos = cfg.DEFAULT_DOF_POS.copy()
        self.stiffness = cfg.STIFFNESS * kp_scale
        self.damping = cfg.DAMPING * kd_scale
        self.torque_limits = cfg.TORQUE_LIMITS

        # isaac_ordered = sdk_ordered[sdk_to_isaac]; sdk_ordered = isaac_ordered[isaac_to_sdk]
        self.sdk_to_isaac = np.array([cfg.SDK_JOINT_NAMES.index(n) for n in cfg.ISAAC_JOINT_NAMES], dtype=np.int64)
        self.isaac_to_sdk = np.array([cfg.ISAAC_JOINT_NAMES.index(n) for n in cfg.SDK_JOINT_NAMES], dtype=np.int64)
        self.default_dof_pos_isaac = self.default_dof_pos[self.sdk_to_isaac]

        self.tracked_body_ids = np.array([self.model.body(n).id for n in cfg.TRACKED_BODY_NAMES], dtype=np.int64)
        self.extended_parent_ids = np.array([self.model.body(p).id for _, p, _ in cfg.EXTENDED_JOINTS], dtype=np.int64)
        self.extended_local_offsets = np.array([o for _, _, o in cfg.EXTENDED_JOINTS], dtype=np.float64)
        self.num_tracked = len(self.tracked_body_ids)
        self.pelvis_id = int(self.tracked_body_ids[0])
        # index of each extended body's parent within the tracked list (velocity inheritance)
        tracked = self.tracked_body_ids.tolist()
        self.extended_parent_tracked_idx = np.array([tracked.index(int(p)) for p in self.extended_parent_ids], dtype=np.int64)

        # Private MjData for FK on the reference frames (diff_body_* / future_motion_* terms).
        self.ref_data = mujoco.MjData(self.model)
        self.future_ref_data = mujoco.MjData(self.model)
        self.ref_fk = RefFK(self.model, self.tracked_body_ids, self.extended_parent_ids,
                            self.extended_local_offsets, self.num_actions)

        self.obs_static = StaticContext(
            model=self.model,
            num_actions=self.num_actions,
            tracked_body_ids=self.tracked_body_ids,
            extended_parent_ids=self.extended_parent_ids,
            extended_local_offsets=self.extended_local_offsets,
            default_dof_pos_isaac=self.default_dof_pos_isaac,
            isaac_joint_names=list(cfg.ISAAC_JOINT_NAMES),
            future_steps=future_steps,
            future_fk_steps=max(int(future_fk_steps), 1),
        )

        # Joint limits (SDK order == MuJoCo joint order; joint 0 is the free joint).
        jnt_range = self.model.jnt_range[1:1 + self.num_actions].astype(np.float64)
        self.joint_limits_sdk = jnt_range
        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(self.model.nbody)]

        self._vel6 = np.zeros(6, dtype=np.float64)
        self._force6 = np.zeros(6, dtype=np.float64)
        self._contact_acc = np.zeros((self.model.nbody, 3), dtype=np.float64)
        self._contact_max = np.zeros(self.model.nbody, dtype=np.float64)
        self.last_torque = np.zeros(self.num_actions, dtype=np.float64)

    # ----- helpers -----
    def soft_joint_limits(self, factor: float) -> np.ndarray:
        """[J, 2] IsaacLab soft limits: mid +- factor * half-range (SDK order)."""
        lo, hi = self.joint_limits_sdk[:, 0], self.joint_limits_sdk[:, 1]
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * factor
        return np.stack([mid - half, mid + half], axis=-1)

    def body_ids_matching(self, patterns: List[str]) -> np.ndarray:
        return np.array([i for i, n in enumerate(self.body_names) if any(re.fullmatch(p, n) for p in patterns)],
                        dtype=np.int64)

    # ----- reset -----
    def reset_to(self, root_pos, root_quat_wxyz, root_lin_vel_w, root_ang_vel_w, dof_pos_sdk, dof_vel_sdk):
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[:3] = root_pos
        d.qpos[3:7] = root_quat_wxyz
        d.qpos[7:7 + self.num_actions] = dof_pos_sdk
        d.qvel[:3] = root_lin_vel_w
        # free-joint angular velocity is expressed in the body frame
        d.qvel[3:6] = quat_rotate_inverse(np.asarray(root_quat_wxyz, dtype=np.float64), np.asarray(root_ang_vel_w, dtype=np.float64))
        d.qvel[6:6 + self.num_actions] = dof_vel_sdk
        d.ctrl[:] = 0.0
        self.last_torque[:] = 0.0
        self._contact_max[:] = 0.0
        mujoco.mj_forward(self.model, d)

    # ----- step -----
    def pd_target_sdk(self, raw_action_isaac: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(raw_action_isaac, dtype=np.float32), -self.action_clip, self.action_clip)
        target_isaac = a * self.action_scale + self.default_dof_pos_isaac
        return target_isaac[self.isaac_to_sdk].astype(np.float64)

    def step(self, raw_action_isaac: np.ndarray):
        """One control step: PD torques over ``decimation`` physics substeps.
        Contact forces are tracked per body (max norm over the substeps)."""
        pd_target = self.pd_target_sdk(raw_action_isaac)
        d, m, n = self.data, self.model, self.num_actions
        self._contact_max[:] = 0.0
        for k in range(1, self.decimation + 1):
            dof_pos = d.qpos[7:7 + n]
            dof_vel = d.qvel[6:6 + n]
            torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            d.ctrl[:] = torque
            mujoco.mj_step(m, d)
            if k in self.contact_substeps:
                self._accumulate_contacts()
        self.last_torque[:] = d.ctrl
        # kinematics / body velocities for the integrated state
        mujoco.mj_forward(m, d)

    def _accumulate_contacts(self):
        d, m = self.data, self.model
        if d.ncon == 0:
            return
        acc = self._contact_acc
        acc[:] = 0.0
        for i in range(d.ncon):
            con = d.contact[i]
            mujoco.mj_contactForce(m, d, i, self._force6)
            f_w = con.frame.reshape(3, 3).T @ self._force6[:3]
            acc[m.geom_bodyid[con.geom1]] -= f_w
            acc[m.geom_bodyid[con.geom2]] += f_w
        np.maximum(self._contact_max, np.linalg.norm(acc, axis=-1), out=self._contact_max)

    # ----- state -----
    def state(self) -> RobotState:
        d, n = self.data, self.num_actions
        dof_pos_sdk = d.qpos[7:7 + n].copy()
        dof_vel_sdk = d.qvel[6:6 + n].copy()
        body_pos = compute_extended_body_pos_w(d, self.tracked_body_ids, self.extended_parent_ids, self.extended_local_offsets)
        body_quat = compute_extended_body_quat_w(d, self.tracked_body_ids, self.extended_parent_ids)
        lin = np.empty((self.num_tracked, 3))
        ang = np.empty((self.num_tracked, 3))
        for k, bid in enumerate(self.tracked_body_ids):
            # link-frame origin velocity in world orientation (IsaacLab body_link_*_vel_w)
            mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_XBODY, int(bid), self._vel6, 0)
            ang[k] = self._vel6[:3]
            lin[k] = self._vel6[3:]
        return RobotState(
            root_pos=d.qpos[:3].copy(),
            root_quat=d.qpos[3:7].copy(),
            root_lin_vel_w=d.qvel[:3].copy(),
            root_ang_vel_b=d.qvel[3:6].copy(),
            dof_pos_sdk=dof_pos_sdk,
            dof_vel_sdk=dof_vel_sdk,
            dof_pos_isaac=dof_pos_sdk[self.sdk_to_isaac],
            dof_vel_isaac=dof_vel_sdk[self.sdk_to_isaac],
            torque_sdk=self.last_torque.copy(),
            body_pos=body_pos.copy(),
            body_quat=body_quat.copy(),
            body_lin_vel=lin,
            body_ang_vel=ang,
            contact_force=self._contact_max.copy(),
            body_lin_vel_ext=np.concatenate([lin, lin[self.extended_parent_tracked_idx]], axis=0),
            body_ang_vel_ext=np.concatenate([ang, ang[self.extended_parent_tracked_idx]], axis=0),
        )
