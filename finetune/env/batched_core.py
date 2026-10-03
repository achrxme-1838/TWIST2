from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from cfg import g1_29dof_cfg as cfg   # deploy_real
from obs_terms import StaticContext
from finetune.helper.torch_math import quat_apply, quat_rotate_inverse

# mj_contactForce packs the contact-frame wrench as [force(3), torque(3)]; mujoco_warp keeps that
# layout in its spatial_vectorf. check_batched_core.py verifies this against mj_contactForce.
_FORCE_SLICE = slice(0, 3)


@dataclass
class BatchedRobotState:
    """Post-step robot state, leading env dim. Field-for-field the batched sim_core.RobotState."""
    root_pos: torch.Tensor          # [N, 3]
    root_quat: torch.Tensor         # [N, 4] wxyz
    root_lin_vel_w: torch.Tensor    # [N, 3]
    root_ang_vel_b: torch.Tensor    # [N, 3] body frame (free-joint qvel[3:6])
    dof_pos_sdk: torch.Tensor       # [N, J]
    dof_vel_sdk: torch.Tensor
    dof_pos_isaac: torch.Tensor
    dof_vel_isaac: torch.Tensor
    torque_sdk: torch.Tensor        # [N, J] last applied torque
    body_pos: torch.Tensor          # [N, B_ext, 3] tracked + extended, world
    body_quat: torch.Tensor         # [N, B_ext, 4]
    body_lin_vel: torch.Tensor      # [N, B, 3] tracked, world (link-frame origin)
    body_ang_vel: torch.Tensor      # [N, B, 3]
    contact_force: torch.Tensor     # [N, nbody] max net contact force norm over the substeps
    body_lin_vel_ext: torch.Tensor  # [N, B_ext, 3] extended bodies inherit their parent
    body_ang_vel_ext: torch.Tensor


@wp.kernel
def _accumulate_contacts(
    nacon: wp.array(dtype=wp.int32),
    worldid: wp.array(dtype=wp.int32),
    geom: wp.array(dtype=wp.vec2i),
    force: wp.array(dtype=wp.spatial_vectorf),
    geom_bodyid: wp.array(dtype=wp.int32),
    acc: wp.array3d(dtype=wp.float32),      # [nworld, nbody, 3], zeroed by the caller
):
    """Net contact force per body, world frame -- the batched _accumulate_contacts of sim_core."""
    i = wp.tid()
    if i >= nacon[0]:
        return
    w = worldid[i]
    f = force[i]
    g = geom[i]
    b1 = geom_bodyid[g[0]]
    b2 = geom_bodyid[g[1]]
    for k in range(3):
        fk = f[k]
        wp.atomic_sub(acc, w, b1, k, fk)
        wp.atomic_add(acc, w, b2, k, fk)


class BatchedG1SimCore:
    def __init__(self, xml_path: str, num_envs: int, sim_dt: float = 0.001, control_hz: int = 50,
                 kp_scale: float = 1.0, kd_scale: float = 1.0,
                 action_scale: float = cfg.ACTION_SCALE, action_clip: float = 10.0,
                 contact_sensor_dt: float = 0.005, contact_history: int = 3,
                 device: str = "cuda:0", njmax: int = 256, contacts_per_env: int = 64,
                 future_steps: int = 1, future_fk_steps: int = 1,
                 model: Optional[mujoco.MjModel] = None):
        self.num_envs = num_envs
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(
                f"BatchedG1SimCore needs a CUDA device, got {device!r}. mujoco_warp has no CPU "
                "path here; pass train.device=cuda:0 (with env.backend=gpu), or use "
                "env.backend=cpu for the single-threaded MuJoCo env."
            )
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = sim_dt
        if self.model.opt.solver == mujoco.mjtSolver.mjSOL_PGS:
            raise ValueError("mujoco_warp does not support mjSOL_PGS; set solver='Newton' in the XML")

        self.sim_dt = sim_dt
        self.control_dt = 1.0 / control_hz
        self.decimation = int(round(self.control_dt / sim_dt))
        assert abs(self.decimation * sim_dt - self.control_dt) < 1e-9, (sim_dt, control_hz)
        period = max(int(round(contact_sensor_dt / sim_dt)), 1)
        self.contact_substeps = {self.decimation - k * period for k in range(contact_history)
                                 if self.decimation - k * period >= 1}

        # ---- warp model / data (one world per env) -------------------------------------------
        wp.init()
        mjd = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, mjd)
        # warp allocates on its *current* device, which is cuda:0 unless scoped -- pin it to the
        # requested one so the warp arrays and the torch constants below agree.
        self.wp_device = str(self.device)
        with wp.ScopedDevice(self.wp_device):
            self.mw = mjwarp.put_model(self.model)
            if hasattr(self.mw.opt, "graph_conditional"):
                # CUDA graph conditional nodes need a >= 12.4 driver; mujoco_warp warns otherwise.
                self.mw.opt.graph_conditional = False
            self.dw = mjwarp.put_data(self.model, mjd, nworld=num_envs, njmax=njmax,
                                      naconmax=contacts_per_env * num_envs)

        v = wp.to_torch
        self.qpos, self.qvel, self.ctrl = v(self.dw.qpos), v(self.dw.qvel), v(self.dw.ctrl)
        self.xpos, self.xquat, self.cvel = v(self.dw.xpos), v(self.dw.xquat), v(self.dw.cvel)
        self.xipos = v(self.dw.xipos)
        self.actuator_force = v(self.dw.actuator_force)

        # ---- constants ------------------------------------------------------------------------
        dev = self.device
        f = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=dev)
        i64 = lambda x: torch.as_tensor(np.asarray(x, dtype=np.int64), device=dev)
        J = self.num_actions = cfg.NUM_ACTIONS
        self.action_scale = float(action_scale)
        self.action_clip = float(action_clip)
        self.stiffness = f(cfg.STIFFNESS * kp_scale)
        self.damping = f(cfg.DAMPING * kd_scale)
        self.torque_limits = f(cfg.TORQUE_LIMITS)
        self.default_dof_pos = f(cfg.DEFAULT_DOF_POS)
        self.sdk_to_isaac = i64([cfg.SDK_JOINT_NAMES.index(n) for n in cfg.ISAAC_JOINT_NAMES])
        self.isaac_to_sdk = i64([cfg.ISAAC_JOINT_NAMES.index(n) for n in cfg.SDK_JOINT_NAMES])
        self.default_dof_pos_isaac = self.default_dof_pos[self.sdk_to_isaac]

        self.tracked_body_ids = i64([self.model.body(n).id for n in cfg.TRACKED_BODY_NAMES])
        self.extended_parent_ids = i64([self.model.body(p).id for _, p, _ in cfg.EXTENDED_JOINTS])
        self.extended_local_offsets = f([o for _, _, o in cfg.EXTENDED_JOINTS])
        self.num_tracked = len(self.tracked_body_ids)
        tracked = self.tracked_body_ids.tolist()
        self.extended_parent_tracked_idx = i64([tracked.index(int(p)) for p in self.extended_parent_ids])

        # whole-robot CoM (cvel is expressed about subtree_com[body_rootid]; rootid == 1 for all
        # bodies of a floating-base robot, so one CoM per world is enough)
        assert np.all(self.model.body_rootid[1:] == 1), "expected a single floating-base root"
        self._mass = f(self.model.body_mass[1:])[None, :, None]
        self._mass_sum = float(self.model.body_mass[1:].sum())

        self.joint_limits_sdk = self.model.jnt_range[1:1 + J].astype(np.float64)
        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                           for b in range(self.model.nbody)]

        # same StaticContext the CPU core exposes, so obs term widths are shared
        self.obs_static = StaticContext(
            model=self.model,
            num_actions=J,
            tracked_body_ids=self.tracked_body_ids.cpu().numpy(),
            extended_parent_ids=self.extended_parent_ids.cpu().numpy(),
            extended_local_offsets=self.extended_local_offsets.cpu().numpy(),
            default_dof_pos_isaac=self.default_dof_pos_isaac.cpu().numpy(),
            isaac_joint_names=list(cfg.ISAAC_JOINT_NAMES),
            future_steps=future_steps,
            future_fk_steps=max(int(future_fk_steps), 1),
        )

        self.last_torque = torch.zeros(num_envs, J, device=dev)
        self._contact_max = torch.zeros(num_envs, self.model.nbody, device=dev)

        # contact-force scratch (device-side; no host sync in the substep loop)
        nacon_max = self.dw.contact.worldid.shape[0]
        self._nacon_max = nacon_max
        self._wp_force = wp.zeros(nacon_max, dtype=wp.spatial_vectorf, device=self.wp_device)
        self._wp_contact_ids = wp.array(np.arange(nacon_max, dtype=np.int32), dtype=wp.int32, device=self.wp_device)
        self._wp_geom_bodyid = wp.array(self.model.geom_bodyid.astype(np.int32), dtype=wp.int32, device=self.wp_device)
        self._wp_acc = wp.zeros((num_envs, self.model.nbody, 3), dtype=wp.float32, device=self.wp_device)
        self._acc = wp.to_torch(self._wp_acc)
        self._force_t = wp.to_torch(self._wp_force)

    # ----- helpers (same semantics as sim_core) -----
    def soft_joint_limits(self, factor: float) -> np.ndarray:
        lo, hi = self.joint_limits_sdk[:, 0], self.joint_limits_sdk[:, 1]
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * factor
        return np.stack([mid - half, mid + half], axis=-1)

    def body_ids_matching(self, patterns: List[str]) -> np.ndarray:
        import re
        return np.array([b for b, n in enumerate(self.body_names)
                         if any(re.fullmatch(p, n) for p in patterns)], dtype=np.int64)

    # ----- reset -----
    def reset_to(self, env_ids: torch.Tensor, root_pos, root_quat_wxyz, root_lin_vel_w,
                 root_ang_vel_w, dof_pos_sdk, dof_vel_sdk):
        """Batched reset of the given envs. All tensors [len(env_ids), ...] on self.device."""
        J = self.num_actions
        self.qpos[env_ids] = 0.0
        self.qvel[env_ids] = 0.0
        self.qpos[env_ids, 0:3] = root_pos
        self.qpos[env_ids, 3:7] = root_quat_wxyz
        self.qpos[env_ids, 7:7 + J] = dof_pos_sdk
        self.qvel[env_ids, 0:3] = root_lin_vel_w
        # the free joint's angular velocity is expressed in the body frame
        self.qvel[env_ids, 3:6] = quat_rotate_inverse(root_quat_wxyz, root_ang_vel_w)
        self.qvel[env_ids, 6:6 + J] = dof_vel_sdk
        self.ctrl[env_ids] = 0.0
        self.last_torque[env_ids] = 0.0
        self._contact_max[env_ids] = 0.0
        mjwarp.forward(self.mw, self.dw)

    # ----- step -----
    def pd_target_sdk(self, raw_action_isaac: torch.Tensor) -> torch.Tensor:
        a = raw_action_isaac.clamp(-self.action_clip, self.action_clip)
        target_isaac = a * self.action_scale + self.default_dof_pos_isaac
        return target_isaac[:, self.isaac_to_sdk]

    def step(self, raw_action_isaac: torch.Tensor):
        """One control step: PD torques over ``decimation`` physics substeps."""
        J = self.num_actions
        pd_target = self.pd_target_sdk(raw_action_isaac)
        self._contact_max.zero_()
        for k in range(1, self.decimation + 1):
            torque = (pd_target - self.qpos[:, 7:7 + J]) * self.stiffness \
                - self.qvel[:, 6:6 + J] * self.damping
            self.ctrl[:] = torque.clamp(-self.torque_limits, self.torque_limits)
            mjwarp.step(self.mw, self.dw)
            if k in self.contact_substeps:
                self._accumulate_contact_forces()
        self.last_torque[:] = self.ctrl
        mjwarp.forward(self.mw, self.dw)   # refresh kinematics / body velocities

    def _accumulate_contact_forces(self):
        mjwarp.contact_force(self.mw, self.dw, self._wp_contact_ids, True, self._wp_force)
        self._acc.zero_()
        wp.launch(_accumulate_contacts, dim=self._nacon_max,
                  inputs=[self.dw.nacon, self.dw.contact.worldid, self.dw.contact.geom,
                          self._wp_force, self._wp_geom_bodyid, self._wp_acc])
        torch.maximum(self._contact_max, self._acc.norm(dim=-1), out=self._contact_max)

    # ----- state -----
    def _com(self) -> torch.Tensor:
        """[N, 3] whole-robot CoM = subtree_com[root]."""
        return (self._mass * self.xipos[:, 1:]).sum(dim=1) / self._mass_sum

    def state(self) -> BatchedRobotState:
        J, N = self.num_actions, self.num_envs
        dof_pos_sdk = self.qpos[:, 7:7 + J]
        dof_vel_sdk = self.qvel[:, 6:6 + J]

        parent_pos = self.xpos[:, self.extended_parent_ids]
        parent_quat = self.xquat[:, self.extended_parent_ids]
        ext_pos = parent_pos + quat_apply(parent_quat, self.extended_local_offsets.expand_as(parent_pos))
        body_pos = torch.cat([self.xpos[:, self.tracked_body_ids], ext_pos], dim=1)
        body_quat = torch.cat([self.xquat[:, self.tracked_body_ids], parent_quat], dim=1)

        # cvel -> link-frame-origin velocity in world orientation (== mj_objectVelocity XBODY)
        off = self.xpos[:, self.tracked_body_ids] - self._com()[:, None, :]
        cv = self.cvel[:, self.tracked_body_ids]
        ang = cv[..., 0:3]
        lin = cv[..., 3:6] + torch.cross(ang, off, dim=-1)

        return BatchedRobotState(
            root_pos=self.qpos[:, 0:3].clone(),
            root_quat=self.qpos[:, 3:7].clone(),
            root_lin_vel_w=self.qvel[:, 0:3].clone(),
            root_ang_vel_b=self.qvel[:, 3:6].clone(),
            dof_pos_sdk=dof_pos_sdk.clone(),
            dof_vel_sdk=dof_vel_sdk.clone(),
            dof_pos_isaac=dof_pos_sdk[:, self.sdk_to_isaac],
            dof_vel_isaac=dof_vel_sdk[:, self.sdk_to_isaac],
            torque_sdk=self.last_torque.clone(),
            body_pos=body_pos,
            body_quat=body_quat,
            body_lin_vel=lin,
            body_ang_vel=ang,
            contact_force=self._contact_max.clone(),
            body_lin_vel_ext=torch.cat([lin, lin[:, self.extended_parent_tracked_idx]], dim=1),
            body_ang_vel_ext=torch.cat([ang, ang[:, self.extended_parent_tracked_idx]], dim=1),
        )
