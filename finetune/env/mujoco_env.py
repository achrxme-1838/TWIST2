"""MuJoCo motion-tracking environment (rsl_rl ``VecEnv``) for sim-to-sim fine-tuning."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import mujoco
import numpy as np
import torch

from data_utils.rot_utils import quatToEuler
from finetune.mdp.critic_terms import TrainObsContext
from finetune.tasks.ft_cfg import FinetuneCfg
from finetune.motion_data.motion_ref import MotionSet
from obs_builder import ObsBuilder
from policy_spec import PolicySpec, load_spec
from finetune.mdp.rewards import RewardComputer, TerminationChecker
from rsl_rl.env import VecEnv
from finetune.env.sim_core import G1SimCore


class _EnvSlot:
    """Per-env mutable state."""

    def __init__(self, core: G1SimCore, actor_spec: PolicySpec, critic_spec: PolicySpec):
        self.core = core
        self.actor_builder = ObsBuilder(actor_spec, core.obs_static)
        self.critic_builder = ObsBuilder(critic_spec, core.obs_static)
        self.mid = 0
        self.t0 = 0.0
        self.t = 0.0
        self.step_count = 0
        self.last_action = np.zeros(core.num_actions, dtype=np.float32)
        self.prev_action = np.zeros(core.num_actions, dtype=np.float32)
        self.anchor_delta_xy = np.zeros(2)
        self.ep_reward = 0.0
        self.ep_terms: Dict[str, float] = {}
        self.ep_max_dist = 0.0
        self.ep_dist_sum = 0.0


class G1MimicVecEnv(VecEnv):
    def __init__(self, cfg: FinetuneCfg, device: str = "cpu", verbose: bool = True):
        self.cfg = cfg
        self.device = torch.device(device)
        ec, mc = cfg.env, cfg.motion

        self.actor_spec = load_spec(cfg.paths.actor_spec)
        self.critic_spec = load_spec(cfg.paths.critic_spec)
        if (self.critic_spec.future_steps, self.critic_spec.future_interval) != \
           (self.actor_spec.future_steps, self.actor_spec.future_interval):
            raise ValueError("critic spec future_steps/future_interval must match the actor spec")
        fk_steps = max(self.actor_spec.used_future_steps, self.critic_spec.used_future_steps, 1)

        self.model = mujoco.MjModel.from_xml_path(cfg.paths.xml)
        self.num_envs = ec.num_envs
        self.slots: List[_EnvSlot] = []
        for _ in range(self.num_envs):
            core = G1SimCore(cfg.paths.xml, self.actor_spec.future_steps, fk_steps,
                             sim_dt=ec.sim_dt, control_hz=ec.control_hz,
                             kp_scale=ec.kp_scale, kd_scale=ec.kd_scale,
                             action_scale=ec.action_scale, action_clip=ec.action_clip, model=self.model)
            self.slots.append(_EnvSlot(core, self.actor_spec, self.critic_spec))
        core0 = self.slots[0].core
        self.num_actions = core0.num_actions
        self.control_dt = core0.control_dt
        self.step_dt = self.control_dt
        self.max_episode_length = int(round(ec.episode_length_s / self.control_dt))
        self.num_actor_obs = self.slots[0].actor_builder.obs_dim
        self.num_critic_obs = self.slots[0].critic_builder.obs_dim

        self.motions = MotionSet(cfg.paths.motion_file, mc, core0.ref_fk, self.num_actions, verbose=verbose)
        self.future_steps = self.actor_spec.future_motion_steps   # control-step offsets of the T future frames

        exclude = core0.body_ids_matching(ec.contact_body_exclude)
        contact_ids = np.array([i for i in range(1, self.model.nbody) if i not in set(exclude.tolist())], dtype=np.int64)
        self.rewards = RewardComputer(cfg.reward, self.step_dt, core0.soft_joint_limits(ec.soft_joint_limit_factor),
                                      contact_ids, ec.contact_force_threshold)
        self.terminator = TerminationChecker(cfg.termination)

        self.rng = np.random.default_rng(ec.seed)
        self.force_motion_id: Optional[int] = None   # evaluation: pin every reset to one motion
        self.auto_reset = True                        # False: leave a done slot as is (caller resets)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._obs = torch.zeros(self.num_envs, self.num_actor_obs, device=self.device)
        self._critic_obs = torch.zeros(self.num_envs, self.num_critic_obs, device=self.device)
        self._wall_t0 = None
        self._wall_steps = 0

        if verbose:
            print(self.slots[0].actor_builder.describe())
            print("critic " + self.slots[0].critic_builder.describe())
            print(f"[env] num_envs={self.num_envs} control_dt={self.control_dt} decimation={core0.decimation} "
                  f"max_episode_length={self.max_episode_length}")
        self.reset()

    # ------------------------------------------------------------------ VecEnv API
    def get_observations(self):
        obs = self._obs.clone()
        return obs, {"observations": {"critic": self._critic_obs.clone()}}

    def reset(self):
        for i, slot in enumerate(self.slots):
            self._reset_slot(slot)
            self._write_obs(i, slot, *self._ref_now(slot))
        self.episode_length_buf.zero_()
        return self.get_observations()

    def step(self, actions: torch.Tensor):
        actions_np = actions.detach().cpu().numpy().astype(np.float32)
        rewards = torch.zeros(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        episodes = []
        step_log = {"max_body_dist": 0.0}

        for i, slot in enumerate(self.slots):
            a = actions_np[i]
            slot.prev_action = slot.last_action
            slot.last_action = a.copy()
            slot.core.step(a)
            slot.step_count += 1
            slot.t = slot.t0 + slot.step_count * self.control_dt

            robot = slot.core.state()
            ref_frame, ref_body = self._ref_now(slot)
            reward, raw = self.rewards(robot, ref_body, ref_frame.root_quat, slot.last_action, slot.prev_action)
            diverged, max_dist = self.terminator.diverged(robot, ref_body)
            motion_end = slot.t >= self.motions.length(slot.mid) - 1e-6
            timeout = motion_end or slot.step_count >= self.max_episode_length
            done = diverged or timeout

            rewards[i] = reward
            dones[i] = done
            time_outs[i] = bool(timeout and not diverged)
            slot.ep_reward += reward
            for k, v in raw.items():
                slot.ep_terms[k] = slot.ep_terms.get(k, 0.0) + self.rewards.weights[k] * v * self.rewards.dt
            slot.ep_max_dist = max(slot.ep_max_dist, max_dist)
            slot.ep_dist_sum += max_dist
            step_log["max_body_dist"] = max(step_log["max_body_dist"], max_dist)

            if done:
                episodes.append({
                    "reward": slot.ep_reward,
                    "length": slot.step_count,
                    "motion": self.motions.names[slot.mid],
                    "diverged": float(diverged),
                    "motion_end": float(motion_end and not diverged),
                    "max_body_dist": slot.ep_max_dist,
                    "mean_max_body_dist": slot.ep_dist_sum / max(slot.step_count, 1),
                    "terms": dict(slot.ep_terms),
                })
                if self.auto_reset:
                    self._reset_slot(slot)
                    ref_frame, ref_body = self._ref_now(slot)
                    robot = None
            self._write_obs(i, slot, ref_frame, ref_body, robot)

        self.episode_length_buf = torch.tensor([s.step_count for s in self.slots], dtype=torch.long, device=self.device)
        if self.cfg.env.realtime:
            self._pace()
        obs, extras = self.get_observations()
        extras.update({"time_outs": time_outs, "episodes": episodes, "log": step_log})
        return obs, rewards, dones, extras

    # ------------------------------------------------------------------ internals
    def _pace(self):
        if self._wall_t0 is None:
            self._wall_t0 = time.time()
            self._wall_steps = 0
        self._wall_steps += 1
        ahead = self._wall_t0 + self._wall_steps * self.control_dt - time.time()
        if ahead > 0:
            time.sleep(ahead)

    def _ref_now(self, slot: _EnvSlot):
        fr = self.motions.frame(slot.mid, slot.t)
        body = self.motions.body_state(slot.mid, slot.t)
        return fr, body

    def _reset_slot(self, slot: _EnvSlot):
        mc, ec = self.cfg.motion, self.cfg.env
        slot.mid = self.force_motion_id if self.force_motion_id is not None else self.motions.sample_motion(self.rng)
        slot.t0 = self.motions.sample_start_time(slot.mid, self.rng)
        slot.t = slot.t0
        slot.step_count = 0
        slot.last_action[:] = 0.0
        slot.prev_action = np.zeros_like(slot.last_action)
        slot.ep_reward = 0.0
        slot.ep_terms = {}
        slot.ep_max_dist = 0.0
        slot.ep_dist_sum = 0.0

        fr = self.motions.frame(slot.mid, slot.t0)
        root_pos = fr.root_pos.copy()
        root_pos[2] += ec.reset_root_z_offset
        slot.core.reset_to(root_pos, fr.root_quat, fr.root_lin_vel, fr.root_ang_vel, fr.dof_pos, fr.dof_vel)
        slot.actor_builder.reset()
        slot.critic_builder.reset()

        # Deploy odom anchor: world origin latched at the robot root when the first
        # reference frame is published (frame t0 + mimic offset in deploy, t0 if disabled).
        t_origin = slot.t0 + (mc.mimic_step_offset * self.control_dt if mc.anchor_origin_at_mimic_frame else 0.0)
        ref_origin_xy = self.motions.frame(slot.mid, t_origin).root_pos[:2]
        slot.anchor_delta_xy = slot.core.data.qpos[:2].copy() - ref_origin_xy

    def _write_obs(self, i: int, slot: _EnvSlot, ref_frame, ref_body, robot=None):
        core = slot.core
        if robot is None:
            robot = core.state()
        mc = self.cfg.motion
        t_mimic = slot.t + mc.mimic_step_offset * self.control_dt
        fr_m = self.motions.frame(slot.mid, t_mimic)
        action_mimic = self.motions.mimic_obs(fr_m)
        ref_root_xy_w = fr_m.root_pos[:2] + slot.anchor_delta_xy
        future_raw = self.motions.future_raw(slot.mid, slot.t, self.future_steps, self.control_dt) \
            if (self.actor_spec.needs_future or self.critic_spec.needs_future) else None
        length = self.motions.length(slot.mid)

        ctx = TrainObsContext(
            static=core.obs_static,
            data=core.data,
            dof_pos_isaac=robot.dof_pos_isaac,
            dof_vel_isaac=robot.dof_vel_isaac,
            ang_vel=robot.root_ang_vel_b,
            rpy=quatToEuler(robot.root_quat),
            last_action=slot.last_action,
            action_mimic=action_mimic,
            ref_data=core.ref_data,
            future_ref_data=core.future_ref_data,
            future_raw=future_raw,
            ref_root_xy_w=ref_root_xy_w,
            odom_on=True,
            anchor_delta_xy=slot.anchor_delta_xy,
            robot=robot,
            ref=ref_body,
            ref_frame=ref_frame,
            motion_phase=float(slot.t / length) if length > 0 else 1.0,
            time_left=float(max(length - slot.t, 0.0)),
        )
        self._obs[i] = torch.from_numpy(slot.actor_builder(ctx)).to(self.device)
        self._critic_obs[i] = torch.from_numpy(slot.critic_builder(ctx)).to(self.device)

    # ------------------------------------------------------------------ misc
    def close(self):
        pass
