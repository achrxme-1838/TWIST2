from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch

import finetune.mdp.batched_terms  # noqa: F401  (registers the batched obs terms)
import finetune.mdp.critic_terms   # noqa: F401  (CPU registry: term widths for priv_*/teacher)
from finetune.env.batched_core import BatchedG1SimCore
from finetune.helper.torch_math import quat_to_rpy, xyzw_to_wxyz
from finetune.mdp.batched_obs import BatchedObsBuilder, BatchedObsContext
from finetune.mdp.batched_rewards import BatchedRewardComputer, BatchedTerminationChecker
from finetune.motion_data.batched_motion_ref import BatchedMotionSet, BatchedRefFK
from finetune.motion_data.motion_ref import MotionSet, RefFK
from finetune.tasks.ft_cfg import FinetuneCfg
from policy_spec import load_spec
from rsl_rl.env import VecEnv


class BatchedG1MimicVecEnv(VecEnv):
    def __init__(self, cfg: FinetuneCfg, device: str = "cuda:0", verbose: bool = True):
        self.cfg = cfg
        self.device = torch.device(device)
        ec, mc = cfg.env, cfg.motion

        self.actor_spec = load_spec(cfg.paths.actor_spec)
        self.critic_spec = load_spec(cfg.paths.critic_spec)
        if (self.critic_spec.future_steps, self.critic_spec.future_interval) != \
           (self.actor_spec.future_steps, self.actor_spec.future_interval):
            raise ValueError("critic spec future_steps/future_interval must match the actor spec")
        fk_steps = max(self.actor_spec.used_future_steps, self.critic_spec.used_future_steps, 1)

        self.num_envs = N = ec.num_envs
        self.core = BatchedG1SimCore(
            cfg.paths.xml, num_envs=N, sim_dt=ec.sim_dt, control_hz=ec.control_hz,
            kp_scale=ec.kp_scale, kd_scale=ec.kd_scale, action_scale=ec.action_scale,
            action_clip=ec.action_clip, device=device,
            future_steps=self.actor_spec.future_steps, future_fk_steps=fk_steps)
        self.num_actions = self.core.num_actions
        self.control_dt = self.core.control_dt
        self.step_dt = self.control_dt
        self.max_episode_length = int(round(ec.episode_length_s / self.control_dt))

        self.obs_static = self.core.obs_static
        self.actor_builder = BatchedObsBuilder(self.actor_spec, self.obs_static, N, self.device)
        self.critic_builder = BatchedObsBuilder(self.critic_spec, self.obs_static, N, self.device)
        self.num_actor_obs = self.actor_builder.obs_dim
        self.num_critic_obs = self.critic_builder.obs_dim

        self.ref_fk = BatchedRefFK(self.core.model, self.core.mw, N, self.core.tracked_body_ids,
                                   self.core.extended_parent_ids, self.core.extended_local_offsets,
                                   self.num_actions, device=self.core.wp_device)
        cpu_fk = RefFK(self.core.model, self.core.tracked_body_ids.cpu().numpy(),
                       self.core.extended_parent_ids.cpu().numpy(),
                       self.core.extended_local_offsets.cpu().numpy(), self.num_actions)
        self.cpu_motions = MotionSet(cfg.paths.motion_file, mc, cpu_fk, self.num_actions, verbose=verbose)
        self.motions = BatchedMotionSet(self.cpu_motions, self.ref_fk, self.device)
        self.future_steps = self.actor_spec.future_motion_steps

        exclude = set(self.core.body_ids_matching(ec.contact_body_exclude).tolist())
        contact_ids = np.array([b for b in range(1, self.core.model.nbody) if b not in exclude],
                               dtype=np.int64)
        self.rewards = BatchedRewardComputer(cfg.reward, self.step_dt,
                                            self.core.soft_joint_limits(ec.soft_joint_limit_factor),
                                            contact_ids, ec.contact_force_threshold, self.device)
        self.terminator = BatchedTerminationChecker(cfg.termination)
        self.term_names = self.rewards.term_names

        self.rng = np.random.default_rng(ec.seed)
        self.force_motion_id: Optional[int] = None
        # [num_envs] explicit per-env motion ids (evaluation); -1 leaves that env to sampling.
        # Takes precedence over force_motion_id.
        self.force_motion_ids: Optional[torch.Tensor] = None
        self.auto_reset = True

        z = lambda *s: torch.zeros(*s, device=self.device)
        self.mid = torch.zeros(N, dtype=torch.long, device=self.device)
        self.t0, self.t = z(N), z(N)
        self.step_count = torch.zeros(N, dtype=torch.long, device=self.device)
        self.last_action, self.prev_action = z(N, self.num_actions), z(N, self.num_actions)
        self.anchor_delta_xy = z(N, 2)
        self.ep_reward, self.ep_max_dist, self.ep_dist_sum = z(N), z(N), z(N)
        self.ep_terms = z(N, len(self.term_names))
        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=self.device)
        self._obs, self._critic_obs = z(N, self.num_actor_obs), z(N, self.num_critic_obs)
        self._wall_t0, self._wall_steps = None, 0

        if verbose:
            print(self.actor_builder.describe())
            print("critic " + self.critic_builder.describe())
            print(f"[env] num_envs={N} control_dt={self.control_dt} decimation={self.core.decimation} "
                  f"max_episode_length={self.max_episode_length} device={device}")
        self.reset()

    # ------------------------------------------------------------------ VecEnv API
    def get_observations(self):
        return self._obs.clone(), {"observations": {"critic": self._critic_obs.clone()}}

    def reset(self):
        self._reset_envs(torch.arange(self.num_envs, device=self.device))
        self._write_obs()
        self.episode_length_buf.zero_()
        return self.get_observations()

    def step(self, actions: torch.Tensor):
        a = actions.detach().to(self.device, torch.float32)
        self.prev_action, self.last_action = self.last_action, a.clone()
        self.core.step(a)
        self.step_count += 1
        self.t = self.t0 + self.step_count.to(self.t0.dtype) * self.control_dt

        robot = self.core.state()
        ref_frame = self.motions.frame(self.mid, self.t)
        ref_body = self.motions.body_state(self.mid, self.t)
        rewards, raw = self.rewards(robot, ref_body, ref_frame.root_quat, self.last_action, self.prev_action)
        diverged, max_dist = self.terminator.diverged(robot, ref_body)

        length = self.motions.length(self.mid)
        motion_end = self.t >= length - 1e-6
        timeout = motion_end | (self.step_count >= self.max_episode_length)
        dones = diverged | timeout
        time_outs = timeout & ~diverged

        self.ep_reward += rewards
        for j, k in enumerate(self.term_names):
            self.ep_terms[:, j] += self.rewards.weights[k] * raw[k] * self.rewards.dt
        self.ep_max_dist = torch.maximum(self.ep_max_dist, max_dist)
        self.ep_dist_sum += max_dist

        done_ids = dones.nonzero(as_tuple=False).flatten()
        episodes = self._episode_records(done_ids, diverged, motion_end) if done_ids.numel() else []
        if done_ids.numel() and self.auto_reset:
            self._reset_envs(done_ids)
        self._write_obs()

        self.episode_length_buf = self.step_count.clone()
        if self.cfg.env.realtime:
            self._pace()
        obs, extras = self.get_observations()
        extras.update({"time_outs": time_outs, "episodes": episodes,
                       "log": {"max_body_dist": float(max_dist.max())}})
        return obs, rewards, dones, extras

    # ------------------------------------------------------------------ internals
    def _episode_records(self, ids: torch.Tensor, diverged: torch.Tensor,
                         motion_end: torch.Tensor) -> List[Dict]:
        """Same per-episode dicts the CPU env emits, for the envs that just finished."""
        ids_c = ids.cpu().numpy()
        rew = self.ep_reward[ids].cpu().numpy()
        ln = self.step_count[ids].cpu().numpy()
        mid = self.mid[ids].cpu().numpy()
        div = diverged[ids].cpu().numpy()
        end = (motion_end & ~diverged)[ids].cpu().numpy()
        mx = self.ep_max_dist[ids].cpu().numpy()
        mean = (self.ep_dist_sum[ids] / self.step_count[ids].clamp_min(1)).cpu().numpy()
        terms = self.ep_terms[ids].cpu().numpy()
        return [{
            "reward": float(rew[i]), "length": int(ln[i]),
            "motion": self.motions.names[int(mid[i])],
            "diverged": float(div[i]), "motion_end": float(end[i]),
            "max_body_dist": float(mx[i]), "mean_max_body_dist": float(mean[i]),
            "terms": {k: float(terms[i, j]) for j, k in enumerate(self.term_names)},
        } for i in range(len(ids_c))]

    def _pace(self):
        if self._wall_t0 is None:
            self._wall_t0, self._wall_steps = time.time(), 0
        self._wall_steps += 1
        ahead = self._wall_t0 + self._wall_steps * self.control_dt - time.time()
        if ahead > 0:
            time.sleep(ahead)

    def _reset_envs(self, ids: torch.Tensor):
        mc, ec = self.cfg.motion, self.cfg.env
        n = ids.numel()
        if self.force_motion_ids is not None:
            mid = self.force_motion_ids[ids].clone()
            free = mid < 0
            if bool(free.any()):
                mid[free] = self.motions.sample_motion(self.rng, int(free.sum()))
        elif self.force_motion_id is not None:
            mid = torch.full((n,), int(self.force_motion_id), dtype=torch.long, device=self.device)
        else:
            mid = self.motions.sample_motion(self.rng, n)
        self.mid[ids] = mid
        t0 = self.motions.sample_start_time(mid, self.rng)
        self.t0[ids] = t0
        self.t[ids] = t0
        self.step_count[ids] = 0
        self.last_action[ids] = 0.0
        self.prev_action[ids] = 0.0
        self.ep_reward[ids] = 0.0
        self.ep_terms[ids] = 0.0
        self.ep_max_dist[ids] = 0.0
        self.ep_dist_sum[ids] = 0.0

        fr = self.motions.frame(mid, t0)
        root_pos = fr.root_pos.clone()
        root_pos[:, 2] += ec.reset_root_z_offset
        self.core.reset_to(ids, root_pos, fr.root_quat, fr.root_lin_vel, fr.root_ang_vel,
                           fr.dof_pos, fr.dof_vel)
        self.actor_builder.reset(ids)
        self.critic_builder.reset(ids)

        # deploy odom anchor: world origin latched at the robot root when the first reference
        # frame is published (t0 + mimic offset, or t0 when disabled)
        t_origin = t0 + (mc.mimic_step_offset * self.control_dt if mc.anchor_origin_at_mimic_frame else 0.0)
        ref_origin_xy = self.motions.frame(mid, t_origin).root_pos[:, :2]
        self.anchor_delta_xy[ids] = self.core.qpos[ids][:, :2] - ref_origin_xy

    def reset_envs_to(self, ids: torch.Tensor, mids: torch.Tensor):
        """Evaluation helper (auto_reset=False): put ``ids`` on motions ``mids`` and refresh the
        observation buffer. ``step`` already wrote one obs frame this tick, so the history ring
        buffer is rewound first and the refreshed frame overwrites it -- envs that were not reset
        recompute the same values, and the reset ones broadcast their new first frame."""
        if self.force_motion_ids is None:
            self.force_motion_ids = torch.full((self.num_envs,), -1, dtype=torch.long,
                                               device=self.device)
        self.force_motion_ids[ids] = mids
        self._reset_envs(ids)
        self.actor_builder.rewind()
        self.critic_builder.rewind()
        self._write_obs()

    def _context(self, robot=None, ref_frame=None, ref_body=None) -> BatchedObsContext:
        mc = self.cfg.motion
        if robot is None:
            robot = self.core.state()
        if ref_frame is None:
            ref_frame = self.motions.frame(self.mid, self.t)
        if ref_body is None:
            ref_body = self.motions.body_state(self.mid, self.t)

        t_mimic = self.t + mc.mimic_step_offset * self.control_dt
        fr_m = self.motions.frame(self.mid, t_mimic)
        action_mimic = self.motions.mimic_obs(fr_m)
        ref_root_xy_w = fr_m.root_pos[:, :2] + self.anchor_delta_xy

        future = None
        if self.actor_spec.needs_future or self.critic_spec.needs_future:
            f_pos, f_rot_xyzw, f_dof, f_lin, f_ang = self.motions.future_raw(
                self.mid, self.t, self.future_steps, self.control_dt)
            f_pos = f_pos.clone()
            f_pos[..., :2] += self.anchor_delta_xy[:, None, :]
            future = (f_pos, xyzw_to_wxyz(f_rot_xyzw), f_dof, f_lin, f_ang)

        length = self.motions.length(self.mid)
        return BatchedObsContext(
            static=self.obs_static, num_envs=self.num_envs, device=self.device,
            dof_pos_isaac=robot.dof_pos_isaac, dof_vel_isaac=robot.dof_vel_isaac,
            ang_vel=robot.root_ang_vel_b, rpy=quat_to_rpy(robot.root_quat),
            last_action=self.last_action, action_mimic=action_mimic,
            robot=robot, ref=ref_body, ref_frame=ref_frame, future_frames=future,
            ref_root_xy_w=ref_root_xy_w, anchor_delta_xy=self.anchor_delta_xy, ref_fk=self.ref_fk,
            motion_phase=torch.where(length > 0, self.t / length.clamp_min(1e-12),
                                     torch.ones_like(self.t)),
            time_left=(length - self.t).clamp_min(0.0))

    def _write_obs(self):
        ctx = self._context()
        self._obs.copy_(self.actor_builder(ctx))
        self._critic_obs.copy_(self.critic_builder(ctx))

    def close(self):
        pass
