from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from cfg import g1_29dof_cfg as rcfg   # deploy_real
from finetune.helper.torch_math import quat_error_magnitude
from finetune.mdp.rewards import RewardComputer, check_terms
from finetune.motion_data.batched_motion_ref import BatchedRefBodyState
from finetune.tasks.ft_cfg import RewardCfg, TerminationCfg


class BatchedRewardComputer:
    def __init__(self, cfg: RewardCfg, step_dt: float, soft_joint_limits_sdk: np.ndarray,
                 contact_body_ids: np.ndarray, contact_force_threshold: float,
                 device: torch.device):
        self.cfg = cfg
        self.device = device
        self.dt = step_dt if cfg.scale_by_dt else 1.0
        f = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
        self.soft_lo = f(soft_joint_limits_sdk[:, 0])
        self.soft_hi = f(soft_joint_limits_sdk[:, 1])
        self.contact_body_ids = torch.as_tensor(np.asarray(contact_body_ids, dtype=np.int64), device=device)
        self.contact_threshold = contact_force_threshold
        self.feet_body_ids = torch.as_tensor(
            np.array([list(rcfg.TRACKED_BODY_NAMES).index(n) for n in cfg.feet_body_names],
                     dtype=np.int64), device=device)
        self.weights = check_terms(cfg.terms)

    @property
    def term_names(self):
        return list(self.weights)

    def describe(self) -> str:
        return RewardComputer.describe(self)

    def __call__(self, robot, ref: BatchedRefBodyState, ref_root_quat: torch.Tensor,
                 action: torch.Tensor, prev_action: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """([N] total reward, {name: [N] unweighted raw value})."""
        c = self.cfg
        n = ref.lin_vel.shape[1]      # tracked bodies (extended excluded from the velocity terms)
        raw: Dict[str, torch.Tensor] = {}

        w = self.weights

        if "root_orientation" in w:
            rot_err = quat_error_magnitude(robot.root_quat, ref_root_quat)
            raw["root_orientation"] = torch.exp(-(rot_err ** 2) / c.root_orientation_sigma)

        if "mimic_extended_body_pos_exp" in w:
            pos_err = (robot.body_pos - ref.pos).pow(2).flatten(1).mean(dim=1)
            raw["mimic_extended_body_pos_exp"] = torch.exp(-pos_err / c.body_pos_sigma)

        if "mimic_extended_body_ori_exp" in w:
            ori_err = quat_error_magnitude(robot.body_quat, ref.quat).mean(dim=1)
            raw["mimic_extended_body_ori_exp"] = torch.exp(-(ori_err ** 2) / c.body_ori_sigma)

        if "mimic_body_lin_vel_exp" in w:
            lin_err = (robot.body_lin_vel[:, :n] - ref.lin_vel).pow(2).flatten(1).mean(dim=1)
            raw["mimic_body_lin_vel_exp"] = torch.exp(-lin_err / c.body_lin_vel_sigma)

        if "mimic_body_ang_vel_exp" in w:
            ang_err = (robot.body_ang_vel[:, :n] - ref.ang_vel).pow(2).flatten(1).mean(dim=1)
            raw["mimic_body_ang_vel_exp"] = torch.exp(-ang_err / c.body_ang_vel_sigma)

        if "feet_pos_exp" in w:
            # DEX mimic_specific_body_pos_exp: mean over the selected bodies *and* xyz
            ids = self.feet_body_ids
            feet_err = (robot.body_pos[:, ids] - ref.pos[:, ids]).pow(2).flatten(1).mean(dim=1)
            raw["feet_pos_exp"] = torch.exp(-feet_err / c.feet_pos_sigma)

        if "action_rate" in w:
            raw["action_rate"] = (action - prev_action).pow(2).sum(dim=1)
        if "joint_pos_limits" in w:
            out = (robot.dof_pos_sdk < self.soft_lo) | (robot.dof_pos_sdk > self.soft_hi)
            raw["joint_pos_limits"] = out.float().sum(dim=1)
        if "undesired_contacts" in w:
            contact = robot.contact_force[:, self.contact_body_ids] > self.contact_threshold
            raw["undesired_contacts"] = contact.float().sum(dim=1)

        total = sum(self.weights[k] * v for k, v in raw.items()) * self.dt
        return total, raw


class BatchedTerminationChecker:
    def __init__(self, cfg: TerminationCfg):
        self.cfg = cfg

    def diverged(self, robot, ref: BatchedRefBodyState) -> Tuple[torch.Tensor, torch.Tensor]:
        """([N] bool diverged, [N] max per-body distance)."""
        dist = (robot.body_pos - ref.pos).norm(dim=-1)        # [N, B_ext]
        limit = self.cfg.max_ref_motion_dist
        flag = (dist > limit).any(dim=1) if self.cfg.diverge_any_body else dist.mean(dim=1) > limit
        return flag, dist.amax(dim=1)
