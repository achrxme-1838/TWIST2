from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from cfg import g1_29dof_cfg as rcfg   # deploy_real
from finetune.helper.torch_math import quat_error_magnitude
from finetune.mdp.adaptive_sigma import AdaptiveSigma
from finetune.mdp.rewards import RewardComputer, check_terms
from finetune.motion_data.batched_motion_ref import BatchedRefBodyState
from finetune.tasks.ft_cfg import SIGMA_FIELDS, RewardCfg, TerminationCfg, UpdateSigmaCfg


class BatchedRewardComputer:
    def __init__(self, cfg: RewardCfg, step_dt: float, soft_joint_limits_sdk: np.ndarray,
                 contact_body_ids: np.ndarray, contact_force_threshold: float,
                 device: torch.device, update_sigma: Optional[UpdateSigmaCfg] = None,
                 num_motions: int = 0, num_envs: int = 0, motion_names: Optional[List[str]] = None):
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

        # -- PBHC adaptive tracking factor; None keeps the fixed RewardCfg sigmas.
        self.adaptive_sigma: Optional[AdaptiveSigma] = None
        if update_sigma is not None and update_sigma.enabled:
            sigma_init = {k: getattr(cfg, SIGMA_FIELDS[k]) for k in update_sigma.terms}
            self.adaptive_sigma = AdaptiveSigma(update_sigma, sigma_init, num_motions, num_envs,
                                                device, motion_names)

    @property
    def term_names(self):
        return list(self.weights)

    def describe(self) -> str:
        text = RewardComputer.describe(self)
        if self.adaptive_sigma is not None:
            text += "\n" + self.adaptive_sigma.describe()
        return text

    def _sigmas(self, mid: Optional[torch.Tensor]) -> Dict[str, object]:
        """{term: sigma} for this step: the RewardCfg float, or the adaptive factor."""
        sig = {k: getattr(self.cfg, f) for k, f in SIGMA_FIELDS.items()}
        a = self.adaptive_sigma
        if a is not None:
            if mid is None:
                raise ValueError("adaptive sigma is on, so the reward needs mid (per-env motion id)")
            a.begin_step(mid)
            for k in a.term_names:
                sig[k] = a.sigma(k)
        return sig

    def __call__(self, robot, ref: BatchedRefBodyState, ref_root_quat: torch.Tensor,
                 action: torch.Tensor, prev_action: torch.Tensor,
                 mid: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """([N] total reward, {name: [N] unweighted raw value}).

        ``mid`` ([N] motion id per env) is needed only when the adaptive sigma is on.
        """
        n = ref.lin_vel.shape[1]      # tracked bodies (extended excluded from the velocity terms)
        raw: Dict[str, torch.Tensor] = {}
        err: Dict[str, torch.Tensor] = {}     # x of each exp(-x / sigma) term
        sig = self._sigmas(mid)

        w = self.weights

        if "root_orientation" in w:
            rot_err = quat_error_magnitude(robot.root_quat, ref_root_quat)
            err["root_orientation"] = rot_err ** 2

        if "mimic_extended_body_pos_exp" in w:
            err["mimic_extended_body_pos_exp"] = (robot.body_pos - ref.pos).pow(2).flatten(1).mean(dim=1)

        if "mimic_extended_body_ori_exp" in w:
            ori_err = quat_error_magnitude(robot.body_quat, ref.quat).mean(dim=1)
            err["mimic_extended_body_ori_exp"] = ori_err ** 2

        if "mimic_body_lin_vel_exp" in w:
            err["mimic_body_lin_vel_exp"] = (robot.body_lin_vel[:, :n] - ref.lin_vel).pow(2).flatten(1).mean(dim=1)

        if "mimic_body_ang_vel_exp" in w:
            err["mimic_body_ang_vel_exp"] = (robot.body_ang_vel[:, :n] - ref.ang_vel).pow(2).flatten(1).mean(dim=1)

        if "feet_pos_exp" in w:
            # DEX mimic_specific_body_pos_exp: mean over the selected bodies *and* xyz
            ids = self.feet_body_ids
            err["feet_pos_exp"] = (robot.body_pos[:, ids] - ref.pos[:, ids]).pow(2).flatten(1).mean(dim=1)

        for k, x in err.items():
            raw[k] = torch.exp(-x / sig[k])
        # -- tighten the factors from this step's error. Done after the reward, so this step
        #    still uses the factor the policy was rolled out with.
        if self.adaptive_sigma is not None:
            for k in self.adaptive_sigma.term_names:
                self.adaptive_sigma.update(k, err[k])

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
