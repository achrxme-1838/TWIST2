from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from finetune.tasks.ft_cfg import RewardCfg, TerminationCfg
from finetune.helper.math_utils import quat_error_magnitude
from finetune.motion_data.motion_ref import RefBodyState
from finetune.env.sim_core import RobotState


class RewardComputer:
    def __init__(self, cfg: RewardCfg, step_dt: float, soft_joint_limits_sdk: np.ndarray,
                 contact_body_ids: np.ndarray, contact_force_threshold: float):
        self.cfg = cfg
        self.dt = step_dt if cfg.scale_by_dt else 1.0
        self.soft_lo = soft_joint_limits_sdk[:, 0]
        self.soft_hi = soft_joint_limits_sdk[:, 1]
        self.contact_body_ids = contact_body_ids       # bodies whose contact is penalised
        self.contact_threshold = contact_force_threshold
        self.weights = {
            "root_orientation": cfg.root_orientation_w,
            "mimic_extended_body_pos_exp": cfg.body_pos_w,
            "mimic_extended_body_ori_exp": cfg.body_ori_w,
            "mimic_body_lin_vel_exp": cfg.body_lin_vel_w,
            "mimic_body_ang_vel_exp": cfg.body_ang_vel_w,
            "action_rate": cfg.action_rate_w,
            "joint_pos_limits": cfg.joint_pos_limits_w,
            "undesired_contacts": cfg.undesired_contacts_w,
        }

    @property
    def term_names(self):
        return list(self.weights)

    def __call__(self, robot: RobotState, ref: RefBodyState, ref_root_quat: np.ndarray,
                 action: np.ndarray, prev_action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Returns (total reward, unweighted raw term values)."""
        c = self.cfg
        n = ref.lin_vel.shape[0]   # tracked bodies (extended excluded from velocity terms)
        raw = {}

        # -- task (mimic_rewards.py)
        rot_err = quat_error_magnitude(robot.root_quat, ref_root_quat)
        raw["root_orientation"] = float(np.exp(-(rot_err ** 2) / c.root_orientation_sigma))

        pos_err = np.mean(np.square(robot.body_pos - ref.pos))
        raw["mimic_extended_body_pos_exp"] = float(np.exp(-pos_err / c.body_pos_sigma))

        ori_err = np.mean(quat_error_magnitude(robot.body_quat, ref.quat))
        raw["mimic_extended_body_ori_exp"] = float(np.exp(-(ori_err ** 2) / c.body_ori_sigma))

        lin_err = np.mean(np.square(robot.body_lin_vel[:n] - ref.lin_vel))
        raw["mimic_body_lin_vel_exp"] = float(np.exp(-lin_err / c.body_lin_vel_sigma))

        ang_err = np.mean(np.square(robot.body_ang_vel[:n] - ref.ang_vel))
        raw["mimic_body_ang_vel_exp"] = float(np.exp(-ang_err / c.body_ang_vel_sigma))

        # -- regularisation
        raw["action_rate"] = float(np.sum(np.square(action - prev_action)))
        out = np.logical_or(robot.dof_pos_sdk < self.soft_lo, robot.dof_pos_sdk > self.soft_hi)
        raw["joint_pos_limits"] = float(np.sum(out))
        raw["undesired_contacts"] = float(np.sum(robot.contact_force[self.contact_body_ids] > self.contact_threshold))

        total = sum(self.weights[k] * v for k, v in raw.items()) * self.dt
        return float(total), raw


class TerminationChecker:
    def __init__(self, cfg: TerminationCfg):
        self.cfg = cfg

    def diverged(self, robot: RobotState, ref: RefBodyState) -> Tuple[bool, float]:
        dist = np.linalg.norm(robot.body_pos - ref.pos, axis=-1)
        if self.cfg.diverge_any_body:
            flag = bool(np.any(dist > self.cfg.max_ref_motion_dist))
        else:
            flag = bool(dist.mean() > self.cfg.max_ref_motion_dist)
        return flag, float(dist.max())
