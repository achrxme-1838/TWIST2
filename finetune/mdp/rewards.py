from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from cfg import g1_29dof_cfg as rcfg   # deploy_real
from finetune.tasks.ft_cfg import RewardCfg, TerminationCfg
from finetune.helper.math_utils import quat_error_magnitude
from finetune.motion_data.motion_ref import RefBodyState
from finetune.env.sim_core import RobotState


IMPLEMENTED_TERMS = (
    "root_orientation",
    "mimic_extended_body_pos_exp",
    "mimic_extended_body_ori_exp",
    "mimic_body_lin_vel_exp",
    "mimic_body_ang_vel_exp",
    "feet_pos_exp",
    "action_rate",
    "joint_pos_limits",
    "undesired_contacts",
)


def check_terms(terms: Dict[str, float]) -> Dict[str, float]:
    unknown = [k for k in terms if k not in IMPLEMENTED_TERMS]
    if unknown:
        raise ValueError(
            f"cfg.reward.terms has unimplemented reward term(s): {unknown}. "
            f"Implemented: {list(IMPLEMENTED_TERMS)}. Add the computation to "
            "finetune/mdp/rewards.py and finetune/mdp/batched_rewards.py first."
        )
    if not terms:
        raise ValueError("cfg.reward.terms is empty -- the reward would always be 0")
    return dict(terms)


class RewardComputer:
    def __init__(self, cfg: RewardCfg, step_dt: float, soft_joint_limits_sdk: np.ndarray,
                 contact_body_ids: np.ndarray, contact_force_threshold: float):
        self.cfg = cfg
        self.dt = step_dt if cfg.scale_by_dt else 1.0
        self.soft_lo = soft_joint_limits_sdk[:, 0]
        self.soft_hi = soft_joint_limits_sdk[:, 1]
        self.contact_body_ids = contact_body_ids       # bodies whose contact is penalised
        self.contact_threshold = contact_force_threshold
        # indices into the extended body list (tracked bodies first, same order as RefBodyState)
        self.feet_body_ids = np.array([list(rcfg.TRACKED_BODY_NAMES).index(n)
                                       for n in cfg.feet_body_names], dtype=np.int64)
        self.weights = check_terms(cfg.terms)

    @property
    def term_names(self):
        return list(self.weights)

    def describe(self) -> str:
        sig = {"root_orientation": self.cfg.root_orientation_sigma,
               "mimic_extended_body_pos_exp": self.cfg.body_pos_sigma,
               "mimic_extended_body_ori_exp": self.cfg.body_ori_sigma,
               "mimic_body_lin_vel_exp": self.cfg.body_lin_vel_sigma,
               "mimic_body_ang_vel_exp": self.cfg.body_ang_vel_sigma,
               "feet_pos_exp": self.cfg.feet_pos_sigma}
        lines = [f"reward composition ({len(self.weights)} active terms, "
                 f"scale_by_dt={self.cfg.scale_by_dt}, dt={self.dt}):"]
        for k, w in self.weights.items():
            s = sig.get(k)
            lines.append(f"  {k:30s} w={w:+8.3f}" + (f"   sigma={s:.6g} (={s ** 0.5:.4g})" if s else ""))
        off = [k for k in IMPLEMENTED_TERMS if k not in self.weights]
        if off:
            lines.append(f"  not in cfg.reward.terms: {', '.join(off)}")
        return "\n".join(lines)

    def __call__(self, robot: RobotState, ref: RefBodyState, ref_root_quat: np.ndarray,
                 action: np.ndarray, prev_action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Returns (total reward, unweighted raw term values)."""
        c = self.cfg
        n = ref.lin_vel.shape[0]   # tracked bodies (extended excluded from velocity terms)
        raw = {}

        w = self.weights

        # -- task (mimic_rewards.py)
        if "root_orientation" in w:
            rot_err = quat_error_magnitude(robot.root_quat, ref_root_quat)
            raw["root_orientation"] = float(np.exp(-(rot_err ** 2) / c.root_orientation_sigma))

        if "mimic_extended_body_pos_exp" in w:
            pos_err = np.mean(np.square(robot.body_pos - ref.pos))
            raw["mimic_extended_body_pos_exp"] = float(np.exp(-pos_err / c.body_pos_sigma))

        if "mimic_extended_body_ori_exp" in w:
            ori_err = np.mean(quat_error_magnitude(robot.body_quat, ref.quat))
            raw["mimic_extended_body_ori_exp"] = float(np.exp(-(ori_err ** 2) / c.body_ori_sigma))

        if "mimic_body_lin_vel_exp" in w:
            lin_err = np.mean(np.square(robot.body_lin_vel[:n] - ref.lin_vel))
            raw["mimic_body_lin_vel_exp"] = float(np.exp(-lin_err / c.body_lin_vel_sigma))

        if "mimic_body_ang_vel_exp" in w:
            ang_err = np.mean(np.square(robot.body_ang_vel[:n] - ref.ang_vel))
            raw["mimic_body_ang_vel_exp"] = float(np.exp(-ang_err / c.body_ang_vel_sigma))

        if "feet_pos_exp" in w:
            # DEX mimic_specific_body_pos_exp: mean over the selected bodies *and* xyz
            feet_err = np.mean(np.square(robot.body_pos[self.feet_body_ids] - ref.pos[self.feet_body_ids]))
            raw["feet_pos_exp"] = float(np.exp(-feet_err / c.feet_pos_sigma))

        # -- regularisation
        if "action_rate" in w:
            raw["action_rate"] = float(np.sum(np.square(action - prev_action)))
        if "joint_pos_limits" in w:
            out = np.logical_or(robot.dof_pos_sdk < self.soft_lo, robot.dof_pos_sdk > self.soft_hi)
            raw["joint_pos_limits"] = float(np.sum(out))
        if "undesired_contacts" in w:
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
