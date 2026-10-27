"""IsaacLab-parity motion-quality metrics (smoothness, foot slip, constraint violations)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from cfg import g1_29dof_cfg as rcfg   # deploy_real
from finetune.helper.torch_math import quat_apply


def _eval_cfg():
    """The IsaacLab evaluation-only constants: constraint limits, foot geometry, keypoints."""
    from dex_rl_lab.robots.g1.g1_29dof.configs.g1_29dof_eval_constraint_cfg import (
        G1_29DOF_EVAL_CONSTRAINTS, G1_29DOF_EVAL_FOOT_CFG, G1_29DOF_EVAL_JERK_KEYPOINTS,
    )
    return G1_29DOF_EVAL_CONSTRAINTS, G1_29DOF_EVAL_FOOT_CFG, G1_29DOF_EVAL_JERK_KEYPOINTS


def _constraint_specs(core, soft_joint_limit_factor: float, constraint_cfg: dict, device):
    """resolve_constraint_specs against the mujoco model, in SDK (qpos) joint order.

    The metrics sum over joints, so the order only has to agree with the joint_pos / joint_vel /
    torque tensors handed to update() -- which are the SDK-order ones. A ``sim`` source reads the
    limit the simulator itself enforces: the XML jnt_range shrunk by soft_joint_limit_factor,
    which is how IsaacLab builds soft_joint_pos_limits (both use 0.9), and cfg.TORQUE_LIMITS,
    which mirrors effort_limit_sim.
    """
    from dex_rl_lab.utils.motion_quality_metrics import resolve_constraint_specs

    entry = (constraint_cfg or {}).get("joint_vel") or {}
    if entry.get("enabled", False) and str(entry.get("source", "sim")) == "sim":
        raise SystemExit(
            "evaluation_constraint_cfg['joint_vel']['source'] == 'sim' has no mujoco counterpart:\n"
            "the XML carries no joint velocity limit. Use 'manual' with G1_29DOF_JOINT_VEL_LIMIT\n"
            "(that is what g1_29dof_eval_constraint_cfg.py ships), or disable the constraint.")

    f = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    robot = SimpleNamespace(
        joint_names=list(rcfg.SDK_JOINT_NAMES),
        data=SimpleNamespace(
            soft_joint_pos_limits=f(core.soft_joint_limits(soft_joint_limit_factor)),  # [J, 2]
            joint_effort_limits=f(rcfg.TORQUE_LIMITS),                                 # [J]
        ),
    )
    return resolve_constraint_specs(robot, constraint_cfg, device)


class BatchedQualityTracker:
    """Per-env quality-metric accumulator for the batched mujoco_warp evaluation."""

    def __init__(self, env, device, jerk_keypoints: list[str] | None = None,
                 warmup_steps: int = 0, verbose: bool = True):
        from dex_rl_lab.utils.motion_quality_metrics import print_constraint_specs

        constraint_cfg, foot_cfg, default_keypoints = _eval_cfg()
        self.device = torch.device(device)
        self.core = env.core

        specs = _constraint_specs(self.core, env.cfg.env.soft_joint_limit_factor,
                                  constraint_cfg, self.device)
        if verbose:
            print_constraint_specs(specs, list(rcfg.SDK_JOINT_NAMES))

        # Same body list the metrics score: tracked + extended, i.e. robot.body_pos's axis 1 and
        # IsaacLab's robot_tracked_body_names_extended.
        body_names = list(rcfg.TRACKED_BODY_NAMES) + [n for n, _, _ in rcfg.EXTENDED_JOINTS]
        keypoints = list(default_keypoints if jerk_keypoints is None else jerk_keypoints)
        unknown = [n for n in keypoints if n not in body_names]
        if unknown:
            raise SystemExit(
                f"Unknown jerk keypoint(s) {unknown}. Valid bodies ({len(body_names)}): {body_names}")
        keypoint_ids = [body_names.index(n) for n in keypoints]
        self.keypoint_names = keypoints

        foot_names = list(foot_cfg["body_names"])
        self._foot_body_ids = torch.as_tensor(
            [self.core.model.body(n).id for n in foot_names], dtype=torch.long, device=self.device)
        self._sole_offsets_b = torch.as_tensor(
            np.asarray(foot_cfg["sole_offsets"], dtype=np.float32), device=self.device)
        self._force_threshold = float(foot_cfg.get("contact_force_threshold", 1.0))
        self._height_threshold = float(foot_cfg.get("contact_height_threshold", 0.1))
        self._num_feet = len(foot_names)
        self._num_sole_points = self._sole_offsets_b.shape[0]

        self._metrics = _PerTrialQualityMetrics(
            num_envs=env.num_envs, device=self.device, dt=env.step_dt,
            num_bodies=len(body_names), keypoint_ids=keypoint_ids, keypoint_names=keypoints,
            num_feet=self._num_feet, num_sole_points=self._num_sole_points,
            constraint_specs=specs, warmup_steps=int(warmup_steps),
        )
        if verbose:
            print(f"[eval] quality metrics on. body_jerk over {len(body_names)} bodies; "
                  f"keypoint_jerk over {keypoints}; feet {foot_names} with "
                  f"{self._num_sole_points} sole points; warmup {int(warmup_steps)} steps.")

    # ------------------------------------------------------------------ per step
    def _sole_points(self, foot_pos_w: torch.Tensor, foot_quat_w: torch.Tensor) -> torch.Tensor:
        """[N, F, P, 3] world sole proxy points, the command term's sole_points."""
        n, f, p = foot_pos_w.shape[0], self._num_feet, self._num_sole_points
        offsets = self._sole_offsets_b.view(1, 1, p, 3).expand(n, f, p, 3)
        quat = foot_quat_w.unsqueeze(2).expand(n, f, p, 4)
        return quat_apply(quat, offsets) + foot_pos_w.unsqueeze(2)

    def update(self, robot, action: torch.Tensor, active: torch.Tensor):
        """Fold one control step in. ``active`` is the pre-step mask: the env that finished on
        this very step still counts, the ones already idle do not -- IsaacLab's ~self._terminated.
        """
        core = self.core
        foot_pos_w = core.xpos[:, self._foot_body_ids]      # [N, F, 3]
        foot_quat_w = core.xquat[:, self._foot_body_ids]     # [N, F, 4] wxyz
        # Same contact test as the command term: net force over the contact substeps (so a
        # contact lasting only part of the control step still counts) AND the foot low enough.
        force_contact = robot.contact_force[:, self._foot_body_ids] > self._force_threshold
        low_enough = foot_pos_w[:, :, 2] <= self._height_threshold

        self._metrics.update(
            body_pos=robot.body_pos,
            action=action,
            sole_points=self._sole_points(foot_pos_w, foot_quat_w),
            foot_contact=force_contact & low_enough,
            joint_pos=robot.dof_pos_sdk,
            joint_vel=robot.dof_vel_sdk,
            # Pre-clip torque: core.step() saturates ctrl at TORQUE_LIMITS, which are the very
            # limits cv_torque scores, so the clipped torque would violate them by exactly zero.
            # IsaacLab reads computed_torque for the same reason.
            torque=core.last_torque_raw,
            active=active,
        )

    def pop(self, ids: list[int]) -> list[dict[str, float]]:
        """Finished trials' metric values, in the order of ``ids``; clears those envs."""
        return self._metrics.pop(ids)


class _PerTrialQualityMetrics:
    """MotionQualityMetrics plus a per-env read-and-clear.

    IsaacLab resets every accumulator between dataset batches; here a trial ends on its own, so
    only that env's buffers are cleared. Zeroing its ``_run_len`` also re-arms the window gates
    (run >= 2 / 3 / 4), so the stale rolling history left behind is never read -- the same
    argument the base class makes for envs that terminated mid-batch.
    """

    def __init__(self, **kwargs):
        from dex_rl_lab.utils.motion_quality_metrics import MotionQualityMetrics
        self._m = MotionQualityMetrics(**kwargs)
        self._device = self._m._device

    def update(self, **signals):
        self._m.update(**signals)

    def pop(self, ids: list[int]) -> list[dict[str, float]]:
        m = self._m
        # finalize() reduces all envs at once; a handful of small device-to-host copies per step
        # is cheaper than reimplementing the reduction for a subset, and keeps it IsaacLab's.
        values = m.finalize()
        out = [{name: float(arr[i]) for name, arr in values.items()} for i in ids]

        sel = torch.as_tensor(ids, dtype=torch.long, device=self._device)
        m._run_len[sel] = 0
        for buf in (m._ar_sum, m._ar_cnt, m._aj_sum, m._aj_cnt, m._bj_sum, m._bj_cnt,
                    m._kj_sum, m._kj_cnt, m._kj_sum_each, m._kj_cnt_each,
                    m._slide_sum, m._slide_cnt, m._slide_max, m._contact_sum, m._contact_cnt):
            buf[sel] = 0
        for store in (m._cv_mag_sum, m._cv_mag_cnt, m._cv_step_cnt, m._cv_step_tot, m._cv_max):
            for buf in store.values():
                buf[sel] = 0
        return out
