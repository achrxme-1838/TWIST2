"""Observation term registry for the deployed G1 policies."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from observations import (
    compute_diff_body_pos_b,
    compute_diff_body_pos_pb,
    compute_diff_body_tannorm_b,
    compute_future_body_pos_w,
    compute_future_motion_anchor,
    future_pos_h_from_body_pos,
    resolve_future_frames,
)
from data_utils.rot_utils import quatToEuler
from utils.math import quat_apply, quat_conj


# --------------------------------------------------------------------------- context
@dataclass
class StaticContext:
    """Per-run constants the term dims and functions depend on."""
    model: Any                               # mujoco.MjModel (FK for ref/future frames)
    num_actions: int
    tracked_body_ids: np.ndarray
    extended_parent_ids: np.ndarray
    extended_local_offsets: np.ndarray
    default_dof_pos_isaac: np.ndarray        # [J] Isaac order
    isaac_joint_names: List[str]             # [J] Isaac order (for regex params)
    future_steps: int = 1                    # frames the motion server publishes (T)
    future_fk_steps: Optional[int] = None    # frames to run body FK on (default: all T)

    @property
    def num_bodies(self) -> int:
        return len(self.tracked_body_ids) + len(self.extended_parent_ids)

    def joint_indices(self, patterns) -> List[int]:
        """Isaac-order indices of joints whose name fully matches any regex (IsaacLab
        `resolve_matching_names` semantics)."""
        if isinstance(patterns, str):
            patterns = [patterns]
        return [i for i, n in enumerate(self.isaac_joint_names)
                if any(re.fullmatch(p, n) for p in patterns)]


@dataclass
class ObsContext:
    """One control tick. Robot state arrays are already in Isaac joint order."""
    static: StaticContext
    data: Any                                # robot MjData (sim GT, or FK-driven on the real robot)
    dof_pos_isaac: np.ndarray                # [J]
    dof_vel_isaac: np.ndarray                # [J]
    ang_vel: np.ndarray                      # [3] base angular velocity (body frame)
    rpy: np.ndarray                          # [3] roll, pitch, yaw
    last_action: np.ndarray                  # [J] previous raw policy output (Isaac order)
    action_mimic: np.ndarray                 # [38] motion-server target (SDK joint order)
    ref_data: Any = None                     # MjData for FK on the current reference frame
    future_ref_data: Any = None              # MjData for FK on the future reference frames
    future_raw: Optional[tuple] = None       # (root_pos [T,3], root_rot xyzw [T,4], dof [T,J]) or None
    ref_root_xy_w: Optional[np.ndarray] = None
    odom_on: bool = False
    anchor_delta_xy: Optional[np.ndarray] = None
    _cache: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ----- shared sub-computations (cached per tick; several terms share them) -----
    @property
    def robot_root_pos_w(self) -> np.ndarray:
        return np.asarray(self.data.qpos[:3], dtype=np.float64)

    @property
    def robot_root_quat_w(self) -> np.ndarray:
        """wxyz."""
        return np.asarray(self.data.qpos[3:7], dtype=np.float64)

    def future_frames(self):
        """The T future reference frames in the robot world frame:
        (root_pos [T,3], root_quat_wxyz [T,4], dof [T,J] SDK order, lin_vel [T,3], ang_vel [T,3])."""
        if "future_frames" not in self._cache:
            s = self.static
            self._cache["future_frames"] = resolve_future_frames(
                s.future_steps, s.num_actions,
                future_raw=self.future_raw,
                anchor_delta_xy=self.anchor_delta_xy if self.odom_on else None,
                fallback_action_mimic=self.action_mimic,
                fallback_root_xy=self.ref_root_xy_w,
            )
        return self._cache["future_frames"]

    @property
    def n_fk(self) -> int:
        """Future frames body FK is run on (the first n_fk of the T published)."""
        n = self.static.future_fk_steps
        return self.static.future_steps if n is None else int(n)

    def future_body_pos_w(self) -> np.ndarray:
        """Extended-body world positions of the first n_fk future frames, [n_fk, B, 3]
        (one FK pass per frame, shared by pos_h / pos_ref_b)."""
        if "future_body_pos_w" not in self._cache:
            s = self.static
            if self.future_ref_data is None:
                raise RuntimeError("future_motion_* body terms need future_ref_data (spec.needs_future).")
            f_pos, f_quat, f_dof, _, _ = self.future_frames()
            n = self.n_fk
            self._cache["future_body_pos_w"] = compute_future_body_pos_w(
                s.model, self.future_ref_data, f_pos[:n], f_quat[:n], f_dof[:n],
                s.tracked_body_ids, s.extended_parent_ids, s.extended_local_offsets, s.num_actions,
            )
        return self._cache["future_body_pos_w"]

    def future_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        """(future_motion_pos_h [n_fk*B*3], future_motion_anchor [n_fk*6]) for this tick."""
        if "future_obs" not in self._cache:
            _, f_quat, _, _, _ = self.future_frames()
            pos_h = future_pos_h_from_body_pos(self.future_body_pos_w(), self.robot_root_pos_w, self.robot_root_quat_w)
            anchor = compute_future_motion_anchor(f_quat[:self.n_fk], self.robot_root_quat_w)
            self._cache["future_obs"] = (pos_h, anchor)
        return self._cache["future_obs"]

    def _require_ref(self):
        if self.ref_data is None:
            raise RuntimeError("diff_body_* terms need ref_data (spec.needs_ref_fk).")


# --------------------------------------------------------------------------- registry
TermFn = Callable[[ObsContext, dict], np.ndarray]   # fn(ctx, params) -> per-frame value
DimFn = Callable[[StaticContext, dict], int]         # dim(static, params) -> per-frame width

REGISTRY: Dict[str, Tuple[TermFn, DimFn]] = {}


def term(name: str, dim: DimFn):
    """Register ``fn`` as the deploy implementation of DEX mdp term ``name``.
    ``dim(static, params)`` is the per-frame width (before history stacking)."""
    def deco(fn: TermFn) -> TermFn:
        if name in REGISTRY:
            raise KeyError(f"obs term '{name}' registered twice")
        REGISTRY[name] = (fn, dim)
        return fn
    return deco


def get_term(name: str) -> Tuple[TermFn, DimFn]:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"obs term '{name}' has no deploy implementation. "
            f"Available: {sorted(REGISTRY)}"
        ) from None


def term_dim(name: str, static: StaticContext, params: Optional[dict] = None) -> int:
    return int(get_term(name)[1](static, params or {}))


def future_num_steps(static: StaticContext, params: dict) -> int:
    """Frames a future_motion_* term consumes: ``params.num_steps`` (first n of the
    published T), else all ``static.future_steps`` (older runs without the param)."""
    n = params.get("num_steps")
    n = static.future_steps if n is None else int(n)
    if not 1 <= n <= static.future_steps:
        raise ValueError(f"num_steps={n} outside 1..{static.future_steps} (published future frames)")
    return n


# --------------------------------------------------------------------------- proprio
# DEX: mdp/observations/robot_observations.py

@term("base_ang_vel", dim=lambda s, p: 3)
def base_ang_vel(c: ObsContext, params: dict) -> np.ndarray:
    return np.asarray(c.ang_vel, dtype=np.float32)


@term("base_roll_pitch", dim=lambda s, p: 2)
def base_roll_pitch(c: ObsContext, params: dict) -> np.ndarray:
    return np.asarray(c.rpy[:2], dtype=np.float32)


@term("joint_pos_rel", dim=lambda s, p: s.num_actions)
def joint_pos_rel(c: ObsContext, params: dict) -> np.ndarray:
    return (c.dof_pos_isaac - c.static.default_dof_pos_isaac).astype(np.float32)


@term("joint_vel_rel", dim=lambda s, p: s.num_actions)
def joint_vel_rel(c: ObsContext, params: dict) -> np.ndarray:
    v = np.asarray(c.dof_vel_isaac, dtype=np.float32).copy()
    mask = params.get("mask_joint_names")
    if mask:
        v[c.static.joint_indices(mask)] = 0.0
    return v


@term("last_action", dim=lambda s, p: s.num_actions)
def last_action(c: ObsContext, params: dict) -> np.ndarray:
    return np.asarray(c.last_action, dtype=np.float32)


# --------------------------------------------------------------------------- mimic target
# DEX: mimic_observations.upcoming_twist_mimic_target

@term("upcoming_twist_mimic_target", dim=lambda s, p: 9 + s.num_actions)
def upcoming_twist_mimic_target(c: ObsContext, params: dict) -> np.ndarray:
    return np.asarray(c.action_mimic, dtype=np.float32)


# --------------------------------------------------------------------------- diff_body_*
# DEX: mimic_observations.diff_body_pos_b_deploy

@term("diff_body_pos_b_deploy", dim=lambda s, p: s.num_bodies * 3)
def diff_body_pos_b_deploy(c: ObsContext, params: dict) -> np.ndarray:
    c._require_ref()
    s = c.static
    return compute_diff_body_pos_b(
        s.model, c.data, c.ref_data, c.action_mimic,
        s.tracked_body_ids, s.extended_parent_ids, s.extended_local_offsets,
        s.num_actions, use_pb=False,
        update_robot_w_odom=False, ref_root_xy_w=None,
    )


# DEX: mimic_observations.diff_body_pos_b

@term("diff_body_pos_b", dim=lambda s, p: s.num_bodies * 3)
def diff_body_pos_b(c: ObsContext, params: dict) -> np.ndarray:
    c._require_ref()
    s = c.static
    return compute_diff_body_pos_b(
        s.model, c.data, c.ref_data, c.action_mimic,
        s.tracked_body_ids, s.extended_parent_ids, s.extended_local_offsets,
        s.num_actions, use_pb=False,
        update_robot_w_odom=c.odom_on, ref_root_xy_w=c.ref_root_xy_w,
    )


# DEX: mimic_observations.diff_body_pos_pb

@term("diff_body_pos_pb", dim=lambda s, p: s.num_bodies * 3)
def diff_body_pos_pb(c: ObsContext, params: dict) -> np.ndarray:
    c._require_ref()
    s = c.static
    return compute_diff_body_pos_pb(
        s.model, c.data, c.ref_data, c.action_mimic,
        s.tracked_body_ids, s.extended_parent_ids, s.extended_local_offsets,
        s.num_actions, z_align=bool(params.get("z_align", False)),
        update_robot_w_odom=c.odom_on, ref_root_xy_w=c.ref_root_xy_w,
    )


# DEX: mimic_observations.diff_body_tannorm_pb

@term("diff_body_tannorm_pb", dim=lambda s, p: s.num_bodies * 6)
def diff_body_tannorm_pb(c: ObsContext, params: dict) -> np.ndarray:
    c._require_ref()
    s = c.static
    return compute_diff_body_tannorm_b(
        s.model, c.data, c.ref_data, c.action_mimic,
        s.tracked_body_ids, s.extended_parent_ids,
        s.num_actions, use_pb=True,
        update_robot_w_odom=c.odom_on, ref_root_xy_w=c.ref_root_xy_w,
    )


# --------------------------------------------------------------------------- future_motion_*
# DEX: mimic_observations.future_motion_pos_h / future_motion_anchor

@term("future_motion_pos_h", dim=lambda s, p: future_num_steps(s, p) * s.num_bodies * 3)
def future_motion_pos_h(c: ObsContext, params: dict) -> np.ndarray:
    n = future_num_steps(c.static, params)
    pos_h = c.future_obs()[0].reshape(c.n_fk, -1)
    return pos_h[:n].reshape(-1)


@term("future_motion_anchor", dim=lambda s, p: future_num_steps(s, p) * 6)
def future_motion_anchor(c: ObsContext, params: dict) -> np.ndarray:
    n = future_num_steps(c.static, params)
    anchor = c.future_obs()[1].reshape(c.n_fk, -1)
    return anchor[:n].reshape(-1)


def _planar_quat_euler(q_wxyz: np.ndarray) -> np.ndarray:
    yaw = float(quatToEuler(np.asarray(q_wxyz, dtype=np.float64))[2])
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)


# DEX: mimic_observations.future_motion_joint_pos / future_motion_root_height /
# future_motion_root_roll_pitch / future_motion_root_lin_vel_pb / future_motion_root_ang_vel_b /
# future_motion_pos_ref_b

@term("future_motion_joint_pos", dim=lambda s, p: future_num_steps(s, p) * s.num_actions)
def future_motion_joint_pos(c: ObsContext, params: dict) -> np.ndarray:
    n = future_num_steps(c.static, params)
    return c.future_frames()[2][:n].astype(np.float32).reshape(-1)


@term("future_motion_root_height", dim=lambda s, p: future_num_steps(s, p))
def future_motion_root_height(c: ObsContext, params: dict) -> np.ndarray:
    n = future_num_steps(c.static, params)
    return c.future_frames()[0][:n, 2].astype(np.float32)


@term("future_motion_root_roll_pitch", dim=lambda s, p: future_num_steps(s, p) * 2)
def future_motion_root_roll_pitch(c: ObsContext, params: dict) -> np.ndarray:
    n = future_num_steps(c.static, params)
    f_quat = c.future_frames()[1][:n]
    rp = np.array([quatToEuler(q)[:2] for q in f_quat], dtype=np.float32)
    return rp.reshape(-1)


@term("future_motion_root_lin_vel_pb", dim=lambda s, p: future_num_steps(s, p) * 3)
def future_motion_root_lin_vel_pb(c: ObsContext, params: dict) -> np.ndarray:
    n = future_num_steps(c.static, params)
    v_w = c.future_frames()[3][:n]
    inv_planar = quat_conj(_planar_quat_euler(c.robot_root_quat_w))
    return quat_apply(np.broadcast_to(inv_planar, (n, 4)), v_w).astype(np.float32).reshape(-1)


@term("future_motion_root_ang_vel_b", dim=lambda s, p: future_num_steps(s, p) * 3)
def future_motion_root_ang_vel_b(c: ObsContext, params: dict) -> np.ndarray:
    n = future_num_steps(c.static, params)
    w_w = c.future_frames()[4][:n]
    inv_base = quat_conj(c.robot_root_quat_w)
    return quat_apply(np.broadcast_to(inv_base, (n, 4)), w_w).astype(np.float32).reshape(-1)


@term("future_motion_pos_ref_b", dim=lambda s, p: future_num_steps(s, p) * s.num_bodies * 3)
def future_motion_pos_ref_b(c: ObsContext, params: dict) -> np.ndarray:
    n = future_num_steps(c.static, params)
    f_pos, f_quat, _, _, _ = c.future_frames()
    body_pos_w = c.future_body_pos_w()[:n]                       # [n, B, 3]
    offset_w = body_pos_w - f_pos[:n, None, :]                   # [n, B, 3]
    inv_q = quat_conj(f_quat[:n])                                # [n, 4]
    B = body_pos_w.shape[1]
    pos_ref_b = quat_apply(np.broadcast_to(inv_q[:, None, :], (n, B, 4)), offset_w)
    return pos_ref_b.astype(np.float32).reshape(-1)
