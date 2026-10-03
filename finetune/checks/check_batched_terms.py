"""Term-by-term parity: batched observation terms (GPU) vs obs_terms / critic_terms (CPU).

Steps a CPU G1SimCore and a BatchedG1SimCore on identical actions along the same motion, builds
the CPU ``TrainObsContext`` and the ``BatchedObsContext`` the way the env does, and compares every
registered batched term against its CPU counterpart.

    python check_batched_terms.py --motion <integrated.pkl> [--envs 4] [--steps 30]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from finetune import _paths  # noqa: E402

_paths.setup()

import warp as wp  # noqa: E402

import obs_terms  # noqa: E402
from cfg import g1_29dof_cfg as cfg  # noqa: E402
from data_utils.rot_utils import quatToEuler  # noqa: E402

import finetune.mdp.batched_terms  # noqa: F401,E402  (registers the batched terms)
import finetune.mdp.critic_terms  # noqa: F401,E402  (registers the CPU priv_*/teacher terms)
from finetune._paths import TWIST2_ROOT  # noqa: E402
from finetune.env.batched_core import BatchedG1SimCore  # noqa: E402
from finetune.env.sim_core import G1SimCore  # noqa: E402
from finetune.helper.torch_math import quat_to_rpy, xyzw_to_wxyz  # noqa: E402
from finetune.mdp.batched_obs import REGISTRY as BREG, BatchedObsContext  # noqa: E402
from finetune.mdp.critic_terms import TrainObsContext  # noqa: E402
from finetune.motion_data.batched_motion_ref import BatchedMotionSet, BatchedRefFK  # noqa: E402
from finetune.motion_data.motion_ref import MotionSet  # noqa: E402
from finetune.tasks.ft_cfg import MotionCfg  # noqa: E402

# (term name, params) -- params as the specs in use pass them
TERMS = [
    ("base_ang_vel", {}), ("base_roll_pitch", {}), ("joint_pos_rel", {}),
    ("joint_vel_rel", {"mask_joint_names": [".*_ankle_pitch_joint", ".*_ankle_roll_joint"]}),
    ("joint_vel_rel", {}), ("last_action", {}),
    ("diff_body_pos_b", {"randomize_motion_ref_xyz": True}),
    ("diff_body_pos_b_deploy", {}),
    ("diff_body_pos_pb", {}), ("diff_body_pos_pb", {"z_align": True}),
    ("diff_body_tannorm_pb", {}), ("upcoming_twist_mimic_target", {}),
    ("future_motion_pos_ref_b", {"num_steps": 1}),
    ("future_motion_pos_h", {"num_steps": 1}), ("future_motion_anchor", {"num_steps": 1}),
    ("future_motion_joint_pos", {"num_steps": 1}), ("future_motion_root_height", {"num_steps": 1}),
    ("future_motion_root_roll_pitch", {"num_steps": 1}),
    ("future_motion_root_lin_vel_pb", {"num_steps": 1}),
    ("future_motion_root_ang_vel_b", {"num_steps": 1}),
    ("priv_root_lin_vel_b", {}), ("priv_root_height", {}), ("priv_projected_gravity", {}),
    ("priv_ref_root_pos_diff_b", {}), ("priv_ref_root_rot_diff", {}),
    ("priv_ref_root_lin_vel_b", {}), ("priv_ref_root_ang_vel_b", {}),
    ("priv_body_pos_diff_b", {}), ("priv_body_lin_vel_diff_b", {}), ("priv_body_ang_vel_diff_b", {}),
    ("priv_feet_contact", {}), ("priv_joint_torque", {"scale": 0.01}), ("priv_motion_phase", {}),
    ("projected_gravity", {}), ("extended_body_pos_h", {}), ("extended_body_quat_h", {}),
    ("extended_body_lin_vel_h", {}), ("extended_body_ang_vel_h", {}),
    ("diff_body_lin_vel_pb", {}), ("diff_body_ang_vel_pb", {}),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", required=True)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--xml", default=os.path.join(TWIST2_ROOT, "assets", "g1", "g1_sim2sim_29dof.xml"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--future-steps", type=int, default=10)
    p.add_argument("--future-interval", type=int, default=5)
    p.add_argument("--mimic-step-offset", type=int, default=5)
    args = p.parse_args()

    dev = torch.device(args.device)
    wp.init()
    J, N, dt = cfg.NUM_ACTIONS, args.envs, 0.02
    rng = np.random.default_rng(0)

    cpu_core = G1SimCore(args.xml, future_steps=args.future_steps, future_fk_steps=1)
    gpu_core = BatchedG1SimCore(args.xml, num_envs=N, device=str(dev))
    cpu_set = MotionSet(args.motion, MotionCfg(), cpu_core.ref_fk, J, verbose=False)
    fk = BatchedRefFK(gpu_core.model, gpu_core.mw, N, gpu_core.tracked_body_ids,
                      gpu_core.extended_parent_ids, gpu_core.extended_local_offsets, J)
    gpu_set = BatchedMotionSet(cpu_set, fk, dev)
    static = cpu_core.obs_static

    mid, mid_g = 0, torch.zeros(N, dtype=torch.long, device=dev)
    length = cpu_set.length(mid)
    fr0 = cpu_set.frame(mid, 0.0)
    root_pos = fr0.root_pos.copy()
    root_pos[2] += 0.03
    cpu_core.reset_to(root_pos, fr0.root_quat, fr0.root_lin_vel, fr0.root_ang_vel, fr0.dof_pos, fr0.dof_vel)
    rep = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=dev).expand(N, -1).contiguous()
    gpu_core.reset_to(torch.arange(N, device=dev), rep(root_pos), rep(fr0.root_quat),
                      rep(fr0.root_lin_vel), rep(fr0.root_ang_vel), rep(fr0.dof_pos), rep(fr0.dof_vel))

    # odom anchor, as _reset_slot computes it
    t_origin = args.mimic_step_offset * dt
    anchor = cpu_core.data.qpos[:2].copy() - cpu_set.frame(mid, t_origin).root_pos[:2]
    anchor_g = torch.as_tensor(anchor, dtype=torch.float32, device=dev).expand(N, 2).contiguous()
    offsets = [(i + 1) * args.future_interval for i in range(args.future_steps)]

    acc = {}
    actions = rng.normal(scale=0.3, size=(args.steps, J)).astype(np.float32)
    last = np.zeros(J, dtype=np.float32)
    for k in range(args.steps):
        a = actions[k]
        cpu_core.step(a)
        gpu_core.step(torch.as_tensor(a, device=dev).expand(N, -1).contiguous())
        last = a
        t = (k + 1) * dt
        t_g = torch.full((N,), t, device=dev)
        t_mimic = t + args.mimic_step_offset * dt
        tm_g = torch.full((N,), t_mimic, device=dev)

        # ---- CPU context (mirrors mujoco_env._write_obs) ----
        robot = cpu_core.state()
        fr_m = cpu_set.frame(mid, t_mimic)
        action_mimic = cpu_set.mimic_obs(fr_m)
        ref_root_xy = fr_m.root_pos[:2] + anchor
        future_raw = cpu_set.future_raw(mid, t, offsets, dt)
        cctx = TrainObsContext(
            static=static, data=cpu_core.data,
            dof_pos_isaac=robot.dof_pos_isaac, dof_vel_isaac=robot.dof_vel_isaac,
            ang_vel=robot.root_ang_vel_b, rpy=quatToEuler(robot.root_quat), last_action=last,
            action_mimic=action_mimic, ref_data=cpu_core.ref_data,
            future_ref_data=cpu_core.future_ref_data, future_raw=future_raw,
            ref_root_xy_w=ref_root_xy, odom_on=True, anchor_delta_xy=anchor,
            robot=robot, ref=cpu_set.body_state(mid, t), ref_frame=cpu_set.frame(mid, t),
            motion_phase=float(t / length), time_left=float(max(length - t, 0.0)))

        # ---- GPU context (what BatchedG1MimicVecEnv will build) ----
        gstate = gpu_core.state()
        gfr_m = gpu_set.frame(mid_g, tm_g)
        gam = gpu_set.mimic_obs(gfr_m)
        f_pos, f_rot_xyzw, f_dof, f_lin, f_ang = gpu_set.future_raw(mid_g, t_g, offsets, dt)
        f_pos = f_pos.clone()
        f_pos[..., :2] += anchor_g[:, None, :]
        gctx = BatchedObsContext(
            static=static, num_envs=N, device=dev,
            dof_pos_isaac=gstate.dof_pos_isaac, dof_vel_isaac=gstate.dof_vel_isaac,
            ang_vel=gstate.root_ang_vel_b, rpy=quat_to_rpy(gstate.root_quat),
            last_action=torch.as_tensor(last, device=dev).expand(N, -1).contiguous(),
            action_mimic=gam, robot=gstate, ref=gpu_set.body_state(mid_g, t_g),
            ref_frame=gpu_set.frame(mid_g, t_g),
            future_frames=(f_pos, xyzw_to_wxyz(f_rot_xyzw), f_dof, f_lin, f_ang),
            ref_root_xy_w=gfr_m.root_pos[:, :2] + anchor_g, ref_fk=fk,
            motion_phase=torch.full((N,), float(t / length), device=dev),
            time_left=torch.full((N,), float(max(length - t, 0.0)), device=dev))
        wp.synchronize()

        for name, params in TERMS:
            cfn, _ = obs_terms.get_term(name)
            c = np.asarray(cfn(cctx, params), dtype=np.float64).reshape(-1)
            g = BREG[name](gctx, params)[0].detach().cpu().numpy().astype(np.float64).reshape(-1)
            assert c.shape == g.shape, f"{name}: cpu{c.shape} gpu{g.shape}"
            suffix = "(mask)" if params.get("mask_joint_names") else ("(z_align)" if params.get("z_align") else "")
            key = f"{name}{suffix}"
            d = float(np.abs(c - g).max())
            s = float(np.abs(c).max())
            cur = acc.get(key, (0.0, 0.0))
            acc[key] = (max(cur[0], d), max(cur[1], s))
        gctx._cache.clear()

    missing = sorted(set(BREG) - {n for n, _ in TERMS})
    print(f"\n{'term':34s} {'dim':>5s} {'max|diff|':>12s} {'rel':>11s} {'|cpu|max':>10s}")
    worst, worst_name = 0.0, ""
    for name, (d, s) in acc.items():
        rel = d / max(s, 1e-9)
        base = name.replace("(mask)", "").replace("(z_align)", "")
        dim = obs_terms.term_dim(base, static, dict(next(p for n, p in TERMS if n == base)))
        if rel > worst:
            worst, worst_name = rel, name
        print(f"  {name:32s} {dim:5d} {d:12.3e} {rel:11.3e} {s:10.4f}")
    print(f"\n{len(acc)} terms compared over {args.steps} steps x {N} envs")
    if missing:
        print(f"registered but untested: {missing}")
    print(f"worst relative difference: {worst:.3e}  ({worst_name})")


if __name__ == "__main__":
    main()
