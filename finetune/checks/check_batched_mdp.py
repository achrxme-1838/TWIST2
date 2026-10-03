"""Parity checks for the batched MDP pieces:

  1. observation history -- BatchedObsBuilder's ring buffer vs ObsBuilder's deque, including
     staggered per-env resets (a synthetic counter term is registered in both registries);
  2. rewards and terminations -- BatchedRewardComputer / BatchedTerminationChecker against the
     CPU versions, driven by the two sim cores stepping on the same actions and motion.

    python check_batched_mdp.py --motion <integrated.pkl> [--envs 8] [--steps 40]
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

import mujoco  # noqa: E402
import mujoco_warp as mjwarp  # noqa: E402
import warp as wp  # noqa: E402

import obs_terms  # noqa: E402
from cfg import g1_29dof_cfg as cfg  # noqa: E402
from obs_builder import ObsBuilder  # noqa: E402
from policy_spec import PolicySpec, TermSpec  # noqa: E402

from finetune._paths import TWIST2_ROOT  # noqa: E402
from finetune.env.batched_core import BatchedG1SimCore  # noqa: E402
from finetune.env.sim_core import G1SimCore  # noqa: E402
from finetune.mdp.batched_obs import BatchedObsBuilder, BatchedObsContext, bterm  # noqa: E402
from finetune.mdp.batched_rewards import BatchedRewardComputer, BatchedTerminationChecker  # noqa: E402
from finetune.mdp.rewards import RewardComputer, TerminationChecker  # noqa: E402
from finetune.motion_data.batched_motion_ref import BatchedMotionSet, BatchedRefFK  # noqa: E402
from finetune.motion_data.motion_ref import MotionSet  # noqa: E402
from finetune.tasks.ft_cfg import MotionCfg, RewardCfg, TerminationCfg  # noqa: E402

_DIM = 3


# ---- synthetic term in both registries: value = env-specific counter, so ordering is visible ----
@obs_terms.term("_hist_probe", dim=lambda s, p: _DIM)
def _cpu_probe(c, params):
    return np.full(_DIM, float(c.action_mimic[0]), dtype=np.float32) + np.arange(_DIM, dtype=np.float32)


@bterm("_hist_probe")
def _gpu_probe(c: BatchedObsContext, params):
    base = c.action_mimic[:, 0:1]
    return base + torch.arange(_DIM, device=c.device, dtype=base.dtype)[None, :]


def check_history(num_envs: int, steps: int, device: torch.device, history: int = 4):
    print(f"\n=== 1. observation history (history={history}, {num_envs} envs, {steps} steps) ===")
    static = obs_terms.StaticContext(
        model=None, num_actions=cfg.NUM_ACTIONS,
        tracked_body_ids=np.arange(len(cfg.TRACKED_BODY_NAMES)),
        extended_parent_ids=np.arange(len(cfg.EXTENDED_JOINTS)),
        extended_local_offsets=np.zeros((len(cfg.EXTENDED_JOINTS), 3)),
        default_dof_pos_isaac=np.zeros(cfg.NUM_ACTIONS),
        isaac_joint_names=list(cfg.ISAAC_JOINT_NAMES), future_steps=1, future_fk_steps=1)
    spec = PolicySpec(terms=[TermSpec(name="_hist_probe", history=history),
                             TermSpec(name="_hist_probe", history=1)])

    gpu = BatchedObsBuilder(spec, static, num_envs, device)
    cpus = [ObsBuilder(spec, static) for _ in range(num_envs)]
    for b in cpus:
        b.reset()
    gpu.reset()

    rng = np.random.default_rng(0)
    worst = 0.0
    for k in range(steps):
        # stagger resets: each step a random subset of envs restarts its episode
        hit = np.where(rng.random(num_envs) < 0.2)[0] if k else np.array([], dtype=np.int64)
        if hit.size:
            gpu.reset(torch.as_tensor(hit, dtype=torch.long, device=device))
            for e in hit:
                cpus[int(e)].reset()
        vals = (np.arange(num_envs) * 100 + k).astype(np.float32)
        ctx = BatchedObsContext(static=static, num_envs=num_envs, device=device,
                                action_mimic=torch.as_tensor(vals, device=device)[:, None].expand(-1, 38).contiguous())
        g = gpu(ctx).cpu().numpy()
        for e in range(num_envs):
            cctx = obs_terms.ObsContext(static=static, data=None, dof_pos_isaac=None, dof_vel_isaac=None,
                                        ang_vel=None, rpy=None, last_action=None,
                                        action_mimic=np.full(38, vals[e], dtype=np.float32))
            c = cpus[e](cctx)
            worst = max(worst, float(np.abs(c - g[e]).max()))
    print(f"  obs_dim = {gpu.obs_dim} (cpu {cpus[0].obs_dim})")
    print(f"  max|diff| over all steps/envs = {worst:.3e}   -> {'PASS' if worst == 0.0 else 'FAIL'}")
    return worst


def check_rewards(args, device: torch.device):
    print(f"\n=== 2. rewards / terminations ({args.envs} envs, {args.steps} steps) ===")
    J = cfg.NUM_ACTIONS
    rng = np.random.default_rng(1)

    cpu_core = G1SimCore(args.xml, future_steps=10, future_fk_steps=1)
    gpu_core = BatchedG1SimCore(args.xml, num_envs=args.envs, device=str(device))
    cpu_set = MotionSet(args.motion, MotionCfg(), cpu_core.ref_fk, J, verbose=False)
    fk = BatchedRefFK(gpu_core.model, gpu_core.mw, args.envs, gpu_core.tracked_body_ids,
                      gpu_core.extended_parent_ids, gpu_core.extended_local_offsets, J)
    gpu_set = BatchedMotionSet(cpu_set, fk, device)

    rcfg, tcfg = RewardCfg(), TerminationCfg()
    exclude = set(cpu_core.body_ids_matching([".*ankle.*", ".*wrist.*"]).tolist())
    contact_ids = np.array([b for b in range(1, cpu_core.model.nbody) if b not in exclude], dtype=np.int64)
    soft = cpu_core.soft_joint_limits(0.9)
    cpu_rew = RewardComputer(rcfg, 0.02, soft, contact_ids, 1.0)
    gpu_rew = BatchedRewardComputer(rcfg, 0.02, soft, contact_ids, 1.0, device)
    cpu_term = TerminationChecker(tcfg)
    gpu_term = BatchedTerminationChecker(tcfg)

    mid = 0
    fr0 = cpu_set.frame(mid, 0.0)
    root_pos = fr0.root_pos.copy()
    root_pos[2] += 0.03
    cpu_core.reset_to(root_pos, fr0.root_quat, fr0.root_lin_vel, fr0.root_ang_vel, fr0.dof_pos, fr0.dof_vel)
    rep = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=device).expand(args.envs, -1).contiguous()
    gpu_core.reset_to(torch.arange(args.envs, device=device), rep(root_pos), rep(fr0.root_quat),
                      rep(fr0.root_lin_vel), rep(fr0.root_ang_vel), rep(fr0.dof_pos), rep(fr0.dof_vel))
    mid_g = torch.zeros(args.envs, dtype=torch.long, device=device)

    actions = rng.normal(scale=0.3, size=(args.steps, J)).astype(np.float32)
    prev = np.zeros(J, dtype=np.float32)
    acc = {}
    for k in range(args.steps):
        a = actions[k]
        cpu_core.step(a)
        gpu_core.step(torch.as_tensor(a, device=device).expand(args.envs, -1).contiguous())
        t = (k + 1) * 0.02
        t_g = torch.full((args.envs,), t, device=device)

        cs, gs = cpu_core.state(), gpu_core.state()
        cb = cpu_set.body_state(mid, t)
        gb = gpu_set.body_state(mid_g, t_g)
        cfr = cpu_set.frame(mid, t)
        gfr = gpu_set.frame(mid_g, t_g)
        wp.synchronize()

        c_tot, c_raw = cpu_rew(cs, cb, cfr.root_quat, a, prev)
        g_tot, g_raw = gpu_rew(gs, gb, gfr.root_quat,
                               torch.as_tensor(a, device=device).expand(args.envs, -1),
                               torch.as_tensor(prev, device=device).expand(args.envs, -1))
        c_div, c_dist = cpu_term.diverged(cs, cb)
        g_div, g_dist = gpu_term.diverged(gs, gb)

        def note(name, c, g):
            c = float(c)
            g = float(g)
            d = abs(c - g)
            cur = acc.get(name, (0.0, 0.0))
            acc[name] = (max(cur[0], d), max(cur[1], abs(c)))

        note("total", c_tot, g_tot[0])
        for name in c_raw:
            note(f"raw.{name}", c_raw[name], g_raw[name][0])
        note("diverged", float(c_div), float(g_div[0]))
        note("max_body_dist", c_dist, g_dist[0])
        prev = a

    print(f"\n{'term':34s} {'max|diff|':>12s} {'rel':>11s} {'|cpu|max':>10s}")
    worst = 0.0
    for name, (d, s) in acc.items():
        rel = d / max(s, 1e-9)
        worst = max(worst, rel)
        print(f"  {name:32s} {d:12.3e} {rel:11.3e} {s:10.4f}")
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", required=True)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--xml", default=os.path.join(TWIST2_ROOT, "assets", "g1", "g1_sim2sim_29dof.xml"))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)
    wp.init()

    h = check_history(args.envs, args.steps, device)
    r = check_rewards(args, device)
    print(f"\nhistory worst |diff| = {h:.3e}   |   reward/termination worst rel = {r:.3e}")


if __name__ == "__main__":
    main()
