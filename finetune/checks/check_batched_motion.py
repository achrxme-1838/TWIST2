"""Parity check: BatchedMotionSet (GPU) vs MotionSet (CPU).

Draws random (motion id, time) pairs, including motion ends and exact frame boundaries, and
compares frame(), mimic_obs(), future_raw() and body_state() field by field.

    python check_batched_motion.py --motion <integrated.pkl> [--envs 64] [--draws 20]
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

from cfg import g1_29dof_cfg as cfg  # noqa: E402
from finetune._paths import TWIST2_ROOT  # noqa: E402
from finetune.env.sim_core import G1SimCore  # noqa: E402
from finetune.motion_data.batched_motion_ref import BatchedMotionSet, BatchedRefFK  # noqa: E402
from finetune.motion_data.motion_ref import MotionSet  # noqa: E402
from finetune.tasks.ft_cfg import MotionCfg  # noqa: E402


class Acc:
    """Collects max |diff| and the matching reference scale per field."""

    def __init__(self):
        self.d, self.s = {}, {}

    def add(self, name, cpu, gpu):
        a = np.asarray(cpu, dtype=np.float64)
        b = np.asarray(gpu, dtype=np.float64)
        assert a.shape == b.shape, f"{name}: cpu{a.shape} gpu{b.shape}"
        self.d[name] = max(self.d.get(name, 0.0), float(np.abs(a - b).max()) if a.size else 0.0)
        self.s[name] = max(self.s.get(name, 0.0), float(np.abs(a).max()) if a.size else 0.0)

    def report(self):
        print(f"\n{'field':26s} {'max|diff|':>12s} {'rel':>11s} {'|cpu|max':>11s}")
        worst = 0.0
        for k in self.d:
            scale = max(self.s[k], 1e-9)
            rel = self.d[k] / scale
            worst = max(worst, rel)
            print(f"  {k:24s} {self.d[k]:12.3e} {rel:11.3e} {scale:11.4f}")
        return worst


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", required=True)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--draws", type=int, default=20)
    p.add_argument("--xml", default=os.path.join(TWIST2_ROOT, "assets", "g1", "g1_sim2sim_29dof.xml"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--control-dt", type=float, default=0.02)
    args = p.parse_args()

    dev = torch.device(args.device)
    rng = np.random.default_rng(0)
    J = cfg.NUM_ACTIONS

    cpu_core = G1SimCore(args.xml, future_steps=10, future_fk_steps=1)
    cpu_set = MotionSet(args.motion, MotionCfg(), cpu_core.ref_fk, J, verbose=True)

    wp.init()
    mjm = mujoco.MjModel.from_xml_path(args.xml)
    mjm.opt.timestep = 0.001
    mw = mjwarp.put_model(mjm)
    if hasattr(mw.opt, "graph_conditional"):
        mw.opt.graph_conditional = False
    i64 = lambda x: torch.as_tensor(np.asarray(x, dtype=np.int64), device=dev)
    f32 = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=dev)
    fk = BatchedRefFK(mjm, mw, args.envs,
                      i64([mjm.body(n).id for n in cfg.TRACKED_BODY_NAMES]),
                      i64([mjm.body(pn).id for _, pn, _ in cfg.EXTENDED_JOINTS]),
                      f32([o for _, _, o in cfg.EXTENDED_JOINTS]), J)
    gpu_set = BatchedMotionSet(cpu_set, fk, dev)
    print(f"[gpu] {gpu_set.num_motions} motions, {gpu_set.root_pos.shape[0]} frames flattened, "
          f"tables {sum(t.numel() * 4 for t in (gpu_set.root_pos, gpu_set.root_quat, gpu_set.root_lin_vel, gpu_set.root_ang_vel, gpu_set.dof_pos, gpu_set.dof_vel)) / 1e6:.2f} MB")

    offsets = list(range(0, 50, 5))     # future step offsets, as the specs use
    acc = Acc()
    for _ in range(args.draws):
        mid = rng.integers(gpu_set.num_motions, size=args.envs)
        L = np.array([cpu_set.length(int(k)) for k in mid])
        u = rng.random(args.envs)
        # mix interior times, exact frame boundaries, t=0 and t=L
        t = u * L
        t[:: 4] = 0.0
        t[1::4] = L[1::4]
        frames = np.array([cpu_set.motions[int(k)].dt for k in mid])
        t[2::4] = np.floor(t[2::4] / frames[2::4]) * frames[2::4]

        mid_g = torch.as_tensor(mid, dtype=torch.long, device=dev)
        t_g = torch.as_tensor(t, dtype=torch.float32, device=dev)
        gfr = gpu_set.frame(mid_g, t_g)
        gmim = gpu_set.mimic_obs(gfr)
        gfut = gpu_set.future_raw(mid_g, t_g, offsets, args.control_dt)
        gbs = gpu_set.body_state(mid_g, t_g)
        wp.synchronize()

        for e in range(args.envs):
            m, tt = int(mid[e]), float(t[e])
            cfr = cpu_set.frame(m, tt)
            for name in ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "dof_pos", "dof_vel"):
                acc.add(f"frame.{name}", getattr(cfr, name), getattr(gfr, name)[e].cpu().numpy())
            acc.add("mimic_obs", cpu_set.mimic_obs(cfr), gmim[e].cpu().numpy())
            cfut = cpu_set.future_raw(m, tt, offsets, args.control_dt)
            for name, c, g in zip(("pos", "rot_xyzw", "dof", "lin_vel", "ang_vel"), cfut, gfut):
                acc.add(f"future.{name}", c, g[e].cpu().numpy())
            cbs = cpu_set.body_state(m, tt)
            for name in ("pos", "quat", "lin_vel", "ang_vel", "lin_vel_ext", "ang_vel_ext"):
                acc.add(f"body_state.{name}", getattr(cbs, name), getattr(gbs, name)[e].cpu().numpy())

    worst = acc.report()
    print(f"\nworst relative difference: {worst:.3e}")


if __name__ == "__main__":
    main()
