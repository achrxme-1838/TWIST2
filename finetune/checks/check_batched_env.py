"""End-to-end parity: BatchedG1MimicVecEnv (GPU) vs G1MimicVecEnv (CPU).

Both envs are pinned to the same motion and driven with the same actions, so their observation,
reward and termination streams must agree to float32 precision. Also reports GPU throughput.

    python check_batched_env.py --motion <integrated.pkl> [--envs 4] [--steps 30] [--bench 4096]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from finetune import _paths  # noqa: E402

_paths.setup()

import warp as wp  # noqa: E402

from finetune.checks._common import add_cfg_args, cfg_from_args  # noqa: E402


def main():
    p = add_cfg_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--bench", type=int, default=0, help="also time the GPU env with this many envs")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg = cfg_from_args(args)

    from finetune.env.batched_mujoco_env import BatchedG1MimicVecEnv
    from finetune.env.mujoco_env import G1MimicVecEnv

    wp.init()
    cfg.env.num_envs = 1
    cpu = G1MimicVecEnv(cfg, device="cpu", verbose=False)
    cfg.env.num_envs = args.envs
    gpu = BatchedG1MimicVecEnv(cfg, device=args.device, verbose=False)
    for e in (cpu, gpu):
        e.force_motion_id = 0
        e.reset()

    assert (cpu.num_actor_obs, cpu.num_critic_obs) == (gpu.num_actor_obs, gpu.num_critic_obs), \
        (cpu.num_actor_obs, cpu.num_critic_obs, gpu.num_actor_obs, gpu.num_critic_obs)
    print(f"obs dims: actor {cpu.num_actor_obs}, critic {cpu.num_critic_obs} (both)")

    oc, xc = cpu.get_observations()
    og, xg = gpu.get_observations()
    d_obs = float((oc[0] - og[0].cpu()).abs().max())
    d_cri = float((xc["observations"]["critic"][0] - xg["observations"]["critic"][0].cpu()).abs().max())
    print(f"after reset:  actor obs {d_obs:.3e}   critic obs {d_cri:.3e}")

    rng = np.random.default_rng(0)
    J = cpu.num_actions
    worst = {"actor_obs": 0.0, "critic_obs": 0.0, "reward": 0.0, "done": 0.0, "time_out": 0.0}
    for _ in range(args.steps):
        a = rng.normal(scale=0.3, size=J).astype(np.float32)
        oc, rc, dc, xc = cpu.step(torch.as_tensor(a)[None, :])
        og, rg, dg, xg = gpu.step(torch.as_tensor(a, device=args.device).expand(args.envs, -1).contiguous())
        worst["actor_obs"] = max(worst["actor_obs"], float((oc[0] - og[0].cpu()).abs().max()))
        worst["critic_obs"] = max(worst["critic_obs"], float(
            (xc["observations"]["critic"][0] - xg["observations"]["critic"][0].cpu()).abs().max()))
        worst["reward"] = max(worst["reward"], float(abs(rc[0] - rg[0].cpu())))
        worst["done"] = max(worst["done"], float(abs(dc[0].float() - dg[0].cpu().float())))
        worst["time_out"] = max(worst["time_out"],
                                float(abs(xc["time_outs"][0].float() - xg["time_outs"][0].cpu().float())))
    print(f"after {args.steps} steps:")
    for k, v in worst.items():
        print(f"  {k:12s} max|diff| = {v:.3e}")

    if args.bench:
        cfg.env.num_envs = args.bench
        big = BatchedG1MimicVecEnv(cfg, device=args.device, verbose=False)
        act = torch.zeros(args.bench, J, device=args.device)
        for _ in range(3):
            big.step(act)
        wp.synchronize()
        K = 20
        t0 = time.perf_counter()
        for _ in range(K):
            big.step(act)
        wp.synchronize()
        dt = (time.perf_counter() - t0) / K
        print(f"\nthroughput at num_envs={args.bench}: {dt * 1000:.1f} ms/step -> "
              f"{args.bench / dt:,.0f} env-steps/s")


if __name__ == "__main__":
    main()
