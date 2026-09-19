from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from finetune.checks._common import add_cfg_args, cfg_from_args  # noqa: E402


def main():
    p = add_cfg_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()
    cfg = cfg_from_args(args)
    from finetune.env.mujoco_env import G1MimicVecEnv
    from finetune.mdp.policy import load_student_state, student_from_state

    env = G1MimicVecEnv(cfg, verbose=True)
    student = student_from_state(load_student_state(cfg.paths.student_pt)).eval()
    assert student[0].in_features == env.num_actor_obs, (student[0].in_features, env.num_actor_obs)
    print(f"[ok] actor obs dim {env.num_actor_obs} == student input; critic obs dim {env.num_critic_obs}")

    # history buffers after reset: every history slot equals the first frame
    obs, _ = env.get_observations()
    slot = env.slots[0]
    for s in slot.actor_builder.slots:
        if s.history > 1:
            frames = obs[0, s.offset:s.offset + s.width].view(s.history, s.dim)
            assert torch.allclose(frames, frames[0:1].expand_as(frames)), s.spec.name
    print("[ok] history buffers filled with the first frame after reset")

    # zero-action (hold default pose) vs student actions: reward / termination sanity + timing
    for label, policy in (("student", lambda o: student(o)), ("zero", lambda o: torch.zeros(o.shape[0], env.num_actions))):
        env.reset()
        obs, _ = env.get_observations()
        t0 = time.time()
        rews, dists, eps = [], [], []
        with torch.no_grad():
            for _ in range(args.steps):
                obs, r, d, ex = env.step(policy(obs))
                rews.append(float(r.mean())); dists.append(ex["log"]["max_body_dist"]); eps += ex["episodes"]
        dt = (time.time() - t0) / args.steps
        print(f"[{label:7s}] mean step reward {np.mean(rews):.4f} | mean max body dist {np.mean(dists):.3f} m "
              f"| episodes {len(eps)} (diverged {sum(e['diverged'] for e in eps):.0f}) | {1000 * dt:.1f} ms/step "
              f"({env.num_envs / dt:.0f} env-steps/s)")
        if eps:
            terms = {k: np.mean([e['terms'][k] for e in eps]) for k in eps[0]['terms']}
            print("          per-episode reward terms:", {k: round(v, 3) for k, v in terms.items()})


if __name__ == "__main__":
    main()
