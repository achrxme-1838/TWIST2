from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from finetune.checks._common import add_cfg_args, cfg_from_args  # noqa: E402


def load_actor(path: str):
    from finetune.mdp.policy import build_mlp, load_student_state, student_from_state
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "merged_actor_state_dict" in ck:
        sd = ck["merged_actor_state_dict"]
        idx = sorted({int(k.split(".")[0]) for k in sd if k.endswith(".weight")})
        dims = [sd[f"{i}.weight"].shape for i in idx]
        actor = build_mlp(dims[0][1], [d[0] for d in dims[:-1]], dims[-1][0], "elu")
        actor.load_state_dict(sd)
        return actor.eval(), "finetuned (merged)"
    return student_from_state(load_student_state(path)).eval(), "student"


def main():
    p = add_cfg_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--episodes-per-motion", type=int, default=1)
    p.add_argument("--out", default=None, help="json summary path")
    args = p.parse_args()
    cfg = cfg_from_args(args)
    cfg.env.num_envs = 1
    from finetune.env.mujoco_env import G1MimicVecEnv

    actor, kind = load_actor(cfg.paths.student_pt)
    env = G1MimicVecEnv(cfg, verbose=False)
    print(f"[zero-shot] {kind} from {cfg.paths.student_pt}; {env.motions.num_motions} motions")

    per_motion = defaultdict(list)
    with torch.no_grad():
        for mid in range(env.motions.num_motions):
            for _ in range(args.episodes_per_motion):
                env.force_motion_id = mid
                env.reset()
                obs, _ = env.get_observations()
                done = False
                while not done:
                    obs, r, d, ex = env.step(actor(obs))
                    done = bool(d[0])
                ep = ex["episodes"][-1]
                per_motion[ep["motion"]].append(ep)
    env.force_motion_id = None

    rows = []
    for name, eps in per_motion.items():
        rows.append({
            "motion": name,
            "success": float(np.mean([1.0 - e["diverged"] for e in eps])),
            "reward": float(np.mean([e["reward"] for e in eps])),
            "length": float(np.mean([e["length"] for e in eps])),
            "mean_max_body_dist": float(np.mean([e["mean_max_body_dist"] for e in eps])),
        })
    rows.sort(key=lambda r: r["success"])
    for r in rows:
        print(f"  {r['motion'][:90]:90s} succ {r['success']:.2f} rew {r['reward']:8.2f} len {r['length']:6.0f} dist {r['mean_max_body_dist']:.3f}")
    summary = {
        "policy": cfg.paths.student_pt, "kind": kind,
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "mean_max_body_dist": float(np.mean([r["mean_max_body_dist"] for r in rows])),
        "motions": rows,
    }
    print(f"[zero-shot] success {summary['success_rate']:.3f} | reward {summary['mean_reward']:.2f} "
          f"| mean max body dist {summary['mean_max_body_dist']:.3f}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[zero-shot] wrote {args.out}")


if __name__ == "__main__":
    main()
