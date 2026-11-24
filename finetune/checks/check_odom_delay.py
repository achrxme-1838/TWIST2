"""Odometry delay checks for the batched (GPU) env.

1. OdomDelayBuffer on a synthetic trajectory: after every update the held pose equals the true
   pose ``latency`` steps ago and it is held (unchanged) between updates.
2. Env level, odom delay on with the default ranges:
   * period=1 / latency=0 makes every use_odom_delay term equal to its true-state value,
   * the delayed terms really differ from the true ones with the default ranges,
   * the critic obs never depends on the odometry (est_pelvis on/off -> same critic terms).

    python check_odom_delay.py --motion <integrated.pkl> --actor-spec <odom spec.yaml> [--envs 8]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from finetune import _paths  # noqa: E402

_paths.setup()

from finetune.checks._common import add_cfg_args, cfg_from_args  # noqa: E402
from finetune.helper.torch_math import quat_to_rpy, rpy_to_quat  # noqa: E402
from finetune.mdp.odom_delay import OdomDelayBuffer  # noqa: E402


def _yaw_quat_noise(n, g, dev):
    rpy = (torch.rand(n, 3, generator=g, device=dev) - 0.5) * torch.tensor([0.4, 0.4, 6.0], device=dev)
    return rpy_to_quat(rpy)


def check_buffer(dev) -> bool:
    N, T = 64, 200
    g = torch.Generator(device=dev).manual_seed(0)
    pos = torch.cumsum(torch.rand(T, N, 3, generator=g, device=dev) * 0.02, dim=0)
    quat = torch.stack([_yaw_quat_noise(N, g, dev) for _ in range(T)])
    yaw = quat_to_rpy(quat)[..., 2]
    buf = OdomDelayBuffer(N, (4, 6), (0, 3), dev, seed=0)
    buf.init(torch.arange(N, device=dev), pos[0], quat[0])
    ok = True
    updates = 0
    for k in range(1, T):
        prev_held = buf.held_pos.clone()
        lat = buf.latency.clone()
        due = buf.countdown == 1
        buf.step(pos[k], quat[k])
        if due.any():
            src = (k - lat[due]).clamp_min(0)
            idx = due.nonzero().squeeze(-1)
            exp_pos = pos[src, idx]
            exp_yaw = yaw[src, idx]
            ok &= torch.allclose(buf.held_pos[due], exp_pos) and torch.allclose(buf.held_yaw[due], exp_yaw)
            updates += int(due.sum())
        if (~due).any():
            ok &= torch.equal(buf.held_pos[~due], prev_held[~due])
        ok &= bool(((buf.countdown >= 1) & (buf.countdown <= 6)).all())
    rate = updates / (N * (T - 1))
    ok &= 1 / 6 - 0.03 < rate < 1 / 4 + 0.03
    # estimated quat keeps the current roll / pitch and takes the held yaw
    est_q = buf.pelvis_pose(quat[-1])[1]
    rpy_e, rpy_t = quat_to_rpy(est_q), quat_to_rpy(quat[-1])
    ok &= torch.allclose(rpy_e[:, :2], rpy_t[:, :2], atol=1e-5)
    ok &= torch.allclose(torch.cos(rpy_e[:, 2] - buf.held_yaw), torch.ones(N, device=dev), atol=1e-5)
    print(f"[buffer] update rate {rate:.3f} (expect ~0.2) -> {'OK' if ok else 'FAIL'}")
    return bool(ok)


def _term_values(env, ctx, spec):
    from finetune.mdp.batched_obs import get_bterm
    return {f"{i}:{t.name}": get_bterm(t.name)(ctx, t.params) for i, t in enumerate(spec.terms)}


def _ctx(env, builder):
    ctx = env._context()
    ctx.root_offset_pos = builder.root_offset
    return ctx


def check_env(args) -> bool:
    import warp as wp
    wp.init()
    from finetune.env.batched_mujoco_env import BatchedG1MimicVecEnv

    ok = True
    for period, latency, expect_equal in (([1, 1], [0, 0], True), ([4, 6], [0, 3], False)):
        cfg = cfg_from_args(args)
        cfg.env.num_envs = args.envs
        cfg.env.backend = "gpu"
        cfg.train.device = args.device
        cfg.odom_delay.enabled = True
        cfg.odom_delay.update_period_steps = period
        cfg.odom_delay.latency_steps = latency
        cfg.odom_delay.drift_enabled = not expect_equal   # exact-equality check needs drift off
        cfg.resolve()
        env = BatchedG1MimicVecEnv(cfg, device=args.device, verbose=False)
        if env.odom is None:
            print(f"[env] actor spec {cfg.paths.actor_spec} has no use_odom_delay term")
            return False
        odom_names = {t.name for t in env.actor_spec.terms if t.params.get("use_odom_delay")}
        g = torch.Generator(device=env.device).manual_seed(1)
        max_err = {n: 0.0 for n in odom_names}
        critic_err = 0.0
        for _ in range(args.steps):
            a = 0.3 * torch.randn(env.num_envs, env.num_actions, generator=g, device=env.device)
            env.step(a)
            ctx_odom = _ctx(env, env.actor_builder)
            ctx_true = _ctx(env, env.actor_builder)
            ctx_true.est_pelvis = None
            vo, vt = _term_values(env, ctx_odom, env.actor_spec), _term_values(env, ctx_true, env.actor_spec)
            for k in vo:
                name = k.split(":", 1)[1]
                err = float((vo[k] - vt[k]).abs().max())
                if name in odom_names:
                    max_err[name] = max(max_err[name], err)
                elif err > 0:
                    print(f"[env] non-odom actor term {name} changed with odom ({err:.2e})")
                    ok = False
            co = _term_values(env, _ctx(env, env.critic_builder), env.critic_spec)
            cc = _ctx(env, env.critic_builder)
            cc.est_pelvis = None
            ct = _term_values(env, cc, env.critic_spec)
            critic_err = max(critic_err, max(float((co[k] - ct[k]).abs().max()) for k in co))
        tag = f"period={period} latency={latency}"
        for name, err in max_err.items():
            good = err < 1e-4 if expect_equal else err > 1e-3
            ok &= good
            print(f"[env] {tag}: {name:22s} max |odom - true| = {err:.3e} "
                  f"({'== true expected' if expect_equal else '!= true expected'}) -> {'OK' if good else 'FAIL'}")
        ok &= critic_err == 0.0
        print(f"[env] {tag}: critic obs max |odom - true| = {critic_err:.1e} -> "
              f"{'OK' if critic_err == 0.0 else 'FAIL'}")
        env.close()
    return bool(ok)


def main():
    p = add_cfg_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    dev = torch.device(args.device)
    ok = check_buffer(dev)
    ok &= check_env(args)
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
