from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from finetune import _paths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from finetune.tasks.ft_cfg import FinetuneCfg, apply_overrides  # noqa: E402
from finetune.tasks.optim import bind_scaled_groups, gaussian_kl, group_lrs, make_param_groups  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion", required=False, help="integrated motion pkl (paths.motion_file)")
    p.add_argument("--name", default=None, help="student checkpoint name: assets/pt/<name>.pt (+ .yaml spec)")
    p.add_argument("--student", default=None, help="explicit student .pt path")
    p.add_argument("--actor-spec", default=None)
    p.add_argument("--critic-spec", default=None)
    p.add_argument("--run", default=None, help="run directory name (default <name>_<time>)")
    p.add_argument("--resume", default=None, help="finetune checkpoint to resume")
    p.add_argument("--device", default=None)
    p.add_argument("--set", nargs="*", default=[], help="cfg overrides section.field=value")
    return p


def make_cfg(args) -> FinetuneCfg:
    cfg = FinetuneCfg()
    if args.name:
        cfg.paths.name = args.name
    if args.motion:
        cfg.paths.motion_file = args.motion
    if args.student:
        cfg.paths.student_pt = args.student
    if args.actor_spec:
        cfg.paths.actor_spec = args.actor_spec
    if args.critic_spec:
        cfg.paths.critic_spec = args.critic_spec
    if args.resume:
        cfg.train.resume = args.resume
    if args.device:
        cfg.train.device = args.device
    apply_overrides(cfg, args.set)
    cfg.resolve()
    if not cfg.paths.motion_file:
        raise SystemExit("--motion (paths.motion_file) is required")
    files = [cfg.paths.student_pt, cfg.paths.actor_spec, cfg.paths.critic_spec, cfg.paths.motion_file, cfg.paths.xml]
    if cfg.critic.source == "teacher":
        files.append(cfg.critic.teacher_ckpt)
    for f in files:
        if not os.path.isfile(f):
            raise SystemExit(f"missing file: {f}")
    return cfg


class Logger:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(log_dir=run_dir, flush_secs=10)
        except Exception as e:
            print(f"[log] tensorboard unavailable ({e}); csv only")
        self.csv_path = os.path.join(run_dir, "log.csv")
        self._csv_keys = None

    def log(self, it: int, scalars: dict):
        if self.tb is not None:
            for k, v in scalars.items():
                self.tb.add_scalar(k, v, it)
        if self._csv_keys is None:
            self._csv_keys = ["iter"] + sorted(scalars)
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self._csv_keys)
        row = {"iter": it, **scalars}
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self._csv_keys])

    def close(self):
        if self.tb is not None:
            self.tb.close()


def build_optimizer(policy, cfg: FinetuneCfg):
    actor_lr = cfg.lora.lr if cfg.lora.mode == "lora" else cfg.lora.fft_lr
    named = [("actor", policy.actor_trainable_parameters(), actor_lr)]
    if cfg.train.std_trainable:
        named.append(("std", [policy.std], cfg.train.std_lr or actor_lr))
    named.append(("critic", policy.critic_parameters(), cfg.critic.lr))
    groups = make_param_groups(named)
    if not any(g["name"] == "actor" for g in groups):
        raise SystemExit("no trainable actor parameters (lora.mode=frozen?)")
    opt = torch.optim.Adam(groups)
    bind_scaled_groups(opt, ref_group="actor", follow_groups=cfg.ppo.adaptive_groups)
    return opt, actor_lr


def save_checkpoint(path: str, policy, optimizer, it: int, cfg: FinetuneCfg, learning_rate: float):
    torch.save({
        "iter": it,
        "cfg": cfg.to_dict(),
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "merged_actor_state_dict": policy.merged_actor().state_dict(),
        "merged_critic_state_dict": policy.merged_critic().state_dict(),
        "std": policy.std.detach().cpu(),
        "learning_rate": learning_rate,
    }, path)


def main():
    args = build_parser().parse_args()
    cfg = make_cfg(args)
    _paths.setup(cfg.paths.rsl_rl_root)

    from finetune.env.mujoco_env import G1MimicVecEnv
    from finetune.mdp.policy import LoRAActorCritic, load_student_state, load_teacher_critic_state
    from rsl_rl.algorithms import PPO

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)
    device = cfg.train.device

    run_name = args.run or f"{cfg.paths.name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = os.path.join(cfg.paths.log_root, run_name)
    logger = Logger(run_dir)
    with open(os.path.join(run_dir, "cfg.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"[train] run dir: {run_dir}")

    env = G1MimicVecEnv(cfg, device=device)
    student_state = load_student_state(cfg.paths.student_pt)
    critic_state = load_teacher_critic_state(cfg.critic.teacher_ckpt) if cfg.critic.source == "teacher" else None
    policy = LoRAActorCritic(student_state, env.num_critic_obs, env.num_actions, cfg.lora, cfg.critic,
                             init_noise_std=cfg.train.init_noise_std, std_trainable=cfg.train.std_trainable,
                             critic_state=critic_state).to(device)
    if policy.num_actor_obs != env.num_actor_obs:
        raise SystemExit(f"student expects {policy.num_actor_obs} obs but the actor spec builds {env.num_actor_obs}")
    print("[policy] " + policy.summary())

    optimizer, actor_lr = build_optimizer(policy, cfg)   # separate LoRA / std / critic learning rates
    ppo_kwargs = dict(
        num_learning_epochs=cfg.ppo.num_learning_epochs, num_mini_batches=cfg.ppo.num_mini_batches,
        clip_param=cfg.ppo.clip_param, gamma=cfg.ppo.gamma, lam=cfg.ppo.lam,
        value_loss_coef=cfg.ppo.value_loss_coef, entropy_coef=cfg.ppo.entropy_coef,
        learning_rate=actor_lr, max_grad_norm=cfg.ppo.max_grad_norm,
        use_clipped_value_loss=cfg.ppo.use_clipped_value_loss, schedule=cfg.ppo.schedule,
        desired_kl=cfg.ppo.desired_kl, normalize_advantage_per_mini_batch=cfg.ppo.normalize_advantage_per_mini_batch,
        device=device,
    )
    alg = PPO(policy, **ppo_kwargs)
    alg.optimizer = optimizer
    alg.init_storage("rl", env.num_envs, cfg.ppo.num_steps_per_env,
                     [env.num_actor_obs], [env.num_critic_obs], [env.num_actions])

    start_it = 0
    if cfg.train.resume:
        ck = torch.load(cfg.train.resume, map_location=device, weights_only=False)
        policy.load_state_dict(ck["model_state_dict"])
        alg.optimizer.load_state_dict(ck["optimizer_state_dict"])
        bind_scaled_groups(alg.optimizer, ref_group="actor", follow_groups=cfg.ppo.adaptive_groups)
        alg.learning_rate = float(ck.get("learning_rate", actor_lr))
        for g in alg.optimizer.param_groups:
            g["lr"] = alg.learning_rate
        start_it = int(ck.get("iter", 0)) + 1
        print(f"[train] resumed from {cfg.train.resume} @ iter {start_it}")

    obs, extras = env.get_observations()
    critic_obs = extras["observations"]["critic"]
    ep_rew, ep_len, ep_div, ep_end, ep_dist = (deque(maxlen=100) for _ in range(5))
    ep_terms = {k: deque(maxlen=100) for k in env.rewards.term_names}
    steps_per_iter = cfg.ppo.num_steps_per_env * env.num_envs
    total_steps = start_it * steps_per_iter
    t_start = time.time()

    for it in range(start_it, cfg.train.max_iterations):
        warmup = it < cfg.critic.warmup_iters
        policy.set_actor_trainable(not warmup)
        alg.schedule = "fixed" if warmup else cfg.ppo.schedule
        t_iter = time.time()
        rew_sum = 0.0
        with torch.inference_mode():
            for _ in range(cfg.ppo.num_steps_per_env):
                actions = alg.act(obs, critic_obs)
                obs, rewards, dones, extras = env.step(actions)
                critic_obs = extras["observations"]["critic"]
                alg.process_env_step(rewards, dones, extras)
                rew_sum += float(rewards.sum())
                for ep in extras["episodes"]:
                    ep_rew.append(ep["reward"]); ep_len.append(ep["length"])
                    ep_div.append(ep["diverged"]); ep_end.append(ep["motion_end"]); ep_dist.append(ep["mean_max_body_dist"])
                    for k, v in ep["terms"].items():
                        ep_terms[k].append(v)
            alg.compute_returns(critic_obs)
        t_collect = time.time() - t_iter
        st = alg.storage
        # critic quality on this rollout, before the update: 1 - Var(R - V) / Var(R)
        # (0 = as good as predicting the mean return, 1 = perfect; unlike the value loss
        # this is independent of the return scale)
        ret, val = st.returns.flatten(), st.values.flatten()
        explained_var = float(1.0 - torch.var(ret - val) / torch.var(ret).clamp_min(1e-8))
        kl_obs = st.observations.flatten(0, 1)
        kl_old_mu, kl_old_sigma = st.mu.flatten(0, 1).clone(), st.sigma.flatten(0, 1).clone()
        losses = alg.update()
        with torch.inference_mode():
            policy.update_distribution(kl_obs)
            kl = gaussian_kl(kl_old_mu, kl_old_sigma, policy.action_mean, policy.action_std)
        t_learn = time.time() - t_iter - t_collect
        total_steps += steps_per_iter

        if it % cfg.train.log_interval == 0:
            fps = steps_per_iter / max(t_collect + t_learn, 1e-9)
            scalars = {
                "Loss/value_function": losses["value_function"],
                "Loss/explained_variance": explained_var,
                "Loss/surrogate": losses["surrogate"],
                "Loss/entropy": losses["entropy"],
                "Policy/kl": kl,
                "Policy/mean_noise_std": float(policy.std.mean()),
                "Policy/actor_trainable": float(not warmup),
                "Train/mean_step_reward": rew_sum / steps_per_iter,
                "Train/fps": fps,
                "Train/total_steps": total_steps,
                "Train/collect_time": t_collect,
                "Train/learn_time": t_learn,
            }
            for name, lr in group_lrs(alg.optimizer).items():
                scalars[f"Train/lr_{name}"] = lr
            if ep_rew:
                scalars.update({
                    "Episode/mean_reward": statistics.fmean(ep_rew),
                    "Episode/mean_length": statistics.fmean(ep_len),
                    "Episode/diverged_rate": statistics.fmean(ep_div),
                    "Episode/motion_end_rate": statistics.fmean(ep_end),
                    "Episode/mean_max_body_dist": statistics.fmean(ep_dist),
                })
                for k, dq in ep_terms.items():
                    if dq:
                        scalars[f"Episode/rew_{k}"] = statistics.fmean(dq)
            logger.log(it, scalars)
            print(f"it {it:5d} | {'warmup ' if warmup else 'lora   '}| step_rew {scalars['Train/mean_step_reward']:7.4f} "
                  f"| ep_rew {scalars.get('Episode/mean_reward', float('nan')):8.3f} "
                  f"| ep_len {scalars.get('Episode/mean_length', float('nan')):6.1f} "
                  f"| div {scalars.get('Episode/diverged_rate', float('nan')):.2f} "
                  f"| vloss {losses['value_function']:.4f} ev {explained_var:.2f} | sloss {losses['surrogate']:.4f} "
                  f"| kl {kl:.4f} | lr {alg.learning_rate:.1e} "
                  f"| std {scalars['Policy/mean_noise_std']:.3f} | fps {fps:6.0f} | {time.time() - t_start:6.0f}s")

        if (it + 1) % cfg.train.save_interval == 0 or it + 1 == cfg.train.max_iterations:
            save_checkpoint(os.path.join(run_dir, f"model_{it + 1}.pt"), policy, alg.optimizer, it, cfg, alg.learning_rate)

    logger.close()
    print(f"[train] done. checkpoints in {run_dir}; export with finetune/tasks/export.py --ckpt <model_N.pt>")


if __name__ == "__main__":
    main()
