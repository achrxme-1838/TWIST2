"""Parallel motion-tracking evaluation on the mujoco_warp batched env.

Usage:
    python evaluation_batched.py --motion <pkl> --policy ckpt.pt --envs 4096 --out eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from finetune import _paths  # noqa: E402

from finetune.checks._common import add_cfg_args, cfg_from_args  # noqa: E402
from finetune.checks.evaluation import (  # noqa: E402
    Aggregate, load_actor, per_frame_errors, print_metrics, trial_variants,
)
from finetune.tasks.ft_cfg import PathsCfg  # noqa: E402

REC_KEYS = ("body_pos", "joint_pos", "root_pos", "root_rot", "root_lin_vel")


def _trial_metrics(payload):
    """(pred, gt, fail_frame, num_frames) -> trial_variants dict. ~2 ms for a 170-frame trial, so
    it runs inline: the rollout produces ~11 trials/s and this handles ~440/s."""
    pred, gt, fail_frame, num_frames = payload
    return trial_variants(per_frame_errors(pred, gt), fail_frame, num_frames)


class Recorder:
    """Per-env trajectory buffer for the trial currently running in that env."""

    def __init__(self, num_envs: int):
        self.buf = [{k: [] for k in REC_KEYS} for _ in range(num_envs)]
        self.gt = [{k: [] for k in REC_KEYS} for _ in range(num_envs)]
        self.fail_frame = [None] * num_envs
        self.job = [None] * num_envs          # (motion id, trial index) or None when idle

    def clear(self, i: int):
        self.buf[i] = {k: [] for k in REC_KEYS}
        self.gt[i] = {k: [] for k in REC_KEYS}
        self.fail_frame[i] = None

    def finish(self, i: int):
        pred = {k: np.asarray(v, dtype=np.float64) for k, v in self.buf[i].items()}
        gt = {k: np.asarray(v, dtype=np.float64) for k, v in self.gt[i].items()}
        n = pred["body_pos"].shape[0]
        ff = n if self.fail_frame[i] is None else self.fail_frame[i]
        return pred, gt, ff, n, self.fail_frame[i] is not None


def main():
    p = add_cfg_args(argparse.ArgumentParser(description=__doc__,
                                             formatter_class=argparse.RawDescriptionHelpFormatter),
                     policy_arg="policy")
    p.add_argument("--envs", type=int, default=4096, help="warp worlds running trials in parallel")
    p.add_argument("--episodes-per-motion", type=int, default=1)
    p.add_argument("--fail-max-dist", type=float, default=0.5)
    p.add_argument("--fail-mode", choices=("mean", "any"), default="mean")
    p.add_argument("--start", choices=("frame0",), default="frame0")
    p.add_argument("--ref-at", choices=("pre", "post"), default="pre",
                   help="reference motion time a recorded state is scored against. pre (default) "
                        "matches IsaacLab humanoid_eval.py; post uses the current motion time.")
    p.add_argument("--max-motions", type=int, default=None, help="evaluate only the first N motions")
    p.add_argument("--no-quality", dest="quality", action="store_false",
                   help="skip the motion-quality metrics (smoothness / foot slip / constraint "
                        "violations). They are reported on the success trials only, as in IsaacLab.")
    p.add_argument("--jerk-keypoints", default=None,
                   help="comma-separated body names reported as keypoint_jerk, e.g. "
                        "'left_ankle_roll_link,right_ankle_roll_link'. Defaults to the IsaacLab "
                        "config's G1_29DOF_EVAL_JERK_KEYPOINTS.")
    p.add_argument("--quality-warmup-steps", type=int, default=0,
                   help="steps skipped at the start of each trial when accumulating the quality "
                        "metrics (IsaacLab --quality_warmup_steps).")
    p.add_argument("--workers", type=int, default=0, help=argparse.SUPPRESS)   # deprecated, ignored
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = cfg_from_args(args)
    cfg.env.num_envs = args.envs
    cfg.env.backend = "gpu"
    cfg.train.device = args.device
    cfg.termination.max_ref_motion_dist = float("inf")
    cfg.env.episode_length_s = 1e4
    cfg.motion.start_time_mode = "zero"
    if args.max_motions:
        cfg.motion.max_motions = args.max_motions
    if cfg.paths.xml == PathsCfg.xml:
        gpu_xml = cfg.paths.xml.replace("_29dof.xml", "_29dof_deploy_gpu.xml")
        if os.path.isfile(gpu_xml):
            cfg.paths.xml = gpu_xml
    cfg.resolve()
    print(f"[eval] xml: {cfg.paths.xml}")
    import mujoco as _mj
    if _mj.MjModel.from_xml_path(cfg.paths.xml).opt.solver == _mj.mjtSolver.mjSOL_PGS:
        raise SystemExit(
            f"{cfg.paths.xml} uses the PGS solver, which mujoco_warp cannot run. Use the Newton\n"
            "variant (assets/g1/g1_sim2sim_29dof_deploy_gpu.xml) here, or run evaluation.py for\n"
            "the PGS deploy physics on the CPU env.")

    from finetune.env.batched_mujoco_env import BatchedG1MimicVecEnv

    dev = torch.device(args.device)
    actor, kind = load_actor(cfg.paths.student_pt)
    actor = actor.to(dev)
    env = BatchedG1MimicVecEnv(cfg, device=args.device, verbose=False)
    env.auto_reset = False
    if actor[0].in_features != env.num_actor_obs:
        raise SystemExit(
            f"obs mismatch: the policy expects {actor[0].in_features} observations but\n"
            f"  {cfg.paths.actor_spec}\nbuilds {env.num_actor_obs}. Pass the right one with --actor-spec.")

    n_motions = env.motions.num_motions
    N = env.num_envs
    print(f"[eval] {kind} from {cfg.paths.student_pt}")
    print(f"[eval] actor spec: {cfg.paths.actor_spec}")
    print(f"[eval] {n_motions} motions x {args.episodes_per_motion} trials on {N} parallel envs "
          f"| fail: body dist > {args.fail_max_dist} m ({args.fail_mode}) | ref-at={args.ref_at}")

    quality = None
    if args.quality:
        from finetune.checks.quality_metrics import BatchedQualityTracker
        keypoints = ([k.strip() for k in args.jerk_keypoints.split(",") if k.strip()]
                     if args.jerk_keypoints else None)
        quality = BatchedQualityTracker(env, args.device, jerk_keypoints=keypoints,
                                        warmup_steps=args.quality_warmup_steps)

    jobs = [(mid, k) for mid in range(n_motions) for k in range(args.episodes_per_motion)]
    total_jobs = len(jobs)
    next_job = 0
    rec = Recorder(N)
    env.force_motion_ids = torch.full((N,), -1, dtype=torch.long, device=dev)

    # seed every env with its first job
    seed_ids, seed_mids = [], []
    for i in range(N):
        if next_job < total_jobs:
            rec.job[i] = jobs[next_job]
            seed_ids.append(i)
            seed_mids.append(jobs[next_job][0])
            next_job += 1
    env.force_motion_ids[torch.as_tensor(seed_ids, device=dev)] = torch.as_tensor(seed_mids, device=dev)
    env.reset()
    active = torch.zeros(N, dtype=torch.bool, device=dev)
    active[torch.as_tensor(seed_ids, device=dev)] = True

    reduce_dist = {"mean": lambda a: a.mean(axis=-1), "any": lambda a: a.max(axis=-1)}[args.fail_mode]
    if args.workers:
        print("[eval] --workers is ignored: the metric reduction costs ~2 ms/trial and runs inline.")
    results = {}                       # job -> (variants, failed, quality values or None)

    obs, _ = env.get_observations()
    t_start = time.time()
    steps = 0
    next_report = max(total_jobs // 20, 1)
    while bool(active.any()):
        t_pre = env.t.clone()                                   # motion time before the step
        with torch.no_grad():
            obs, _, dones, _ = env.step(actor(obs))
        steps += 1
        t_ref = t_pre if args.ref_at == "pre" else env.t

        robot = env.core.state()
        ref_frame = env.motions.frame(env.mid, t_ref)
        ref_body = env.motions.body_state(env.mid, t_ref)
        act_np = active.cpu().numpy()
        snap = {
            "body_pos": robot.body_pos.cpu().numpy(),
            "joint_pos": robot.dof_pos_sdk.cpu().numpy(),
            "root_pos": robot.root_pos.cpu().numpy(),
            "root_rot": robot.root_quat.cpu().numpy(),
            "root_lin_vel": robot.root_lin_vel_w.cpu().numpy(),
        }
        gsnap = {
            "body_pos": ref_body.pos.cpu().numpy(),
            "joint_pos": ref_frame.dof_pos.cpu().numpy(),
            "root_pos": ref_frame.root_pos.cpu().numpy(),
            "root_rot": ref_frame.root_quat.cpu().numpy(),
            "root_lin_vel": ref_frame.root_lin_vel.cpu().numpy(),
        }
        dist = np.linalg.norm(snap["body_pos"] - gsnap["body_pos"], axis=-1)   # [N, B_ext]
        failed_now = reduce_dist(dist) > args.fail_max_dist
        done_np = dones.cpu().numpy()

        for i in np.nonzero(act_np)[0]:
            for k in REC_KEYS:
                rec.buf[i][k].append(snap[k][i])
                rec.gt[i][k].append(gsnap[k][i])
            if rec.fail_frame[i] is None and failed_now[i]:
                rec.fail_frame[i] = len(rec.buf[i]["body_pos"])

        if quality is not None:
            # ``active`` is still the pre-step mask here, so the trial that ended on this very
            # step is counted and the idle envs are not -- IsaacLab's ~self._terminated.
            quality.update(robot, env.last_action, active)

        finished = [int(i) for i in np.nonzero(act_np & done_np)[0]]
        if finished:
            qvals = quality.pop(finished) if quality is not None else [None] * len(finished)
            refill_ids, refill_mids = [], []
            for i, qv in zip(finished, qvals):
                job = rec.job[i]
                pred, gt, ff, n, failed = rec.finish(i)
                results[job] = (_trial_metrics((pred, gt, ff, n)), failed, qv)
                rec.clear(i)
                if next_job < total_jobs:
                    rec.job[i] = jobs[next_job]
                    refill_ids.append(i)
                    refill_mids.append(jobs[next_job][0])
                    next_job += 1
                else:
                    rec.job[i] = None
                    active[i] = False
            if refill_ids:
                env.reset_envs_to(torch.as_tensor(refill_ids, dtype=torch.long, device=dev),
                                  torch.as_tensor(refill_mids, dtype=torch.long, device=dev))
                obs, _ = env.get_observations()
            done_n = len(results)
            if done_n >= next_report or done_n == total_jobs:
                el = time.time() - t_start
                eta = el / max(done_n, 1) * (total_jobs - done_n)
                print(f"  [{done_n:6d}/{total_jobs}] {el:6.1f}s elapsed  "
                      f"{done_n / max(el, 1e-9):6.2f} trials/s  {steps} steps  eta {eta / 60:5.1f} min")
                next_report = done_n + max(total_jobs // 20, 1)

    # ---- aggregate over every trial; no per-motion breakdown ----
    agg = {k: Aggregate() for k in ("all", "success", "keep", "continue")}
    num_trials, num_success = 0, 0
    for var, failed, qv in results.values():
        agg["all"].add(var["all"]); agg["keep"].add(var["keep"]); agg["continue"].add(var["continue"])
        if not failed:
            agg["success"].add(var["all"])
            # Quality metrics have no ground truth and are only defined on a motion tracked to
            # the end, so they join the Success variant only -- IsaacLab record_quality_metrics.
            # One trial, one value: weight 1 makes the aggregate their plain mean.
            if qv:
                agg["success"].add({k: (v, 1) for k, v in qv.items()})
            num_success += 1
        num_trials += 1

    summary = OrderedDict([
        ("policy", cfg.paths.student_pt),
        ("kind", kind),
        ("xml", cfg.paths.xml),
        ("motion_file", cfg.paths.motion_file),
        ("fail_criterion", {"mode": args.fail_mode, "max_dist": args.fail_max_dist}),
        ("start", {"mode": args.start, "settle_time": 0.0}),
        ("ref_at", args.ref_at),
        ("quality_metrics", {"enabled": bool(args.quality),
                             "warmup_steps": args.quality_warmup_steps,
                             "jerk_keypoints": quality.keypoint_names} if args.quality
                            else {"enabled": False}),
        ("num_envs", N),
        ("num_unique_motions", n_motions),
        ("trials_per_motion", args.episodes_per_motion),
        ("success_rate", num_success / max(num_trials, 1)),
        ("num_motions", num_trials),
        ("all", agg["all"].result()),
        ("success", agg["success"].result()),
        ("keep", agg["keep"].result()),
        ("continue", agg["continue"].result()),
    ])
    print_metrics(summary)
    el = time.time() - t_start
    print(f"[eval] {num_trials} trials in {el:.1f}s ({num_trials / max(el, 1e-9):.1f} trials/s, {steps} env steps)")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
