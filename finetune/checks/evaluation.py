"""Motion-tracking evaluation of a policy in the MuJoCo env.
Usage:
    python evaluation.py --motion <pkl> --policy ckpt.pt --episodes-per-motion N --out metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from finetune.checks._common import add_cfg_args, cfg_from_args  # noqa: E402

LINK_METRICS = ("mpjpe_g", "mpjpe_l", "mpjpe_pa", "vel_dist", "accel_dist")
JOINT_METRICS = ("upper_body_joints_dist", "lower_body_joints_dist")
ROOT_METRICS = ("root_height_error", "root_vel_tracking_error", "root_r_error", "root_p_error", "root_y_error")
NUM_LOWER_JOINTS = 12   # DEX TRACKED_JOINT_NAMES[:12] = legs (SDK order)


# ----------------------------------------------------------------------------- policy
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


# ----------------------------------------------------------------------------- per-frame errors
def _quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    from utils.math import quat_apply, quat_conj
    return quat_apply(quat_conj(q), v)


def _euler_xyz(q: np.ndarray) -> tuple:
    """Vectorised euler_xyz_from_quat (wxyz -> roll, pitch, yaw), IsaacLab formulas."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.where(np.abs(sinp) >= 1.0, np.copysign(np.pi / 2, sinp), np.arcsin(sinp))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def per_frame_errors(pred: dict, gt: dict) -> "OrderedDict[str, np.ndarray]":
    """Per-frame (per-joint where applicable) error arrays of one trial; axis 0 = frame.
    Same quantities as MotionTrackingMetrics, without the cross-env reduction."""
    from dex_rl_lab.utils.smpl_evaluator import compute_metrics_lite
    out = OrderedDict()
    raw = compute_metrics_lite([pred["body_pos"]], [gt["body_pos"]], use_tqdm=False, concatenate=False)
    for name in LINK_METRICS:
        out[name] = np.asarray(raw[name][0])
    out["upper_body_joints_dist"] = np.abs(pred["joint_pos"][:, NUM_LOWER_JOINTS:] - gt["joint_pos"][:, NUM_LOWER_JOINTS:])
    out["lower_body_joints_dist"] = np.abs(pred["joint_pos"][:, :NUM_LOWER_JOINTS] - gt["joint_pos"][:, :NUM_LOWER_JOINTS])
    out["root_height_error"] = np.abs(pred["root_pos"][:, 2] - gt["root_pos"][:, 2])
    v_l = _quat_apply_inverse(pred["root_rot"], pred["root_lin_vel"])
    v_l_gt = _quat_apply_inverse(gt["root_rot"], gt["root_lin_vel"])
    out["root_vel_tracking_error"] = np.linalg.norm(v_l - v_l_gt, axis=-1)
    from utils.math import quat_conj, quat_mul
    roll, pitch, yaw = _euler_xyz(quat_mul(pred["root_rot"], quat_conj(gt["root_rot"])))
    out["root_r_error"], out["root_p_error"], out["root_y_error"] = np.abs(roll), np.abs(pitch), np.abs(yaw)
    return out


def _freeze(arr: np.ndarray, fail_frame: int) -> np.ndarray:
    """MotionTrackingMetrics._freeze_per_env_arrays: hold the value at fail_frame-1 from fail_frame on."""
    if fail_frame <= 0 or fail_frame >= arr.shape[0]:
        return arr
    new = arr.copy()
    new[fail_frame:] = arr[fail_frame - 1]
    return new


def trial_variants(errors: dict, fail_frame: int, num_frames: int) -> dict:
    """{variant: {metric: (mean, weight)}} of one trial. Weights follow MotionTrackingMetrics:
    element count for the link metrics, one per trial (env) for joint / root metrics."""
    def reduce(arrs: dict) -> dict:
        res = OrderedDict()
        for name, a in arrs.items():
            if a.size == 0:
                continue
            w = int(a.size) if name in LINK_METRICS else 1
            res[name] = (float(np.mean(a)), w)
        return res

    cut = min(fail_frame, num_frames)
    trimmed = {k: v[: max(cut - (num_frames - v.shape[0]), 0)] for k, v in errors.items()}
    frozen = {k: _freeze(v, fail_frame) for k, v in errors.items()}
    return {"all": reduce(trimmed), "keep": reduce(errors), "continue": reduce(frozen)}


class Aggregate:
    """Weighted average of (mean, weight) pairs -- MotionTrackingMetrics._record_metrics / conclude."""

    def __init__(self):
        self.store: dict = OrderedDict()

    def add(self, values: dict):
        for name, (mean, w) in values.items():
            if np.isfinite(mean) and w > 0:
                self.store.setdefault(name, [[], []])
                self.store[name][0].append(mean)
                self.store[name][1].append(w)

    def result(self) -> dict:
        return OrderedDict((k, float(np.average(m, weights=w))) for k, (m, w) in self.store.items() if m)


# ----------------------------------------------------------------------------- rollout
def start_from_default_pose(env, mid: int, settle_time: float):
    """Deploy-style start (sim2sim_batch.sh): the robot stands in the default pose at motion
    frame 0's xy / heading and tracks frame 0 for ``settle_time`` s (motion time < 0 clamps
    to frame 0) before the motion starts -- instead of IsaacLab's reset onto frame 0."""
    from cfg import g1_29dof_cfg as rcfg
    from utils.math import heading_quat_from_quat
    slot = env.slots[0]
    fr0 = env.motions.frame(mid, 0.0)
    root_pos = np.array([fr0.root_pos[0], fr0.root_pos[1], 0.793])
    slot.core.reset_to(root_pos, heading_quat_from_quat(fr0.root_quat), np.zeros(3), np.zeros(3),
                       rcfg.DEFAULT_DOF_POS, np.zeros(env.num_actions))
    slot.t0 = -float(settle_time)
    slot.t = slot.t0
    slot.step_count = 0
    slot.anchor_delta_xy = np.zeros(2)          # robot already sits on frame 0's xy
    slot.actor_builder.reset()
    slot.critic_builder.reset()
    env._write_obs(0, slot, *env._ref_now(slot))


def run_trial(env, actor, mid: int, fail_max_dist: float, fail_mode: str = "mean",
              start: str = "frame0", settle_time: float = 1.0) -> dict:
    reduce_dist = {"mean": np.mean, "any": np.max}[fail_mode]
    env.force_motion_id = mid
    env.reset()
    if start == "default":
        start_from_default_pose(env, mid, settle_time)
    obs, _ = env.get_observations()
    slot = env.slots[0]
    ref_shift_xy = np.zeros(2)
    mc = env.cfg.motion
    pred = {k: [] for k in ("body_pos", "joint_pos", "root_pos", "root_rot", "root_lin_vel")}
    gt = {k: [] for k in pred}
    fail_frame = None
    done = False
    with torch.no_grad():
        while not done:
            t_cmd = slot.t
            if start == "default" and t_cmd < 0.0 <= t_cmd + env.control_dt + 1e-9:
                t_origin = mc.mimic_step_offset * env.control_dt if mc.anchor_origin_at_mimic_frame else 0.0
                ref_shift_xy = slot.core.data.qpos[:2] - env.motions.frame(mid, t_origin).root_pos[:2]
                slot.anchor_delta_xy = ref_shift_xy.copy()
            obs, _, d, _ = env.step(actor(obs))
            done = bool(d[0])
            if t_cmd < 0.0:
                continue
            robot = slot.core.state()
            fr = env.motions.frame(mid, t_cmd)
            fr.root_pos = fr.root_pos + np.r_[ref_shift_xy, 0.0]
            ref_pos, _ = env.motions.fk.body_pose(fr.root_pos, fr.root_quat, fr.dof_pos)
            pred["body_pos"].append(robot.body_pos); gt["body_pos"].append(ref_pos)
            pred["joint_pos"].append(robot.dof_pos_sdk); gt["joint_pos"].append(fr.dof_pos.copy())
            pred["root_pos"].append(robot.root_pos); gt["root_pos"].append(fr.root_pos.copy())
            pred["root_rot"].append(robot.root_quat); gt["root_rot"].append(fr.root_quat.copy())
            pred["root_lin_vel"].append(robot.root_lin_vel_w); gt["root_lin_vel"].append(fr.root_lin_vel.copy())
            if fail_frame is None and reduce_dist(np.linalg.norm(robot.body_pos - ref_pos, axis=-1)) > fail_max_dist:
                fail_frame = len(pred["body_pos"])
    pred = {k: np.asarray(v, dtype=np.float64) for k, v in pred.items()}
    gt = {k: np.asarray(v, dtype=np.float64) for k, v in gt.items()}
    n = pred["body_pos"].shape[0]
    return {"errors": per_frame_errors(pred, gt), "fail_frame": n if fail_frame is None else fail_frame,
            "num_frames": n, "failed": fail_frame is not None}


# ----------------------------------------------------------------------------- main
def print_metrics(summary: dict):
    print(f"Number of reference motions: {summary['num_motions']}")
    print(f"Success Rate: {summary['success_rate']:.10f}")
    for label, key in (("All:      ", "all"), ("Succ:     ", "success"), ("Keep:     ", "keep"), ("Continue: ", "continue")):
        print(label, " \t\t".join(f"{k}: {v:.3f}" for k, v in summary[key].items()))


def main():
    p = add_cfg_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter),
                     policy_arg="policy")
    p.add_argument("--episodes-per-motion", type=int, default=1, help="trials per motion (IsaacLab repeat_per_pkl)")
    p.add_argument("--fail-max-dist", type=float, default=0.5,
                   help="extended-body distance [m] that marks a trial as failed (IsaacLab eval_fail_max_dist)")
    p.add_argument("--start", choices=("frame0", "default"), default="frame0",
                   help="frame0: reset the robot onto motion frame 0 (IsaacLab evaluator, default); "
                        "default: stand in the default pose and track frame 0 for --settle-time s first "
                        "(deploy / sim2sim_batch.sh style)")
    p.add_argument("--settle-time", type=float, default=1.0, help="seconds on frame 0 before the motion (--start default)")
    p.add_argument("--fail-mode", choices=("mean", "any"), default="mean",
                   help="mean: mean body distance > threshold (IsaacLab evaluator, default); "
                        "any: any single body > threshold (training termination, training_mode=True)")
    p.add_argument("--out", default=None, help="json summary path")
    args = p.parse_args()
    cfg = cfg_from_args(args)
    cfg.env.num_envs = 1
    cfg.termination.max_ref_motion_dist = float("inf")
    cfg.env.episode_length_s = 1e4
    from finetune.env.mujoco_env import G1MimicVecEnv

    actor, kind = load_actor(cfg.paths.student_pt)
    env = G1MimicVecEnv(cfg, verbose=False)
    env.auto_reset = False
    n_motions = env.motions.num_motions
    print(f"[eval] {kind} from {cfg.paths.student_pt}")
    print(f"[eval] actor spec: {cfg.paths.actor_spec}")
    print(f"[eval] {n_motions} motions x {args.episodes_per_motion} trials from {os.path.basename(cfg.paths.motion_file)} "
          f"| fail: {'mean' if args.fail_mode == 'mean' else 'max'} body dist > {args.fail_max_dist} m ({args.fail_mode}) "
          f"| start: {args.start}" + (f" (settle {args.settle_time}s)" if args.start == "default" else ""))

    agg = {k: Aggregate() for k in ("all", "success", "keep", "continue")}
    motions = []
    num_trials, num_success = 0, 0
    for mid in range(n_motions):
        name = env.motions.names[mid]
        succ = 0
        per_motion = {"all": {}, "continue": {}}
        for _ in range(args.episodes_per_motion):
            tr = run_trial(env, actor, mid, args.fail_max_dist, args.fail_mode, args.start, args.settle_time)
            var = trial_variants(tr["errors"], tr["fail_frame"], tr["num_frames"])
            agg["all"].add(var["all"]); agg["keep"].add(var["keep"]); agg["continue"].add(var["continue"])
            if not tr["failed"]:
                agg["success"].add(var["all"])
                succ += 1
            for v in ("all", "continue"):
                for k, (mean, _) in var[v].items():
                    if np.isfinite(mean):
                        e = per_motion[v].setdefault(k, [0.0, 0]); e[0] += mean; e[1] += 1
            num_trials += 1
        num_success += succ
        entry = {
            "motion_name": name,
            "success_count": succ,
            "trial_count": args.episodes_per_motion,
            "success_rate": succ / args.episodes_per_motion,
            "metrics": {v: OrderedDict((k, s / c) for k, (s, c) in m.items() if c > 0) for v, m in per_motion.items()},
        }
        motions.append(entry)
        a = entry["metrics"]["all"]
        print(f"  [{mid + 1:4d}/{n_motions}] {name[:70]:70s} succ {entry['success_rate']:.2f} "
              f"mpjpe_g {a.get('mpjpe_g', float('nan')):7.1f} mm  root_height {a.get('root_height_error', float('nan')):.3f}")
    env.force_motion_id = None

    summary = OrderedDict([
        ("policy", cfg.paths.student_pt),
        ("kind", kind),
        ("fail_criterion", {"mode": args.fail_mode, "max_dist": args.fail_max_dist}),
        ("start", {"mode": args.start, "settle_time": args.settle_time if args.start == "default" else 0.0}),
        ("success_rate", num_success / max(num_trials, 1)),
        ("num_motions", num_trials),
        ("all", agg["all"].result()),
        ("success", agg["success"].result()),
        ("keep", agg["keep"].result()),
        ("continue", agg["continue"].result()),
        ("motions", motions),
    ])
    print_metrics(summary)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
