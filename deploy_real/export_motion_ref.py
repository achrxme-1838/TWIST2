#!/usr/bin/env python
"""Precompute a reference motion for in-process playback.
Takes the same flags as server_motion_lib.py

    python export_motion_ref.py --motion_file X.pkl --obs_cfg ckpt.yaml --steps 5 --out X.npz
"""
import argparse
import json
import os

import numpy as np
import torch

from pose.utils.motion_lib_pkl import MotionLib
from data_utils.rot_utils import euler_from_quaternion_torch
from cfg import g1_29dof_cfg as cfg
from policy_spec import load_spec
from server_motion_lib import build_mimic_obs

EXPORT_VERSION = 1
CONTROL_DT = 0.02          # same as server_motion_lib.py
RETURN_TIME = 2.0          # server_motion_lib.py time_back_to_default
YAW_INDEX = 5              # yaw field in the 38-dim mimic frame


def export(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    motion_lib = MotionLib(args.motion_file, device=device, motion_height_adjust=args.motion_height_adjust)

    tar_motion_steps = [int(x.strip()) for x in args.steps.split(",")]
    tar_motion_steps_tensor = torch.tensor(tar_motion_steps, device=device, dtype=torch.long)
    if args.obs_cfg:
        future_motion_steps = load_spec(args.obs_cfg).future_motion_steps
    else:
        future_motion_steps = list(cfg.FUTURE_MOTION_STEPS)
    tar_future_steps_tensor = torch.tensor(future_motion_steps, device=device, dtype=torch.long)
    T = len(future_motion_steps)

    # Motion frame 0: root-alignment reference + yaw used for the start-heading anchor.
    n = len(tar_motion_steps)
    root_pos_0, root_rot_0, *_ = motion_lib.calc_motion_frame(
        torch.zeros(n, dtype=torch.long, device=device), torch.zeros(n, dtype=torch.float, device=device))
    root_pos_ref = root_pos_0[0].detach().clone()
    root_rot_ref = root_rot_0[0].detach().clone()
    _, _, yaw_0 = euler_from_quaternion_torch(root_rot_ref.unsqueeze(0), scalar_first=False)

    motion_length = motion_lib.get_motion_length(torch.tensor([0], device=device, dtype=torch.long))
    num_steps = int(motion_length / (CONTROL_DT * args.playback_speed))
    if args.max_step is not None:
        num_steps = min(num_steps, args.max_step)
    # The server publishes start_step first and stops once t_step >= num_steps.
    t_steps = range(args.start_step, max(num_steps, args.start_step + 1))

    common = dict(
        motion_lib=motion_lib, control_dt=CONTROL_DT, robot_type=args.robot,
        fix_root_pos=args.fix_root_pos, fix_root_heading=args.fix_root_heading,
        root_pos_ref=root_pos_ref, root_rot_ref=root_rot_ref,
        motion_yaw_anchor_delta=None, playback_speed=args.playback_speed,
    )
    out = {k: [] for k in ("mimic_obs", "ref_root", "future_root_pos", "future_root_rot",
                           "future_dof_pos", "future_root_lin_vel", "future_root_ang_vel")}
    for t in t_steps:
        mimic_obs, root_pos, *_ = build_mimic_obs(t_step=t, tar_motion_steps=tar_motion_steps_tensor, **common)
        _, fr_pos, fr_rot, fr_dof, fr_lin, fr_ang = build_mimic_obs(
            t_step=t, tar_motion_steps=tar_future_steps_tensor, **common)
        out["mimic_obs"].append(np.asarray(mimic_obs).reshape(-1))
        out["ref_root"].append(np.asarray(root_pos).reshape(-1)[:3])
        out["future_root_pos"].append(np.asarray(fr_pos).reshape(T, 3))
        out["future_root_rot"].append(np.asarray(fr_rot).reshape(T, 4))
        out["future_dof_pos"].append(np.asarray(fr_dof).reshape(T, -1))
        out["future_root_lin_vel"].append(np.asarray(fr_lin).reshape(T, 3))
        out["future_root_ang_vel"].append(np.asarray(fr_ang).reshape(T, 3))
    arrays = {k: np.stack(v).astype(np.float32) for k, v in out.items()}

    meta = dict(
        version=EXPORT_VERSION,
        motion_file=os.path.abspath(args.motion_file),
        robot=args.robot,
        control_dt=CONTROL_DT,
        playback_speed=args.playback_speed,
        start_step=args.start_step,
        num_steps=len(t_steps),
        steps=tar_motion_steps,
        future_motion_steps=list(future_motion_steps),
        frame_dim=arrays["mimic_obs"].shape[1] // len(tar_motion_steps),
        yaw_index=YAW_INDEX,
        motion_yaw_0=float(yaw_0.item()),
        fix_root_pos=bool(args.fix_root_pos),
        fix_root_heading=bool(args.fix_root_heading),
        motion_height_adjust=bool(args.motion_height_adjust),
        align_heading=bool(args.align_motion_start_to_robot_heading),
        # playback defaults (LocalMotionPlayer may override)
        blend_in_time=args.blend_in_time,
        return_time=RETURN_TIME,
        send_start_frame_as_end_frame=bool(args.send_start_frame_as_end_frame),
        obs_cfg=os.path.abspath(args.obs_cfg) if args.obs_cfg else None,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, meta=np.array(json.dumps(meta)), **arrays)
    print(f"[export] {len(t_steps)} steps ({len(t_steps) * CONTROL_DT:.2f}s), "
          f"future steps {future_motion_steps}, mimic dim {arrays['mimic_obs'].shape[1]} -> {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion_file", required=True, help="*.pkl motion file for MotionLib")
    parser.add_argument("--out", required=True, help="Output .npz")
    parser.add_argument("--obs_cfg", type=str, default=None,
                        help="Policy spec yaml of the controller's checkpoint (future-frame schedule).")
    parser.add_argument("--robot", type=str, default="unitree_g1_with_hands",
                        choices=["unitree_g1", "unitree_g1_with_hands"])
    parser.add_argument("--steps", type=str, default="1", help="Comma-separated mimic target steps")
    parser.add_argument("--playback_speed", type=float, default=1.0)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--max_step", type=int, default=None)
    parser.add_argument("--fix_root_pos", action="store_true")
    parser.add_argument("--fix_root_heading", action="store_true")
    parser.add_argument("--motion_height_adjust", action="store_true")
    parser.add_argument("--align_motion_start_to_robot_heading", action="store_true", default=True)
    parser.add_argument("--blend_in_time", type=float, default=3.0)
    parser.add_argument("--send_start_frame_as_end_frame", action="store_true")
    # accepted for flag compatibility with server_motion_lib.py; no effect here
    parser.add_argument("--redis_ip", type=str, default=None)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--use_remote_control", action="store_true")
    export(parser.parse_args())
