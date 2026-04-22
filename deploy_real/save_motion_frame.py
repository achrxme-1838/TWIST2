#!/usr/bin/env python
"""Compute one mimic_obs frame from a motion pkl and save to JSON so it can be
replayed (published to Redis) later without running the full motion server.

Usage:
    python save_motion_frame.py \\
        --motion_file ../assets/example_motions/A1-Stand_poses.pkl \\
        --t_step 20 \\
        --out ./saved_motion_frames/A1-Stand_frame20.json
"""
import argparse
import json
import os

import isaacgym  # noqa: F401  (required before torch for cuda coexistence)
import torch

from pose.utils.motion_lib_pkl import MotionLib
from server_motion_lib import build_mimic_obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", type=str,
                        default="../assets/example_motions/A1-Stand_poses.pkl")
    parser.add_argument("--t_step", type=int, default=20,
                        help="Motion step to snapshot (0.02s per step).")
    parser.add_argument("--steps", type=str, default="1",
                        help="Comma-separated tar_motion_steps (matches motion server).")
    parser.add_argument("--control_dt", type=float, default=0.02)
    parser.add_argument("--robot", type=str, default="unitree_g1_with_hands")
    parser.add_argument("--fix_root_pos", action="store_true", default=True)
    parser.add_argument("--fix_root_heading", action="store_true", default=True)
    parser.add_argument("--out", type=str,
                        default="./saved_motion_frames/A1-Stand_frame20.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    motion_lib = MotionLib(args.motion_file, device=device)
    tar_motion_steps = [int(x.strip()) for x in args.steps.split(",")]
    tar_motion_steps_tensor = torch.tensor(tar_motion_steps, device=device, dtype=torch.long)

    # Reference at motion frame 0 for fix_root_pos / fix_root_heading.
    root_pos_ref = None
    root_rot_ref = None
    if args.fix_root_pos or args.fix_root_heading:
        ref_motion_ids = torch.zeros(len(tar_motion_steps), dtype=torch.long, device=device)
        ref_times = torch.zeros(len(tar_motion_steps), dtype=torch.float, device=device)
        root_pos_0, root_rot_0, *_ = motion_lib.calc_motion_frame(ref_motion_ids, ref_times)
        root_pos_ref = root_pos_0[0].detach().clone()
        root_rot_ref = root_rot_0[0].detach().clone()

    mimic_obs, root_pos, root_rot, dof_pos, root_vel, root_ang_vel = build_mimic_obs(
        motion_lib=motion_lib,
        t_step=args.t_step,
        control_dt=args.control_dt,
        tar_motion_steps=tar_motion_steps_tensor,
        robot_type=args.robot,
        fix_root_pos=args.fix_root_pos,
        fix_root_heading=args.fix_root_heading,
        root_pos_ref=root_pos_ref,
        root_rot_ref=root_rot_ref,
    )

    mimic_obs_list = mimic_obs.tolist() if mimic_obs.ndim == 1 else mimic_obs.flatten().tolist()
    payload = {
        "meta": {
            "motion_file": os.path.abspath(args.motion_file),
            "t_step": args.t_step,
            "control_dt": args.control_dt,
            "tar_motion_steps": tar_motion_steps,
            "robot": args.robot,
            "fix_root_pos": bool(args.fix_root_pos),
            "fix_root_heading": bool(args.fix_root_heading),
            "motion_time_s": args.t_step * args.control_dt,
            "mimic_obs_dim": len(mimic_obs_list),
            "root_pos": [float(v) for v in (root_pos.tolist() if hasattr(root_pos, "tolist") else list(root_pos))],
            "root_rot": [float(v) for v in (root_rot.tolist() if hasattr(root_rot, "tolist") else list(root_rot))],
        },
        f"action_body_{args.robot}": mimic_obs_list,
        f"action_hand_left_{args.robot}": [0.0] * 7,
        f"action_hand_right_{args.robot}": [0.0] * 7,
        f"action_neck_{args.robot}": [0.0] * 2,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {len(mimic_obs_list)}-dim mimic_obs to {os.path.abspath(args.out)}")
    print(f"  motion_time = {args.t_step * args.control_dt:.3f}s, robot = {args.robot}")


if __name__ == "__main__":
    main()
