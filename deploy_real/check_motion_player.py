#!/usr/bin/env python
"""Check LocalMotionPlayer against the motion server's build_mimic_obs.

For a few robot start headings, plays an exported .npz through LocalMotionPlayer and
compares every MOTION step (mimic target, future frames, reference root) with what
server_motion_lib.py would publish for the same anchor delta, plus the BLEND / RETURN
step counts and endpoints.

    python check_motion_player.py --npz X.npz [--yaws 0 1.2 -2.9]
"""
import argparse

import numpy as np
import torch

from pose.utils.motion_lib_pkl import MotionLib
from data_utils.params import DEFAULT_MIMIC_OBS
from motion_player import BLEND, IDLE, MOTION, RETURN, LocalMotionPlayer
from server_motion_lib import build_mimic_obs


def _err(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def check(npz, robot_yaw, motion_lib, device):
    player = LocalMotionPlayer(npz, DEFAULT_MIMIC_OBS["unitree_g1_with_hands"])
    m = player.meta
    player.start(robot_yaw)

    n = len(m["steps"])
    root_pos_0, root_rot_0, *_ = motion_lib.calc_motion_frame(
        torch.zeros(n, dtype=torch.long, device=device), torch.zeros(n, dtype=torch.float, device=device))
    common = dict(
        motion_lib=motion_lib, control_dt=m["control_dt"], robot_type=m["robot"],
        fix_root_pos=m["fix_root_pos"], fix_root_heading=m["fix_root_heading"],
        root_pos_ref=root_pos_0[0], root_rot_ref=root_rot_0[0],
        motion_yaw_anchor_delta=player.delta if m["align_heading"] else None,
        playback_speed=m["playback_speed"],
    )
    tar = torch.tensor(m["steps"], device=device, dtype=torch.long)
    fut = torch.tensor(m["future_motion_steps"], device=device, dtype=torch.long)
    T = len(m["future_motion_steps"])

    phases = []
    worst = dict(mimic=0.0, ref_root=0.0, f_pos=0.0, f_rot=0.0, f_dof=0.0, f_lin=0.0, f_ang=0.0)
    k = 0
    while True:
        s = player.step()
        phases.append(s.phase)
        if s.phase == MOTION:
            t = m["start_step"] + k
            mimic, root_pos, *_ = build_mimic_obs(t_step=t, tar_motion_steps=tar, **common)
            _, fp, fr, fd, fl, fa = build_mimic_obs(t_step=t, tar_motion_steps=fut, **common)
            mimic = np.asarray(mimic).reshape(-1)
            d = s.action_mimic - mimic
            yaw_cols = player._yaw_cols
            d[yaw_cols] = np.arctan2(np.sin(d[yaw_cols]), np.cos(d[yaw_cols]))
            worst["mimic"] = max(worst["mimic"], float(np.abs(d).max()))
            worst["ref_root"] = max(worst["ref_root"], _err(s.ref_root, np.asarray(root_pos).reshape(-1)[:3]))
            # quaternion sign is irrelevant
            ref_rot = np.asarray(fr).reshape(T, 4)
            sign = np.sign(np.sum(s.future_raw[1] * ref_rot, axis=-1, keepdims=True))
            for key, a, b in (("f_pos", s.future_raw[0], np.asarray(fp).reshape(T, 3)),
                              ("f_rot", s.future_raw[1] * sign, ref_rot),
                              ("f_dof", s.future_raw[2], np.asarray(fd).reshape(T, -1)),
                              ("f_lin", s.future_raw[3], np.asarray(fl).reshape(T, 3)),
                              ("f_ang", s.future_raw[4], np.asarray(fa).reshape(T, 3))):
                worst[key] = max(worst[key], _err(a, b))
            k += 1
        if player.done:
            break

    counts = {p: phases.count(p) for p in (BLEND, MOTION, RETURN, IDLE)}
    ok_counts = (counts[BLEND] == player.n_blend and counts[MOTION] == player.num_steps
                 and counts[RETURN] == player.n_return and counts[IDLE] == 1)
    end_ok = _err(s.action_mimic, player._return_target) < 1e-6
    # The server rotates the root quat / velocities in float32 and converts back to local
    # velocity + yaw, which alone moves those mimic fields by ~2e-4 versus delta=None; the
    # player adds delta to the yaw directly. Hence the looser mimic tolerance.
    ok = (ok_counts and end_ok and worst["mimic"] < 1e-3
          and max(v for k, v in worst.items() if k != "mimic") < 1e-4)
    print(f"robot_yaw={robot_yaw:+.3f} delta={player.delta:+.3f} phases={counts} "
          f"max_err={ {k: f'{v:.1e}' for k, v in worst.items()} } -> {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--yaws", type=float, nargs="+", default=[0.0, 1.2, -2.9, 3.1])
    args = parser.parse_args()

    meta = LocalMotionPlayer(args.npz, DEFAULT_MIMIC_OBS["unitree_g1_with_hands"]).meta
    device = "cuda" if torch.cuda.is_available() else "cpu"
    motion_lib = MotionLib(meta["motion_file"], device=device, motion_height_adjust=meta["motion_height_adjust"])
    results = [check(args.npz, y, motion_lib, device) for y in args.yaws]
    print("ALL OK" if all(results) else "SOME CHECKS FAILED")
    raise SystemExit(0 if all(results) else 1)
