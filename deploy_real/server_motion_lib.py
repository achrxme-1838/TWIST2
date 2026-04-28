#!/usr/bin/env python
import argparse
import time
import redis
import json
from typing import Optional
import numpy as np
import isaacgym
import torch
from rich import print
import os
import mujoco
from mujoco.viewer import launch_passive
import matplotlib.pyplot as plt
from pose.utils.motion_lib_pkl import MotionLib
from data_utils.rot_utils import euler_from_quaternion_torch, quat_rotate_inverse_torch

from data_utils.params import DEFAULT_MIMIC_OBS


def build_mimic_obs(
    motion_lib: MotionLib,
    t_step: int,
    control_dt: float,
    tar_motion_steps,
    robot_type: str = "g1",
    mask_indicator: bool = False,
    fix_root_pos: bool = False,
    fix_root_heading: bool = False,
    root_pos_ref: torch.Tensor = None,
    root_rot_ref: torch.Tensor = None,
    motion_yaw_anchor_delta: Optional[float] = None,
):
    """
    Build the mimic_obs at time-step t_step, referencing the code in MimicRunner.

    Output layout (38-dim, ``mask_indicator=False``):
        [xy_vel_local(2), z(1), roll(1), pitch(1), yaw(1), ang_vel_local(3), dof_pos(29)]

    The ``yaw`` field is the motion's root yaw in the published frame. When
    ``motion_yaw_anchor_delta`` is provided, every frame is rotated by that
    constant angle around the world z-axis -- this is how the caller anchors
    motion frame 0's yaw to the robot's heading at playback start.

    Matches DEX_RL_LAB ``upcoming_twist_mimic_target``.
    """
    device = torch.device("cuda")
    # Build times
    motion_times = torch.tensor([t_step * control_dt], device=device).unsqueeze(-1)
    obs_motion_times = tar_motion_steps * control_dt + motion_times
    obs_motion_times = obs_motion_times.flatten()

    # Suppose we only have a single motion in the .pkl
    motion_ids = torch.zeros(len(tar_motion_steps), dtype=torch.long, device=device)

    # Retrieve motion frames
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos, root_pos_delta_local, root_rot_delta_local = motion_lib.calc_motion_frame(motion_ids, obs_motion_times)

    # Align root heading / horizontal position to the reference (motion frame 0), assuming the
    # robot's root (horizontal) and heading always match the reference motion. root_rot/root_vel/
    # root_ang_vel are rotated together, so local-frame obs (root_vel_local, roll/pitch,
    # root_ang_vel_local) remain invariant; only the world-frame outputs used for viz change.
    if fix_root_heading and root_rot_ref is not None:
        _, _, yaw_cur = euler_from_quaternion_torch(root_rot, scalar_first=False)
        _, _, yaw_ref = euler_from_quaternion_torch(root_rot_ref.unsqueeze(0), scalar_first=False)
        delta_yaw = yaw_ref - yaw_cur  # [N]

        # Rotate quaternion around world z (scalar-last: x, y, z, w)
        half = delta_yaw * 0.5
        cw = torch.cos(half)
        sz = torch.sin(half)
        x, y, z, w = root_rot[..., 0], root_rot[..., 1], root_rot[..., 2], root_rot[..., 3]
        root_rot = torch.stack([cw * x - sz * y, sz * x + cw * y, cw * z + sz * w, cw * w - sz * z], dim=-1)

        # Rotate world-frame linear / angular velocities around z by delta_yaw
        cos_d = torch.cos(delta_yaw)
        sin_d = torch.sin(delta_yaw)
        vx, vy, vz = root_vel[..., 0], root_vel[..., 1], root_vel[..., 2]
        root_vel = torch.stack([cos_d * vx - sin_d * vy, sin_d * vx + cos_d * vy, vz], dim=-1)
        ax, ay, az = root_ang_vel[..., 0], root_ang_vel[..., 1], root_ang_vel[..., 2]
        root_ang_vel = torch.stack([cos_d * ax - sin_d * ay, sin_d * ax + cos_d * ay, az], dim=-1)

    # Anchor motion start to robot heading: rotate every frame by a CONSTANT
    # delta_yaw around world z. This preserves the motion's yaw progression
    # (unlike fix_root_heading, which freezes yaw) and just rotates the entire
    # trajectory so that motion frame 0's yaw matches robot's start yaw.
    if motion_yaw_anchor_delta is not None:
        delta = float(motion_yaw_anchor_delta)
        half = 0.5 * delta
        cw = float(np.cos(half))
        sz = float(np.sin(half))
        x, y, z, w = root_rot[..., 0], root_rot[..., 1], root_rot[..., 2], root_rot[..., 3]
        root_rot = torch.stack([cw * x - sz * y, sz * x + cw * y, cw * z + sz * w, cw * w - sz * z], dim=-1)

        cos_d = float(np.cos(delta))
        sin_d = float(np.sin(delta))
        vx, vy, vz = root_vel[..., 0], root_vel[..., 1], root_vel[..., 2]
        root_vel = torch.stack([cos_d * vx - sin_d * vy, sin_d * vx + cos_d * vy, vz], dim=-1)
        ax, ay, az = root_ang_vel[..., 0], root_ang_vel[..., 1], root_ang_vel[..., 2]
        root_ang_vel = torch.stack([cos_d * ax - sin_d * ay, sin_d * ax + cos_d * ay, az], dim=-1)
        # Rotate root_pos xy too so the world-frame viz stays consistent (z invariant under z-rot).
        px, py, pz = root_pos[..., 0], root_pos[..., 1], root_pos[..., 2]
        root_pos = torch.stack([cos_d * px - sin_d * py, sin_d * px + cos_d * py, pz], dim=-1)

    if fix_root_pos and root_pos_ref is not None:
        root_pos = torch.stack([
            torch.full_like(root_pos[..., 0], float(root_pos_ref[0])),
            torch.full_like(root_pos[..., 1], float(root_pos_ref[1])),
            root_pos[..., 2],
        ], dim=-1)

    # Convert to euler (roll, pitch, yaw)
    roll, pitch, yaw = euler_from_quaternion_torch(root_rot, scalar_first=False)
    roll = roll.reshape(1, -1, 1)
    pitch = pitch.reshape(1, -1, 1)
    yaw = yaw.reshape(1, -1, 1)

    # Transform velocities to root frame
    root_vel_local = quat_rotate_inverse_torch(root_rot, root_vel, scalar_first=False).reshape(1, -1, 3)
    root_ang_vel_local = quat_rotate_inverse_torch(root_rot, root_ang_vel, scalar_first=False).reshape(1, -1, 3)
    root_vel = root_vel.reshape(1, -1, 3)
    root_ang_vel = root_ang_vel.reshape(1, -1, 3)

    root_pos = root_pos.reshape(1, -1, 3)
    dof_pos = dof_pos.reshape(1, -1, dof_pos.shape[-1])

    # 38-dim layout: xy_vel(2) + z(1) + roll/pitch(2) + yaw(1) + ang_vel_local(3) + dof(29).
    if mask_indicator:
        mimic_obs_buf = torch.cat((
                    root_vel_local[..., :2],          # 2: xy linear vel (root-local)
                    root_pos[..., 2:3],               # 1: z
                    roll, pitch,                      # 2: roll, pitch
                    yaw,                              # 1: yaw (in published frame)
                    root_ang_vel_local[..., :],       # 3: full angular velocity (root-local)
                    dof_pos,                          # 29
                ), dim=-1)[:, :]  # shape (1, 1, 9 + num_dof)
        # append mask indicator 1
        mask_indicator = torch.ones(1, mimic_obs_buf.shape[1], 1).to(device)
        mimic_obs_buf = torch.cat((mimic_obs_buf, mask_indicator), dim=-1)
    else:
        mimic_obs_buf = torch.cat((
                    root_vel_local[..., :2],          # 2
                    root_pos[..., 2:3],               # 1
                    roll, pitch,                      # 2
                    yaw,                              # 1
                    root_ang_vel_local[..., :],       # 3
                    dof_pos,                          # 29
                ), dim=-1)[:, :]  # shape (1, 1, 9 + num_dof)

    # print("root height: ", root_pos[..., 2:3].detach().cpu().numpy().squeeze())
    mimic_obs_buf = mimic_obs_buf.reshape(1, -1)
    
    return mimic_obs_buf.detach().cpu().numpy().squeeze(), root_pos.detach().cpu().numpy().squeeze(), \
        root_rot.detach().cpu().numpy().squeeze(), dof_pos.detach().cpu().numpy().squeeze(), \
            root_vel.detach().cpu().numpy().squeeze(), root_ang_vel.detach().cpu().numpy().squeeze()


def main(args, xml_file, robot_base):
    # Remote control state  
    motion_started = False if args.use_remote_control else True
    
    if args.use_remote_control:
        print("[Motion Server] Remote control enabled. Waiting for start signal from robot controller...")

    if args.vis:
        sim_model = mujoco.MjModel.from_xml_path(xml_file)
        sim_data = mujoco.MjData(sim_model)
        viewer = launch_passive(model=sim_model, data=sim_data, show_left_ui=False, show_right_ui=False)
            
    # 1. Connect to Redis
    redis_ip = args.redis_ip
    # redis_client = redis.Redis(host="localhost", port=6379, db=0)
    # redis_client = redis.Redis(host="127.0.0.1", port=6379, db=0)
    # redis_client = redis.Redis(host="192.168.110.24", port=6379, db=0)
    redis_client = redis.Redis(host=redis_ip, port=6379, db=0)
    redis_client.ping()


    # 2. Load motion library
    device = "cuda" if torch.cuda.is_available() else "cpu"
    motion_lib = MotionLib(args.motion_file, device=device)
    
    # 3. Prepare the steps array
    tar_motion_steps = [int(x.strip()) for x in args.steps.split(",")]
    tar_motion_steps_tensor = torch.tensor(tar_motion_steps, device=device, dtype=torch.long)

    # 4. Loop over time steps and publish mimic obs
    control_dt = 0.02

    # 4.1 Cache motion frame 0 as the root-alignment reference for fix_root_pos/fix_root_heading.
    # Assumes the robot's root (horizontal) and heading always match the reference motion.
    root_pos_ref = None
    root_rot_ref = None
    motion_yaw_0 = None
    if args.fix_root_pos or args.fix_root_heading or args.align_motion_start_to_robot_heading:
        ref_motion_ids = torch.zeros(len(tar_motion_steps), dtype=torch.long, device=device)
        ref_times = torch.zeros(len(tar_motion_steps), dtype=torch.float, device=device)
        root_pos_0, root_rot_0, *_ = motion_lib.calc_motion_frame(ref_motion_ids, ref_times)
        root_pos_ref = root_pos_0[0].detach().clone()
        root_rot_ref = root_rot_0[0].detach().clone()
        # Cache motion frame 0's yaw so we can compute a one-time anchor delta
        # to robot's heading at playback start.
        _, _, _yaw_0 = euler_from_quaternion_torch(root_rot_ref.unsqueeze(0), scalar_first=False)
        motion_yaw_0 = float(_yaw_0.item())

    # motion_yaw_anchor_delta is set ONCE at motion start (constant for the entire
    # playback). When set, build_mimic_obs rotates every frame by this delta so
    # motion frame 0's yaw aligns with the robot's heading at playback start.
    motion_yaw_anchor_delta = None

    def _read_robot_heading_from_redis(timeout_s: float = 2.0) -> Optional[float]:
        """Poll Redis for the robot's current planar yaw. Returns None on timeout."""
        key = f"state_heading_{args.robot}"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            raw = redis_client.get(key)
            if raw is not None:
                try:
                    return float(json.loads(raw))
                except Exception:
                    return float(raw)
            time.sleep(0.05)
        return None

    def _compute_anchor_delta() -> Optional[float]:
        if not args.align_motion_start_to_robot_heading:
            return None
        if motion_yaw_0 is None:
            print("[Motion Server] align_motion_start_to_robot_heading set but motion_yaw_0 unavailable; skipping anchor.")
            return None
        robot_yaw = _read_robot_heading_from_redis()
        if robot_yaw is None:
            print(f"[Motion Server] No state_heading_{args.robot} on Redis; defaulting anchor delta to 0.")
            return 0.0
        delta = robot_yaw - motion_yaw_0
        print(f"[Motion Server] Anchored motion start yaw to robot heading "
              f"(robot_yaw={robot_yaw:.4f}, motion_yaw_0={motion_yaw_0:.4f}, delta={delta:.4f}).")
        return delta

    # If motion plays immediately (no remote control), anchor right now.
    if not args.use_remote_control:
        motion_yaw_anchor_delta = _compute_anchor_delta()

    # 4.5 Extract start frame for end frame if option is enabled.
    # NOTE: start_frame_mimic_obs uses the CURRENT anchor delta (None until motion
    # actually starts under remote control); recomputed after the start signal
    # so the idle frame matches the streamed frames once playback begins.
    start_frame_mimic_obs = None
    if args.send_start_frame_as_end_frame:
        start_frame_mimic_obs, _, _, _, _, _ = build_mimic_obs(
            motion_lib=motion_lib,
            t_step=0,
            control_dt=control_dt,
            tar_motion_steps=tar_motion_steps_tensor,
            robot_type=args.robot,
            fix_root_pos=args.fix_root_pos,
            fix_root_heading=args.fix_root_heading,
            root_pos_ref=root_pos_ref,
            root_rot_ref=root_rot_ref,
            motion_yaw_anchor_delta=motion_yaw_anchor_delta,
        )
    # compute num_steps based on motion length
    motion_id = torch.tensor([0], device=device, dtype=torch.long)
    motion_length = motion_lib.get_motion_length(motion_id)
    num_steps = int(motion_length / control_dt)
    
    print(f"[Motion Server] Streaming for {num_steps} steps at dt={control_dt:.3f} seconds...")

    last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]
    
    # Helper function to check remote control signals
    def check_remote_control_signals():
        if not args.use_remote_control:
            return True, False  # motion_active, should_exit
        
        try:
            # Check for start signal (B button from robot controller)
            start_signal = redis_client.get("motion_start_signal")
            start_pressed = start_signal == b"1" if start_signal else False
            
            # Check for exit signal (Select button from robot controller)
            exit_signal = redis_client.get("motion_exit_signal") 
            exit_pressed = exit_signal == b"1" if exit_signal else False
            
            return start_pressed, exit_pressed
        except Exception as e:
            return False, False
    
    if args.use_remote_control:
        # reset start and exit signal to 0
        redis_client.set("motion_start_signal", "0")
        redis_client.set("motion_exit_signal", "0")
    
    try:
        # for t_step in range(num_steps):
        t_step = 0
        while True:
            t0 = time.time()
            
            # Handle remote control logic
            if args.use_remote_control:
                # Check remote control signals
                start_pressed, exit_pressed = check_remote_control_signals()

                if exit_pressed:
                    print("[Motion Server] Exit signal received, stopping...")
                    break
                    
                if not motion_started and start_pressed:
                    print("[Motion Server] Start signal received, beginning motion...")
                    motion_started = True
                    # Anchor motion frame 0's yaw to robot's current heading (read once at start).
                    motion_yaw_anchor_delta = _compute_anchor_delta()
                    if args.send_start_frame_as_end_frame:
                        start_frame_mimic_obs, _, _, _, _, _ = build_mimic_obs(
                            motion_lib=motion_lib,
                            t_step=0,
                            control_dt=control_dt,
                            tar_motion_steps=tar_motion_steps_tensor,
                            robot_type=args.robot,
                            fix_root_pos=args.fix_root_pos,
                            fix_root_heading=args.fix_root_heading,
                            root_pos_ref=root_pos_ref,
                            root_rot_ref=root_rot_ref,
                            motion_yaw_anchor_delta=motion_yaw_anchor_delta,
                        )
                elif not motion_started:
                    # Keep sending default pose while waiting for start signal
                    idle_mimic_obs = start_frame_mimic_obs if args.send_start_frame_as_end_frame and start_frame_mimic_obs is not None else DEFAULT_MIMIC_OBS[args.robot]
                    redis_client.set(f"action_body_{args.robot}", json.dumps(idle_mimic_obs.tolist()))
                    redis_client.set(f"action_hand_left_{args.robot}", json.dumps(np.zeros(7).tolist()))
                    redis_client.set(f"action_hand_right_{args.robot}", json.dumps(np.zeros(7).tolist()))

                    # Sleep and continue to next iteration
                    elapsed = time.time() - t0
                    if elapsed < control_dt:
                        time.sleep(control_dt - elapsed)
                    continue

            # Build a mimic obs from the motion library
            mimic_obs, root_pos, root_rot, dof_pos, root_vel, root_ang_vel = build_mimic_obs(
                motion_lib=motion_lib,
                t_step=t_step,
                control_dt=control_dt,
                tar_motion_steps=tar_motion_steps_tensor,
                robot_type=args.robot,
                fix_root_pos=args.fix_root_pos,
                fix_root_heading=args.fix_root_heading,
                root_pos_ref=root_pos_ref,
                root_rot_ref=root_rot_ref,
                motion_yaw_anchor_delta=motion_yaw_anchor_delta,
            )
            
            # Convert to JSON (list) to put into Redis
            mimic_obs_list = mimic_obs.tolist() if mimic_obs.ndim == 1 else mimic_obs.flatten().tolist()
            redis_client.set(f"action_body_{args.robot}", json.dumps(mimic_obs_list))
            redis_client.set(f"action_hand_left_{args.robot}", json.dumps(np.zeros(7).tolist()))
            redis_client.set(f"action_hand_right_{args.robot}", json.dumps(np.zeros(7).tolist()))
            redis_client.set(f"action_neck_{args.robot}", json.dumps(np.zeros(2).tolist()))
            last_mimic_obs = mimic_obs
            
            # Print or log it
            print(f"Step {t_step:4d} => mimic_obs shape = {mimic_obs.shape} published...", end="\r")

            if args.vis:
                sim_data.qpos[:3] = root_pos
                # filp rot
                # root_rot = root_rot[[1,2,3,0]]
                root_rot = root_rot[[3,0,1,2]]
                sim_data.qpos[3:7] = root_rot
                sim_data.qpos[7:] = dof_pos
                mujoco.mj_forward(sim_model, sim_data)
                robot_base_pos = sim_data.xpos[sim_model.body(robot_base).id]
                viewer.cam.lookat = robot_base_pos
                # set distance to pelvis
                viewer.cam.distance = 2.0
                viewer.sync()
            
            t_step += 1
            if t_step >= num_steps:
                break
            # Sleep to maintain real-time pace
            elapsed = time.time() - t0
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
    
      
    except Exception as e:
        print(f"[Motion Server] Error: {e}")
        print("[Motion Server] Keyboard interrupt. Interpolating to default mimic_obs...")
        # do linear interpolation to the last mimic_obs
        time_back_to_default = 2.0
        target_mimic_obs = start_frame_mimic_obs if args.send_start_frame_as_end_frame and start_frame_mimic_obs is not None else DEFAULT_MIMIC_OBS[args.robot]
        for i in range(int(time_back_to_default / control_dt)):
            interp_mimic_obs = last_mimic_obs + (target_mimic_obs - last_mimic_obs) * (i / (time_back_to_default / control_dt))
            redis_client.set(f"action_body_{args.robot}", json.dumps(interp_mimic_obs.tolist()))
            time.sleep(control_dt)
        redis_client.set(f"action_body_{args.robot}", json.dumps(target_mimic_obs.tolist()))
        last_mimic_obs = target_mimic_obs
        viewer.close()
        time.sleep(0.5)
        exit()
    finally:
        print("[Motion Server] Exiting...Interpolating to default mimic_obs...")
        # do linear interpolation to the last mimic_obs
        time_back_to_default = 2.0
        target_mimic_obs = start_frame_mimic_obs if args.send_start_frame_as_end_frame and start_frame_mimic_obs is not None else DEFAULT_MIMIC_OBS[args.robot]
        for i in range(int(time_back_to_default / control_dt)):
            interp_mimic_obs = last_mimic_obs + (target_mimic_obs - last_mimic_obs) * (i / (time_back_to_default / control_dt))
            redis_client.set(f"action_body_{args.robot}", json.dumps(interp_mimic_obs.tolist()))
            time.sleep(control_dt)
        redis_client.set(f"action_body_{args.robot}", json.dumps(target_mimic_obs.tolist()))
        last_mimic_obs = target_mimic_obs
        viewer.close()
        time.sleep(0.5)
        exit()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", help="Path to your *.pkl motion file for MotionLib", 
                        default="../motion_data/OMOMO_g1_GMR/sub1_clothesstand_067.pkl"
                        )
    parser.add_argument("--robot", type=str, default="unitree_g1_with_hands", choices=["unitree_g1", "unitree_g1_with_hands"])
    parser.add_argument("--steps", type=str,
                        # default="1,3,5,10,15,20,30,40,50",
                        default="1",
                        help="Comma-separated steps for future frames (tar_motion_steps)")
    parser.add_argument("--vis", action="store_true", help="Visualize the motion")
    parser.add_argument("--use_remote_control", action="store_true", help="Use remote control signals from robot controller")
    parser.add_argument("--send_start_frame_as_end_frame", action="store_true", help="Use motion's first frame as end frame instead of default pose")
    parser.add_argument("--redis_ip", type=str, default="localhost", help="Redis IP")
    parser.add_argument("--fix_root_pos", action="store_true",
                        help="Fix the motion's root horizontal (xy) position to the frame-0 reference. "
                             "Assumes the robot's horizontal root always matches the reference motion.")
    parser.add_argument("--fix_root_heading", action="store_true",
                        help="Fix the motion's root heading (yaw) to the frame-0 reference. "
                             "Assumes the robot's heading always matches the reference motion.")
    parser.add_argument("--align_motion_start_to_robot_heading", action="store_true",
                        default=True,
                        help="At motion start, read the robot's planar yaw from "
                             "Redis (key: state_heading_<robot>) and rotate the entire "
                             "motion by a CONSTANT delta so motion frame 0's yaw matches "
                             "robot's heading. Preserves the motion's yaw progression "
                             "(unlike --fix_root_heading which freezes yaw).")
    args = parser.parse_args()

    args.vis = True
    

    print("Robot type: ", args.robot)
    print("Motion file: ", args.motion_file)
    print("Steps: ", args.steps)
    
    HERE = os.path.dirname(os.path.abspath(__file__))
    
    if args.robot == "unitree_g1" or args.robot == "unitree_g1_with_hands":
        xml_file = f"{HERE}/../assets/g1/g1_mocap_29dof.xml"
        robot_base = "pelvis"
    else:
        raise ValueError(f"robot type {args.robot} not supported")
    
    
    main(args, xml_file, robot_base)
