#!/usr/bin/env python3
"""TWIST2 real-robot controller.

Mirrors server_low_level_g1_sim.py: same term-major PolicyCfg observation layout,
same SDK<->Isaac permutation, same ONNX policy interface. Only the I/O surface
differs — robot state comes from G1RealWorldEnv (IMU + motor state) and PD runs
on the robot via env.send_robot_action instead of locally inside MuJoCo.

If --use_diff_body_pos / --use_diff_body_tannorm is set, an auxiliary MjModel
is loaded and driven by the robot's measured (root_quat, joint_pos) each tick
so we can run mj_kinematics for the FK-based diff terms.
"""
import argparse
import json
import os
import time
from collections import deque

import numpy as np
import redis
import torch
from rich import print

from data_utils.rot_utils import quatToEuler
from data_utils.params import DEFAULT_MIMIC_OBS

from cfg import g1_29dof_cfg as cfg
from observations import (
    compute_diff_body_pos_b,
    compute_diff_body_tannorm_b,
)
from utils.math import yaw_from_quat

from robot_control.g1_wrapper import G1RealWorldEnv
from robot_control.config import Config
from robot_control.dex_hand_wrapper import Dex3_1_Controller

try:
    import mujoco
except ImportError:
    mujoco = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class OnnxPolicyWrapper:
    """Minimal wrapper so ONNXRuntime policies mimic TorchScript call signature."""

    def __init__(self, session, input_name, output_index=0):
        self.session = session
        self.input_name = input_name
        self.output_index = output_index

    def __call__(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        result = outputs[self.output_index]
        if not isinstance(result, np.ndarray):
            result = np.asarray(result, dtype=np.float32)
        return torch.from_numpy(result.astype(np.float32))


class EMASmoother:
    """Exponential Moving Average smoother for body actions (real-only safety)."""

    def __init__(self, alpha=0.1, initial_value=None):
        self.alpha = alpha
        self.initialized = False
        self.smoothed_value = initial_value

    def smooth(self, new_value):
        if not self.initialized:
            self.smoothed_value = new_value.copy() if hasattr(new_value, 'copy') else new_value
            self.initialized = True
            return self.smoothed_value
        self.smoothed_value = self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
        return self.smoothed_value

    def reset(self):
        self.initialized = False
        self.smoothed_value = None


def load_onnx_policy(policy_path: str, device: str) -> OnnxPolicyWrapper:
    if ort is None:
        raise ImportError("onnxruntime is required for ONNX policy inference but is not installed.")
    providers = []
    available = ort.get_available_providers()
    if device.startswith('cuda'):
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        else:
            print("CUDAExecutionProvider not available in onnxruntime; falling back to CPUExecutionProvider.")
    providers.append('CPUExecutionProvider')
    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"ONNX policy loaded from {policy_path} using providers: {session.get_providers()}")
    return OnnxPolicyWrapper(session, input_name)


class RealTimePolicyController(object):
    def __init__(self,
                 policy_path,
                 config_path,
                 device='cuda',
                 net='eno1',
                 use_hand=False,
                 record_proprio=False,
                 smooth_body=0.0,
                 xml_file=None,
                 use_diff_body_pos=False,
                 use_diff_body_tannorm=False,
                 ):
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_pipeline = self.redis_client.pipeline()
        except Exception as e:
            print(f"Error connecting to Redis: {e}")
            exit()

        self.config = Config(config_path)
        # Policy was trained against cfg.DEFAULT_DOF_POS / STIFFNESS / DAMPING —
        # override the YAML so move_to_default_pos, the hardware PD loop, and
        # the obs offsets all match the sim2sim setup.
        self.config.default_angles = cfg.DEFAULT_DOF_POS.astype(np.float32).copy()
        self.config.kps = cfg.STIFFNESS.astype(np.float32).tolist()
        self.config.kds = cfg.DAMPING.astype(np.float32).tolist()

        self.env = G1RealWorldEnv(net=net, config=self.config)
        self.use_hand = use_hand
        if use_hand:
            self.hand_ctrl = Dex3_1_Controller(net, re_init=False)

        self.device = device
        self.policy = load_onnx_policy(policy_path, device)

        self.num_actions = cfg.NUM_ACTIONS

        # Robot params (SDK order). Hardware PD uses YAML kps/kds via env.
        self.default_dof_pos = cfg.DEFAULT_DOF_POS.copy()
        self.action_scale = np.full(self.num_actions, cfg.ACTION_SCALE, dtype=np.float32)

        # Joint-order permutation between SDK (robot/MuJoCo) and Isaac (policy I/O).
        self.sdk_to_isaac = np.array(
            [cfg.SDK_JOINT_NAMES.index(n) for n in cfg.ISAAC_JOINT_NAMES], dtype=np.int64
        )
        self.isaac_to_sdk = np.array(
            [cfg.ISAAC_JOINT_NAMES.index(n) for n in cfg.SDK_JOINT_NAMES], dtype=np.int64
        )
        self.default_dof_pos_isaac = self.default_dof_pos[self.sdk_to_isaac]
        self.ankle_idx_isaac = [cfg.ISAAC_JOINT_NAMES.index(n) for n in cfg.ANKLE_JOINT_NAMES]

        # ----- observation layout (matches DEX_RL_LAB PolicyCfg, term-major) -----
        self.history_len = cfg.HISTORY_LEN
        self._hist_dims = cfg.HIST_TERM_DIMS
        self._hist_term_order = cfg.HIST_TERM_ORDER
        self._mimic_dim = cfg.MIMIC_DIM

        self.use_diff_body_pos = use_diff_body_pos
        self._diff_body_pos_per_step = cfg.NUM_TRACKED_BODIES * 3
        self._diff_body_pos_total = (
            self.history_len * self._diff_body_pos_per_step if use_diff_body_pos else 0
        )
        self.use_diff_body_tannorm = use_diff_body_tannorm
        self._diff_body_tannorm_per_step = cfg.NUM_TRACKED_BODIES * 6
        self._diff_body_tannorm_total = (
            self.history_len * self._diff_body_tannorm_per_step if use_diff_body_tannorm else 0
        )

        self.total_obs_size = (
            self.history_len * sum(self._hist_dims.values())
            + self._mimic_dim
            + self._diff_body_pos_total
            + self._diff_body_tannorm_total
        )

        # MuJoCo model for FK-based diff terms (no viewer, no stepping).
        self.fk_model = None
        self.fk_data = None
        self.ref_data = None
        self.tracked_body_ids = None
        self.extended_parent_ids = None
        self.extended_local_offsets = None
        if use_diff_body_pos or use_diff_body_tannorm:
            if mujoco is None:
                raise ImportError("mujoco is required when --use_diff_body_pos / --use_diff_body_tannorm is set.")
            if xml_file is None:
                raise ValueError("xml_file must be provided when diff_body_* terms are enabled.")
            self.fk_model = mujoco.MjModel.from_xml_path(xml_file)
            self.fk_data = mujoco.MjData(self.fk_model)
            self.ref_data = mujoco.MjData(self.fk_model)
            self.tracked_body_ids = np.array(
                [self.fk_model.body(n).id for n in cfg.TRACKED_BODY_NAMES], dtype=np.int64
            )
            self.extended_parent_ids = np.array(
                [self.fk_model.body(parent).id for _, parent, _ in cfg.EXTENDED_JOINTS], dtype=np.int64,
            )
            self.extended_local_offsets = np.array(
                [offset for _, _, offset in cfg.EXTENDED_JOINTS], dtype=np.float64
            )

        # Per-term ring buffers (zero-init mimics IsaacLab CircularBuffer first-push fill).
        self._hist_bufs = {
            name: deque(
                [np.zeros(d, dtype=np.float32) for _ in range(self.history_len)],
                maxlen=self.history_len,
            )
            for name, d in self._hist_dims.items()
        }
        self._diff_body_pos_hist = (
            deque(
                [np.zeros(self._diff_body_pos_per_step, dtype=np.float32) for _ in range(self.history_len)],
                maxlen=self.history_len,
            )
            if use_diff_body_pos else None
        )
        self._diff_body_tannorm_hist = (
            deque(
                [np.zeros(self._diff_body_tannorm_per_step, dtype=np.float32) for _ in range(self.history_len)],
                maxlen=self.history_len,
            )
            if use_diff_body_tannorm else None
        )

        print("TWIST2 Real Controller obs layout (term-major):")
        for name in self._hist_term_order:
            d = self._hist_dims[name]
            print(f"  {name}: history={self.history_len} x dim={d} = {self.history_len * d}")
        print(f"  future_motion_mimic_target (no history): {self._mimic_dim}")
        if use_diff_body_pos:
            print(f"  diff_body_pos_b: history={self.history_len} x dim={self._diff_body_pos_per_step} = {self._diff_body_pos_total}")
        if use_diff_body_tannorm:
            print(f"  diff_body_tannorm_b: history={self.history_len} x dim={self._diff_body_tannorm_per_step} = {self._diff_body_tannorm_total}")
        print(f"  total_obs_size: {self.total_obs_size}")

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.control_dt = self.config.control_dt

        self.record_proprio = record_proprio
        self.proprio_recordings = [] if record_proprio else None

        self.smooth_body = smooth_body
        if smooth_body > 0.0:
            self.body_smoother = EMASmoother(alpha=smooth_body)
            print(f"Body action smoothing enabled with alpha={smooth_body}")
        else:
            self.body_smoother = None

    def reset_robot(self):
        print("Press START on remote to move to default position ...")
        self.env.move_to_default_pos()

        print("Now in default position, press A to continue ...")
        self.env.default_pos_state()

        print("Robot will hold default pos. If needed, do other checks here.")

    def _update_fk_data(self, dof_pos, quat):
        """Drive fk_data from measured (quat, dof_pos) and run kinematics."""
        self.fk_data.qpos[:3] = 0.0  # diff_body_pos_b cancels root translation
        self.fk_data.qpos[3:7] = quat  # IMU quat (wxyz)
        self.fk_data.qpos[7:7 + self.num_actions] = dof_pos  # SDK order matches XML
        mujoco.mj_kinematics(self.fk_model, self.fk_data)

    def compute_observation(self, dof_pos, dof_vel, ang_vel, rpy, action_mimic):
        """Build the flat observation tensor in PolicyCfg term-major order.

        dof_pos / dof_vel come in SDK order; permuted to Isaac order before the
        policy sees them. last_action is already in Isaac order (raw policy out).
        Updates all history buffers as a side effect.
        """
        dof_pos_isaac = dof_pos[self.sdk_to_isaac]
        dof_vel_isaac = dof_vel[self.sdk_to_isaac].copy()
        dof_vel_isaac[self.ankle_idx_isaac] = 0.0

        term_current = {
            "base_ang_vel":    (ang_vel * 0.25).astype(np.float32),
            "base_roll_pitch": rpy[:2].astype(np.float32),
            "joint_pos_rel":   (dof_pos_isaac - self.default_dof_pos_isaac).astype(np.float32),
            "joint_vel_rel":   (dof_vel_isaac * 0.05).astype(np.float32),
            "last_action":     self.last_action.astype(np.float32),
        }
        for name in self._hist_term_order:
            self._hist_bufs[name].append(term_current[name])

        flat_parts = [
            np.asarray(self._hist_bufs[name], dtype=np.float32).reshape(-1)
            for name in self._hist_term_order
        ]
        flat_parts.append(action_mimic)

        if self.use_diff_body_pos:
            diff = compute_diff_body_pos_b(
                self.fk_model, self.fk_data, self.ref_data, action_mimic,
                self.tracked_body_ids, self.extended_parent_ids, self.extended_local_offsets,
                self.num_actions, use_pb=True,
            )
            self._diff_body_pos_hist.append(diff)
            flat_parts.append(np.asarray(self._diff_body_pos_hist, dtype=np.float32).reshape(-1))

        if self.use_diff_body_tannorm:
            diff = compute_diff_body_tannorm_b(
                self.fk_model, self.fk_data, self.ref_data, action_mimic,
                self.tracked_body_ids, self.extended_parent_ids,
                self.num_actions, use_pb=True,
            )
            self._diff_body_tannorm_hist.append(diff)
            flat_parts.append(np.asarray(self._diff_body_tannorm_hist, dtype=np.float32).reshape(-1))

        obs_buf = np.concatenate(flat_parts)
        assert obs_buf.shape[0] == self.total_obs_size, \
            f"Expected {self.total_obs_size} obs, got {obs_buf.shape[0]}"
        return obs_buf

    def run(self):
        self.reset_robot()
        print("Begin main TWIST2 policy loop. Press [Select] on remote to exit.")

        # Idle-default seed for action_* keys (avoids chasing stale targets).
        default_mimic_obs = DEFAULT_MIMIC_OBS["unitree_g1_with_hands"]
        self.redis_pipeline.set("action_body_unitree_g1_with_hands", json.dumps(default_mimic_obs.tolist()))
        self.redis_pipeline.set("action_hand_left_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.set("action_hand_right_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.set("action_neck_unitree_g1_with_hands", json.dumps(np.zeros(2).tolist()))
        self.redis_pipeline.execute()

        last_policy_time = None
        policy_execution_times = []
        policy_step_count = 0
        policy_fps_print_interval = 100

        try:
            while True:
                t_start = time.time()

                # Forward remote-controller signals to the motion server.
                if self.redis_client:
                    controller_input = self.env.read_controller_input()
                    b_pressed = controller_input.keys == self.env.controller_mapping["B"]
                    select_pressed = controller_input.keys == self.env.controller_mapping["select"]
                    self.redis_client.set("motion_start_signal", "1" if b_pressed else "0")
                    self.redis_client.set("motion_exit_signal", "1" if select_pressed else "0")

                if self.env.read_controller_input().keys == self.env.controller_mapping["select"]:
                    print("Select pressed, exiting main loop.")
                    break

                dof_pos, dof_vel, quat, ang_vel, dof_temp, dof_tau, dof_vol = self.env.get_robot_state()
                rpy = quatToEuler(quat)

                # state_body (teleop bridge input) stays in MuJoCo/SDK order.
                state_body = np.concatenate([ang_vel, rpy[:2], dof_pos])  # 3+2+29 = 34
                self.redis_pipeline.set("state_body_unitree_g1_with_hands", json.dumps(state_body.tolist()))

                if self.use_hand:
                    left_hand_state, right_hand_state = self.hand_ctrl.get_hand_state()
                    lh_pos, rh_pos, lh_temp, rh_temp, lh_tau, rh_tau = self.hand_ctrl.get_hand_all_state()
                    self.redis_pipeline.set("state_hand_left_unitree_g1_with_hands", json.dumps(left_hand_state.tolist()))
                    self.redis_pipeline.set("state_hand_right_unitree_g1_with_hands", json.dumps(right_hand_state.tolist()))
                else:
                    self.redis_pipeline.set("state_hand_left_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
                    self.redis_pipeline.set("state_hand_right_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))

                self.redis_pipeline.set("state_neck_unitree_g1_with_hands", json.dumps(np.zeros(2).tolist()))
                # motion_server reads state_heading once at playback start to anchor frame 0.
                robot_heading = yaw_from_quat(quat)
                self.redis_pipeline.set("state_heading_unitree_g1_with_hands", json.dumps(robot_heading))
                self.redis_pipeline.set("t_state", int(time.time() * 1000))
                self.redis_pipeline.execute()

                keys = [
                    "action_body_unitree_g1_with_hands",
                    "action_hand_left_unitree_g1_with_hands",
                    "action_hand_right_unitree_g1_with_hands",
                    "action_neck_unitree_g1_with_hands",
                ]
                for key in keys:
                    self.redis_pipeline.get(key)
                redis_results = self.redis_pipeline.execute()
                action_mimic = np.asarray(json.loads(redis_results[0]), dtype=np.float32)
                action_hand_left_raw = json.loads(redis_results[1])
                action_hand_right_raw = json.loads(redis_results[2])

                if self.body_smoother is not None:
                    action_mimic = self.body_smoother.smooth(action_mimic)

                if self.use_hand:
                    action_hand_left = np.array(action_hand_left_raw, dtype=np.float32)
                    action_hand_right = np.array(action_hand_right_raw, dtype=np.float32)
                else:
                    action_hand_left = np.zeros(7, dtype=np.float32)
                    action_hand_right = np.zeros(7, dtype=np.float32)

                # Drive FK from measured robot state for diff_body_* terms.
                if self.fk_data is not None:
                    self._update_fk_data(dof_pos, quat)

                obs_buf = self.compute_observation(dof_pos, dof_vel, ang_vel, rpy, action_mimic)

                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()

                current_time = time.time()
                if last_policy_time is not None:
                    policy_execution_times.append(current_time - last_policy_time)
                    policy_step_count += 1
                    if policy_step_count % policy_fps_print_interval == 0:
                        recent = policy_execution_times[-policy_fps_print_interval:]
                        avg_interval = float(np.mean(recent))
                        print(f"Policy Execution FPS (last {policy_fps_print_interval} steps): {1.0/avg_interval:.2f} Hz (avg interval: {avg_interval*1000:.2f}ms)")
                last_policy_time = current_time

                # raw_action is in Isaac order; permute back to SDK before sending to the robot.
                self.last_action = raw_action
                raw_action = np.clip(raw_action, -10.0, 10.0)
                pd_target_isaac = raw_action * self.action_scale + self.default_dof_pos_isaac
                target_dof_pos = pd_target_isaac[self.isaac_to_sdk]

                kp_scale = 1.0
                kd_scale = 1.0
                self.env.send_robot_action(target_dof_pos, kp_scale, kd_scale)

                if self.use_hand:
                    self.hand_ctrl.ctrl_dual_hand(action_hand_left, action_hand_right)

                if self.record_proprio:
                    proprio_data = {
                        'timestamp': time.time(),
                        'body_dof_pos': dof_pos.tolist(),
                        'target_dof_pos': action_mimic.tolist()[-self.num_actions:],
                        'temperature': dof_temp.tolist(),
                        'tau': dof_tau.tolist(),
                        'voltage': dof_vol.tolist(),
                    }
                    if self.use_hand:
                        proprio_data['lh_pos'] = lh_pos.tolist()
                        proprio_data['rh_pos'] = rh_pos.tolist()
                        proprio_data['lh_temp'] = lh_temp.tolist()
                        proprio_data['rh_temp'] = rh_temp.tolist()
                        proprio_data['lh_tau'] = lh_tau.tolist()
                        proprio_data['rh_tau'] = rh_tau.tolist()
                    self.proprio_recordings.append(proprio_data)

                elapsed = time.time() - t_start
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)

        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.record_proprio and self.proprio_recordings:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f'logs/twist2_real_recordings_{timestamp}.json'
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w') as f:
                    json.dump(self.proprio_recordings, f)
                print(f"Proprioceptive recordings saved as {filename}")

            self.env.close()
            if self.use_hand:
                self.hand_ctrl.close()
            print("TWIST2 real controller finished.")


def main():
    parser = argparse.ArgumentParser(description='Run TWIST2 policy on real G1 robot')
    parser.add_argument('--policy', type=str, required=True,
                        help='Path to TWIST2 ONNX policy file')
    parser.add_argument('--config', type=str, default="robot_control/configs/g1.yaml",
                        help='Path to robot configuration file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run policy on (cuda/cpu)')
    parser.add_argument('--net', type=str, default='wlp0s20f3',
                        help='Network interface for robot communication')
    parser.add_argument('--use_hand', action='store_true',
                        help='Enable hand control')
    parser.add_argument('--record_proprio', action='store_true',
                        help='Record proprioceptive data')
    parser.add_argument('--smooth_body', type=float, default=0.0,
                        help='Smoothing factor for body actions (0.0=no smoothing, 1.0=maximum smoothing)')
    parser.add_argument('--xml', type=str, default='../assets/g1/g1_sim2sim_29dof.xml',
                        help='MuJoCo XML used for FK when diff_body_* terms are enabled.')
    parser.add_argument('--use_diff_body_pos', action='store_true',
                        help='Append diff_body_pos_b observation (33 bodies * 3 = 99 dims).')
    parser.add_argument('--use_diff_body_tannorm', action='store_true',
                        help='Append diff_body_tannorm_b observation (33 bodies * 6 = 198 dims).')

    args = parser.parse_args()

    if not os.path.exists(args.policy):
        print(f"Error: Policy file {args.policy} does not exist")
        return
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} does not exist")
        return
    if (args.use_diff_body_pos or args.use_diff_body_tannorm) and not os.path.exists(args.xml):
        print(f"Error: XML file {args.xml} does not exist (required for diff_body_* terms)")
        return

    print(f"Starting TWIST2 real robot controller...")
    print(f"  Policy file: {args.policy}")
    print(f"  Config file: {args.config}")
    print(f"  Device: {args.device}")
    print(f"  Network interface: {args.net}")
    print(f"  Use hand: {args.use_hand}")
    print(f"  Record proprio: {args.record_proprio}")
    print(f"  Smooth body: {args.smooth_body}")
    print(f"  use_diff_body_pos: {args.use_diff_body_pos}")
    print(f"  use_diff_body_tannorm: {args.use_diff_body_tannorm}")

    print("\n" + "="*50)
    print("SAFETY WARNING:")
    print("You are about to run a policy on a real robot.")
    print("Make sure the robot is in a safe environment.")
    print("Press Ctrl+C to stop at any time.")
    print("Use the remote controller [Select] button to exit.")
    print("="*50 + "\n")

    controller = RealTimePolicyController(
        policy_path=args.policy,
        config_path=args.config,
        device=args.device,
        net=args.net,
        use_hand=args.use_hand,
        record_proprio=args.record_proprio,
        smooth_body=args.smooth_body,
        xml_file=args.xml,
        use_diff_body_pos=args.use_diff_body_pos,
        use_diff_body_tannorm=args.use_diff_body_tannorm,
    )
    controller.run()


if __name__ == "__main__":
    main()
