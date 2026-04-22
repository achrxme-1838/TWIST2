import argparse
import json
import time
import numpy as np
import redis
import mujoco
import torch
from rich import print
from collections import deque
import mujoco.viewer as mjv
from tqdm import tqdm
import os
from data_utils.rot_utils import quatToEuler

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


class RealTimePolicyController:
    def __init__(self, 
                 xml_file, 
                 policy_path, 
                 device='cuda', 
                 record_video=False,
                 record_proprio=False,
                 measure_fps=False,
                 limit_fps=True,
                 policy_frequency=50,
                 ):
        self.measure_fps = measure_fps
        self.limit_fps = limit_fps
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_pipeline = self.redis_client.pipeline()
        except Exception as e:
            print(f"Error connecting to Redis: {e}")

        self.device = device
        self.policy = load_onnx_policy(policy_path, device)

        # Create MuJoCo sim
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        
        self.viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
        self.viewer.cam.distance = 2.0

        self.num_actions = 29
        self.sim_duration = 100000.0
        self.sim_dt = 0.001
        # real frequency = 1 / (decimation * sim_dt)
        # ==> decimation = 1 / (real frequency * sim_dt)
        self.sim_decimation = 1 / (policy_frequency * self.sim_dt * 4)
        print(f"sim_decimation: {self.sim_decimation}")

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        # G1 specific configuration
        # Values below mirror the Isaac training config at
        #   DEX_RL_LAB/dex_rl_lab/robots/g1/g1_29dof/configs/g1_29dof_lab_cfg.py
        # Joint order follows `joint_sdk_names` (== MuJoCo g1_sim2sim_29dof.xml order):
        #   0-5   left leg  (hip_p, hip_r, hip_y, knee, ankle_p, ankle_r)
        #   6-11  right leg (hip_p, hip_r, hip_y, knee, ankle_p, ankle_r)
        #   12-14 torso     (waist_yaw, waist_roll, waist_pitch)
        #   15-21 left arm  (sh_p, sh_r, sh_y, elbow, wrist_r, wrist_p, wrist_y)
        #   22-28 right arm (sh_p, sh_r, sh_y, elbow, wrist_r, wrist_p, wrist_y)
        self.default_dof_pos = np.array([
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,           # left leg
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,           # right leg
                0.0, 0.0, 0.0,                            # torso
                0.3, 0.25, 0.0, 0.97, 0.15, 0.0, 0.0,     # left arm
                0.3, -0.25, 0.0, 0.97, -0.15, 0.0, 0.0,   # right arm
            ])

        # MuJoCo initial qpos: [xyz(3), quat_wxyz(4), joint_pos(29)].
        # Joint part mirrors default_dof_pos so sim spawns at the training init pose.
        self.mujoco_default_dof_pos = np.concatenate([
            np.array([0, 0, 0.793]),
            np.array([1, 0, 0, 0]),
            self.default_dof_pos.copy(),
        ])

        # Kp / Kd from ImplicitActuatorCfg stiffness/damping in the G1 lab cfg.
        self.stiffness = np.array([
                100, 100, 100, 150, 40, 40,        # left leg
                100, 100, 100, 150, 40, 40,        # right leg
                200, 40, 40,                       # waist_yaw, roll, pitch
                40, 40, 40, 40, 40, 40, 40,        # left arm
                40, 40, 40, 40, 40, 40, 40,        # right arm
            ])
        self.damping = np.array([
                2, 2, 2, 4, 2, 2,                  # left leg
                2, 2, 2, 4, 2, 2,                  # right leg
                5, 5, 5,                           # waist_yaw, roll, pitch
                1, 1, 1, 1, 1, 1, 1,               # left arm
                1, 1, 1, 1, 1, 1, 1,               # right arm
            ])

        # Torque limits = effort_limit_sim from each actuator group.
        # wrist_pitch/yaw use W4010-25 (limit 5); everything else on the arm uses N5020-16 (25).
        self.torque_limits = np.array([
                88, 139, 88, 139, 25, 25,          # left leg
                88, 139, 88, 139, 25, 25,          # right leg
                88, 25, 25,                        # waist_yaw, roll, pitch
                25, 25, 25, 25, 25, 5, 5,          # left arm
                25, 25, 25, 25, 25, 5, 5,          # right arm
            ])

        # Training: JointPositionActionCfg(scale=0.25, use_default_offset=True)
        #   -> pd_target = raw_action * 0.25 + default_dof_pos
        self.action_scale = np.full(self.num_actions, 0.25, dtype=np.float32)

        # ------------------------------------------------------------------
        # Joint-order permutation (Isaac <-> MuJoCo/SDK)
        # ------------------------------------------------------------------
        # The trained policy operates on Isaac's internal joint order (USD DFS),
        # which differs from `joint_sdk_names` / MuJoCo XML order. Verified via
        # DEX_RL_LAB/scripts/check_joint_order.py.
        #
        # Everything in THIS file that indexes ``data.qpos`` / ``data.qvel`` /
        # ``data.ctrl`` stays in MuJoCo (SDK) order. Observations fed to the
        # policy and raw actions produced by the policy are in Isaac order. We
        # bridge the two with ``sdk_to_isaac`` / ``isaac_to_sdk``.
        ISAAC_JOINT_NAMES = [
            "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
            "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
            "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
            "left_knee_joint", "right_knee_joint",
            "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
            "left_ankle_pitch_joint", "right_ankle_pitch_joint",
            "left_shoulder_roll_joint", "right_shoulder_roll_joint",
            "left_ankle_roll_joint", "right_ankle_roll_joint",
            "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
            "left_elbow_joint", "right_elbow_joint",
            "left_wrist_roll_joint", "right_wrist_roll_joint",
            "left_wrist_pitch_joint", "right_wrist_pitch_joint",
            "left_wrist_yaw_joint", "right_wrist_yaw_joint",
        ]
        SDK_JOINT_NAMES = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]
        #   isaac_ordered = sdk_ordered[self.sdk_to_isaac]
        #   sdk_ordered   = isaac_ordered[self.isaac_to_sdk]
        self.sdk_to_isaac = np.array(
            [SDK_JOINT_NAMES.index(n) for n in ISAAC_JOINT_NAMES], dtype=np.int64
        )
        self.isaac_to_sdk = np.array(
            [ISAAC_JOINT_NAMES.index(n) for n in SDK_JOINT_NAMES], dtype=np.int64
        )

        # Pre-compute Isaac-order views of constants that the policy sees.
        self.default_dof_pos_isaac = self.default_dof_pos[self.sdk_to_isaac]
        # Ankle indices in Isaac order -- used for dof_vel zero-masking to match
        # the training-side ObsTerm(mask_joint_names=[".*_ankle_pitch_joint", ...]).
        self.ankle_idx_isaac = [
            ISAAC_JOINT_NAMES.index(n) for n in (
                "left_ankle_pitch_joint", "right_ankle_pitch_joint",
                "left_ankle_roll_joint",  "right_ankle_roll_joint",
            )
        ]
        # Kept for reference / legacy logging; no longer used for masking.
        self.ankle_idx = [4, 5, 10, 11]

        # ------------------------------------------------------------------
        # Observation layout -- matches DEX_RL_LAB PolicyCfg (term-major).
        # ------------------------------------------------------------------
        # IsaacLab ObservationManager flattens each ObsTerm's CircularBuffer as
        # [oldest, ..., newest] and concatenates terms in declaration order.
        # PolicyCfg order (see g1_29dof_sonic_distill.py):
        #   base_ang_vel              (3, history=10, scale 0.25)
        #   base_roll_pitch           (2, history=10)
        #   joint_pos_rel             (29, history=10, Isaac order)
        #   joint_vel_rel             (29, history=10, Isaac order, ankle mask, scale 0.05)
        #   last_action               (29, history=10, Isaac order)
        #   future_motion_mimic_target(35, history=1)
        #
        # Final flat dim = 10*(3+2+29+29+29) + 35 = 920 + 35 = 955.
        #
        # NOTE: PolicyCfg may also contain ``diff_body_pos_b_deploy``. It is NOT
        # reproduced here (deploy has no robot/reference body FK). If training
        # keeps it, this obs will be short by that term's width — either remove
        # the term from training, or implement the FK diff here.
        self.history_len = 10
        self._hist_dims = {
            "base_ang_vel":     3,
            "base_roll_pitch":  2,
            "joint_pos_rel":   29,
            "joint_vel_rel":   29,
            "last_action":     29,
        }
        self._hist_term_order = [
            "base_ang_vel", "base_roll_pitch",
            "joint_pos_rel", "joint_vel_rel", "last_action",
        ]

        self._mimic_dim = 35   # TODO: MOTION_OBS

        self.total_obs_size = (
            self.history_len * sum(self._hist_dims.values()) + self._mimic_dim
        )

        print(f"TWIST2 Controller Configuration (term-major, DEX_RL_LAB layout):")
        for name in self._hist_term_order:
            d = self._hist_dims[name]
            print(f"  {name}: history={self.history_len} x dim={d} = {self.history_len * d}")
        print(f"  future_motion_mimic_target (no history): {self._mimic_dim}")
        print(f"  total_obs_size: {self.total_obs_size}")

        # One ring buffer per term. IsaacLab CircularBuffer, on first push,
        # back-fills every slot with that first observation; we replicate that
        # by initializing each deque with zeros and then relying on the first
        # step to fill them (the behaviour difference at step 0 is negligible).
        self._hist_bufs = {
            name: deque(
                [np.zeros(d, dtype=np.float32) for _ in range(self.history_len)],
                maxlen=self.history_len,
            )
            for name, d in self._hist_dims.items()
        }

        # Recording
        self.record_video = record_video
        self.record_proprio = record_proprio
        self.proprio_recordings = [] if record_proprio else None
        

    def reset_sim(self):
        """Reset simulation to initial state"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, init_pos):
        """Reset robot to initial position"""
        self.data.qpos[:] = init_pos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def extract_data(self):
        """Extract robot state data"""
        n_dof = self.num_actions
        dof_pos = self.data.qpos[7:7+n_dof]
        dof_vel = self.data.qvel[6:6+n_dof]
        quat = self.data.qpos[3:7]
        ang_vel = self.data.qvel[3:6]
        sim_torque = self.data.ctrl
        return dof_pos, dof_vel, quat, ang_vel, sim_torque

    def run(self):
        """Main simulation loop"""
        print("Starting TWIST2 simulation...")

        # Video recording setup
        if self.record_video:
            import imageio
            mp4_writer = imageio.get_writer('twist2_simulation.mp4', fps=30)
        else:
            mp4_writer = None

        self.reset_sim()
        self.reset(self.mujoco_default_dof_pos)

        steps = int(self.sim_duration / self.sim_dt)
        pbar = tqdm(range(steps), desc="Simulating TWIST2...")

        # Send initial proprio to redis -- state_body has shape 3+2+29 = 34
        # (same layout the main loop will publish, just zero-filled so the teleop
        # bridge has something to read on its first tick).
        initial_state_body = np.zeros(3 + 2 + self.num_actions, dtype=np.float32)
        self.redis_pipeline.set("state_body_unitree_g1_with_hands", json.dumps(initial_state_body.tolist()))
        self.redis_pipeline.set("state_hand_left_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.set("state_hand_right_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.execute()

        measure_fps = self.measure_fps
        fps_measurements = []
        fps_iteration_count = 0
        fps_measurement_target = 1000
        last_policy_time = None

        # Add policy execution FPS tracking for frequent printing
        policy_execution_times = []
        policy_step_count = 0
        policy_fps_print_interval = 100

        try:
            for i in pbar:
                t_start = time.time()
                dof_pos, dof_vel, quat, ang_vel, sim_torque = self.extract_data()
                
                if i % self.sim_decimation == 0:
                    # ---- 1) Compute the current-frame value of each ObsTerm ----
                    # dof_pos / dof_vel come out of MuJoCo in SDK order; the policy
                    # expects Isaac order, so we permute before feeding them in.
                    # last_action is already stored in Isaac order (raw policy output).
                    rpy = quatToEuler(quat)
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

                    # ---- 2) Redis: publish state_body, pull action_mimic ----
                    # state_body (teleop bridge input) stays in MuJoCo/SDK order.
                    state_body = np.concatenate([
                        ang_vel,
                        rpy[:2],
                        dof_pos]) # 3+2+29 = 34 dims

                    self.redis_pipeline.set("state_body_unitree_g1_with_hands", json.dumps(state_body.tolist()))
                    self.redis_pipeline.set("state_hand_left_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
                    self.redis_pipeline.set("state_hand_right_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
                    self.redis_pipeline.set("state_neck_unitree_g1_with_hands", json.dumps(np.zeros(2).tolist()))
                    self.redis_pipeline.set("t_state", int(time.time() * 1000))
                    self.redis_pipeline.execute()

                    keys = ["action_body_unitree_g1_with_hands", "action_hand_left_unitree_g1_with_hands", "action_hand_right_unitree_g1_with_hands", "action_neck_unitree_g1_with_hands"]
                    for key in keys:
                        self.redis_pipeline.get(key)
                    redis_results = self.redis_pipeline.execute()
                    action_mimic = np.asarray(json.loads(redis_results[0]), dtype=np.float32)
                    action_left_hand = json.loads(redis_results[1])
                    action_right_hand = json.loads(redis_results[2])
                    action_neck = json.loads(redis_results[3])

                    # assert action_mimic.shape[0] == self._mimic_dim, \
                    #     f"Expected mimic dim {self._mimic_dim}, got {action_mimic.shape[0]}"

                    # ---- 3) Append current values into per-term ring buffers ----
                    for name in self._hist_term_order:
                        self._hist_bufs[name].append(term_current[name])

                    # ---- 4) Build the flat obs tensor in training term-major order ----
                    # Each term: CircularBuffer flattened as [oldest, ..., newest].
                    # Final layout: concat over terms, then append current mimic target.
                    flat_parts = [
                        np.asarray(self._hist_bufs[name], dtype=np.float32).reshape(-1)
                        for name in self._hist_term_order
                    ]
                    
                    flat_parts.append(action_mimic)   # TODO: MOTION_OBS

                    obs_buf = np.concatenate(flat_parts)

                    assert obs_buf.shape[0] == self.total_obs_size, \
                        f"Expected {self.total_obs_size} obs, got {obs_buf.shape[0]}"
                    
                    # Run policy
                    obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()

                    # Measure and track policy execution FPS
                    current_time = time.time()
                    if last_policy_time is not None:
                        policy_interval = current_time - last_policy_time
                        current_policy_fps = 1.0 / policy_interval
                        
                        # For frequent printing (every 100 steps)  
                        policy_execution_times.append(policy_interval)
                        policy_step_count += 1
                        
                        # Print policy execution FPS every 100 steps
                        if policy_step_count % policy_fps_print_interval == 0:
                            recent_intervals = policy_execution_times[-policy_fps_print_interval:]
                            avg_interval = np.mean(recent_intervals)
                            avg_execution_fps = 1.0 / avg_interval
                            print(f"Policy Execution FPS (last {policy_fps_print_interval} steps): {avg_execution_fps:.2f} Hz (avg interval: {avg_interval*1000:.2f}ms)")
                        
                        # For detailed measurement (every 1000 steps)
                        if measure_fps:
                            fps_measurements.append(current_policy_fps)
                            fps_iteration_count += 1
                            
                            if fps_iteration_count == fps_measurement_target:
                                avg_fps = np.mean(fps_measurements)
                                max_fps = np.max(fps_measurements)
                                min_fps = np.min(fps_measurements)
                                std_fps = np.std(fps_measurements)
                                print(f"\n=== Policy Execution FPS Results (steps {fps_iteration_count-fps_measurement_target+1}-{fps_iteration_count}) ===")
                                print(f"Average Policy FPS: {avg_fps:.2f}")
                                print(f"Max Policy FPS: {max_fps:.2f}")
                                print(f"Min Policy FPS: {min_fps:.2f}")
                                print(f"Std Policy FPS: {std_fps:.2f}")
                                print(f"Expected FPS (from decimation): {1.0/(self.sim_decimation * self.sim_dt):.2f}")
                                print(f"=================================================================================\n")
                                # Reset for next 1000 measurements
                                fps_measurements = []
                                fps_iteration_count = 0
                    last_policy_time = current_time
                    
                    # raw_action is in Isaac joint order (policy output).
                    self.last_action = raw_action
                    raw_action = np.clip(raw_action, -10., 10.)
                    scaled_actions_isaac = raw_action * self.action_scale
                    pd_target_isaac = scaled_actions_isaac + self.default_dof_pos_isaac
                    # Back to MuJoCo/SDK order before applying PD on data.ctrl.
                    pd_target = pd_target_isaac[self.isaac_to_sdk]

                    # self.redis_client.set("action_low_level_unitree_g1", json.dumps(raw_action.tolist()))
                    
                    # Update camera to follow pelvis
                    pelvis_pos = self.data.xpos[self.model.body("pelvis").id]
                    self.viewer.cam.lookat = pelvis_pos
                    self.viewer.sync()
                    
                    if mp4_writer is not None:
                        img = self.viewer.read_pixels()
                        mp4_writer.append_data(img)

                    # Record proprio if enabled
                    if self.record_proprio:
                        proprio_data = {
                            'timestamp': time.time(),
                            'dof_pos': dof_pos.tolist(),
                            'dof_vel': dof_vel.tolist(),
                            'rpy': rpy.tolist(),
                            'ang_vel': ang_vel.tolist(),
                            'target_dof_pos': action_mimic.tolist()[-29:],
                        }
                        self.proprio_recordings.append(proprio_data)

               
                # PD control
                torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)

                self.data.ctrl[:] = torque
                mujoco.mj_step(self.model, self.data)
                
                # Sleep to maintain real-time pace
                if self.limit_fps:
                    elapsed = time.time() - t_start
                    if elapsed < self.sim_dt:
                        time.sleep(self.sim_dt - elapsed)

                    
        except Exception as e:
            print(f"Error in run: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if mp4_writer is not None:
                mp4_writer.close()
                print("Video saved as twist2_simulation.mp4")
            
            # Save proprio recordings if enabled
            if self.record_proprio and self.proprio_recordings:
                import pickle
                with open('twist2_proprio_recordings.pkl', 'wb') as f:
                    pickle.dump(self.proprio_recordings, f)
                print("Proprioceptive recordings saved as twist2_proprio_recordings.pkl")

            if self.viewer:
                self.viewer.close()
            print("Simulation finished.")


def main():
    parser = argparse.ArgumentParser(description='Run TWIST2 policy in simulation')
    parser.add_argument('--xml', type=str, default='../assets/g1/g1_sim2sim.xml',
                        help='Path to MuJoCo XML file')
    parser.add_argument('--policy', type=str, required=True,
                        help='Path to TWIST2 ONNX policy file')
    parser.add_argument('--device', type=str, 
                        default='cuda',
                        help='Device to run policy on (cuda/cpu)')
    parser.add_argument('--record_video', action='store_true',
                        help='Record video of simulation')
    parser.add_argument('--record_proprio', action='store_true',
                        help='Record proprioceptive data')
    parser.add_argument("--measure_fps", help="Measure FPS", default=0, type=int)
    parser.add_argument("--limit_fps", help="Limit FPS with sleep", default=1, type=int)
    parser.add_argument("--policy_frequency", help="Policy frequency", default=100, type=int)
    args = parser.parse_args()
    
    # Verify policy file exists
    if not os.path.exists(args.policy):
        print(f"Error: Policy file {args.policy} does not exist")
        return
    
    # Verify XML file exists
    if not os.path.exists(args.xml):
        print(f"Error: XML file {args.xml} does not exist")
        return
    
    print(f"Starting TWIST2 simulation controller...")
    print(f"  XML file: {args.xml}")
    print(f"  Policy file: {args.policy}")
    print(f"  Device: {args.device}")
    print(f"  Record video: {args.record_video}")
    print(f"  Record proprio: {args.record_proprio}")
    print(f"  Measure FPS: {args.measure_fps}")
    print(f"  Limit FPS: {args.limit_fps}")
    controller = RealTimePolicyController(
        xml_file=args.xml,
        policy_path=args.policy,
        device=args.device,
        record_video=args.record_video,
        record_proprio=args.record_proprio,
        measure_fps=args.measure_fps,
        limit_fps=args.limit_fps,
        policy_frequency=args.policy_frequency,
    )
    controller.run()


if __name__ == "__main__":
    main()
