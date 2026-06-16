import argparse
import json
import time
import os
from collections import deque

import numpy as np
import redis
import mujoco
import mujoco.viewer as mjv
import torch
from rich import print
from tqdm import tqdm

from data_utils.rot_utils import quatToEuler
from data_utils.params import DEFAULT_MIMIC_OBS

from cfg import g1_29dof_cfg as cfg
from observations import (
    compute_diff_body_pos_b,
    compute_diff_body_tannorm_b,
)
from safety import SafetyController
from utils.math import yaw_from_quat

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


class EMASmoother:
    """Exponential Moving Average smoother for the policy's output action."""

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
                 use_diff_body_pos=False,
                 use_diff_body_tannorm=False,
                 smooth_action=0.0,
                 kp_scale=1.0,
                 kd_scale=1.0,
                 update_robot_w_odom=False,
                 odom_topic="/twist2/sim_odom",
                 robot="unitree_g1_with_hands",
                 ):
        self.measure_fps = measure_fps
        self.limit_fps = limit_fps
        self.robot = robot
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_pipeline = self.redis_client.pipeline()
        except Exception as e:
            print(f"Error connecting to Redis: {e}")

        self.device = device
        self.policy = load_onnx_policy(policy_path, device)

        # MuJoCo sim
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)

        # self.safety = SafetyController(initial_scale=0.5)
        self.safety = SafetyController(initial_scale=1.0)
        self.viewer = mjv.launch_passive(
            self.model, self.data,
            key_callback=self.safety.handle_keycode,
            show_left_ui=False, show_right_ui=False,
        )
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
        self.viewer.cam.distance = 2.0

        self.num_actions = cfg.NUM_ACTIONS
        self.sim_duration = 100000.0
        self.sim_dt = 0.001
        self.sim_decimation = 1 / (policy_frequency * self.sim_dt)
        print(f"sim_decimation: {self.sim_decimation}")

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        # Optional EMA smoothing of the policy's OUTPUT action (motor command).
        # Does not affect last_action fed back into the observation.
        self.smooth_action = smooth_action
        if smooth_action > 0.0:
            self.action_smoother = EMASmoother(alpha=smooth_action)
            print(f"Output action smoothing enabled with alpha={smooth_action}")
        else:
            self.action_smoother = None

        # Robot params (SDK order).
        self.default_dof_pos = cfg.DEFAULT_DOF_POS.copy()
        self.stiffness = cfg.STIFFNESS * kp_scale
        self.damping = cfg.DAMPING  * kd_scale
        self.torque_limits = cfg.TORQUE_LIMITS
        self.action_scale = np.full(self.num_actions, cfg.ACTION_SCALE, dtype=np.float32)

        # MuJoCo init qpos: [xyz(3), quat_wxyz(4), joint_pos(29)] at training init pose.
        self.mujoco_default_dof_pos = np.concatenate([
            np.array([0, 0, 0.793]),
            np.array([1, 0, 0, 0]),
            self.default_dof_pos.copy(),
        ])

        # Joint-order permutation between SDK (MuJoCo qpos) and Isaac (policy I/O).
        #   isaac_ordered = sdk_ordered[sdk_to_isaac]
        #   sdk_ordered   = isaac_ordered[isaac_to_sdk]
        self.sdk_to_isaac = np.array(
            [cfg.SDK_JOINT_NAMES.index(n) for n in cfg.ISAAC_JOINT_NAMES], dtype=np.int64
        )
        self.isaac_to_sdk = np.array(
            [cfg.ISAAC_JOINT_NAMES.index(n) for n in cfg.SDK_JOINT_NAMES], dtype=np.int64
        )

        self.default_dof_pos_isaac = self.default_dof_pos[self.sdk_to_isaac]
        self.ankle_idx_isaac = [cfg.ISAAC_JOINT_NAMES.index(n) for n in cfg.ANKLE_JOINT_NAMES]

        # ----- observation layout (matches DEX_RL_LAB PolicyCfg, term-major) -----
        # CircularBuffer flatten = [oldest..newest]; terms concatenated in declaration order.
        self.history_len = cfg.HISTORY_LEN
        self._hist_dims = cfg.HIST_TERM_DIMS
        self._hist_term_order = cfg.HIST_TERM_ORDER
        self._mimic_dim = cfg.MIMIC_DIM

        # Optional diff_body_* terms (declared after mimic in PolicyCfg).
        self.use_diff_body_pos = use_diff_body_pos
        self._diff_body_pos_per_step = cfg.NUM_TRACKED_BODIES * 3   # 99
        self._diff_body_pos_total = (
            self.history_len * self._diff_body_pos_per_step if use_diff_body_pos else 0
        )
        self.use_diff_body_tannorm = use_diff_body_tannorm
        self._diff_body_tannorm_per_step = cfg.NUM_TRACKED_BODIES * 6  # 198
        self._diff_body_tannorm_total = (
            self.history_len * self._diff_body_tannorm_per_step if use_diff_body_tannorm else 0
        )

        self.total_obs_size = (
            self.history_len * sum(self._hist_dims.values())
            + self._mimic_dim
            + self._diff_body_pos_total
            + self._diff_body_tannorm_total
        )

        # Body-id caches for FK-based diff terms.
        self.tracked_body_ids = np.array(
            [self.model.body(n).id for n in cfg.TRACKED_BODY_NAMES], dtype=np.int64
        )
        self.extended_parent_ids = np.array(
            [self.model.body(parent).id for _, parent, _ in cfg.EXTENDED_JOINTS], dtype=np.int64,
        )
        self.extended_local_offsets = np.array(
            [offset for _, _, offset in cfg.EXTENDED_JOINTS], dtype=np.float64
        )
        # Secondary MjData for FK on the reference motion frame.
        self.ref_data = (
            mujoco.MjData(self.model)
            if (use_diff_body_pos or use_diff_body_tannorm) else None
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

        print("TWIST2 Controller obs layout (term-major):")
        for name in self._hist_term_order:
            d = self._hist_dims[name]
            print(f"  {name}: history={self.history_len} x dim={d} = {self.history_len * d}")
        print(f"  future_motion_mimic_target (no history): {self._mimic_dim}")
        if use_diff_body_pos:
            print(f"  diff_body_pos_b: history={self.history_len} x dim={self._diff_body_pos_per_step} = {self._diff_body_pos_total}")
        if use_diff_body_tannorm:
            print(f"  diff_body_tannorm_b: history={self.history_len} x dim={self._diff_body_tannorm_per_step} = {self._diff_body_tannorm_total}")
        print(f"  total_obs_size: {self.total_obs_size}")

        # ----- odom-based world-root tracking for diff_body_* terms -----
        # When enabled, the reference motion root is anchored (per motion) to the
        # robot's current world root so motion frame 0 coincides with the robot.
        # In sim the robot world root is MuJoCo GT (data.qpos); we also publish it
        # as a nav_msgs/Odometry topic so the same topic-based pipeline as the
        # real robot is exercised.
        self.update_robot_w_odom = update_robot_w_odom
        self.odom_topic = odom_topic
        self.odom_pub = None
        self._motion_epoch = None      # last seen motion epoch (new motion -> reset)
        self._odom_origin_xy = None    # robot world xy at motion start
        self._ref_origin_xy = None     # reference motion root xy at motion start
        if self.update_robot_w_odom:
            try:
                from ros_odom import OdomPublisher
                self.odom_pub = OdomPublisher(self.odom_topic)
                print(f"[odom] publishing MuJoCo GT root to ROS topic '{self.odom_topic}'.")
            except Exception as e:
                print(f"[odom] OdomPublisher unavailable ({e}); continuing without ROS publish.")
            print("[odom] update_robot_w_odom=True: diff_body_* uses absolute world root.")

        self.record_video = record_video
        self.record_proprio = record_proprio
        self.proprio_recordings = [] if record_proprio else None

    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, init_pos):
        self.data.qpos[:] = init_pos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def extract_data(self):
        n = self.num_actions
        dof_pos = self.data.qpos[7:7 + n]
        dof_vel = self.data.qvel[6:6 + n]
        quat = self.data.qpos[3:7]
        ang_vel = self.data.qvel[3:6]
        sim_torque = self.data.ctrl
        return dof_pos, dof_vel, quat, ang_vel, sim_torque

    def _compute_ref_root_xy_w(self, robot_root_xy):
        """Resolve the reference motion root xy in the (odom) world frame.

        Reads the motion server's published root + a per-motion epoch from Redis.
        On a new motion (epoch change) the world origin is re-anchored to the
        robot's current world root, so the returned ref xy makes motion frame 0
        coincide with the robot root and then translates with the motion.
        Returns None when odom mode is off (callers then skip the offset).
        """
        if not self.update_robot_w_odom or self.redis_client is None:
            return None
        try:
            raw_epoch = self.redis_client.get(f"motion_epoch_{self.robot}")
            raw_ref = self.redis_client.get(f"ref_root_world_{self.robot}")
        except Exception:
            raw_epoch, raw_ref = None, None

        ref_xy = None
        if raw_ref is not None:
            try:
                ref_root = np.asarray(json.loads(raw_ref), dtype=np.float64)
                ref_xy = ref_root[:2]
            except Exception:
                ref_xy = None

        epoch = raw_epoch.decode() if isinstance(raw_epoch, bytes) else raw_epoch
        if epoch != self._motion_epoch:
            # New motion: reset and re-latch the origin on the first frame whose
            # reference root is actually available (= motion frame 0).
            self._motion_epoch = epoch
            self._odom_origin_xy = None
            self._ref_origin_xy = None

        if ref_xy is None:
            # Ref root not published yet: pin ref root to the current robot root.
            return np.asarray(robot_root_xy, dtype=np.float64).copy()

        if self._odom_origin_xy is None:
            self._odom_origin_xy = np.asarray(robot_root_xy, dtype=np.float64).copy()
            self._ref_origin_xy = ref_xy.copy()
        return ref_xy - self._ref_origin_xy + self._odom_origin_xy

    def compute_observation(self, dof_pos, dof_vel, ang_vel, rpy, action_mimic,
                            ref_root_xy_w=None):
        """Build the flat observation tensor in PolicyCfg term-major order.

        dof_pos / dof_vel come in SDK order; permuted to Isaac order before the
        policy sees them. last_action is already in Isaac order (raw policy out).
        Updates all history buffers as a side effect.
        """
        # Only run the odom (absolute world-root) diff once a world root has been
        # resolved; otherwise fall back to the legacy pelvis-relative diff so the
        # ref/robot frames stay consistent.
        odom_on = self.update_robot_w_odom and ref_root_xy_w is not None

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
                self.model, self.data, self.ref_data, action_mimic,
                self.tracked_body_ids, self.extended_parent_ids, self.extended_local_offsets,
                self.num_actions, use_pb=True,
                update_robot_w_odom=odom_on,
                ref_root_xy_w=ref_root_xy_w,
            )
            self._diff_body_pos_hist.append(diff)
            flat_parts.append(np.asarray(self._diff_body_pos_hist, dtype=np.float32).reshape(-1))
            # diag: world-frame root tracking error (ref - robot) [x,y,z]. Should
            # hover ~0 when tracking well; a persistent z bias => height calibration,
            # persistent xy => anchor/translation lag. (~0.5s cadence)
            self._diff_log_i = getattr(self, "_diff_log_i", 0) + 1
            if self._diff_log_i % 25 == 0:
                ref_xy = ref_root_xy_w if ref_root_xy_w is not None else self.data.qpos[:2]
                dx = float(ref_xy[0]) - float(self.data.qpos[0])
                dy = float(ref_xy[1]) - float(self.data.qpos[1])
                dz = float(action_mimic[2]) - float(self.data.qpos[2])
                print(f"[root err ref-robot] odom={odom_on} x={dx:+.3f} y={dy:+.3f} z={dz:+.3f}")

        if self.use_diff_body_tannorm:
            diff = compute_diff_body_tannorm_b(
                self.model, self.data, self.ref_data, action_mimic,
                self.tracked_body_ids, self.extended_parent_ids,
                self.num_actions, use_pb=True,
                update_robot_w_odom=odom_on,
                ref_root_xy_w=ref_root_xy_w,
            )
            self._diff_body_tannorm_hist.append(diff)
            flat_parts.append(np.asarray(self._diff_body_tannorm_hist, dtype=np.float32).reshape(-1))

        obs_buf = np.concatenate(flat_parts)
        assert obs_buf.shape[0] == self.total_obs_size, \
            f"Expected {self.total_obs_size} obs, got {obs_buf.shape[0]}"
        return obs_buf

    def run(self):
        print("Starting TWIST2 simulation...")

        if self.record_video:
            import imageio
            mp4_writer = imageio.get_writer('twist2_simulation.mp4', fps=30)
        else:
            mp4_writer = None

        self.reset_sim()
        self.reset(self.mujoco_default_dof_pos)

        steps = int(self.sim_duration / self.sim_dt)
        pbar = tqdm(range(steps), desc="Simulating TWIST2...")

        # Seed Redis so the teleop bridge has something to read on its first tick.
        initial_state_body = np.zeros(3 + 2 + self.num_actions, dtype=np.float32)
        self.redis_pipeline.set("state_body_unitree_g1_with_hands", json.dumps(initial_state_body.tolist()))
        self.redis_pipeline.set("state_hand_left_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.set("state_hand_right_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.set("state_heading_unitree_g1_with_hands", json.dumps(0.0))

        # Idle-default seed for action_* keys (avoids chasing stale targets).
        default_mimic_obs = DEFAULT_MIMIC_OBS["unitree_g1_with_hands"]
        self.redis_pipeline.set("action_body_unitree_g1_with_hands", json.dumps(default_mimic_obs.tolist()))
        self.redis_pipeline.set("action_hand_left_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.set("action_hand_right_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
        self.redis_pipeline.set("action_neck_unitree_g1_with_hands", json.dumps(np.zeros(2).tolist()))
        self.redis_pipeline.execute()

        measure_fps = self.measure_fps
        fps_measurements = []
        fps_iteration_count = 0
        fps_measurement_target = 1000
        last_policy_time = None

        policy_execution_times = []
        policy_step_count = 0
        policy_fps_print_interval = 100

        try:
            for i in pbar:
                t_start = time.time()
                self.safety.drain()
                dof_pos, dof_vel, quat, ang_vel, sim_torque = self.extract_data()

                if i % self.sim_decimation == 0:
                    rpy = quatToEuler(quat)

                    # state_body (teleop bridge input) stays in MuJoCo/SDK order.
                    state_body = np.concatenate([ang_vel, rpy[:2], dof_pos])  # 3+2+29 = 34
                    self.redis_pipeline.set("state_body_unitree_g1_with_hands", json.dumps(state_body.tolist()))
                    self.redis_pipeline.set("state_hand_left_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
                    self.redis_pipeline.set("state_hand_right_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
                    self.redis_pipeline.set("state_neck_unitree_g1_with_hands", json.dumps(np.zeros(2).tolist()))
                    # Robot heading; motion_server reads once at playback start to anchor frame 0.
                    robot_heading = yaw_from_quat(self.data.qpos[3:7])
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

                    # Odom world-root tracking. In sim the robot world root is GT
                    # (data.qpos); publish it to the ROS odom topic so the same
                    # pipeline as the real robot is exercised, then resolve the
                    # reference root xy in the anchored world frame.
                    ref_root_xy_w = None
                    if self.update_robot_w_odom:
                        if self.odom_pub is not None:
                            self.odom_pub.publish(self.data.qpos[:3], self.data.qpos[3:7])
                        ref_root_xy_w = self._compute_ref_root_xy_w(self.data.qpos[:2])

                    obs_buf = self.compute_observation(
                        dof_pos, dof_vel, ang_vel, rpy, action_mimic,
                        ref_root_xy_w=ref_root_xy_w,
                    )

                    obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()

                    current_time = time.time()
                    if last_policy_time is not None:
                        policy_interval = current_time - last_policy_time
                        current_policy_fps = 1.0 / policy_interval

                        policy_execution_times.append(policy_interval)
                        policy_step_count += 1
                        if policy_step_count % policy_fps_print_interval == 0:
                            recent_intervals = policy_execution_times[-policy_fps_print_interval:]
                            avg_interval = np.mean(recent_intervals)
                            avg_execution_fps = 1.0 / avg_interval
                            print(f"Policy Execution FPS (last {policy_fps_print_interval} steps): {avg_execution_fps:.2f} Hz (avg interval: {avg_interval*1000:.2f}ms)")

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
                                fps_measurements = []
                                fps_iteration_count = 0
                    last_policy_time = current_time

                    # raw_action is in Isaac order; permute back to SDK before PD.
                    # Keep last_action as the RAW policy output (obs consistency);
                    # smoothing only affects the motor command sent to the robot.
                    self.last_action = raw_action
                    raw_action = np.clip(raw_action, -10., 10.)
                    if self.action_smoother is not None:
                        raw_action = self.action_smoother.smooth(raw_action)
                    pd_target_isaac = raw_action * self.action_scale + self.default_dof_pos_isaac
                    pd_target = pd_target_isaac[self.isaac_to_sdk]

                    pelvis_pos = self.data.xpos[self.model.body("pelvis").id]
                    self.viewer.cam.lookat = pelvis_pos
                    self.viewer.sync()

                    if mp4_writer is not None:
                        img = self.viewer.read_pixels()
                        mp4_writer.append_data(img)

                    if self.record_proprio:
                        self.proprio_recordings.append({
                            'timestamp': time.time(),
                            'dof_pos': dof_pos.tolist(),
                            'dof_vel': dof_vel.tolist(),
                            'rpy': rpy.tolist(),
                            'ang_vel': ang_vel.tolist(),
                            'target_dof_pos': action_mimic.tolist()[-29:],
                        })

                # PD control
                torque = (
                    (pd_target - dof_pos) * self.stiffness * self.safety.kp_scale
                    - dof_vel * self.damping * self.safety.kd_scale
                )
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)

                self.data.ctrl[:] = torque
                mujoco.mj_step(self.model, self.data)

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
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run policy on (cuda/cpu)')
    parser.add_argument('--record_video', action='store_true', help='Record video of simulation')
    parser.add_argument('--record_proprio', action='store_true', help='Record proprioceptive data')
    parser.add_argument("--measure_fps", help="Measure FPS", default=0, type=int)
    parser.add_argument("--limit_fps", help="Limit FPS with sleep", default=1, type=int)
    parser.add_argument("--policy_frequency", help="Policy frequency", default=100, type=int)
    parser.add_argument("--use_diff_body_pos", action="store_true",
                        help="Append diff_body_pos_b observation (33 bodies * 3 = 99 dims).")
    parser.add_argument("--use_diff_body_tannorm", action="store_true",
                        help="Append diff_body_tannorm_b observation (33 bodies * 6 = 198 dims).")
    parser.add_argument("--smooth_action", type=float, default=0.0,
                        help="EMA alpha for smoothing the policy's OUTPUT action (motor command). "
                             "0 disables (default). Smaller alpha = stronger smoothing but more lag. "
                             "1.0 = no smoothing.")
    parser.add_argument("--kp_scale", type=float, default=1.0,
                        help="Scale factor applied to cfg.STIFFNESS for PD control.")
    parser.add_argument("--kd_scale", type=float, default=1.0,
                        help="Scale factor applied to cfg.DAMPING for PD control.")
    parser.add_argument("--update_robot_w_odom", action="store_true",
                        help="Track the robot world root via odometry so diff_body_* "
                             "uses absolute root translation. In sim the MuJoCo GT root "
                             "is published to --odom_topic; the reference motion root is "
                             "anchored to the robot root at each motion start.")
    parser.add_argument("--odom_topic", type=str, default="/twist2/sim_odom",
                        help="nav_msgs/Odometry topic the sim publishes GT root to "
                             "(used when --update_robot_w_odom is set).")
    args = parser.parse_args()

    if not os.path.exists(args.policy):
        print(f"Error: Policy file {args.policy} does not exist")
        return
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
    print(f"  Kp scale: {args.kp_scale}")
    print(f"  Kd scale: {args.kd_scale}")
    controller = RealTimePolicyController(
        xml_file=args.xml,
        policy_path=args.policy,
        device=args.device,
        record_video=args.record_video,
        record_proprio=args.record_proprio,
        measure_fps=args.measure_fps,
        limit_fps=args.limit_fps,
        policy_frequency=args.policy_frequency,
        use_diff_body_pos=args.use_diff_body_pos,
        use_diff_body_tannorm=args.use_diff_body_tannorm,
        smooth_action=args.smooth_action,
        kp_scale=args.kp_scale,
        kd_scale=args.kd_scale,
        update_robot_w_odom=args.update_robot_w_odom,
        odom_topic=args.odom_topic,
    )
    controller.run()


if __name__ == "__main__":
    main()
