#!/usr/bin/env python3
"""TWIST2 real-robot controller.

Mirrors server_low_level_g1_sim.py: same policy-spec observation layout (ObsBuilder),
same SDK<->Isaac permutation, same ONNX policy interface, same reference sources (a
precomputed motion played in-process via --motion_npz, or the Redis motion server).
Only the I/O surface differs: robot state comes from G1RealWorldEnv (IMU + motor
state), PD runs on the motors via env.send_robot_action, and the robot's world root
pose -- which the sim reads from MuJoCo -- is estimated (--root_source):

  odom       FAST-LIO odometry of the MID-360 (rigid on torso_link) -> pelvis via FK
             of the measured joints; aligned to the IMU heading + kinematic height
             once at policy start. roll/pitch always from the IMU.
  kinematic  no odometry: height from leg FK (lowest foot contact on the ground),
             orientation from the IMU, xy assumed to follow the reference root
             (translation is open loop -- for in-place motions / first tests).
  gt         --mock only: MuJoCo ground truth.

The estimated root + measured joints drive an FK-only MjData (``self.data``), which is
what the obs terms read where the sim reads its simulated MjData.

Remote: START -> move to default pose, A -> policy loop, B -> start (or restart) the
motion, Select -> damping + exit. Terminal keys 2..9 / X set the PD gain scale.
"""
import argparse
import json
import os
import time

import numpy as np
import mujoco
import torch
from rich import print

from data_utils.rot_utils import quatToEuler
from data_utils.params import DEFAULT_MIMIC_OBS
from motion_player import LocalMotionPlayer

from cfg import g1_29dof_cfg as cfg
from observations import parse_future_raw
from obs_builder import ObsBuilder
from obs_terms import ObsContext, StaticContext
from policy_spec import resolve_spec
from safety import SafetyController, StdinKeyListener
from utils.math import quat_apply, quat_conj, quat_mul, rpy_to_quat, yaw_from_quat

from robot_control.config import Config

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# MID-360 mount from g1_29dof_rev_1_0.urdf (mid360_joint, parent torso_link): xyz + rpy.
# FAST-LIO reports the pose of its IMU/body frame -- check it against this before use.
MID360_OFFSET = (0.0002835, 0.00003, 0.41618, 0.0, 0.04014257279586953, 0.0)
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
# Proprio logs go to <repo>/logs/real regardless of the working directory.
DEFAULT_LOG_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "real"))


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
    inp = session.get_inputs()[0]
    print(f"ONNX policy loaded from {policy_path} using providers: {session.get_providers()}")
    wrapper = OnnxPolicyWrapper(session, inp.name)
    wrapper.input_dim = int(inp.shape[-1]) if isinstance(inp.shape[-1], int) else None
    return wrapper


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


class SafetyStop(Exception):
    """Raised inside the policy loop to drop the robot into damping."""


def _pose_inv(pos, quat):
    qi = quat_conj(quat)
    return -quat_apply(qi, pos), qi


def _pose_mul(pos_a, quat_a, pos_b, quat_b):
    return pos_a + quat_apply(quat_a, pos_b), quat_mul(quat_a, quat_b)


class RealTimePolicyController:
    def __init__(self,
                 xml_file,
                 policy_path,
                 config_path,
                 obs_cfg=None,
                 device='cuda',
                 net='eno1',
                 use_hand=False,
                 record_proprio=False,
                 log_dir=DEFAULT_LOG_DIR,
                 smooth_action=0.0,
                 kp_scale=1.0,
                 kd_scale=1.0,
                 root_source="odom",
                 odom_topic="/Odometry",
                 odom_body="torso_link",
                 odom_offset=MID360_OFFSET,
                 odom_timeout=0.3,
                 robot="unitree_g1_with_hands",
                 motion_npz=None,
                 motion_start_delay=-1.0,
                 motion_blend_in_time=None,
                 exit_after_motion=-1.0,
                 initial_gain_scale=0.1,
                 max_tilt=0.8,
                 env=None,
                 odom_source=None,
                 realtime=True,
                 ):
        self.robot = robot
        self.motion_npz = motion_npz
        self.redis_client = None
        self.redis_pipeline = None
        if motion_npz is None:
            try:
                import redis
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
                self.redis_pipeline = self.redis_client.pipeline()
            except Exception as e:
                raise SystemExit(f"Error connecting to Redis ({e}); pass --motion_npz for local playback.")

        self.config = Config(config_path)
        # Policy was trained against cfg.DEFAULT_DOF_POS / STIFFNESS / DAMPING -- override
        # the YAML so move_to_default_pos, the motor PD and the obs offsets all match sim.
        self.config.default_angles = cfg.DEFAULT_DOF_POS.astype(np.float32).copy()
        self.config.kps = (cfg.STIFFNESS.astype(np.float32) * kp_scale).tolist()
        self.config.kds = (cfg.DAMPING.astype(np.float32) * kd_scale).tolist()
        self.control_dt = float(self.config.control_dt)
        self.realtime = realtime

        if env is None:
            from robot_control.g1_wrapper import G1RealWorldEnv  # needs unitree_interface
            env = G1RealWorldEnv(net=net, config=self.config)
        self.env = env
        self.use_hand = use_hand
        if use_hand:
            from robot_control.dex_hand_wrapper import Dex3_1_Controller
            self.hand_ctrl = Dex3_1_Controller(net, re_init=False)

        self.device = device
        self.policy = load_onnx_policy(policy_path, device)

        # Robot params (SDK order). Hardware PD uses YAML kps/kds via env.
        self.num_actions = cfg.NUM_ACTIONS
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

        # ----- observation layout: from the policy spec (see policy_spec.py) -----
        self.spec = resolve_spec(policy_path, obs_cfg)
        print(self.spec.describe())

        # MuJoCo model for FK only (no stepping): self.data is driven by the estimated
        # root + measured joints each tick and stands in for the sim's MjData.
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        # Body-id caches for FK-based diff / future terms.
        self.tracked_body_ids = np.array(
            [self.model.body(n).id for n in cfg.TRACKED_BODY_NAMES], dtype=np.int64
        )
        self.extended_parent_ids = np.array(
            [self.model.body(parent).id for _, parent, _ in cfg.EXTENDED_JOINTS], dtype=np.int64,
        )
        self.extended_local_offsets = np.array(
            [offset for _, _, offset in cfg.EXTENDED_JOINTS], dtype=np.float64
        )
        # MjData for FK on the current reference frame (diff_body_* terms)
        # and on the future reference frames (future_motion_* terms)
        self.ref_data = mujoco.MjData(self.model) if self.spec.needs_ref_fk else None
        self.future_ref_data = mujoco.MjData(self.model) if self.spec.needs_future else None

        self.obs_static = StaticContext(
            model=self.model,
            num_actions=self.num_actions,
            tracked_body_ids=self.tracked_body_ids,
            extended_parent_ids=self.extended_parent_ids,
            extended_local_offsets=self.extended_local_offsets,
            default_dof_pos_isaac=self.default_dof_pos_isaac,
            isaac_joint_names=list(cfg.ISAAC_JOINT_NAMES),
            future_steps=self.spec.future_steps,
            future_fk_steps=max(self.spec.used_future_steps, 1),
        )
        self.obs_builder = ObsBuilder(self.spec, self.obs_static)
        self.total_obs_size = self.obs_builder.obs_dim
        print(self.obs_builder.describe())
        policy_dim = getattr(self.policy, "input_dim", None)
        if policy_dim is not None and policy_dim != self.total_obs_size:
            raise ValueError(
                f"policy {policy_path} expects {policy_dim} obs but the spec builds "
                f"{self.total_obs_size} -- wrong --obs_cfg for this checkpoint?"
            )

        # ----- in-process reference motion (replaces the Redis motion server) -----
        self.motion_player = None
        if motion_npz is not None:
            self.motion_player = LocalMotionPlayer(
                motion_npz, DEFAULT_MIMIC_OBS[self.robot],
                control_dt=self.control_dt,
                future_motion_steps=self.spec.future_motion_steps if self.spec.needs_future else None,
                blend_in_time=motion_blend_in_time,
            )
            start = (f"auto start {motion_start_delay:.1f}s after A" if motion_start_delay >= 0
                     else "press B to start")
            print(f"[motion] local playback of {motion_npz} ({self.motion_player.num_steps} steps); {start}")
        self.motion_start_delay = float(motion_start_delay)
        self.exit_after_motion = float(exit_after_motion)
        self._post_motion_steps = 0
        self._b_prev = False

        # ----- world root estimation (see module docstring) -----
        self.root_source = root_source
        self.odom_timeout = float(odom_timeout)
        self.odom_sub = None
        self._odom_body_id = self.model.body(odom_body).id
        self._odom_off_pos = np.asarray(odom_offset[:3], dtype=np.float64)
        self._odom_off_quat = rpy_to_quat(*odom_offset[3:])
        self._odom_cal = None          # (pos, quat): odom-derived pelvis frame -> robot world
        self._kin_xy = np.zeros(2)     # kinematic mode: robot xy (follows the reference)
        self._root_yaw = None          # last estimated heading (motion start anchor)
        foot_ids = {self.model.body(n).id for n in FOOT_BODIES}
        self._foot_geom_ids = np.array([
            g for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] in foot_ids and self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
        ], dtype=np.int64)
        self._foot_geom_r = self.model.geom_size[self._foot_geom_ids, 0].copy()
        if root_source == "odom":
            if odom_source is None:
                from ros_odom import OdomSubscriber
                odom_source = OdomSubscriber(odom_topic)
                print(f"[odom] subscribing to '{odom_topic}' (sensor on {odom_body}, offset {tuple(odom_offset)})")
            self.odom_sub = odom_source
        elif root_source == "gt" and not hasattr(self.env, "gt_root"):
            raise ValueError("--root_source gt is only available with --mock")
        print(f"[root] source={root_source}")

        # Reference-root anchoring, same as the sim controller: on each new motion the
        # world origin is re-latched so motion frame 0 coincides with the robot root.
        self._motion_epoch = None
        self._odom_origin_xy = None
        self._ref_origin_xy = None
        self._idle_anchor_xy = None

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.smooth_action = smooth_action
        if smooth_action > 0.0:
            self.action_smoother = EMASmoother(alpha=smooth_action)
            print(f"Output action smoothing enabled with alpha={smooth_action}")
        else:
            self.action_smoother = None

        # Real robot starts at a low PD gain -- operator ramps up via 2..9 / X.
        self.safety = SafetyController(initial_scale=initial_gain_scale)
        self.max_tilt = float(max_tilt)

        self.record_proprio = record_proprio
        self.proprio_recordings = [] if record_proprio else None
        self.log_dir = log_dir

    # ------------------------------------------------------------------ robot sequence
    def reset_robot(self):
        print("Press START on remote to move to default position ...")
        self.env.move_to_default_pos()
        print("Now in default position, press A to start the policy ...")
        self.env.default_pos_state()

    def _damping(self, until_select=True, max_time=None):
        """Zero stiffness, nominal damping: the robot goes limp slowly (keep the hoist on).
        Held until Select (or max_time seconds)."""
        print("[safety] DAMPING. " + ("Press Select / Ctrl+C to exit." if until_select else ""))
        hold = self.env.get_robot_state()[0]
        t0 = time.time()
        steps = 0
        try:
            while True:
                self.env.send_robot_action(hold, 0.0, 1.0)
                steps += 1
                if until_select and self.env.read_controller_input().keys == self.env.controller_mapping["select"]:
                    break
                if max_time is not None and steps * self.control_dt >= max_time:
                    break
                if self.realtime:
                    time.sleep(max(0.0, t0 + steps * self.control_dt - time.time()))
        except KeyboardInterrupt:
            pass

    # ------------------------------------------------------------------ root estimation
    def _fk(self, root_pos, root_quat, dof_pos):
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_quat
        self.data.qpos[7:7 + self.num_actions] = dof_pos  # SDK order matches XML
        mujoco.mj_kinematics(self.model, self.data)

    def _body_frame_kinematics(self, dof_pos, imu_quat):
        """FK with the pelvis at the origin, oriented by the IMU. Returns the pelvis
        height above the lowest foot contact, and the odom sensor pose in the pelvis
        frame (depends on the waist joints)."""
        self._fk(np.zeros(3), imu_quat, dof_pos)
        z_feet = self.data.geom_xpos[self._foot_geom_ids, 2] - self._foot_geom_r
        height = -float(z_feet.min())
        b_pos = self.data.xpos[self._odom_body_id].copy()
        b_quat = self.data.xquat[self._odom_body_id].copy()
        s_pos, s_quat = _pose_mul(b_pos, b_quat, self._odom_off_pos, self._odom_off_quat)
        # world(origin, imu) -> pelvis frame
        qi = quat_conj(np.asarray(imu_quat, dtype=np.float64))
        return height, (quat_apply(qi, s_pos), quat_mul(qi, s_quat))

    def _estimate_root(self, dof_pos, imu_quat, rpy, ref_root, ref_epoch):
        """Robot world root (pos, quat wxyz) + the anchored reference root xy; drives
        self.data FK for the obs terms."""
        imu_quat = np.asarray(imu_quat, dtype=np.float64)
        height, (p_ps, q_ps) = self._body_frame_kinematics(dof_pos, imu_quat)

        if self.root_source == "kinematic":
            ref_root_xy_w = self._compute_ref_root_xy_w(self._kin_xy, ref_root, ref_epoch)
            self._kin_xy = np.asarray(ref_root_xy_w, dtype=np.float64).copy()
            root_pos = np.array([self._kin_xy[0], self._kin_xy[1], height])
            root_quat = imu_quat
        elif self.root_source == "gt":
            root_pos, root_quat = self.env.gt_root()
            ref_root_xy_w = self._compute_ref_root_xy_w(root_pos[:2], ref_root, ref_epoch)
        else:
            odom = self.odom_sub.get_pose_with_age()
            if odom is None:
                raise SafetyStop("no odometry received")
            o_pos, o_quat, age = odom
            if age > self.odom_timeout:
                raise SafetyStop(f"odometry stale ({age * 1e3:.0f} ms > {self.odom_timeout * 1e3:.0f} ms)")
            # sensor pose in odom frame -> pelvis pose in odom frame
            p_sp, q_sp = _pose_inv(p_ps, q_ps)
            p_op, q_op = _pose_mul(o_pos, o_quat, p_sp, q_sp)
            if self._odom_cal is None:
                # FAST-LIO's frame is the sensor start pose (neither gravity- nor
                # heading-aligned): map it once so the pelvis starts at (0, 0, height)
                # with the IMU orientation.
                p_inv, q_inv = _pose_inv(p_op, q_op)
                self._odom_cal = _pose_mul(np.array([0.0, 0.0, height]), imu_quat, p_inv, q_inv)
                print(f"[odom] calibrated: pelvis height {height:.3f} m, IMU yaw {rpy[2]:+.3f}")
            p_w, q_w = _pose_mul(*self._odom_cal, p_op, q_op)
            root_pos = p_w
            # heading from odom (drift-free), roll/pitch from the IMU
            root_quat = rpy_to_quat(float(rpy[0]), float(rpy[1]), yaw_from_quat(q_w))
            ref_root_xy_w = self._compute_ref_root_xy_w(root_pos[:2], ref_root, ref_epoch)

        self._fk(root_pos, root_quat, dof_pos)
        self._root_yaw = yaw_from_quat(root_quat)
        return np.asarray(root_pos, dtype=np.float64), np.asarray(root_quat, dtype=np.float64), ref_root_xy_w

    def _read_ref_root_redis(self):
        """Motion server's published reference root xyz + per-motion epoch from Redis."""
        try:
            raw_epoch = self.redis_client.get(f"motion_epoch_{self.robot}")
            raw_ref = self.redis_client.get(f"ref_root_world_{self.robot}")
        except Exception:
            return None, None
        ref_root = None
        if raw_ref is not None:
            try:
                ref_root = np.asarray(json.loads(raw_ref), dtype=np.float64)
            except Exception:
                ref_root = None
        epoch = raw_epoch.decode() if isinstance(raw_epoch, bytes) else raw_epoch
        return ref_root, epoch

    def _compute_ref_root_xy_w(self, robot_root_xy, ref_root, epoch):
        """Reference motion root xy in the robot world frame (same logic as the sim
        controller): re-anchored to the robot root at each new motion, a world-fixed
        idle anchor while no reference root is published."""
        ref_xy = None if ref_root is None else np.asarray(ref_root, dtype=np.float64)[:2]

        if epoch != self._motion_epoch:
            self._motion_epoch = epoch
            self._odom_origin_xy = None
            self._ref_origin_xy = None

        if ref_xy is None:
            robot_xy = np.asarray(robot_root_xy, dtype=np.float64)
            if (self._idle_anchor_xy is None
                    or np.linalg.norm(robot_xy - self._idle_anchor_xy) > 0.5):
                self._idle_anchor_xy = robot_xy.copy()
            return self._idle_anchor_xy.copy()
        self._idle_anchor_xy = None

        if self._odom_origin_xy is None:
            self._odom_origin_xy = np.asarray(robot_root_xy, dtype=np.float64).copy()
            self._ref_origin_xy = ref_xy.copy()
        return ref_xy - self._ref_origin_xy + self._odom_origin_xy

    def _anchor_delta_xy(self):
        """xy translation mapping the published motion frame -> robot world frame."""
        if self._odom_origin_xy is not None and self._ref_origin_xy is not None:
            return self._odom_origin_xy - self._ref_origin_xy
        return None

    # ------------------------------------------------------------------ observation
    def compute_observation(self, dof_pos, dof_vel, ang_vel, rpy, action_mimic,
                            ref_root_xy_w=None, future_raw=None):
        """Build the flat observation tensor in policy-spec order (self.data already
        holds the FK of the estimated root + measured joints)."""
        odom_on = ref_root_xy_w is not None
        ctx = ObsContext(
            static=self.obs_static,
            data=self.data,
            dof_pos_isaac=dof_pos[self.sdk_to_isaac],
            dof_vel_isaac=dof_vel[self.sdk_to_isaac],
            ang_vel=ang_vel,
            rpy=rpy,
            last_action=self.last_action,
            action_mimic=action_mimic,
            ref_data=self.ref_data,
            future_ref_data=self.future_ref_data,
            future_raw=future_raw,
            ref_root_xy_w=ref_root_xy_w,
            odom_on=odom_on,
            anchor_delta_xy=self._anchor_delta_xy() if odom_on else None,
        )
        return self.obs_builder(ctx)

    # ------------------------------------------------------------------ reference
    def _reference_local(self, policy_step, b_edge):
        """Advance the in-process motion player by one step."""
        p = self.motion_player
        auto = (self.motion_start_delay >= 0 and not p.started
                and policy_step * self.control_dt >= self.motion_start_delay)
        if (b_edge and (not p.started or p.done)) or auto:
            self._post_motion_steps = 0
            p.start(self._root_yaw)
        s = p.step()
        future_raw = s.future_raw if self.spec.needs_future else None
        return s.action_mimic, future_raw, s.ref_root, s.epoch, s.phase, np.zeros(7), np.zeros(7)

    def _reference_redis(self, keys_pressed, dof_pos, ang_vel, rpy):
        """Publish robot state / remote signals and read the motion server's targets."""
        m = self.env.controller_mapping
        pipe = self.redis_pipeline
        pipe.set("motion_start_signal", "1" if keys_pressed == m["B"] else "0")
        pipe.set("motion_exit_signal", "1" if keys_pressed == m["select"] else "0")
        state_body = np.concatenate([ang_vel, rpy[:2], dof_pos])  # 3+2+29 = 34, SDK order
        pipe.set("state_body_unitree_g1_with_hands", json.dumps(state_body.tolist()))
        if self.use_hand:
            lh, rh = self.hand_ctrl.get_hand_state()
        else:
            lh, rh = np.zeros(7), np.zeros(7)
        pipe.set("state_hand_left_unitree_g1_with_hands", json.dumps(np.asarray(lh).tolist()))
        pipe.set("state_hand_right_unitree_g1_with_hands", json.dumps(np.asarray(rh).tolist()))
        pipe.set("state_neck_unitree_g1_with_hands", json.dumps(np.zeros(2).tolist()))
        # motion server reads the heading once at playback start to anchor frame 0
        pipe.set("state_heading_unitree_g1_with_hands",
                 json.dumps(self._root_yaw if self._root_yaw is not None else float(rpy[2])))
        pipe.set("t_state", int(time.time() * 1000))
        pipe.execute()

        keys = [
            "action_body_unitree_g1_with_hands",
            "action_hand_left_unitree_g1_with_hands",
            "action_hand_right_unitree_g1_with_hands",
        ]
        future_idx = len(keys)
        if self.spec.needs_future:
            keys += [f"{k}_{self.robot}" for k in cfg.FUTURE_MOTION_KEYS]
        phase_idx = len(keys)
        keys.append(f"{cfg.MOTION_PHASE_KEY}_{self.robot}")
        for key in keys:
            pipe.get(key)
        res = pipe.execute()
        action_mimic = np.asarray(json.loads(res[0]), dtype=np.float32)
        hand_l = np.asarray(json.loads(res[1]), dtype=np.float32)
        hand_r = np.asarray(json.loads(res[2]), dtype=np.float32)
        future_raw = None
        if self.spec.needs_future:
            fr = res[future_idx:future_idx + len(cfg.FUTURE_MOTION_KEYS)]
            future_raw = parse_future_raw(fr[0], fr[1], fr[2], self.spec.future_steps, self.num_actions,
                                          raw_lin_vel=fr[3], raw_ang_vel=fr[4])
        phase = res[phase_idx].decode().upper() if res[phase_idx] else "IDLE"
        ref_root, epoch = self._read_ref_root_redis()
        return action_mimic, future_raw, ref_root, epoch, phase, hand_l, hand_r

    # ------------------------------------------------------------------ main loop
    def run(self):
        self.reset_robot()
        print("Begin main TWIST2 policy loop. "
              + ("Press [B] to play the motion, " if self.motion_player is not None else "")
              + "[Select] to exit.")

        if self.redis_pipeline is not None:
            # Idle-default seed for action_* keys + drop stale future frames.
            default_mimic_obs = DEFAULT_MIMIC_OBS["unitree_g1_with_hands"]
            self.redis_pipeline.set("action_body_unitree_g1_with_hands", json.dumps(default_mimic_obs.tolist()))
            self.redis_pipeline.set("action_hand_left_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
            self.redis_pipeline.set("action_hand_right_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
            self.redis_pipeline.set("action_neck_unitree_g1_with_hands", json.dumps(np.zeros(2).tolist()))
            self.redis_pipeline.execute()
            if self.spec.needs_future:
                self.redis_client.delete(*[f"{k}_{self.robot}" for k in cfg.FUTURE_MOTION_KEYS])

        self.obs_builder.reset()
        policy_intervals = []
        overruns = 0
        last_policy_time = None
        key_listener = StdinKeyListener(self.safety.handle_keycode)
        key_listener.start()
        mapping = self.env.controller_mapping
        stop_reason = None

        loop_t0 = time.time()
        step = 0
        try:
            while True:
                self.safety.drain()
                keys_pressed = self.env.read_controller_input().keys
                if keys_pressed == mapping["select"]:
                    print("Select pressed, exiting main loop.")
                    break
                b_pressed = keys_pressed == mapping["B"]
                b_edge = b_pressed and not self._b_prev
                self._b_prev = b_pressed

                dof_pos, dof_vel, quat, ang_vel, dof_temp, dof_tau, dof_vol = self.env.get_robot_state()
                rpy = quatToEuler(quat)
                if self._root_yaw is None:
                    self._root_yaw = yaw_from_quat(quat)
                if max(abs(rpy[0]), abs(rpy[1])) > self.max_tilt:
                    raise SafetyStop(f"tilt roll={rpy[0]:+.2f} pitch={rpy[1]:+.2f} > {self.max_tilt:.2f} rad")

                if self.motion_player is not None:
                    action_mimic, future_raw, ref_root, ref_epoch, phase, hand_l, hand_r = \
                        self._reference_local(step, b_edge)
                    if self.motion_player.done and self.exit_after_motion >= 0:
                        self._post_motion_steps += 1
                        if self._post_motion_steps * self.control_dt > self.exit_after_motion:
                            print("[motion] playback finished; exiting.")
                            break
                else:
                    action_mimic, future_raw, ref_root, ref_epoch, phase, hand_l, hand_r = \
                        self._reference_redis(keys_pressed, dof_pos, ang_vel, rpy)

                root_pos, root_quat, ref_root_xy_w = self._estimate_root(dof_pos, quat, rpy, ref_root, ref_epoch)

                obs_buf = self.compute_observation(
                    dof_pos, dof_vel, ang_vel, rpy, action_mimic,
                    ref_root_xy_w=ref_root_xy_w, future_raw=future_raw,
                )
                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                if not np.all(np.isfinite(raw_action)):
                    raise SafetyStop("non-finite policy action")

                # raw_action is in Isaac order; permute back to SDK before sending.
                # last_action stays the RAW policy output (obs consistency).
                self.last_action = raw_action
                raw_action = np.clip(raw_action, -10.0, 10.0)
                if self.action_smoother is not None:
                    raw_action = self.action_smoother.smooth(raw_action)
                pd_target_isaac = raw_action * self.action_scale + self.default_dof_pos_isaac
                target_dof_pos = pd_target_isaac[self.isaac_to_sdk]
                self.env.send_robot_action(target_dof_pos, self.safety.kp_scale, self.safety.kd_scale)
                if self.use_hand:
                    self.hand_ctrl.ctrl_dual_hand(hand_l, hand_r)

                now = time.time()
                if last_policy_time is not None:
                    policy_intervals.append(now - last_policy_time)
                    if len(policy_intervals) % 100 == 0:
                        avg = float(np.mean(policy_intervals[-100:]))
                        print(f"Policy FPS (last 100): {1.0 / avg:.2f} Hz ({avg * 1e3:.2f} ms), "
                              f"overruns {overruns}, phase {phase}, root "
                              f"[{root_pos[0]:+.2f} {root_pos[1]:+.2f} {root_pos[2]:.3f}] yaw {self._root_yaw:+.2f}")
                last_policy_time = now

                if self.record_proprio:
                    rec = {
                        'timestamp': now,
                        'phase': phase,
                        'body_dof_pos': dof_pos.tolist(),
                        'body_dof_vel': dof_vel.tolist(),
                        'rpy': rpy.tolist(),
                        'ang_vel': np.asarray(ang_vel).tolist(),
                        'root_pos': root_pos.tolist(),
                        'root_quat': root_quat.tolist(),
                        'action_mimic': action_mimic.tolist(),
                        'pd_target': target_dof_pos.tolist(),
                        'kp_scale': self.safety.kp_scale,
                        'temperature': dof_temp.tolist(),
                        'tau': dof_tau.tolist(),
                        'voltage': dof_vol.tolist(),
                    }
                    if hasattr(self.env, "gt_root"):
                        rec['gt_root_pos'] = self.env.gt_root()[0].tolist()
                    self.proprio_recordings.append(rec)

                step += 1
                if self.realtime:
                    # absolute schedule: one late tick does not shift the following ones
                    ahead = loop_t0 + step * self.control_dt - time.time()
                    if ahead > 0:
                        time.sleep(ahead)
                    elif ahead < -self.control_dt:
                        overruns += 1
                        loop_t0 = time.time() - step * self.control_dt
        except SafetyStop as e:
            stop_reason = str(e)
            print(f"[red][safety] STOP: {stop_reason}[/red]")
        except KeyboardInterrupt:
            print("Ctrl+C, exiting main loop.")
        except Exception as e:
            stop_reason = f"error: {e}"
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore the terminal first so the operator sees the cleanup output.
            key_listener.stop()
            if stop_reason is not None:
                self._damping(until_select=self.realtime, max_time=None if self.realtime else 1.0)
            else:
                self._damping(until_select=False, max_time=0.5)
            if self.record_proprio and self.proprio_recordings:
                filename = os.path.join(self.log_dir, f'twist2_real_recordings_{time.strftime("%Y%m%d_%H%M%S")}.json')
                os.makedirs(self.log_dir, exist_ok=True)
                with open(filename, 'w') as f:
                    json.dump(self.proprio_recordings, f)
                print(f"Proprioceptive recordings saved as {filename}")
            if self.odom_sub is not None:
                self.odom_sub.close()
            if self.use_hand:
                self.hand_ctrl.close()
            print("TWIST2 real controller finished.")
            self.env.close()


def main():
    parser = argparse.ArgumentParser(description='Run TWIST2 policy on the real G1 robot')
    parser.add_argument('--policy', type=str, required=True, help='Path to TWIST2 ONNX policy file')
    parser.add_argument('--obs_cfg', type=str, default=None,
                        help='Policy observation spec yaml. Default: <policy>.yaml next to the ONNX.')
    parser.add_argument('--config', type=str, default="robot_control/configs/g1.yaml",
                        help='Robot configuration (control_dt, motor indices)')
    parser.add_argument('--xml', type=str, default='../assets/g1/g1_sim2sim_29dof_deploy.xml',
                        help='MuJoCo XML for FK (same as sim2sim)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run policy on (cuda/cpu)')
    parser.add_argument('--net', type=str, default='eth0', help='Network interface connected to the robot')
    parser.add_argument('--use_hand', action='store_true', help='Enable Dex3-1 hand control')
    parser.add_argument('--record_proprio', action='store_true', help='Record proprioceptive data (json)')
    parser.add_argument('--log_dir', type=str, default=DEFAULT_LOG_DIR,
                        help='Where --record_proprio saves (default: <repo>/logs/real)')
    parser.add_argument('--smooth_action', type=float, default=0.0,
                        help="EMA alpha on the policy's output action (0 disables)")
    parser.add_argument('--kp_scale', type=float, default=1.0, help='Scale on cfg.STIFFNESS (motor kps)')
    parser.add_argument('--kd_scale', type=float, default=1.0, help='Scale on cfg.DAMPING (motor kds)')
    parser.add_argument('--initial_gain_scale', type=float, default=0.1,
                        help='PD gain scale when the policy starts (raise with keys 2..9 / X)')
    parser.add_argument('--max_tilt', type=float, default=3.0,
                        help='|roll| or |pitch| above this [rad] -> damping stop')
    # world root
    parser.add_argument('--root_source', choices=['odom', 'kinematic', 'gt'], default='odom',
                        help='Robot world root: FAST-LIO odometry, leg kinematics (no odom), '
                             'or MuJoCo GT (--mock only)')
    parser.add_argument('--odom_topic', type=str, default='/Odometry', help='nav_msgs/Odometry topic')
    parser.add_argument('--odom_body', type=str, default='torso_link',
                        help='Body the odometry sensor is mounted on')
    parser.add_argument('--odom_offset', type=float, nargs=6, default=list(MID360_OFFSET),
                        metavar=('X', 'Y', 'Z', 'ROLL', 'PITCH', 'YAW'),
                        help='Odometry frame pose in --odom_body (default: URDF mid360_joint)')
    parser.add_argument('--odom_timeout', type=float, default=0.3,
                        help='Odometry older than this [s] -> damping stop')
    # reference motion
    parser.add_argument('--motion_npz', type=str, default=None,
                        help='Precomputed motion (export_motion_ref.py) played in-process; '
                             'without it the Redis motion server is used')
    parser.add_argument('--motion_start_delay', type=float, default=-1.0,
                        help='With --motion_npz: auto start this many seconds after A (<0: wait for B)')
    parser.add_argument('--motion_blend_in_time', type=float, default=None,
                        help='With --motion_npz: override the exported blend-in time [s]')
    parser.add_argument('--exit_after_motion', type=float, default=-1.0,
                        help='With --motion_npz: exit this many seconds after the motion returned '
                             'to the default pose (<0: keep running, B replays)')
    # dry run without the robot
    parser.add_argument('--mock', action='store_true',
                        help='Run against a MuJoCo stand-in of the robot (no unitree_interface / ROS2)')
    parser.add_argument('--mock_buttons', type=str, default='start@0,A@3,B@5',
                        help='--mock: scripted remote buttons, name@sim_seconds')
    parser.add_argument('--mock_realtime', action='store_true',
                        help='--mock: pace the loop in real time (default: as fast as possible)')
    parser.add_argument('--mock_odom_rate', type=float, default=50.0, help='--mock: odometry rate [Hz]')
    args = parser.parse_args()

    for path, what in ((args.policy, "Policy"), (args.config, "Config"), (args.xml, "XML")):
        if not os.path.exists(path):
            print(f"Error: {what} file {path} does not exist")
            return

    print("Starting TWIST2 real robot controller" + (" [MOCK]" if args.mock else "") + "...")
    for k in ("policy", "obs_cfg", "config", "xml", "device", "net", "use_hand", "root_source",
              "motion_npz", "kp_scale", "kd_scale", "initial_gain_scale", "record_proprio"):
        print(f"  {k}: {getattr(args, k)}")

    env, odom_source = None, None
    if args.mock:
        from robot_control.mock_g1_env import MockG1Env, MockOdom
        cfg_mock = Config(args.config)
        cfg_mock.default_angles = cfg.DEFAULT_DOF_POS.astype(np.float32).copy()
        cfg_mock.kps = (cfg.STIFFNESS.astype(np.float32) * args.kp_scale).tolist()
        cfg_mock.kds = (cfg.DAMPING.astype(np.float32) * args.kd_scale).tolist()
        env = MockG1Env(args.xml, cfg_mock, args.mock_buttons, realtime=args.mock_realtime)
        if args.root_source == 'odom':
            odom_source = MockOdom(env, args.odom_body, args.odom_offset, rate=args.mock_odom_rate)
    else:
        print("\n" + "=" * 50)
        print("SAFETY WARNING: running a policy on the real robot.")
        print("Keep the robot on the hoist; Select = damping + exit, Ctrl+C = same.")
        print("=" * 50 + "\n")

    controller = RealTimePolicyController(
        xml_file=args.xml,
        policy_path=args.policy,
        config_path=args.config,
        obs_cfg=args.obs_cfg,
        device=args.device,
        net=args.net,
        use_hand=args.use_hand,
        record_proprio=args.record_proprio,
        log_dir=args.log_dir,
        smooth_action=args.smooth_action,
        kp_scale=args.kp_scale,
        kd_scale=args.kd_scale,
        root_source=args.root_source,
        odom_topic=args.odom_topic,
        odom_body=args.odom_body,
        odom_offset=args.odom_offset,
        odom_timeout=args.odom_timeout,
        motion_npz=args.motion_npz,
        motion_start_delay=args.motion_start_delay,
        motion_blend_in_time=args.motion_blend_in_time,
        exit_after_motion=args.exit_after_motion,
        initial_gain_scale=args.initial_gain_scale,
        max_tilt=args.max_tilt,
        env=env,
        odom_source=odom_source,
        realtime=not args.mock or args.mock_realtime,
    )
    controller.run()


if __name__ == "__main__":
    main()
