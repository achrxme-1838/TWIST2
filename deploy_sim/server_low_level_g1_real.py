#!/usr/bin/env python3
"""TWIST2 G1 controller: LIO odometry (livox IMU frame) + a global origin set at A.
  S (START)   move to the default pose, then hold it with PD (100%, robot on the hoist)
  (LIO)       start the LIO outside this script (sim: key L launches the emulator)
  A           set the global origin G at the current robot pose (refused only on stale odometry),
              start the policy (gain --initial_gain_scale, default 100%) on the stand motion
              (--stand_npz) anchored on the robot: blend -> play -> hold its last frame (STAND)
  B           after the stand / a motion has finished: play the motion (--motion_npz), frame 0 anchored
              on the robot's current xy / yaw in G: blend -> play -> return to the A1-Stand pose at the
              last frame's xy / yaw; the policy keeps running (MOTION). B again replays it.
  C (Select)  damping + exit
  2..9 / X    policy PD gain scale 20..90% / 100%
  P / H       push the robot / toggle the virtual hoist (sim only)

Robot root: odometry in G (xyz, yaw) + IMU roll / pitch. Reference: the playing motion, rotated by
(robot yaw - frame 0 yaw) and shifted so frame 0's point X (--origin_ref) lands on the robot's X.
"""
import argparse
import builtins
import json
import os
import time
from collections import defaultdict
from queue import SimpleQueue

import numpy as np
import mujoco
import torch
from rich import print
from rich.markup import escape

from data_utils.rot_utils import quatToEuler
from data_utils.params import DEFAULT_MIMIC_OBS
from motion_player import BLEND, IDLE, LocalMotionPlayer

from cfg import g1_29dof_cfg as cfg
from obs_builder import ObsBuilder
from obs_terms import ObsContext, StaticContext
from odom_estimator import (IMU_IN_PELVIS, LIVOX_IMU_IN_MID360, MID360_IN_TORSO, ODOM_FRAMES,
                            ORIGIN_REFS, FrameChain, OdomEstimator, roll_pitch, yaw_quat)
from policy_spec import resolve_spec
from safety import SafetyController, StdinKeyListener
from utils.math import rpy_to_quat, yaw_from_quat

from robot_control.config import Config

try:
    import onnxruntime as ort
except ImportError:
    ort = None

DEFAULT_LOG_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "real"))
PASSIVE, MOVING, HOLD = "PASSIVE", "MOVING", "HOLD"
STAND, MOTION = "STAND", "MOTION"
STATES = (PASSIVE, MOVING, HOLD, STAND, MOTION)
POLICY_STATES = (STAND, MOTION)
PHASES = ("-", "IDLE", "BLEND", "MOTION", "RETURN")
REMOTE_KEYS = (("start", "s"), ("A", "a"), ("B", "b"), ("select", "c"))
EVENT_KEYS = "sablcph"
DEFAULT_STAND_NPZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets",
                                                  "motions", "A1-Stand_poses.npz"))


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

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.smoothed_value = None

    def smooth(self, new_value):
        if self.smoothed_value is None:
            self.smoothed_value = new_value.copy()
        else:
            self.smoothed_value = self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
        return self.smoothed_value

    def reset(self):
        self.smoothed_value = None


class SafetyStop(Exception):
    """Raised inside the control loop to drop the robot into damping."""


class _Exit(Exception):
    """C / Select pressed."""


class ObsPipe:
    """FK MjData (robot root + measured joints) + reference MjData + ObsBuilder."""

    def __init__(self, model, spec, static):
        self.model = model
        self.data = mujoco.MjData(model)
        self.ref_data = mujoco.MjData(model) if spec.needs_ref_fk else None
        self.future_ref_data = mujoco.MjData(model) if spec.needs_future else None
        self.builder = ObsBuilder(spec, static)
        self.static = static
        self.spec = spec

    def reset(self):
        self.builder.reset()

    def __call__(self, root_pos, root_quat, dof_pos_isaac, dof_vel_isaac, dof_pos_sdk, ang_vel,
                 rpy, last_action, action_mimic, future_raw, ref_root_xy, anchor_delta_xy):
        d = self.data
        d.qpos[:3] = root_pos
        d.qpos[3:7] = root_quat
        d.qpos[7:7 + self.static.num_actions] = dof_pos_sdk
        mujoco.mj_kinematics(self.model, d)
        ctx = ObsContext(
            static=self.static, data=d,
            dof_pos_isaac=dof_pos_isaac, dof_vel_isaac=dof_vel_isaac,
            ang_vel=ang_vel, rpy=rpy, last_action=last_action, action_mimic=action_mimic,
            ref_data=self.ref_data, future_ref_data=self.future_ref_data,
            future_raw=future_raw if self.spec.needs_future else None,
            ref_root_xy_w=ref_root_xy, odom_on=True, anchor_delta_xy=anchor_delta_xy,
        )
        return self.builder(ctx)


class RealTimePolicyController:
    def __init__(self,
                 xml_file,
                 policy_path,
                 config_path,
                 motion_npz,
                 stand_npz=DEFAULT_STAND_NPZ,
                 obs_cfg=None,
                 device='cpu',
                 net='eth0',
                 kp_scale=1.0,
                 kd_scale=1.0,
                 initial_gain_scale=1.0,
                 smooth_action=0.0,
                 max_tilt=0.8,
                 move_time=2.0,
                 motion_blend_in_time=None,
                 # odometry
                 odom_topic="/dreamstep_odom_grav",
                 odom_frame="livox_imu",
                 mid360_in_torso=MID360_IN_TORSO,
                 livox_imu_in_mid360=LIVOX_IMU_IN_MID360,
                 imu_in_pelvis=IMU_IN_PELVIS,
                 odom_timeout=0.3,
                 odom_ma=1,
                 odom_sync="stamp",
                 # global origin
                 origin_ref="pelvis",
                 origin_window=0.5,
                 print_odom_hz=0.0,
                 log_dir=DEFAULT_LOG_DIR,
                 run_name=None,
                 robot="unitree_g1_with_hands",
                 env=None,
                 odom_source=None,
                 realtime=True,
                 keyboard=True,
                 ):
        self.robot = robot
        self.config = Config(config_path)
        # Policy was trained against cfg.DEFAULT_DOF_POS / STIFFNESS / DAMPING -- the PD hold
        # and the policy use the same gains.
        self.config.default_angles = cfg.DEFAULT_DOF_POS.astype(np.float32).copy()
        self.config.kps = (cfg.STIFFNESS.astype(np.float32) * kp_scale).tolist()
        self.config.kds = (cfg.DAMPING.astype(np.float32) * kd_scale).tolist()
        self.control_dt = float(self.config.control_dt)
        self.realtime = realtime
        self.keyboard = keyboard

        if env is None:
            from robot_control.g1_wrapper import G1RealWorldEnv  # needs unitree_interface
            env = G1RealWorldEnv(net=net, config=self.config)
        self.env = env

        self.device = device
        self.policy = load_onnx_policy(policy_path, device)

        self.num_actions = cfg.NUM_ACTIONS
        self.default_dof_pos = cfg.DEFAULT_DOF_POS.copy()
        self.action_scale = np.full(self.num_actions, cfg.ACTION_SCALE, dtype=np.float32)
        self.sdk_to_isaac = np.array([cfg.SDK_JOINT_NAMES.index(n) for n in cfg.ISAAC_JOINT_NAMES], dtype=np.int64)
        self.isaac_to_sdk = np.array([cfg.ISAAC_JOINT_NAMES.index(n) for n in cfg.SDK_JOINT_NAMES], dtype=np.int64)
        self.default_dof_pos_isaac = self.default_dof_pos[self.sdk_to_isaac]

        # ----- observation layout (policy spec) -----
        self.spec = resolve_spec(policy_path, obs_cfg)
        print(self.spec.describe())
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.tracked_body_ids = np.array([self.model.body(n).id for n in cfg.TRACKED_BODY_NAMES], dtype=np.int64)
        self.obs_static = StaticContext(
            model=self.model,
            num_actions=self.num_actions,
            tracked_body_ids=self.tracked_body_ids,
            extended_parent_ids=np.array([self.model.body(p).id for _, p, _ in cfg.EXTENDED_JOINTS], dtype=np.int64),
            extended_local_offsets=np.array([o for _, _, o in cfg.EXTENDED_JOINTS], dtype=np.float64),
            default_dof_pos_isaac=self.default_dof_pos_isaac,
            isaac_joint_names=list(cfg.ISAAC_JOINT_NAMES),
            future_steps=self.spec.future_steps,
            future_fk_steps=max(self.spec.used_future_steps, 1),
        )
        self.obs_pipe = ObsPipe(self.model, self.spec, self.obs_static)
        print(self.obs_pipe.builder.describe())
        policy_dim = getattr(self.policy, "input_dim", None)
        if policy_dim is not None and policy_dim != self.obs_pipe.builder.obs_dim:
            raise ValueError(f"policy {policy_path} expects {policy_dim} obs but the spec builds "
                             f"{self.obs_pipe.builder.obs_dim} -- wrong --obs_cfg for this checkpoint?")
        # Optional second pipeline fed with a different root (sim: ground truth) to log
        # how much the odometry error changes the observation.
        self.shadow_root_fn = None
        self._shadow_pipe = None

        # ----- reference motion -----
        self.motion_npz, self.stand_npz = motion_npz, stand_npz
        fut = self.spec.future_motion_steps if self.spec.needs_future else None
        self.stand_player = LocalMotionPlayer(stand_npz, DEFAULT_MIMIC_OBS[self.robot], control_dt=self.control_dt,
                                              future_motion_steps=fut, blend_in_time=motion_blend_in_time,
                                              end_hold=True)
        # B ends by returning to the A1-Stand pose (the stand's held last frame) at the motion's
        # last root xy / heading (the player swaps in the last frame's yaw), and idles on it.
        self._idle_mimic = self.stand_player.last_frame()
        self.player = LocalMotionPlayer(motion_npz, self._idle_mimic, control_dt=self.control_dt,
                                        future_motion_steps=fut, blend_in_time=motion_blend_in_time)
        print(f"[motion] B: {motion_npz} ({self.player.num_steps} steps)")
        print(f"[motion] A (stand, last frame held): {stand_npz} ({self.stand_player.num_steps} steps)")
        self._default_mimic = np.asarray(DEFAULT_MIMIC_OBS[self.robot], dtype=np.float64).copy()
        self.active = None                 # player of the current reference (stand / motion)

        # ----- odometry + global origin -----
        self.origin_ref = origin_ref
        self.chain = FrameChain(self.model, odom_frame, mid360_in_torso, livox_imu_in_mid360, imu_in_pelvis)
        self.est = OdomEstimator(self.chain, origin_ref=origin_ref, timeout=odom_timeout, ma_len=odom_ma,
                                 sync=odom_sync, origin_window=origin_window)
        if odom_source is None:
            from ros_odom import OdomSubscriber
            odom_source = OdomSubscriber(odom_topic)
            print(f"[odom] subscribing to '{odom_topic}'")
        self.odom = odom_source
        print(f"[odom] frame={odom_frame} ma={odom_ma} sync={odom_sync} timeout={odom_timeout}s; "
              f"origin_ref={origin_ref}")
        self.print_odom_period = 1.0 / print_odom_hz if print_odom_hz > 0 else None
        self._last_odom_print = -np.inf
        self.root_override = None          # sim: callable -> (pos, quat) in G (ground truth)

        # ----- control state -----
        self.initial_gain_scale = float(initial_gain_scale)
        self.safety = SafetyController(initial_scale=initial_gain_scale)
        self.max_tilt = float(max_tilt)
        self.move_steps = max(1, int(round(move_time / self.control_dt)))
        self.smooth_action = smooth_action
        self.action_smoother = EMASmoother(alpha=smooth_action) if smooth_action > 0.0 else None
        self.state = PASSIVE
        self.phase = "-"
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.anchor_delta = np.zeros(2)    # motion frame -> G xy shift of the active player
        self.ref_root_xy = None            # reference pelvis xy in G (this tick)
        self._last_mimic = None            # reference mimic obs (this tick)
        self._blend_from_xy = np.zeros(2)  # reference pelvis xy at the start of the blend
        self._frame0_xy = np.zeros(2)      # anchored frame-0 pelvis xy (blend target)
        self._done_said = False
        self._prev_obs = None
        self.last_root = None              # (pos, quat) in G used by the policy this tick
        self.status = ""                   # last event result (for the sim recorder overlay)
        self._move_from = None
        self._move_k = 0
        self._events = SimpleQueue()
        self._remote_prev = {name: False for name, _ in REMOTE_KEYS}

        # ----- logging / hooks -----
        self.tick_hooks = []               # callables(ctrl) after every control tick (sim monitor/recorder)
        self.tick_extra = {}               # extra per-tick log fields filled by hooks
        self.log_dir = log_dir
        self.run_name = run_name or time.strftime("%Y%m%d_%H%M%S")
        self._log = defaultdict(list)
        self.event_log = []

    # ------------------------------------------------------------------ input
    def _now(self):
        return self.env.now() if hasattr(self.env, "now") else time.time()

    def _on_key(self, keycode):
        """Keyboard thread: S/A/B/C (+ sim L/P/H) go to the event queue, digits/X to the gain control."""
        ch = chr(keycode).lower() if 0 <= keycode < 0x110000 else ""
        if ch and ch in EVENT_KEYS:
            self._events.put(ch)
        else:
            self.safety.handle_keycode(keycode)

    def _gather_events(self):
        events = []
        while not self._events.empty():
            events.append(self._events.get_nowait())
        if hasattr(self.env, "poll_events"):              # sim: scripted keys
            for ch in self.env.poll_events():
                if ch in EVENT_KEYS:
                    events.append(ch)
                else:
                    self.safety.handle_keycode(ord(ch))
        if hasattr(self.env, "read_controller_input"):    # real: remote edges
            keys = int(self.env.read_controller_input().keys)
            mapping = self.env.controller_mapping
            for name, ch in REMOTE_KEYS:
                pressed = bool(keys & mapping[name])
                if pressed and not self._remote_prev[name]:
                    events.append(ch)
                self._remote_prev[name] = pressed
        return events

    def _say(self, now, ev, msg, color=None):
        self.status = msg
        self.event_log.append((float(now), ev, msg))
        text = escape(msg)                  # "[hoist] ..." etc. are not rich markup
        print(f"[{color}]{text}[/{color}]" if color else text)

    # ------------------------------------------------------------------ events
    def _handle(self, ev, dof_pos, quat, now):
        if ev == "c":
            raise _Exit()
        if ev == "p":
            if hasattr(self.env, "push"):
                self._say(now, ev, f"[push] {self.env.push()}")
            return
        if ev == "h":
            if hasattr(self.env, "toggle_hoist"):
                self._say(now, ev, f"[hoist] {self.env.toggle_hoist()}")
            return
        if ev == "s":
            if self.state == PASSIVE:
                self._move_from = np.asarray(dof_pos, dtype=np.float64).copy()
                self._move_k = 0
                self.state = MOVING
                self._say(now, ev, f"[S] moving to the default pose ({self.move_steps * self.control_dt:.1f}s)", "green")
            else:
                self._say(now, ev, f"[S] ignored in {self.state}")
            return
        if ev == "l":
            if hasattr(self.odom, "launch"):                  # sim: LIO emulator
                if getattr(self.odom, "world_from_W", None) is not None:
                    self._say(now, ev, "[L] LIO already running", "yellow")
                else:
                    self.odom.launch()
                    self._say(now, ev, f"[L] LIO launched at t={now:.2f}s", "green")
            else:
                self._say(now, ev, "[L] ignored: the LIO runs outside this script", "yellow")
            return
        if ev == "a":
            if self.state != HOLD:
                self._say(now, ev, f"[A] ignored in {self.state} (starts the policy from the PD hold)", "yellow")
                return
            ok, msg = self.est.reset_origin(now, dof_pos, quat)
            if not ok:
                self._say(now, ev, f"[A] refused: {msg}", "red")
                return
            self._say(now, ev, f"[A] {msg}", "green")
            self._start_policy(now)
            return
        if ev == "b":
            if self.state not in POLICY_STATES:
                self._say(now, ev, f"[B] ignored in {self.state} -- press A first", "yellow")
            elif not self.active.done:
                self._say(now, ev, f"[B] ignored: {self.state} still playing ({self.phase})", "yellow")
            elif not self.est.fresh(now):
                self._say(now, ev, f"[B] ignored: odometry stale ({self.est.age(now) * 1e3:.0f} ms)", "red")
            else:
                self._start_motion(now)

    def _play(self, player, blend_from, blend_from_xy):
        """Start a player with its frame 0 anchored on the robot: heading -> robot yaw in G,
        frame 0's point X -> robot X xy in G. The blend starts from (blend_from, blend_from_xy)."""
        xy, yaw = self.est.point_in_G()
        player.start(yaw, blend_from=blend_from)
        p0, rpy0 = player.frame0_root()
        off = self.chain.point_offset_xy(rpy_to_quat(*rpy0), self.origin_ref)
        self.anchor_delta = np.asarray(xy, dtype=np.float64) - (np.asarray(p0[:2], dtype=np.float64) + off)
        self._frame0_xy = np.asarray(p0[:2], dtype=np.float64) + self.anchor_delta
        self._blend_from_xy = np.asarray(blend_from_xy, dtype=np.float64).copy()
        self.active = player
        self._done_said = False
        return xy, yaw

    def _start_policy(self, now):
        """A (origin just set): fresh policy (obs history refilled on the first tick) on the
        stand motion, blended in from the default stand at the robot's pose."""
        self.obs_pipe.reset()
        if self._shadow_pipe is not None:
            self._shadow_pipe.reset()
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        if self.action_smoother is not None:
            self.action_smoother.reset()
        self._prev_obs = None
        self.safety.kp_scale = self.safety.kd_scale = self.initial_gain_scale
        xy, yaw = self.est.point_in_G()
        start = self._default_mimic.copy()
        start[3] = start[4] = 0.0
        start[5] = yaw
        start_xy = np.asarray(xy, dtype=np.float64) - self.chain.point_offset_xy(yaw_quat(yaw), self.origin_ref)
        self._play(self.stand_player, start, start_xy)
        self.state = STAND
        self._say(now, "a", f"[A] policy start (gain {self.initial_gain_scale * 100:.0f}%) on the stand motion "
                            f"({self.stand_player.n_blend} blend + {self.stand_player.num_steps} steps, then held)",
                  "green")
        if hasattr(self.env, "on_policy_start"):
            m = self.env.on_policy_start()
            if m:
                self._say(now, "h", f"[hoist] {m}")

    def _start_motion(self, now):
        """B: the motion's frame 0 anchored on the robot's current xy / yaw in G; blended in
        from the reference tracked so far. The obs history is kept."""
        xy, yaw = self._play(self.player, self._last_mimic, self.ref_root_xy)
        self.state = MOTION
        self._say(now, "b", f"[B] motion start at the robot (xy {xy[0]:+.3f} {xy[1]:+.3f}, yaw "
                            f"{np.rad2deg(yaw):+.1f} deg; anchor {self.anchor_delta[0]:+.3f} "
                            f"{self.anchor_delta[1]:+.3f})", "green")

    # ------------------------------------------------------------------ control
    def _send(self, target):
        """Policy PD target at the operator gain (keys 2..9 / X)."""
        self.env.send_robot_action(target, self.safety.kp_scale, self.safety.kd_scale)

    def _send_pd(self, target):
        """Default-pose move / hold: full gain (as the stock G1 default-pose state)."""
        self.env.send_robot_action(target, 1.0, 1.0)

    def _ref_xy(self, sample):
        """Reference pelvis xy in G: during the blend from the previous reference to the anchored
        frame 0; afterwards the anchored motion root (the last one while returning / idle)."""
        if sample.phase == BLEND or sample.ref_root is None:
            a = self.active.blend_alpha if sample.phase == BLEND else 1.0
            return self._blend_from_xy + (self._frame0_xy - self._blend_from_xy) * a
        return np.asarray(sample.ref_root[:2], dtype=np.float64) + self.anchor_delta

    def _policy_root(self, quat, now):
        if self.root_override is not None:
            return self.root_override()
        if not self.est.fresh(now):
            raise SafetyStop(f"odometry stale ({self.est.age(now) * 1e3:.0f} ms)")
        if self.est.jump:
            raise SafetyStop(self.est.jump)
        root = self.est.root_in_G(quat)
        if root is None:
            raise SafetyStop("no root estimate")
        return root

    def _reference(self, quat, now):
        """-> (robot root (pos, quat), action_mimic, future_raw, reference pelvis xy, anchor delta)."""
        root = self._policy_root(quat, now)
        sample = self.active.step()
        self.phase = sample.phase
        return root, sample.action_mimic, sample.future_raw, self._ref_xy(sample), self.anchor_delta

    def _policy_tick(self, dof_pos, dof_vel, quat, ang_vel, rpy, now):
        if max(abs(rpy[0]), abs(rpy[1])) > self.max_tilt:
            raise SafetyStop(f"tilt roll={rpy[0]:+.2f} pitch={rpy[1]:+.2f} > {self.max_tilt:.2f} rad")
        (root_pos, root_quat), action_mimic, future_raw, ref_xy, anchor_delta = self._reference(quat, now)
        self.ref_root_xy = ref_xy
        self._last_mimic = np.asarray(action_mimic, dtype=np.float64).copy()
        self.last_root = (root_pos, root_quat)
        rpy_obs = np.array([rpy[0], rpy[1], yaw_from_quat(root_quat)])
        args = (np.asarray(dof_pos)[self.sdk_to_isaac], np.asarray(dof_vel)[self.sdk_to_isaac], dof_pos,
                ang_vel, rpy_obs, self.last_action, action_mimic, future_raw, ref_xy, anchor_delta)
        obs = self.obs_pipe(root_pos, root_quat, *args)
        slices = self.obs_pipe.builder.slices().values()
        if self._prev_obs is not None:         # per-term max |obs(t) - obs(t-1)| (A / B continuity)
            self.tick_extra["obs_step"] = np.array([np.max(np.abs(obs[s] - self._prev_obs[s])) for s in slices])
        self._prev_obs = obs.copy()
        if self.shadow_root_fn is not None:
            if self._shadow_pipe is None:
                self._shadow_pipe = ObsPipe(self.model, self.spec, self.obs_static)
            s_pos, s_quat = self.shadow_root_fn()
            obs_s = self._shadow_pipe(s_pos, s_quat, *args)
            self.tick_extra["obs_err"] = np.array([np.max(np.abs(obs[s] - obs_s[s])) for s in slices])
        with torch.no_grad():
            raw_action = self.policy(torch.from_numpy(obs).float().unsqueeze(0)).cpu().numpy().squeeze()
        if not np.all(np.isfinite(raw_action)):
            raise SafetyStop("non-finite policy action")
        self.last_action = raw_action               # RAW policy output (obs consistency)
        raw_action = np.clip(raw_action, -10.0, 10.0)
        if self.action_smoother is not None:
            raw_action = self.action_smoother.smooth(raw_action)
        target = (raw_action * self.action_scale + self.default_dof_pos_isaac)[self.isaac_to_sdk]
        self._send(target)
        if self.active.done and not self._done_said:
            self._done_said = True
            if self.state == STAND:
                self._say(now, "", "[stand] finished -> holding its last frame; B: play the motion", "green")
            else:
                self._say(now, "", f"[motion] finished -> policy holds the A1-Stand pose at the last frame "
                                   f"(xy {ref_xy[0]:+.2f} {ref_xy[1]:+.2f}, yaw "
                                   f"{np.rad2deg(float(action_mimic[5])):+.1f} deg); B: play again", "green")
        return target

    def _damping(self, max_time=None):
        """Zero stiffness, nominal damping (the robot goes limp slowly -- keep the hoist on).
        Until C / Select, or max_time seconds."""
        print("[safety] DAMPING. " + ("Press C / Select to exit." if max_time is None else ""))
        hold = np.asarray(self.env.get_robot_state()[0]).copy()
        t0 = time.time()
        steps = 0
        try:
            while True:
                self.env.send_robot_action(hold, 0.0, 1.0)
                for h in self.tick_hooks:
                    h(self)
                steps += 1
                if max_time is not None and steps * self.control_dt >= max_time:
                    break
                if max_time is None and "c" in self._gather_events():
                    break
                if self.realtime:
                    time.sleep(max(0.0, t0 + steps * self.control_dt - time.time()))
        except KeyboardInterrupt:
            pass

    # ------------------------------------------------------------------ logging
    def _record(self, now, rpy, dof_pos, target):
        L = self._log
        nan3 = np.full(3, np.nan)
        L["t"].append(now)
        L["state"].append(STATES.index(self.state))
        L["phase"].append(PHASES.index(self.phase) if self.phase in PHASES else 0)
        L["gain"].append(self.safety.kp_scale)
        L["odom_age"].append(min(self.est.age(now), 1e3))
        L["odom_msgs"].append(self.est.n_msgs)
        L["origin_count"].append(self.est.origin_count)
        L["imu_rpy"].append(np.asarray(rpy, dtype=np.float64))
        T = self.est.T_W_P
        L["est_W_pos"].append(T[0] if T is not None else nan3)
        L["est_W_yaw"].append(yaw_from_quat(T[1]) if T is not None else np.nan)
        G = self.est.pelvis_in_G()
        L["est_G_pos"].append(G[0] if G is not None else nan3)
        L["est_G_yaw"].append(yaw_from_quat(G[1]) if G is not None else np.nan)
        on = self.state in POLICY_STATES and self.last_root is not None
        L["root_pos"].append(self.last_root[0] if on else nan3)
        L["root_yaw"].append(yaw_from_quat(self.last_root[1]) if on else np.nan)
        L["dof_pos"].append(np.asarray(dof_pos, dtype=np.float64))
        L["target"].append(np.asarray(target, dtype=np.float64) if target is not None
                           else np.full(self.num_actions, np.nan))
        n = len(L["t"])
        for k, v in self.tick_extra.items():
            v = np.asarray(v, dtype=np.float64)
            if len(L[k]) < n - 1:      # field first seen now: back-fill earlier ticks
                L[k].extend([np.full_like(v, np.nan)] * (n - 1 - len(L[k])))
            L[k].append(v)
        # fields not refreshed next tick are logged as NaN
        self.tick_extra = {k: np.full_like(np.asarray(v, dtype=np.float64), np.nan) for k, v in self.tick_extra.items()}

    def save_log(self, extra_meta=None):
        if not self._log["t"]:
            return None
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, f"odom_{self.run_name}.npz")
        n = len(self._log["t"])
        arrays = {k: np.asarray(v) for k, v in self._log.items() if len(v) == n}
        meta = dict(states=STATES, phases=PHASES, origin_ref=self.origin_ref,
                    odom_frame=self.chain.odom_frame, control_dt=self.control_dt,
                    motion_npz=self.motion_npz, obs_terms=list(self.obs_pipe.builder.slices().keys()),
                    events=self.event_log, **(extra_meta or {}))
        np.savez(path, meta=np.array(json.dumps(meta, default=float)), **arrays)
        print(f"[log] {n} ticks -> {path}")
        return path

    def _print_odom(self, now):
        if self.print_odom_period is None or now - self._last_odom_print < self.print_odom_period:
            return
        self._last_odom_print = now
        msg = self.odom.get_latest()
        if msg is None:
            builtins.print(f"[LIO] no message yet ({self.state})")
            return
        p, q = np.asarray(msg.pos, dtype=np.float64), np.asarray(msg.quat, dtype=np.float64)
        r, pi = roll_pitch(q)
        s = (f"[LIO] xyz [{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}] rpy [{np.rad2deg(r):+.1f} {np.rad2deg(pi):+.1f} "
             f"{np.rad2deg(yaw_from_quat(q)):+.1f}] deg  age {(self._now() - msg.recv_time) * 1e3:.0f} ms")
        G = self.est.pelvis_in_G()
        if G is not None:
            s += (f"  | G pelvis [{G[0][0]:+.3f} {G[0][1]:+.3f} {G[0][2]:+.3f}] "
                  f"yaw {np.rad2deg(yaw_from_quat(G[1])):+.1f}")
        builtins.print(s + f"  ({self.state})")

    # ------------------------------------------------------------------ main loop
    def run(self):
        listener = StdinKeyListener(self._on_key) if self.keyboard else None
        if listener is not None:
            listener.start()
        print("[green]Keys: S default pose  A origin + policy (stand)  B play motion  C exit  "
              "2..9/X policy gain  (sim: L launch LIO, P push, H hoist)[/green]")
        stop_reason = None
        loop_t0 = time.time()
        step = 0
        overruns = 0
        try:
            while True:
                self.safety.drain()
                now = self._now()
                dof_pos, dof_vel, quat, ang_vel = self.env.get_robot_state()[:4]
                quat = np.asarray(quat, dtype=np.float64)
                rpy = quatToEuler(quat)
                self.est.push_joints(now, dof_pos)
                self.est.update(self.odom.get_latest())
                for ev in self._gather_events():
                    self._handle(ev, dof_pos, quat, now)
                target = None
                if self.state == PASSIVE:
                    if hasattr(self.env, "passive_step"):
                        self.env.passive_step()
                elif self.state == MOVING:
                    a = min(1.0, (self._move_k + 1) / self.move_steps)
                    target = self._move_from * (1 - a) + self.default_dof_pos * a
                    self._send_pd(target)
                    self._move_k += 1
                    if self._move_k >= self.move_steps:
                        self.state = HOLD
                        self._say(now, "", "[S] default pose reached -> PD hold "
                                           "(gain 100%); A: start the policy", "green")
                elif self.state == HOLD:
                    target = self.default_dof_pos.copy()
                    self._send_pd(target)
                else:
                    target = self._policy_tick(dof_pos, dof_vel, quat, np.asarray(ang_vel), rpy, now)
                if self.state not in POLICY_STATES:
                    self.last_root, self.ref_root_xy = None, None
                self._print_odom(now)
                for h in self.tick_hooks:
                    h(self)
                self._record(now, rpy, dof_pos, target)

                step += 1
                if self.realtime:
                    ahead = loop_t0 + step * self.control_dt - time.time()
                    if ahead > 0:
                        time.sleep(ahead)
                    elif ahead < -self.control_dt:
                        overruns += 1
                        loop_t0 = time.time() - step * self.control_dt
        except _Exit:
            print("C / Select pressed, exiting.")
        except SafetyStop as e:
            stop_reason = str(e)
            self._say(self._now(), "stop", f"[safety] STOP: {stop_reason}", "red")
        except KeyboardInterrupt:
            print("Ctrl+C, exiting.")
        finally:
            if listener is not None:
                listener.stop()
            if stop_reason is not None:
                self._damping(max_time=None if self.realtime and self.keyboard else 1.0)
            else:
                self._damping(max_time=0.5)
            self.save_log()
            if hasattr(self.odom, "close"):
                self.odom.close()
            print(f"controller finished (overruns {overruns}).")
            self.env.close()
        return stop_reason


def add_controller_args(parser):
    parser.add_argument('--policy', type=str, required=True, help='ONNX policy')
    parser.add_argument('--motion_npz', type=str, required=True, help='Motion exported by export_motion_ref.py (B)')
    parser.add_argument('--stand_npz', type=str, default=DEFAULT_STAND_NPZ,
                        help='Stand motion played at A (last frame held until B)')
    parser.add_argument('--obs_cfg', type=str, default=None, help='Policy spec yaml (default: <policy>.yaml)')
    parser.add_argument('--config', type=str, default="robot_control/configs/g1.yaml")
    parser.add_argument('--xml', type=str, default='../assets/g1/g1_sim2sim_29dof_deploy.xml')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--kp_scale', type=float, default=1.0, help='Scale on cfg.STIFFNESS')
    parser.add_argument('--kd_scale', type=float, default=1.0, help='Scale on cfg.DAMPING')
    parser.add_argument('--initial_gain_scale', type=float, default=1.0,
                        help='Policy PD gain scale at A (keys 2..9 / X change it)')
    parser.add_argument('--smooth_action', type=float, default=0.0, help='EMA alpha on the action (0: off)')
    parser.add_argument('--max_tilt', type=float, default=1.5, help='|roll| or |pitch| [rad] -> damping')
    parser.add_argument('--motion_blend_in_time', type=float, default=None)
    parser.add_argument('--odom_frame', choices=ODOM_FRAMES, default='livox_imu',
                        help='Frame the odometry pose refers to (child_frame_id)')
    parser.add_argument('--mid360_in_torso', type=float, nargs=6, default=list(MID360_IN_TORSO),
                        metavar=('X', 'Y', 'Z', 'R', 'P', 'Y_'))
    parser.add_argument('--livox_imu_in_mid360', type=float, nargs=6, default=list(LIVOX_IMU_IN_MID360),
                        metavar=('X', 'Y', 'Z', 'R', 'P', 'Y_'))
    parser.add_argument('--imu_in_pelvis', type=float, nargs=6, default=list(IMU_IN_PELVIS),
                        metavar=('X', 'Y', 'Z', 'R', 'P', 'Y_'))
    parser.add_argument('--odom_timeout', type=float, default=0.3, help='Odometry older than this [s] -> stop')
    parser.add_argument('--odom_ma', type=int, default=1,
                        help='Moving-average length on the odom position (1: off; >1 adds (N-1)/2 scans of lag)')
    parser.add_argument('--odom_sync', choices=['stamp', 'latest'], default='stamp',
                        help='Joint state for the FK: at the message stamp, or the latest')
    parser.add_argument('--origin_ref', choices=ORIGIN_REFS, default='pelvis', help='Point the origin is set on')
    parser.add_argument('--origin_window', type=float, default=0.5, help='Averaging window of the origin (A) [s]')
    parser.add_argument('--print_odom_hz', type=float, default=0.0,
                        help='Print the raw LIO pose (xyz rpy) at this rate [Hz] (0: off)')
    parser.add_argument('--log_dir', type=str, default=DEFAULT_LOG_DIR)
    return parser


def controller_kwargs(args):
    return dict(
        xml_file=args.xml, policy_path=args.policy, config_path=args.config, motion_npz=args.motion_npz,
        stand_npz=args.stand_npz,
        obs_cfg=args.obs_cfg, device=args.device, kp_scale=args.kp_scale, kd_scale=args.kd_scale,
        initial_gain_scale=args.initial_gain_scale, smooth_action=args.smooth_action, max_tilt=args.max_tilt,
        motion_blend_in_time=args.motion_blend_in_time, odom_frame=args.odom_frame,
        mid360_in_torso=args.mid360_in_torso, livox_imu_in_mid360=args.livox_imu_in_mid360,
        imu_in_pelvis=args.imu_in_pelvis, odom_timeout=args.odom_timeout, odom_ma=args.odom_ma,
        odom_sync=args.odom_sync, origin_ref=args.origin_ref, origin_window=args.origin_window,
        print_odom_hz=args.print_odom_hz, log_dir=args.log_dir,
    )


def main():
    parser = add_controller_args(argparse.ArgumentParser(description='TWIST2 policy on the real G1 (LIO odom)'))
    parser.add_argument('--net', type=str, default='eth0', help='Network interface connected to the robot')
    parser.add_argument('--odom_topic', type=str, default='/dreamstep_odom_grav', help='nav_msgs/Odometry topic')
    args = parser.parse_args()
    for path, what in ((args.policy, "Policy"), (args.config, "Config"), (args.xml, "XML"), (args.motion_npz, "Motion")):
        if not os.path.exists(path):
            raise SystemExit(f"Error: {what} file {path} does not exist")
    print("\n" + "=" * 50 + "\nSAFETY WARNING: running a policy on the real robot.\n"
          "Keep the robot on the hoist; C / Select = damping + exit.\n" + "=" * 50 + "\n")
    controller = RealTimePolicyController(net=args.net, odom_topic=args.odom_topic, **controller_kwargs(args))
    controller.run()


if __name__ == "__main__":
    main()
