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
from data_utils.params import DEFAULT_MIMIC_OBS

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
    # Mirrors DEX_RL_LAB TRACKED_BODY_NAMES (29) for g1_29dof_sonic_distill.
    # Order must match so the deployed policy sees the same body indexing the
    # teacher/student were trained on.
    TRACKED_BODY_NAMES = [
        "pelvis",
        "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
        "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
        "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
        "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
        "waist_yaw_link", "waist_roll_link", "torso_link",
        "left_shoulder_pitch_link", "left_shoulder_roll_link",
        "left_shoulder_yaw_link", "left_elbow_link",
        "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
        "right_shoulder_pitch_link", "right_shoulder_roll_link",
        "right_shoulder_yaw_link", "right_elbow_link",
        "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
    ]
    # Matches SMPLCfg.extending.extended_joints in DEX_RL_LAB (g1_29dof_smpl_cfg.py):
    # (name, parent_body, local_offset) -- local_offset is in parent-body frame.
    EXTENDED_JOINTS = [
        ("left_hand_link_ext",  "left_wrist_yaw_link",  (0.0415,  0.003, 0.0)),
        ("right_hand_link_ext", "right_wrist_yaw_link", (0.0415, -0.003, 0.0)),
        ("head_link_ext",       "torso_link",           (0.0,     0.0,   0.4)),
    ]
    NUM_TRACKED_BODIES = 30 + 3  # 33

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
        # self.sim_decimation = 1 / (policy_frequency * self.sim_dt * 4)
        self.sim_decimation = 1 / (policy_frequency * self.sim_dt)
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
        # self.action_scale = np.full(self.num_actions, 0.25, dtype=np.float32)
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
        #   upcoming_twist_mimic_target(38, history=1)
        #
        # Final flat dim = 10*(3+2+29+29+29) + 38 = 920 + 38 = 958.
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

        # mimic_obs (action_body) layout (motion_server build_mimic_obs):
        #   [xy_vel_local(2), z(1), roll(1), pitch(1), yaw(1), ang_vel_local(3), dof_pos(29)]
        # = 38 dims. yaw is the ref root yaw in the published frame; motion_server
        # anchors motion frame 0's yaw to robot's heading at playback start so
        # this yaw is comparable to the robot's actual yaw in the same world frame.
        self._mimic_dim = 38

        # diff_body_pos_b observation (ref-robot per-body position diff in robot base
        # frame). Ref FK uses (z, roll, pitch, yaw, dof_pos) from mimic_obs --
        # since motion_server publishes ref yaw in the same frame as the robot's
        # heading (anchored at start), we no longer need to substitute the robot's
        # current yaw and yaw mismatches now show up in the diff.
        # In PolicyCfg it has history_length=10 and is declared AFTER
        # upcoming_twist_mimic_target, so it's appended post-mimic in flat_parts.
        self.use_diff_body_pos = use_diff_body_pos
        self._diff_body_pos_per_step = self.NUM_TRACKED_BODIES * 3  # 99
        self._diff_body_pos_total = (
            self.history_len * self._diff_body_pos_per_step if use_diff_body_pos else 0
        )

        # diff_body_tannorm_pb: per-body 6D tan-norm of the ref/robot rotation diff
        # in robot planar (yaw-only) base frame. Declared AFTER diff_body_pos_b in
        # PolicyCfg so it's appended last in flat_parts.
        self.use_diff_body_tannorm = use_diff_body_tannorm
        self._diff_body_tannorm_per_step = self.NUM_TRACKED_BODIES * 6  # 198
        self._diff_body_tannorm_total = (
            self.history_len * self._diff_body_tannorm_per_step if use_diff_body_tannorm else 0
        )

        self.total_obs_size = (
            self.history_len * sum(self._hist_dims.values())
            + self._mimic_dim
            + self._diff_body_pos_total
            + self._diff_body_tannorm_total
        )

        # Body-layout caches used when use_diff_body_pos is on (built here so we
        # can also print their dims below).
        self.tracked_body_ids = np.array(
            [self.model.body(n).id for n in self.TRACKED_BODY_NAMES], dtype=np.int64
        )
        self.extended_parent_ids = np.array(
            [self.model.body(parent).id for _, parent, _ in self.EXTENDED_JOINTS],
            dtype=np.int64,
        )
        self.extended_local_offsets = np.array(
            [offset for _, _, offset in self.EXTENDED_JOINTS], dtype=np.float64
        )
        # Secondary MjData for running FK on the reference motion frame each step.
        # Shared between diff_body_pos_b and diff_body_tannorm_pb when both are on.
        self.ref_data = (
            mujoco.MjData(self.model)
            if (use_diff_body_pos or use_diff_body_tannorm) else None
        )

        # diff_body_pos_b has its own history buffer (history=10, separate from
        # _hist_bufs because it's emitted *after* action_mimic in the obs layout).
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

        print(f"TWIST2 Controller Configuration (term-major, DEX_RL_LAB layout):")
        for name in self._hist_term_order:
            d = self._hist_dims[name]
            print(f"  {name}: history={self.history_len} x dim={d} = {self.history_len * d}")
        print(f"  future_motion_mimic_target (no history): {self._mimic_dim}")
        if use_diff_body_pos:
            print(f"  diff_body_pos_b: history={self.history_len} x dim={self._diff_body_pos_per_step} = {self._diff_body_pos_total}")
        if use_diff_body_tannorm:
            print(f"  diff_body_tannorm_pb: history={self.history_len} x dim={self._diff_body_tannorm_per_step} = {self._diff_body_tannorm_total}")
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

    def _compute_extended_body_pos_w(self, data):
        """Stack world positions for the 29 tracked bodies + 3 extended joints.

        Returns [32, 3] in TRACKED_BODY_NAMES order followed by EXTENDED_JOINTS order.
        Extended joints are computed as ``parent_xpos + parent_R @ local_offset``.
        """
        tracked_pos = data.xpos[self.tracked_body_ids]
        parent_pos = data.xpos[self.extended_parent_ids]
        parent_mat = data.xmat[self.extended_parent_ids].reshape(-1, 3, 3)
        ext_pos = parent_pos + np.einsum("bij,bj->bi", parent_mat, self.extended_local_offsets)
        return np.concatenate([tracked_pos, ext_pos], axis=0)

    def _compute_diff_body_pos_b(self, action_mimic, use_pb=False):
        """Compute diff_body_pos_b/pb (matches DEX_RL_LAB mdp.diff_body_pos_pb
        under root xy alignment + motion-start heading anchor) as a flat
        [NUM_TRACKED_BODIES * 3] float32 array.

        ``action_mimic`` layout (motion server ``build_mimic_obs``, 36-dim):
            [xy_vel_local(2), z(1), roll(1), pitch(1), yaw(1),
             yaw_ang_vel_local(1), dof_pos(29)]
        ref yaw is read directly from mimic_obs[5] -- motion_server anchors
        motion frame 0's yaw to robot's heading at playback start, so the ref
        yaw is in the same world frame as the robot's actual yaw and yaw
        mismatches show up in the diff. Ref root xy stays at 0 (per-side root
        subtraction below cancels it).

        If ``use_pb=True``, rotates the world-frame diff into the robot *planar*
        (yaw-only) base frame -- matches DEX_RL_LAB mdp.diff_body_pos_pb /
        extended_body_pos_pb. Otherwise uses the full base frame (pelvis xmat).
        """
        ref_z = float(action_mimic[2])
        ref_roll = float(action_mimic[3])
        ref_pitch = float(action_mimic[4])
        ref_yaw = float(action_mimic[5])
        ref_dof_pos = np.asarray(action_mimic[-self.num_actions:], dtype=np.float64)

        # Build ref root quat (w, x, y, z) from (roll, pitch, ref_yaw)
        # using ZYX-extrinsic order: q = q_yaw * q_pitch * q_roll.
        hr, hp, hy = 0.5 * ref_roll, 0.5 * ref_pitch, 0.5 * ref_yaw
        cr, sr = np.cos(hr), np.sin(hr)
        cp, sp = np.cos(hp), np.sin(hp)
        cy, sy = np.cos(hy), np.sin(hy)
        ref_qw = cy * cp * cr + sy * sp * sr
        ref_qx = cy * cp * sr - sy * sp * cr
        ref_qy = cy * sp * cr + sy * cp * sr
        ref_qz = sy * cp * cr - cy * sp * sr

        self.ref_data.qpos[:3] = (0.0, 0.0, ref_z)
        self.ref_data.qpos[3:7] = (ref_qw, ref_qx, ref_qy, ref_qz)
        self.ref_data.qpos[7:7 + self.num_actions] = ref_dof_pos
        mujoco.mj_kinematics(self.model, self.ref_data)

        ref_body_pos_w = self._compute_extended_body_pos_w(self.ref_data)      # [N, 3]
        robot_body_pos_w = self._compute_extended_body_pos_w(self.data)        # [N, 3]

        # Subtract each side's own root (pelvis == index 0) before diffing so any
        # residual root xy/z offset between sim and ref is cancelled -- matches
        # diff_body_pos_b_deploy, which collapses to diff_body_pos_b under alignment.
        ref_body_rel = ref_body_pos_w - ref_body_pos_w[0:1]
        robot_body_rel = robot_body_pos_w - robot_body_pos_w[0:1]
        diff_w = ref_body_rel - robot_body_rel                                  # [N, 3]

        if use_pb:
            # Planar (yaw-only) base frame; same convention as planar_root_quat_w
            # in DEX_RL_LAB (calc_heading: yaw is the x-axis heading).
            # R_planar maps planar-base -> world, so v_pb = R_planar.T @ v_w == v_w @ R_planar.
            qw, qx, qy, qz = (float(v) for v in self.data.qpos[3:7])
            dir_x = 1.0 - 2.0 * (qy * qy + qz * qz)
            dir_y = 2.0 * (qw * qz + qx * qy)
            robot_yaw = np.arctan2(dir_y, dir_x)
            c, s = np.cos(robot_yaw), np.sin(robot_yaw)
            R_planar = np.array(
                [[c, -s, 0.0],
                 [s,  c, 0.0],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            diff_b = diff_w @ R_planar
        else:
            # Rotate world-frame diff into robot base (pelvis) frame.
            # xmat[pelvis] maps body->world, so (R.T @ v_w) == v_b, i.e. v_b = v_w @ R.
            pelvis_id = int(self.tracked_body_ids[0])
            R_body_to_world = self.data.xmat[pelvis_id].reshape(3, 3)
            diff_b = diff_w @ R_body_to_world
        return diff_b.astype(np.float32).reshape(-1)

    def _compute_extended_body_quat_w(self, data):
        """Stack world quats for the 30 tracked bodies + 3 extended joints.

        Extended bodies inherit their parent body's world orientation (matches
        ``extending_body_quat_w`` in DEX_RL_LAB). Returns [33, 4] in
        TRACKED_BODY_NAMES order followed by EXTENDED_JOINTS parent-order, with
        (w, x, y, z) convention (MuJoCo's ``data.xquat``).
        """
        tracked_quat = data.xquat[self.tracked_body_ids]       # [30, 4]
        parent_quat = data.xquat[self.extended_parent_ids]     # [3, 4]
        return np.concatenate([tracked_quat, parent_quat], axis=0)

    @staticmethod
    def _quat_mul_np(q1, q2):
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.stack([w, x, y, z], axis=-1)

    @staticmethod
    def _quat_conj_np(q):
        out = q.copy()
        out[..., 1:] *= -1.0
        return out

    @staticmethod
    def _quat_apply_np(q, v):
        qw = q[..., 0:1]
        qxyz = q[..., 1:4]
        t = 2.0 * np.cross(qxyz, v)
        return v + qw * t + np.cross(qxyz, t)

    def _compute_diff_body_tannorm_pb(self, action_mimic):
        """Compute diff_body_tannorm_pb (matches DEX_RL_LAB mdp.diff_body_tannorm_pb
        under root xy alignment + motion-start heading anchor) as flat
        [NUM_TRACKED_BODIES * 6] float32.

        ref yaw is read directly from mimic_obs[5] (motion_server anchors
        motion frame 0's yaw to robot's heading at playback start), so the ref
        is in the same world frame as the robot and yaw mismatches show up.

        Math:
            diff_q_w  = q_ref_w * conj(q_robot_w)             per body
            diff_q_pb = conj(q_yaw) * diff_q_w * q_yaw        (robot root yaw-only)
            tannorm   = [ diff_q_pb * (1,0,0), diff_q_pb * (0,0,1) ]   (6D per body)
        """
        ref_z = float(action_mimic[2])
        ref_roll = float(action_mimic[3])
        ref_pitch = float(action_mimic[4])
        ref_yaw = float(action_mimic[5])
        ref_dof_pos = np.asarray(action_mimic[-self.num_actions:], dtype=np.float64)

        # Build ref root quat from (roll, pitch, ref_yaw) via ZYX-extrinsic order.
        hr, hp, hy = 0.5 * ref_roll, 0.5 * ref_pitch, 0.5 * ref_yaw
        cr, sr = np.cos(hr), np.sin(hr)
        cp, sp = np.cos(hp), np.sin(hp)
        cy, sy = np.cos(hy), np.sin(hy)
        ref_qw = cy * cp * cr + sy * sp * sr
        ref_qx = cy * cp * sr - sy * sp * cr
        ref_qy = cy * sp * cr + sy * cp * sr
        ref_qz = sy * cp * cr - cy * sp * sr

        self.ref_data.qpos[:3] = (0.0, 0.0, ref_z)
        self.ref_data.qpos[3:7] = (ref_qw, ref_qx, ref_qy, ref_qz)
        self.ref_data.qpos[7:7 + self.num_actions] = ref_dof_pos
        mujoco.mj_kinematics(self.model, self.ref_data)

        ref_body_quat_w = self._compute_extended_body_quat_w(self.ref_data)     # [N, 4]
        robot_body_quat_w = self._compute_extended_body_quat_w(self.data)       # [N, 4]

        diff_body_quat_w = self._quat_mul_np(
            ref_body_quat_w,
            self._quat_conj_np(robot_body_quat_w),
        )

        # Robot planar (yaw-only) root quat (calc_heading convention),
        # broadcast across bodies for the planar-frame rotation.
        qw, qx, qy, qz = (float(v) for v in self.data.qpos[3:7])
        dir_x = 1.0 - 2.0 * (qy * qy + qz * qz)
        dir_y = 2.0 * (qw * qz + qx * qy)
        robot_yaw = np.arctan2(dir_y, dir_x)
        planar_root_quat = np.array(
            [np.cos(0.5 * robot_yaw), 0.0, 0.0, np.sin(0.5 * robot_yaw)], dtype=np.float64
        )
        inv_planar_root_quat = self._quat_conj_np(planar_root_quat)

        planar_b = np.broadcast_to(planar_root_quat, diff_body_quat_w.shape)
        inv_planar_b = np.broadcast_to(inv_planar_root_quat, diff_body_quat_w.shape)

        diff_body_quat_pb = self._quat_mul_np(
            self._quat_mul_np(inv_planar_b, diff_body_quat_w),
            planar_b,
        )

        # Tan-norm: rotate reference tangent (1,0,0) and normal (0,0,1) by each
        # diff quat and concatenate -> 6D per body.
        ref_tan = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        ref_norm = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        tan = self._quat_apply_np(diff_body_quat_pb, ref_tan)    # [N, 3]
        norm = self._quat_apply_np(diff_body_quat_pb, ref_norm)  # [N, 3]
        tannorm = np.concatenate([tan, norm], axis=-1)           # [N, 6]
        return tannorm.astype(np.float32).reshape(-1)

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
        # Publish robot's current planar yaw so motion_server can anchor motion
        # frame 0's heading to the robot at playback start.
        self.redis_pipeline.set("state_heading_unitree_g1_with_hands", json.dumps(0.0))

        # Seed action_* Redis keys with the idle default so we don't chase a stale
        # target left over from a prior motion server session.
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
                    # Robot's planar yaw (calc_heading: x-axis heading) — motion_server
                    # reads this once at playback start to anchor motion frame 0.
                    qw, qx, qy, qz = (float(v) for v in self.data.qpos[3:7])
                    _dirx = 1.0 - 2.0 * (qy * qy + qz * qz)
                    _diry = 2.0 * (qw * qz + qx * qy)
                    robot_heading = float(np.arctan2(_diry, _dirx))
                    self.redis_pipeline.set("state_heading_unitree_g1_with_hands", json.dumps(robot_heading))
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

                    if self.use_diff_body_pos:
                        diff_body_pos_b = self._compute_diff_body_pos_b(action_mimic, use_pb=True)
                        self._diff_body_pos_hist.append(diff_body_pos_b)
                        flat_parts.append(
                            np.asarray(self._diff_body_pos_hist, dtype=np.float32).reshape(-1)
                        )

                    if self.use_diff_body_tannorm:
                        diff_body_tannorm_pb = self._compute_diff_body_tannorm_pb(action_mimic)
                        self._diff_body_tannorm_hist.append(diff_body_tannorm_pb)
                        flat_parts.append(
                            np.asarray(self._diff_body_tannorm_hist, dtype=np.float32).reshape(-1)
                        )

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
    parser.add_argument("--use_diff_body_pos", action="store_true",
                        help="Append diff_body_pos_b observation (33 bodies * 3 = 99 dims) "
                             "to the obs. Ref root yaw is taken from the robot's current "
                             "planar yaw at FK time (mimic_obs has no absolute yaw).")
    parser.add_argument("--use_diff_body_tannorm", action="store_true",
                        help="Append diff_body_tannorm_pb observation (33 bodies * 6 = 198 dims) "
                             "to the obs. Ref root yaw is taken from the robot's current "
                             "planar yaw at FK time (mimic_obs has no absolute yaw).")
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
        use_diff_body_pos=args.use_diff_body_pos,
        use_diff_body_tannorm=args.use_diff_body_tannorm,
    )
    controller.run()


if __name__ == "__main__":
    main()
