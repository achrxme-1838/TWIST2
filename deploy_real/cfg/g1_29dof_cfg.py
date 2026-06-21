"""Deploy config for G1 29-DoF (mirrors DEX_RL_LAB g1_29dof_lab_cfg / SMPLCfg)."""
from __future__ import annotations

import numpy as np


NUM_ACTIONS = 29

# MuJoCo XML / SDK joint order. data.qpos[7:36] / data.qvel[6:35] use this.
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

# Isaac USD-DFS joint order. The trained policy reads/writes joints in this order.
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

# Ankle joints zero-masked in the dof_vel observation term.
ANKLE_JOINT_NAMES = (
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint",  "right_ankle_roll_joint",
)

# Robot init pose (SDK order). Mirrors g1_29dof_beyond_mimic_cfg init_state.joint_pos.
# Used as the JointPositionAction default offset (use_default_offset=True) and the
# joint_pos_rel observation reference, so it must match training exactly.
DEFAULT_DOF_POS = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,     # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,     # right leg
    0.0, 0.0, 0.0,                            # waist yaw/roll/pitch
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,        # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,       # right arm
])

# PD gains mirror DEX_RL_LAB g1_29dof_beyond_mimic_cfg ImplicitActuatorCfg. They are
# derived per-motor from armature with a 10 Hz natural frequency and damping ratio 2.0:
#   stiffness = armature * (2*pi*10)**2,  damping = 2 * zeta * armature * (2*pi*10).
# Feet (ankle) and waist roll/pitch use the 5020 actuator doubled (2x armature/gain).
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425
_NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
_DAMPING_RATIO = 2.0
_stiff = lambda a: a * _NATURAL_FREQ ** 2
_damp = lambda a: 2.0 * _DAMPING_RATIO * a * _NATURAL_FREQ
_S5020, _S7514, _S7522, _S4010 = (_stiff(a) for a in
    (_ARMATURE_5020, _ARMATURE_7520_14, _ARMATURE_7520_22, _ARMATURE_4010))
_D5020, _D7514, _D7522, _D4010 = (_damp(a) for a in
    (_ARMATURE_5020, _ARMATURE_7520_14, _ARMATURE_7520_22, _ARMATURE_4010))

# (SDK order) leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll.
STIFFNESS = np.array([
    _S7514, _S7522, _S7514, _S7522, 2 * _S5020, 2 * _S5020,   # left leg
    _S7514, _S7522, _S7514, _S7522, 2 * _S5020, 2 * _S5020,   # right leg
    _S7514, 2 * _S5020, 2 * _S5020,                           # waist yaw/roll/pitch
    _S5020, _S5020, _S5020, _S5020, _S5020, _S4010, _S4010,   # left arm
    _S5020, _S5020, _S5020, _S5020, _S5020, _S4010, _S4010,   # right arm
])
DAMPING = np.array([
    _D7514, _D7522, _D7514, _D7522, 2 * _D5020, 2 * _D5020,
    _D7514, _D7522, _D7514, _D7522, 2 * _D5020, 2 * _D5020,
    _D7514, 2 * _D5020, 2 * _D5020,
    _D5020, _D5020, _D5020, _D5020, _D5020, _D4010, _D4010,
    _D5020, _D5020, _D5020, _D5020, _D5020, _D4010, _D4010,
])

# effort_limit_sim from g1_29dof_beyond_mimic_cfg (SDK order). Feet + waist roll/pitch
# = 50 Nm, waist_yaw/hip_yaw/hip_pitch = 88, hip_roll/knee = 139, arm = 25 except
# wrist pitch/yaw (W4010-25) = 5 Nm.
TORQUE_LIMITS = np.array([
    88, 139, 88, 139, 50, 50,
    88, 139, 88, 139, 50, 50,
    88, 50, 50,
    25, 25, 25, 25, 25, 5, 5,
    25, 25, 25, 25, 25, 5, 5,
])

# JointPositionActionCfg(scale=0.25, use_default_offset=True).
ACTION_SCALE = 0.25

# Bodies tracked in body-position observations.
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

# (name, parent_body, local_offset_in_parent_frame). Mirrors SMPLCfg.extending.
EXTENDED_JOINTS = [
    ("left_hand_link_ext",  "left_wrist_yaw_link",  (0.0415,  0.003, 0.0)),
    ("right_hand_link_ext", "right_wrist_yaw_link", (0.0415, -0.003, 0.0)),
    ("head_link_ext",       "torso_link",           (0.0,     0.0,   0.4)),
]

NUM_TRACKED_BODIES = len(TRACKED_BODY_NAMES) + len(EXTENDED_JOINTS)  # 33

# Observation history (term-major IsaacLab CircularBuffer).
HISTORY_LEN = 10
HIST_TERM_DIMS = {
    "base_ang_vel":    3,
    "base_roll_pitch": 2,
    "joint_pos_rel":   NUM_ACTIONS,
    "joint_vel_rel":   NUM_ACTIONS,
    "last_action":     NUM_ACTIONS,
}
HIST_TERM_ORDER = [
    "base_ang_vel", "base_roll_pitch",
    "joint_pos_rel", "joint_vel_rel", "last_action",
]

# upcoming_twist_mimic_target dim: xy_vel(2)+z(1)+rpy(3)+ang_vel(3)+dof_pos(29).
MIMIC_DIM = 38

# ----- future motion observations (future_motion_pos_h / future_motion_anchor) -----
# Mirrors DEX_RL_LAB RefMotionCommandCfg(num_future_steps=10, step_interval=5).
# Future step i (0-indexed) is (i+1)*FUTURE_STEP_INTERVAL env steps ahead. With the
# env dt = decimation*sim.dt = 0.02s, that is 0.1s .. 1.0s ahead (1s horizon).
NUM_FUTURE_STEPS = 10
FUTURE_STEP_INTERVAL = 5
# Control-step offsets the motion server samples for the future frames
# ([5, 10, ..., 50]). Units of the motion server control_dt (0.02s), which equals
# the training env dt, so the sampled times match training exactly.
FUTURE_MOTION_STEPS = [(i + 1) * FUTURE_STEP_INTERVAL for i in range(NUM_FUTURE_STEPS)]
# future_motion_pos_h:    T * B * 3 (heading-frame extended body positions).
# future_motion_anchor:   T * 6     (heading-corrected base->ref-root rot, 6D).
FUTURE_MOTION_POS_DIM = NUM_FUTURE_STEPS * NUM_TRACKED_BODIES * 3  # 10*33*3 = 990
FUTURE_MOTION_ANCHOR_DIM = NUM_FUTURE_STEPS * 6                    # 10*6   = 60

# Redis keys the motion server publishes the future reference frames to. Each holds
# a flattened JSON list: root_pos [T*3], root_rot [T*4] (xyzw), dof_pos [T*J] (SDK order).
FUTURE_MOTION_ROOT_POS_KEY = "future_motion_root_pos"
FUTURE_MOTION_ROOT_ROT_KEY = "future_motion_root_rot"
FUTURE_MOTION_DOF_POS_KEY = "future_motion_dof_pos"
