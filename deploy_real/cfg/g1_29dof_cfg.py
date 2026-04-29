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

# Robot init pose (SDK order).
DEFAULT_DOF_POS = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,           # left leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,           # right leg
    0.0, 0.0, 0.0,                            # torso
    0.3, 0.25, 0.0, 0.97, 0.15, 0.0, 0.0,     # left arm
    0.3, -0.25, 0.0, 0.97, -0.15, 0.0, 0.0,   # right arm
])

# PD gains (ImplicitActuatorCfg stiffness/damping, SDK order).
STIFFNESS = np.array([
    100, 100, 100, 150, 40, 40,
    100, 100, 100, 150, 40, 40,
    200, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
])
DAMPING = np.array([
    2, 2, 2, 4, 2, 2,
    2, 2, 2, 4, 2, 2,
    5, 5, 5,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
])

# Effort limits: wrist pitch/yaw use W4010-25 (5 Nm), rest of arm N5020-16 (25 Nm).
TORQUE_LIMITS = np.array([
    88, 139, 88, 139, 25, 25,
    88, 139, 88, 139, 25, 25,
    88, 25, 25,
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
