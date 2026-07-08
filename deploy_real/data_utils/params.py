import numpy as np

# Idle reference for the mimic-distill policies (MAPO/HuB generation): a natural
# human standing frame (A1-Stand frame 0). Those policies only ever see human
# motions as mimic targets during training, so idling on the PD-rest / training-
# init squat (knee 0.669) was out-of-distribution and made the robot rattle
# whenever no motion was playing (before the motion server starts and again
# after the motion ends). Root vel / roll / pitch / yaw are zeroed = stationary,
# upright, in the robot's start-heading frame.
DEFAULT_MIMIC_OBS_G1 = np.concatenate([
                    np.array([0, 0]), # xy velocity (root-local)
                    np.array([0.7805]), # z position (A1-Stand frame-0 root height)
                    np.array([0, 0]), # roll/pitch
                    np.array([0]), # yaw (in robot's start-heading frame)
                    np.array([0, 0, 0]), # full angular velocity (root-local)
                    # 29 dof (SDK order) -- A1-Stand_poses.pkl frame 0.
                    np.array([0.0439, -0.0396, -0.0196, -0.0385, -0.1205, 0.0221,  # left leg (6)
                            0.0306, -0.0033, -0.1691, 0.0109, -0.0917, 0.1091,  # right leg (6)
                            -0.0004, 0.0365, -0.0608, # waist yaw/roll/pitch (3)
                            -0.0090, 0.1574, -0.2743, 1.1622, 0.0456, -0.1705, -0.0969, # left arm (7)
                            -0.0722, -0.1758, 0.4069, 1.1640, -0.0873, -0.1338, 0.1233, # right arm (7)
                        ])
                ])

# Legacy idle reference: the PD rest pose / training init squat. The older TWIST
# policies chattered when the idle reference differed from the action=0
# equilibrium pose (knee 0.4 vs 0.669, elbow 1.2 vs 0.6), so they want THIS as
# the idle target. Swap it back in for pre-future (3928-dim) checkpoints.
DEFAULT_MIMIC_OBS_G1_SQUAT = np.concatenate([
                    np.array([0, 0]), # xy velocity (root-local)
                    np.array([0.793]), # z position (root height at the DEFAULT_DOF_POS stand)
                    np.array([0, 0]), # roll/pitch
                    np.array([0]), # yaw (in robot's start-heading frame)
                    np.array([0, 0, 0]), # full angular velocity (root-local)
                    # 29 dof -- mirror cfg.DEFAULT_DOF_POS.
                    np.array([-0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg (6)
                            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg (6)
                            0.0, 0.0, 0.0, # waist yaw/roll/pitch (3)
                            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0, # left arm (7)
                            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0, # right arm (7)
                        ])
                ])

DEFAULT_MIMIC_OBS_G1_MIXED_MODE = np.concatenate([
                    np.array([0, 0]), # xy velocity (root-local)
                    np.array([0.8]), # z position
                    np.array([0, 0]), # roll/pitch
                    np.array([0]), # yaw (in robot's start-heading frame)
                    np.array([0, 0, 0]), # full angular velocity (root-local)
                    # 29 dof
                    np.array([-0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                            0.0, 0.0, 0.0, # torso (1)
                            0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0, # left arm (7)
                            0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0, # right arm (7)
                        ]),
                    np.array([1.0, 0.0]) # mode indicator
                ])

DEFAULT_MIMIC_OBS_T1 = np.concatenate([
                    np.array([ 0.6]),
                    np.array([0, 0, 0]),
                    np.array([0, 0, 0]),
                    np.array([ 0.0]),
                    # 21 dof
                    np.array([
                        0.25, -1.4, 0.0, -0.5, # left arm
                        0.25, 1.4, 0.0, 0.5, # right arm
                        0.0, # waist
                        -0.1, 0.0, 0.0, 0.2, -0.1, 0.0, # left leg
                        -0.1, 0.0, 0.0, 0.2, -0.1, 0.0, # right leg
                    ])
                ])

DEFAULT_MIMIC_OBS_TODDY = np.concatenate([
                    np.array([ 0.3]),
                    np.array([0, 0, 0]),
                    np.array([0, 0, 0]),
                    np.array([ 0.0]),
                    # 21 dof
                    np.array([
                        0.0, 0.0, # waist (2)
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # left leg (6)
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # right leg (6)
                        0, -0.3, 0, 0.0, # left arm (4)
                        0, -0.3, 0, 0.0, # right arm (4)
                      
                    ])
                ])

DEFAULT_MIMIC_OBS = {
    "unitree_g1": DEFAULT_MIMIC_OBS_G1,
    "unitree_g1_mixed_mode": DEFAULT_MIMIC_OBS_G1_MIXED_MODE,
    "unitree_g1_with_hands": DEFAULT_MIMIC_OBS_G1,
    "booster_t1": DEFAULT_MIMIC_OBS_T1,
    "stanford_toddy": DEFAULT_MIMIC_OBS_TODDY,
}


DEFAULT_HAND_POSE = {
    "unitree_g1": 
    {
        "left": {
            "open": np.array([0, 0, 0, 0, 0, 0, 0]),
            "close": np.array([
                    # left (thumb, middle, index)
                    0, 1.0, 1.74, -1.57, -1.74, -1.57, -1.74,
                ]),
            # "open_pinch":
            #             np.array([0, 0, 0, -1.57, -1.74, 0, 0,]),
            # "close_pinch": np.array([
            #         # left (thumb, middle, index)
            #         -0.8, 0.7037, 0.2937, -1.57, -1.74, -1.2, -1.4,
            #     ])
            "open_pinch":
                        np.array([0, 0, 0, 0, 0, -1.57, -1.74]),
            "close_pinch": np.array([
                    # left (thumb, middle, index)
                    0.8, 0.7037, 0.2937,  -1.57, -1.74, -1.57, -1.74,
                ])
        },
        "right": {
            "open": np.array([0, 0, 0, 0, 0, 0, 0]),
            "close": np.array([
                    # right (thumb, middle, index)
                    0, -1.0, -1.74, 1.57, 1.74, 1.57, 1.74,
                ]),
            # "open_pinch":
            #             np.array([0, 0, 0, 1.57, 1.74, 0, 0,]),
            # "close_pinch": np.array([
            #         # right (thumb, middle, index)
            #         -0.8, -0.7037, -0.2937, 1.57, 1.74, 1.2, 1.4, 
            #     ])
            "open_pinch":
                        np.array([0, 0, 0, 0, 0, 1.57, 1.74]),
            "close_pinch": np.array([
                    # right (thumb, middle, index)
                    0.8, -0.7037, -0.2937, 1.57, 1.74, 1.57, 1.74
                ])
        },
    },
    "unitree_g1_with_hands": 
    {
        "left": {
            "open": np.array([0, 0, 0, 0, 0, 0, 0]),
            "close": np.array([
                    # left (thumb, index, middle)
                    0, 1.0, 1.74, -1.57, -1.74, -1.57, -1.74,
                ])
        },
        "right": {
            "open": np.array([0, 0, 0, 0, 0, 0, 0]),
            "close": np.array([
                    # right (thumb, index, middle)
                    0, -1.0, -1.74, 1.57, 1.74, 1.57, 1.74,
                ])
        },
    },
    "booster_t1": 
    {
        "left": {
            "open": np.array([0, 0]), # parallel gripper
            "close": np.array([0, 0]), # parallel gripper
        },
        "right": {
            "open": np.array([0, 0]), # parallel gripper
            "close": np.array([0, 0]), # parallel gripper
        },
    },
    "stanford_toddy": 
    {
        "left": {
            "open": np.array([0, 0]), # parallel gripper
            "close": np.array([0, 0]), # parallel gripper
        },
        "right": {
            "open": np.array([0, 0]), # parallel gripper
            "close": np.array([0, 0]), # parallel gripper
        },
    },
}
