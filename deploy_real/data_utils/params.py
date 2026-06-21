import numpy as np

DEFAULT_MIMIC_OBS_G1 = np.concatenate([
                    np.array([0, 0]), # xy velocity (root-local)
                    np.array([0.793]), # z position (root height at the DEFAULT_DOF_POS stand)
                    np.array([0, 0]), # roll/pitch
                    np.array([0]), # yaw (in robot's start-heading frame)
                    np.array([0, 0, 0]), # full angular velocity (root-local)
                    # 29 dof -- mirror cfg.DEFAULT_DOF_POS (the robot PD rest pose / training
                    # init). Keeping the idle reference == the action=0 equilibrium pose makes
                    # the idle tracking error ~0, avoiding the idle chatter seen when the
                    # reference asked for a different pose (knee 0.4 vs 0.669, elbow 1.2 vs 0.6)
                    # than where the robot physically rests.
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
