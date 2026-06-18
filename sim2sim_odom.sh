#!/bin/bash
# sim2sim with --update_robot_w_odom (world-root tracking via odometry).
#
# The low-level controller needs BOTH rclpy (ROS2 Humble) and the deploy deps
# (mujoco/onnxruntime/torch). Those live in the `twist2_ros` conda env (py3.10),
# NOT in `twist2` (py3.8, which cannot import rclpy). This script sets up that
# env, sources ROS2, and launches the controller with the odom flags.
#
# Run the motion server separately (its own isaacgym/twist2 env) WITHOUT
# --fix_root_pos so the reference root actually translates:
#     bash run_motion_server.sh        # after removing --fix_root_pos (see notes)
#
# redis-server must be running.

SCRIPT_DIR=$(dirname $(realpath $0))
# ckpt_path=${SCRIPT_DIR}/assets/ckpts/test/260613.onnx
# ckpt_path=${SCRIPT_DIR}/assets/ckpts/2_MAPO_wo_fix_260616.onnx
ckpt_path=${SCRIPT_DIR}/assets/ckpts/3_MAPO_add_future_260718.onnx


# ---- env: conda twist2_ros (rclpy + mujoco/onnx/torch) + ROS2 Humble ----
source ~/anaconda3/etc/profile.d/conda.sh
conda activate twist2_ros
export PYTHONNOUSERSITE=1     # ignore ~/.local so it can't shadow env deps
unset PYTHONPATH             # drop DEX_RL_LAB / isaaclab from PYTHONPATH
source /opt/ros/humble/setup.bash   # puts rclpy / nav_msgs on PYTHONPATH

cd ${SCRIPT_DIR}/deploy_real

# torch/onnxruntime in twist2_ros are CPU builds -> run the policy on CPU.
python server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${ckpt_path} \
    --device cpu \
    --policy_frequency 50 \
    --limit_fps 1 \
    --use_diff_body_pos \
    --use_diff_body_tannorm \
    --use_future_motion \
    --update_robot_w_odom \
    --odom_topic /twist2/sim_odom
