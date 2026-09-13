#!/bin/bash
# sim2sim with --update_robot_w_odom (world-root tracking via odometry).
#
# Container (Isaac Sim python) version: no conda / ROS2 needed.
#   * ROS2 is optional here -- without rclpy the controller prints
#     "[odom] OdomPublisher unavailable" and still does the odom-anchored diff_body_*
#     from the MuJoCo ground-truth root + Redis.
#   * --headless renders offscreen (MUJOCO_GL=egl) and --record_video writes an mp4.
#
# Run the motion server in another terminal WITHOUT --fix_root_pos so the reference
# root actually translates:
#     bash run_motion_server.sh
#
# Usage: bash sim2sim_odom.sh [ckpt.onnx] [video_out.mp4]

SCRIPT_DIR=$(dirname $(realpath $0))
PY=/workspace/isaaclab/_isaac_sim/python.sh

ckpt_path=${1:-${SCRIPT_DIR}/assets/ckpts/prop5+upcoming+diff-pos-b-deploy.onnx}
video_path=${2:-${SCRIPT_DIR}/videos/$(basename "${ckpt_path%.onnx}")_$(date +%Y%m%d_%H%M%S).mp4}
mkdir -p "$(dirname "${video_path}")"

# redis-server has no init system in the container: start it on demand.
redis-cli ping >/dev/null 2>&1 || redis-server --daemonize yes --bind 127.0.0.1 --save "" --appendonly no
redis-cli ping >/dev/null 2>&1 || { echo "redis-server failed to start"; exit 1; }

export PYTHONNOUSERSITE=1     # ignore ~/.local so it can't shadow env deps
unset PYTHONPATH             # drop DEX_RL_LAB / isaaclab from PYTHONPATH

cd ${SCRIPT_DIR}/deploy_real

# onnxruntime here is the CPU build -> run the policy on CPU.
"$PY" server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy "${ckpt_path}" \
    --device cpu \
    --policy_frequency 50 \
    --limit_fps 1 \
    --use_diff_body_pos \
    --update_robot_w_odom \
    --odom_topic /twist2/sim_odom \
    --kp_scale 1.0 \
    --kd_scale 1.0 \
    --headless \
    --record_video \
    --video_path "${video_path}" \
    # --sim_duration 60 \
