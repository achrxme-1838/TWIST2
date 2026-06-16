

source ~/miniconda3/bin/activate twist2

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/2_MAPO_260523.onnx

# change the network interface name to your own that connects to the robot
# net=enp0s31f6
# net=eno1
net=eth1

cd deploy_real

python server_low_level_g1_real.py \
    --policy ${ckpt_path} \
    --net ${net} \
    --device cpu \
    --use_hand \
    --use_diff_body_pos \
    --use_diff_body_tannorm \
    # --- world-root tracking via FAST-LIO odometry (groundwork) ---
    # Requires rclpy: launch from a sourced ROS2 (Humble) env with a matching Python
    # (the twist2 py3.8 env above cannot import rclpy). When enabled, the robot world
    # root (xy, z, yaw) comes from --odom_topic and the reference motion root is
    # anchored to the robot root at each motion start. Run the motion server too.
    # --update_robot_w_odom \
    # --odom_topic /Odometry \
    # --kp_scale 1.0 \
    # --kd_scale 1.0
    # --smooth_body 0.5
    # --record_proprio \
