# SCRIPT_DIR=$(dirname $(realpath $0))

# # ckpt_path=${SCRIPT_DIR}/assets/ckpts/260429_2nd.onnx
# # ckpt_path=${SCRIPT_DIR}/assets/ckpts/2_MAPO_260523.onnx
# # ckpt_path=${SCRIPT_DIR}/assets/ckpts/2_MAPO_260524.onnx  # Seems too premitive teacher is used


# # ckpt_path=${SCRIPT_DIR}/assets/ckpts/2_MAPO_260523_10k.onnx  # current SOTA


# # ckpt_path=${SCRIPT_DIR}/assets/ckpts/2_DA_frc_260525.onnx


# # ckpt_path=${SCRIPT_DIR}/assets/ckpts/2_MAPO_wo_fix_260616.onnx
# ckpt_path=${SCRIPT_DIR}/assets/ckpts/3_MAPO_add_future_260718.onnx


# cd deploy_real

# python server_low_level_g1_sim.py \
#     --xml ../assets/g1/g1_sim2sim_29dof.xml \
#     --policy ${ckpt_path} \
#     --device cpu \
#     --measure_fps 1 \
#     --policy_frequency 50 \
#     --limit_fps 1 \
#     --use_diff_body_pos \
#     --use_diff_body_tannorm \
#     --use_future_motion \
#     --update_robot_w_odom \
#     --odom_topic /twist2/sim_odom \

#     # --kp_scale 0.85 \
#     # --kd_scale 1.2 \

#     # --smooth_action 0.8
#     # --policy_frequency 100 \
#     # --record_proprio \


# !! USE sim2sim_odom.sh instead of this script. This script is deprecated. It does not use the future motion feature, which is important for sim2sim with odometry.
