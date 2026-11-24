#!/bin/bash
# Real G1 deploy of deploy_sim/server_low_level_g1_real.py (LIO odometry + global origin set at A).

#source ~/miniconda3/bin/activate twist2
set -u
SCRIPT_DIR=$(dirname $(realpath $0))
ckpt=${CKPT:-${SCRIPT_DIR}/assets/ckpts/prop5+diff-pos-b+future-pos-h+future-anchor_upde-sig-per_mimic-body-w5.0_sig-init0.3sq.onnx}
motion=${MOTION_NPZ:-${SCRIPT_DIR}/assets/motions/B3-walk1_poses.npz}
stand=${STAND_NPZ:-${SCRIPT_DIR}/assets/motions/A1-Stand_poses.npz}
cd "${SCRIPT_DIR}/deploy_sim"
export LD_PRELOAD=/home/unitree/miniconda3/envs/twist2/lib/python3.8/site-packages/torch.libs/libgomp-804f19d4.so.1.0.0
${PY:-python} server_low_level_g1_real.py \
    --policy "$(realpath "${ckpt}")" \
    --motion_npz "$(realpath "${motion}")" \
    --stand_npz "$(realpath "${stand}")" \
    --net ${NET:-eth0} \
    --odom_topic ${ODOM_TOPIC:-/dreamstep_odom_grav} \
    --odom_frame ${ODOM_FRAME:-livox_imu} \
    --origin_ref ${ORIGIN_REF:-pelvis} \
    --print_odom_hz ${PRINT_ODOM_HZ:-2.0} \
    "$@"
