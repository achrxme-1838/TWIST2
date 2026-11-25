#!/bin/bash
# Real G1 deploy of deploy_sim/server_low_level_g1_real.py (LIO odometry + global origin set at A).

#source ~/miniconda3/bin/activate twist2
set -u
SCRIPT_DIR=$(dirname $(realpath $0))

ckpt=${CKPT:-${SCRIPT_DIR}/assets/ckpts/prop5+diff-pos-b+future-pos-anchor_hist25_rt-imu_odom-dly-drift_micpos-u-l-w3.02.0sig1e-42e-3_odom-dly-drift.onnx}
# ckpt=${CKPT:-${SCRIPT_DIR}/assets/ckpts/ft/prop5+diff-pos-b+future-pos-anchor_hist25_rt-imu_odom-dly-drift_micpos-u-l-w3.02.0sig1e-42e-3_odom-dly-drift_ft3000.onnx}

motion=${MOTION_NPZ:-${SCRIPT_DIR}/assets/motions/humanml_004669.npz}
# motion=${MOTION_NPZ:-${SCRIPT_DIR}/assets/motions/humanml_002153.npz}
# motion=${MOTION_NPZ:-${SCRIPT_DIR}/assets/motions/humanml_011875.npz}

stand=${STAND_NPZ:-${SCRIPT_DIR}/assets/motions/A1-Stand_poses_blend1.0.npz}

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
