#!/bin/bash
# Real G1 deploy of deploy_sim/server_low_level_g1_real.py (LIO odometry + global origin set at A).

#source ~/miniconda3/bin/activate twist2
set -u
SCRIPT_DIR=$(dirname $(realpath $0))

ckpt=${CKPT:-${SCRIPT_DIR}/assets/ckpts/prop5+diff-pos-b+future-pos-anchor_hist25_rt-imu_odom-dly-drift_imu-error_micpos-u-l-w3.02.0sig1e-42e-3_odom-dly-drift.onnx}
# ckpt=${CKPT:-${SCRIPT_DIR}/assets/ckpts/ft/prop5+diff-pos-b+future-pos-anchor_hist25_rt-imu_odom-dly-drift_micpos-u-l-w3.02.0sig1e-42e-3_odom-dly-drift_ft3000.onnx}

# motion=${MOTION_NPZ:-${SCRIPT_DIR}/assets/motions/humanml_004669.npz}
# motion=${MOTION_NPZ:-${SCRIPT_DIR}/assets/motions/humanml_002153.npz}
motion=${MOTION_NPZ:-${SCRIPT_DIR}/assets/motions/LAFAN1_LAFAN1_robotwalk1_subject1_seg19.npz}

stand=${STAND_NPZ:-${SCRIPT_DIR}/assets/motions/A1-Stand_poses_blend1.0.npz}

# This G1's physical MID360 axes are rolled 180 degrees relative to the URDF mount.
# Keep the URDF translation/pitch; apply the roll in the sensor-to-torso chain,
# so both sensor orientation and the Livox IMU lever arm are handled consistently.
# The sim uses the upright URDF mount. A trailing --mid360_in_torso can override this.
mid360_in_torso=(0.0002835 0.00003 0.41618 3.141592653589793 0.04014257279586953 0.0)

cd "${SCRIPT_DIR}/deploy_sim"
export LD_PRELOAD=/home/unitree/miniconda3/envs/twist2/lib/python3.8/site-packages/torch.libs/libgomp-804f19d4.so.1.0.0
${PY:-python} server_low_level_g1_real.py \
    --policy "$(realpath "${ckpt}")" \
    --motion_npz "$(realpath "${motion}")" \
    --stand_npz "$(realpath "${stand}")" \
    --net ${NET:-eth0} \
    --odom_topic ${ODOM_TOPIC:-/dreamstep_odom_base} \
    --odom_frame ${ODOM_FRAME:-livox_imu} \
    --mid360_in_torso "${mid360_in_torso[@]}" \
    --origin_ref ${ORIGIN_REF:-pelvis} \
    --print_odom_hz ${PRINT_ODOM_HZ:-2.0} \
    "$@"
