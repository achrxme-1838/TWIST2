#!/bin/bash
# MuJoCo sim2sim of the real-robot controller with LIO odometry + global origin set at A

set -u
SCRIPT_DIR=$(dirname $(realpath $0))
PY=${PY:-/workspace/isaaclab/_isaac_sim/python.sh}
ckpt=${CKPT:-${SCRIPT_DIR}/assets/ckpts/prop5+diff-pos-b+future-pos-anchor_hist25_rt-imu_odom-dly-drift_imu-error_micpos-u-l-w3.02.0sig1e-42e-3_odom-dly-drift.onnx}
# ckpt=${CKPT:-${SCRIPT_DIR}/assets/ckpts/ft/prop5+diff-pos-b+future-pos-anchor_hist25_rt-imu_odom-dly-drift_micpos-u-l-w3.02.0sig1e-42e-3_odom-dly-drift_ft3000.onnx}

# motion=${MOTION_NPZ:-${SCRIPT_DIR}/assets/motions/humanml_002153.npz}
# motion=${MOTION_NPZ:-${SCRIPT_DIR}/assets/motions/humanml_004669.npz}
# motion=${MOTION_NPZ:-${SCRIPT_DIR}/assets/motions/AMASS_MPI_mosh_00093_stretches_poses_seg2_blend1.0.npz}
motion=${MOTION_NPZ:-${SCRIPT_DIR}/assets/motions/G2-Sidekick_leading_left_poses_blend2.0_return1.0.npz}

# stand=${STAND_NPZ:-${SCRIPT_DIR}/assets/motions/A1-Stand_poses.npz}
stand=${STAND_NPZ:-${SCRIPT_DIR}/assets/motions/A1-Stand_poses_blend1.0.npz}
key_args=()
if [ -n "${KEYS:-}" ]; then key_args+=(--keys "${KEYS}"); fi
if [ -n "${MAX_TIME:-}" ]; then key_args+=(--max_time "${MAX_TIME}"); fi

cd "${SCRIPT_DIR}/deploy_sim"
export MUJOCO_GL=${MUJOCO_GL:-egl}
"$PY" sim_odom.py \
    --policy "$(realpath "${ckpt}")" \
    --motion_npz "$(realpath "${motion}")" \
    --stand_npz "$(realpath "${stand}")" \
    --print_odom_hz ${PRINT_ODOM_HZ:-1.0} \
    "${key_args[@]}" \
    "$@"
