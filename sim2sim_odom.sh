#!/bin/bash
# MuJoCo sim2sim of the real-robot controller with LIO odometry + global origin set at A
# (deploy_sim/sim_odom.py). Headless: the run is recorded to logs/sim2sim_odom/<run>/video.mp4
# together with odom_<run>.npz / .png (estimate vs ground truth) and summary.json.
#
# Interactive (keyboard, real time; needs a tty -> docker exec -it ...):
#     bash sim2sim_odom.sh
#   keys: S default pose | L launch the LIO | A origin + policy (stand; hoist lowered)
#         | B play motion (after the stand / previous motion) | C exit | 2..9/X gain | P push | H hoist
# Scripted (as fast as possible):
#     KEYS="s@0.5,l@3,a@4,b@11" MAX_TIME=30 bash sim2sim_odom.sh
# Any extra flag goes to sim_odom.py, e.g.
#     bash sim2sim_odom.sh --origin_ref imu --lio_rate 50 --lio_pos_std 0.001 --odom_ma 3
#     bash sim2sim_odom.sh --root_source gt          # baseline: GT root in the same G
set -u
SCRIPT_DIR=$(dirname $(realpath $0))
PY=${PY:-/workspace/isaaclab/_isaac_sim/python.sh}
# ckpt=${CKPT:-${SCRIPT_DIR}/assets/ckpts/prop5+diff-pos-b+future-pos-anchor_hist25_rt-imu_odom-dly-drift_micpos-u-l-w3.02.0sig1e-42e-3_odom-dly-drift.onnx}
ckpt=${CKPT:-${SCRIPT_DIR}/assets/ckpts/prop5+diff-pos-b+future-pos-anchor_hist25_rt-imu_odom-dly-drift_micpos-u-l-w3.02.0sig1e-43e-3_feet-posw3.0sig01sq_odom-dly-drift.onnx}
# ckpt=${CKPT:-${SCRIPT_DIR}/assets/ckpts/ft/prop5+diff-pos-b+future-pos-anchor_hist25_rt-imu_odom-dly-drift_micpos-u-l-w3.02.0sig1e-42e-3_odom-dly-drift_ft3000.onnx}

motion=${MOTION_NPZ:-${SCRIPT_DIR}/assets/motions/B3-walk1_poses.npz}
# motion=${MOTION_NPZ:-${SCRIPT_DIR}/assets/motions/fitness_subset_0110_Windmill_clip_2.npz}
# motion=${MOTION_NPZ:-${SCRIPT_DIR}/assets/motions/humanml_011875.npz}
# motion=${MOTION_NPZ:-${SCRIPT_DIR}/assets/motions/fitness_subset_0066_Single_To_Double_Butt_Kick_clip_10.npz}

stand=${STAND_NPZ:-${SCRIPT_DIR}/assets/motions/A1-Stand_poses.npz}
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
