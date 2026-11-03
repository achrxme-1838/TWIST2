#!/bin/bash
# Real G1 deploy: server_low_level_g1_real.py with the motion played in-process (no Redis).
#
#   1. export the motion once, in the Isaac env (needs MotionLib / CUDA), e.g.
#        cd deploy_real && PYTHONPATH=../pose $ISAAC_PY export_motion_ref.py \
#            --motion_file ../assets/PHUMA/xxx.pkl --obs_cfg <ckpt>.yaml \
#            --steps 5 --blend_in_time 1.0 --playback_speed 1.0 --out ../assets/motions/xxx.npz
#   2. robot: hoist, power on, zero-torque, L2+R2 (debug mode); FAST-LIO publishing /Odometry
#   3. bash sim2real.sh
#      remote: START -> default pose, A -> policy, B -> play motion (again after it ends),
#      Select -> damping + exit. Terminal keys 2..9 / X raise the PD gain (starts at 10%).
#
#   no robot: MOCK=1 bash sim2real.sh [--record_video [--video_ref] [--video_path X.mp4]]
#      --record_video runs deploy_real/mock_video.py (mp4 -> logs/real/, MOCK only)

# source ~/miniconda3/bin/activate twist2

set -u
SCRIPT_DIR=$(dirname $(realpath $0))

# ckpt_path=${CKPT:-${SCRIPT_DIR}/assets/ckpts/prop5+future-pos-h+future-anchor_mimic-body_w20.0_sig0.3sq.onnx}
ckpt_path=${CKPT:-${SCRIPT_DIR}/assets/ckpts/prop5+diff-pos-b+future-pos-h+future-anchor_upde-sig-per_mimic-body-w5.0_sig-init0.3sq.onnx}

# ckpt_path=${CKPT:-${SCRIPT_DIR}/assets/ckpts/ft/prop5+future-pos-h+future-anchor_mimic-body_w20.0_sig0.3sq_ft2000.onnx}

motion_npz=${MOTION_NPZ:-${SCRIPT_DIR}/assets/motions/LAFAN1_fight1_subject3_chunk_0001-0002.npz}

root_source=${ROOT_SOURCE:-odom}
# PY=${PY:-python}
PY=/workspace/isaaclab/_isaac_sim/python.sh

# net=enp0s31f6
# net=eno1
net=eth0

ckpt_path=$(realpath "${ckpt_path}") || exit 1
motion_npz=$(realpath "${motion_npz}") || exit 1

if [ "${root_source}" = "odom" ] && [ -z "${MOCK:-}" ] && [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

# no robot (MuJoCo)
mock_args=()
if [ -n "${MOCK:-}" ]; then
    mock_args=(--mock --initial_gain_scale 1.0 --exit_after_motion 1.0)
fi

# --record_video: mp4 of the mock run via mock_video.py (wraps server_low_level_g1_real.py)
script=server_low_level_g1_real.py
extra_args=()
for a in "$@"; do
    if [ "$a" = "--record_video" ]; then
        script=mock_video.py
    else
        extra_args+=("$a")
    fi
done
if [ "${script}" = "mock_video.py" ] && [ -z "${MOCK:-}" ]; then
    echo "--record_video is only available with MOCK=1"; exit 1
fi

cd "${SCRIPT_DIR}/deploy_real"
"$PY" "${script}" \
    --policy "${ckpt_path}" \
    --motion_npz "${motion_npz}" \
    --net "${net}" \
    --device cpu \
    --root_source "${root_source}" \
    --odom_topic /Odometry \
    --record_proprio \
    "${mock_args[@]}" \
    "${extra_args[@]}"
    # --use_hand \
    # --kp_scale 1.0 --kd_scale 1.0 \
    # --smooth_action 0.5 \
    # --odom_offset X Y Z ROLL PITCH YAW   # FAST-LIO body frame in torso_link (default: URDF mid360)
