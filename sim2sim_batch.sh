#!/bin/bash

set -u
SCRIPT_DIR=$(dirname $(realpath $0))
PY=/workspace/isaaclab/_isaac_sim/python.sh

# motion_file=${MOTION:-${SCRIPT_DIR}/assets/LAAM/A1-Stand_poses.pkl}
# motion_file=${MOTION:-${SCRIPT_DIR}/assets/LAAM/B3-walk1_poses.pkl}
# motion_file=${MOTION:-${SCRIPT_DIR}/assets/LAAM/G2-Sidekick_leading_left_poses.pkl}
# motion_file=${MOTION:-${SCRIPT_DIR}/assets/LAAM/G6-axe_kick_poses.pkl}
# motion_file=${MOTION:-${SCRIPT_DIR}/assets/LAAM/Subject_69_F_21_poses.pkl}

motion_file=${MOTION:-${SCRIPT_DIR}/assets/PHUMA/LAFAN1_walk1_subject1_chunk_0030.pkl}
# motion_file=${MOTION:-${SCRIPT_DIR}/assets/PHUMA/humanml_007615_chunk_0002.pkl}
# motion_file=${MOTION:-${SCRIPT_DIR}/assets/PHUMA/idea400_Simultaneously_walking_and_Kicking_a_soccer_ball_clip1_chunk_0000.pkl}
# motion_file=${MOTION:-${SCRIPT_DIR}/assets/PHUMA/idea400_standing_during_Shuttlecock_Kicking_1_clip1_chunk_0000.pkl}

# motion_file=${MOTION:-${SCRIPT_DIR}/assets/CLAW/CLAW_locrawl_0.pkl}

# ckpt_path=${CKPT:-${SCRIPT_DIR}/assets/ckpts/prop5+GMT+diff-pos-b-deploy_hist25_futu1.onnx}
# ckpt_path=${CKPT:-${SCRIPT_DIR}/assets/ckpts/prop5+GMT+diff-pos-b_hist25_futu1.onnx}
# ckpt_path=${CKPT:-${SCRIPT_DIR}/assets/ckpts/prop5+future-pos-h+future-anchor_hist15_futu1.onnx}
# ckpt_path=${CKPT:-${SCRIPT_DIR}/assets/ckpts/prop5+future-pos-h+future-anchor_feet-pos-exp.onnx}
# ckpt_path=${CKPT:-${SCRIPT_DIR}/assets/ckpts/prop5+future-pos-h+future-anchor_mimic-body_w20.0_sig0.3sq.onnx}

# ckpt_path=${CKPT:-${SCRIPT_DIR}/assets/ckpts/ft/prop5+GMT+diff-pos-b-deploy_hist25_futu1_ft3000.onnx}
# ckpt_path=${CKPT:-${SCRIPT_DIR}/assets/ckpts/ft/prop5+future-pos-h+future-anchor_hist15_futu1_ft3000.onnx}
# ckpt_path=${CKPT:-${SCRIPT_DIR}/assets/ckpts/ft/prop5+future-pos-h+future-anchor_feet-pos-exp_ft3000.onnx}
# ckpt_path=${CKPT:-${SCRIPT_DIR}/assets/ckpts/ft/prop5+future-pos-h+future-anchor_feet-pos-exp_weight40.0_ft500.onnx}
ckpt_path=${CKPT:-${SCRIPT_DIR}/assets/ckpts/ft/prop5+future-pos-h+future-anchor_mimic-body_w20.0_sig0.3sq_ft2000.onnx}

video_path=${VIDEO:-}
# MOTION_SOURCE=local (default): precompute the reference (export_motion_ref.py) and play it
#   inside the controller -- no motion server / Redis. MOTION_NPZ=<file> reuses an export.
# MOTION_SOURCE=redis: legacy motion server streaming over Redis.
MOTION_SOURCE=${MOTION_SOURCE:-local}
MOTION_START_DELAY=${MOTION_START_DELAY:-2.0}   # local: stand time before the motion starts [s]
RECORD_FROM_MOTION_START=${TRIM:+1}
motion_args=("$@")

# absolute paths: the servers run from deploy_real/
motion_file=$(realpath "${motion_file}") || exit 1
ckpt_path=$(realpath "${ckpt_path}") || exit 1
obs_cfg_path=$(realpath "${OBS_CFG:-${ckpt_path%.onnx}.yaml}") || exit 1   # policy spec, shared by both servers
[ -n "${video_path}" ] && video_path=$(realpath -m "${video_path}")

motion_name=$(basename "${motion_file%.pkl}")
ckpt_name=$(basename "${ckpt_path%.onnx}")
[ -z "${video_path}" ] && video_path="${SCRIPT_DIR}/videos/${ckpt_name}__${motion_name}_$(date +%Y%m%d_%H%M%S).mp4"
mkdir -p "$(dirname "${video_path}")"
log_dir="${video_path%.mp4}_logs"; mkdir -p "${log_dir}"

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${GPU}
export MUJOCO_EGL_DEVICE_ID=${GPU}

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH=${SCRIPT_DIR}/pose
cd "${SCRIPT_DIR}/deploy_real"

controller_args=(
    --xml ../assets/g1/g1_sim2sim_29dof_deploy.xml
    --policy "${ckpt_path}"
    --obs_cfg "${obs_cfg_path}"
    --device cpu
    --policy_frequency 50
    --update_robot_w_odom
    --headless --record_video --video_ref --video_path "${video_path}"
    ${RECORD_FROM_MOTION_START:+--record_from_motion_start}
)

## motion - local
if [ "${MOTION_SOURCE}" = "local" ]; then
    # ---- 1. precompute the reference motion (same flags as the motion server) ----
    motion_npz=${MOTION_NPZ:-${log_dir}/${motion_name}.npz}
    if [ -z "${MOTION_NPZ:-}" ]; then
        "$PY" export_motion_ref.py \
            --motion_file "${motion_file}" \
            --obs_cfg "${obs_cfg_path}" \
            --robot unitree_g1_with_hands \
            --steps 5 \
            --blend_in_time 1.0 \
            --playback_speed 1.0 \
            --out "${motion_npz}" \
            "${motion_args[@]}" \
            2>&1 | tee "${log_dir}/export.log"
        [ ${PIPESTATUS[0]} -eq 0 ] || { echo "export failed, see ${log_dir}/export.log"; exit 1; }
    fi
    motion_npz=$(realpath "${motion_npz}") || exit 1

    # ---- 2. controller plays the motion in-process, exits 1s after the return-to-default ----
    # Nothing external to stay in sync with, so the sim runs as fast as it can (LIMIT_FPS=1: realtime).
    "$PY" server_low_level_g1_sim.py "${controller_args[@]}" \
        --limit_fps ${LIMIT_FPS:-0} \
        --motion_npz "${motion_npz}" \
        --motion_start_delay "${MOTION_START_DELAY}" \
        --exit_after_motion 1.0 \
        < /dev/null > "${log_dir}/controller.log" 2>&1
    rc=$?
    [ ${rc} -eq 0 ] || { echo "controller failed (${rc}), see ${log_dir}/controller.log"; tail -20 "${log_dir}/controller.log"; exit 1; }
    echo "[batch] video: ${video_path}"
    echo "[batch] logs : ${log_dir}/"
    exit 0
fi

## motion - redis
[ "${MOTION_SOURCE}" = "redis" ] || { echo "MOTION_SOURCE must be local or redis"; exit 1; }
redis-cli ping >/dev/null 2>&1 || redis-server --daemonize yes --bind 127.0.0.1 --save "" --appendonly no
redis-cli ping >/dev/null 2>&1 || { echo "redis-server failed to start"; exit 1; }
redis-cli flushall >/dev/null   # no stale targets / anchors from a previous run

# ---- 1. controller (background, headless, records video) ----
setsid "$PY" server_low_level_g1_sim.py "${controller_args[@]}" \
    --limit_fps 1 \
    < /dev/null > "${log_dir}/controller.log" 2>&1 &
ctrl_pid=$!
stop_controller() { kill -TERM -- -${ctrl_pid} 2>/dev/null; wait ${ctrl_pid} 2>/dev/null; }
trap 'stop_controller' EXIT

# wait until the controller is publishing its state, then let it settle in the stand pose
for _ in $(seq 1 100); do
    redis-cli exists t_state | grep -q 1 && break
    kill -0 ${ctrl_pid} 2>/dev/null || break
    sleep 0.2
done
kill -0 ${ctrl_pid} 2>/dev/null || { echo "controller died, see ${log_dir}/controller.log"; tail -20 "${log_dir}/controller.log"; exit 1; }
echo "[batch] controller up (pid ${ctrl_pid}); settling 1s"; sleep 1

# ---- 2. motion server (foreground; exits after playback + 2s return-to-default) ----
"$PY" server_motion_lib.py \
    --motion_file "${motion_file}" \
    --obs_cfg "${obs_cfg_path}" \
    --robot unitree_g1_with_hands \
    --redis_ip localhost \
    --steps 5 \
    --blend_in_time 1.0 \
    --playback_speed 1.0 \
    "${motion_args[@]}" \
    2>&1 | tee "${log_dir}/motion_server.log"

# ---- 3. give the controller 1s after the motion ends, then stop it (finally: saves the mp4) ----
sleep 1
stop_controller
trap - EXIT
echo "[batch] video: ${video_path}"
echo "[batch] logs : ${log_dir}/"
