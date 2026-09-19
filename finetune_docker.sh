#!/bin/bash
# Sim-to-sim LoRA fine-tuning (SLowRL) inside the isaaclab docker
# Usage:
#   bash finetune_docker.sh <motion.pkl> <student name>
# bashfinetune_docker.sh ~/isaaclab_ws/DEX_RL_LAB_PHUMA/data_processing/motion_retargeting/data/integrated_motions/Representative_set_260629.pkl prop5+GMT+diff-pos-b_hist25_futu1 --set env.num_envs=1 critic.warmup_iters=200

SCRIPT_DIR=$(dirname $(realpath $0))
PY=${PY:-/workspace/isaaclab/_isaac_sim/python.sh}

motion=${1:?motion pkl required}
name=${2:-prop5+GMT+diff-pos-b_hist25_futu1}
shift 2 2>/dev/null || shift $#

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export DEX_RSL_RL_ROOT=${DEX_RSL_RL_ROOT:-/root/isaaclab_ws/DEX_RL_LAB_PHUMA}
export MUJOCO_GL=${MUJOCO_GL:-egl}

cd ${SCRIPT_DIR}/finetune/scripts
"$PY" train.py --motion "${motion}" --name "${name}" "$@"
