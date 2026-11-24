#!/bin/bash

# Sim-to-sim LoRA fine-tuning (SLowRL) inside the isaaclab docker
# Usage:
#   bash finetune_docker.sh <student name> [--set section.field=value ...]


SCRIPT_DIR=$(dirname $(realpath $0))
PY=${PY:-/root/isaaclab_ws/.venvs/mjwarp/bin/python}

name=${1:?student name required}
shift

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export DEX_RSL_RL_ROOT=${DEX_RSL_RL_ROOT:-/root/isaaclab_ws/DEX_RL_LAB_PHUMA}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-/root/isaaclab_ws/.cache/warp}

cd ${SCRIPT_DIR}/finetune/scripts
"$PY" train.py --name "${name}" "$@"
