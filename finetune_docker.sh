#!/bin/bash
# Sim-to-sim LoRA fine-tuning (SLowRL) inside the isaaclab docker
# Usage:
#   bash finetune_docker.sh <motion.pkl> <student name> [--set section.field=value ...]
#
# critic from scratch (default), adaptive-KL lr schedule (default, ppo.schedule=adaptive):
#   bash finetune_docker.sh ~/isaaclab_ws/DEX_RL_LAB_PHUMA/data_processing/motion_retargeting/data/integrated_motions/Representative_set_260629.pkl prop5+GMT+diff-pos-b_hist25_futu1 --set env.num_envs=1 critic.warmup_iters=200
#
# teacher critic (paper setting: frozen teacher critic + LoRA). Its obs spec must match the
# teacher's critic group: finetune/specs/critic_teacher.yaml (regenerate for another teacher with
# finetune/tasks/export_critic_spec.py --run <teacher run> --ckpt <model_N.pt>):
#   bash finetune_docker.sh <motion.pkl> prop5+GMT+diff-pos-b_hist25_futu1 --set \
#       critic.source=teacher critic.adapt=lora \
#       critic.teacher_ckpt=/root/isaaclab_ws/DEX_RL_LAB_PHUMA/scripts/logs/rsl_rl/g1_29dof_mapo/PHUMA_rt-v30_ap0.99_2026-08-21_06-59-55/model_30000.pt

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
