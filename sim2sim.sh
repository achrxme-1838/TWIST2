SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/test/student_fix_both.onnx
# ckpt_path=${SCRIPT_DIR}/assets/ckpts/test/student.onnx
# ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx
# ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_25k.onnx

cd deploy_real

python server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --measure_fps 1 \
    --policy_frequency 50 \
    --limit_fps 1 \
    
    # --policy_frequency 100 \
    # --record_proprio \
