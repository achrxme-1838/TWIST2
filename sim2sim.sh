SCRIPT_DIR=$(dirname $(realpath $0))
# ckpt_path=${SCRIPT_DIR}/assets/ckpts/test/student_first_success.onnx
# ckpt_path=${SCRIPT_DIR}/assets/ckpts/test/student_mass_rand3.onnx
# ckpt_path=${SCRIPT_DIR}/assets/ckpts/test/student_narrow_frc.onnx
ckpt_path=${SCRIPT_DIR}/assets/ckpts/test/student_dp_dr_rsmp_20k.onnx
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
    --use_diff_body_pos \
    --use_diff_body_tannorm


    # --policy_frequency 100 \
    # --record_proprio \
