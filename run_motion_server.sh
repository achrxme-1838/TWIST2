#!/bin/bash

script_dir=$(dirname $(realpath $0))

# motion_file="${script_dir}/assets/example_motions/A1-Stand_poses.pkl"

#### Motion Class (suc / fail)
# Standing 
# motion_file="${script_dir}/assets/MAPO_demo/Subject_44_F_12_poses.pkl"  # 1 / 0
# motion_file="${script_dir}/assets/MAPO_demo/Subject_67_F_7_poses.pkl"  # 1 / 0
# motion_file="${script_dir}/assets/MAPO_demo/Subject_71_F_13_poses.pkl"  # 1 / 0  (hi~)
# motion_file="${script_dir}/assets/MAPO_demo/chicken04_poses.pkl"  # 1 / 0
# motion_file="${script_dir}/assets/MAPO_demo/29_11_poses.pkl"  # 2 / 0
# motion_file="${script_dir}/assets/MAPO_demo/wave_both10_poses.pkl"  # 5 / 0
# motion_file="${script_dir}/assets/example_motions/E5-hook_left_poses.pkl"       
# motion_file="${script_dir}/assets/example_motions/E4-cross_right_poses.pkl"       
# motion_file="${script_dir}/assets/example_motions/E9-body_hook_left_poses.pkl"       
# motion_file="${script_dir}/assets/MAPO_demo/throw_toss-15-pass_to_left_light-hamada_poses.pkl"  # 1 / 0
# motion_file="${script_dir}/assets/MAPO_demo/0022_throwing_hard1_poses.pkl"  # 5 / 0
# motion_file="${script_dir}/assets/MAPO_demo/0024_throwing_hard3_poses.pkl"  # 5 / 0
# motion_file="${script_dir}/assets/MAPO_demo/Subject_17_F_2_poses.pkl"  # 5 / 0  (raise hands vertically) (speed 1.0 recommended)
# motion_file="${script_dir}/assets/MAPO_demo/G15-roundhouse_body_right_poses.pkl"  # DANGER!!  (kick) (speed 0.5 is recommended)


# Slight Bending
# motion_file="${script_dir}/assets/MAPO_demo/bow_deep03_poses.pkl"  # 3 / 0  
# motion_file="${script_dir}/assets/example_motions/E22-duck_right_poses.pkl"       
# motion_file="${script_dir}/assets/example_motions/E19-dodge_left_t2_poses.pkl"       
# motion_file="${script_dir}/assets/MAPO_demo/Anna_Angry_C3D_poses.pkl"  # (0.5 speed recommended, ~2000 step) [FIG motion]

# Slight forward moving
# motion_file="${script_dir}/assets/MAPO_demo/0026_kicking2_poses.pkl"  # 5 / 0  (speed 1.0 is recommended)
# motion_file="${script_dir}/assets/MAPO_demo/Subject_74_F_10_poses.pkl"  # 1 / 0 (walking)
# motion_file="${script_dir}/assets/MAPO_demo/walking_run04_poses.pkl"  # 5 / 0 (speed 0.5 MUST)
# motion_file="${script_dir}/assets/MAPO_demo/dance_turntwist180_poses.pkl"  # 5 / 0 (speed 1.0 recommended)

# Squatting
# motion_file="${script_dir}/assets/example_motions/A6_lift_box_poses.pkl"  # 5 / 0 (speed 1.0 is recommended)
# motion_file="${script_dir}/assets/MAPO_demo/22_19_poses.pkl"  # sitting (speed 1.0 is recommended)
# motion_file="${script_dir}/assets/MAPO_demo/Subject_69_F_21_poses.pkl"  # LIL DANGER!! (MUST speed 2.0) [FIG motion]

# Walking farward
# motion_file="${script_dir}/assets/example_motions/B3-walk1_poses.pkl"  # 5 / 0 (speed 0.5 is recommended)
# motion_file="${script_dir}/assets/example_motions/defaults/0807_yanjie_walk_003.pkl"   # 5 / 0 (speed 0.5 / 1000~ step is recommended)
# motion_file="${script_dir}/assets/MAPO_demo/0017_lifting_light2_poses.pkl"  # 9 / 1  (speed 0.5 / 200~ step is recommended)
# motion_file="${script_dir}/assets/MAPO_demo/0020_lifting_heavy2_poses.pkl"  # 3 / 0 (speed 0.5 / 200~ step is recommended)
# motion_file="${script_dir}/assets/MAPO_demo/0030_scamper_poses.pkl"  # DANGER!! (MUST speed 0.5, ~1200 step) [FIG motion]


# Seems unnecessary
# motion_file="${script_dir}/assets/example_motions/Trial_upper_left_225_poses.pkl"
# motion_file="${script_dir}/assets/MAPO_demo/84_05_poses.pkl"  # 9 / 1
# motion_file="${script_dir}/assets/MAPO_demo/PushBeam_10_poses.pkl"  # DANGER!! (default speed/gain is recommended)
# motion_file="${script_dir}/assets/example_motions/0022_jumping1_poses.pkl"  # DANGER!! (default speed/gain is recommended)
# motion_file="${script_dir}/assets/MAPO_demo/Experiment3b_04_poses.pkl"  # DANGER!! (default speed/gain is recommended)
# motion_file="${script_dir}/assets/example_motions/E8-bounce_poses.pkl"       
# motion_file="${script_dir}/assets/example_motions/0014_catching_and_throwing_poses.pkl"  # 5 / 0
# motion_file="${script_dir}/assets/example_motions/A6-lift_box_t2_poses.pkl"  # 5 / 0
# motion_file="${script_dir}/assets/MAPO_demo/step_over_gap08_poses.pkl"  # 3 / 0 (speed 1.0 recommended)
# motion_file="${script_dir}/assets/MAPO_demo/0016_sitting2_poses.pkl"  #  (speed 0.5 / 200~ step is recommended)


###
# motion_file="${script_dir}/assets/MAPO_demo2/G2-Sidekick-leading-left_poses.pkl"
motion_file="${script_dir}/assets/MAPO_demo2/G6-axe-kick_poses.pkl"
# motion_file="${script_dir}/assets/example_motions/A1-Stand_poses.pkl"
# motion_file="${script_dir}/assets/example_motions/B3-walk1_poses.pkl"
# motion_file="${script_dir}/assets/MAPO_demo/Subject_69_F_21_poses.pkl"




# Change to deploy_real directory
cd deploy_real

# by default we use our own laptop as the redis server
redis_ip="localhost"
# this is my unitree g1's ip in wifi
# redis_ip="192.168.0.23"


# Run the motion server
python server_motion_lib.py \
    --motion_file ${motion_file} \
    --robot unitree_g1_with_hands \
    --vis \
    --redis_ip ${redis_ip} \
    --steps 5 \
    --blend_in_time 1.0 \
    --playback_speed 1.0 \
    --max_step 2000 \

    # --fix_root_heading \
    # --fix_root_pos \

    # --start_step 200 \
    # --send_start_frame_as_end_frame \
    # --use_remote_control \
