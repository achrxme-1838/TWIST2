#!/bin/bash

script_dir=$(dirname $(realpath $0))
# motion_file="${script_dir}/assets/example_motions/0807_yanjie_walk_005.pkl"

motion_file="${script_dir}/assets/example_motions/A1-Stand_poses.pkl"
# motion_file="${script_dir}/assets/example_motions/A6_lift_box_poses.pkl"
# motion_file="${script_dir}/assets/example_motions/B3-walk1_poses.pkl"
# motion_file="${script_dir}/assets/example_motions/A6-lift_box_t2_poses.pkl"

# motion_file="${script_dir}/assets/example_motions/0014_catching_and_throwing_poses.pkl"
# motion_file="${script_dir}/assets/example_motions/0022_jumping1_poses.pkl"       
# motion_file="${script_dir}/assets/example_motions/0016_sitting2_poses.pkl"       
# motion_file="${script_dir}/assets/example_motions/Trial_upper_left_225_poses.pkl"       
# motion_file="${script_dir}/assets/example_motions/Subject_1_F_1_poses.pkl"       
# motion_file="${script_dir}/assets/example_motions/E5-hook_left_poses.pkl"


#### MAPO TEST (suc / fail)
# STABLE
# motion_file="${script_dir}/assets/MAPO_demo/0022_throwing_hard1_poses.pkl"  # 3 / 0
# motion_file="${script_dir}/assets/MAPO_demo/0024_throwing_hard3_poses.pkl"  # 3 / 0
# motion_file="${script_dir}/assets/MAPO_demo/0026_kicking2_poses.pkl"  # 3 / 0
# motion_file="${script_dir}/assets/MAPO_demo/0026_kicking2_poses.pkl"  # 3 / 0
# motion_file="${script_dir}/assets/MAPO_demo/29_11_poses.pkl"  # 3 / 0
# motion_file="${script_dir}/assets/MAPO_demo/bow_deep03_poses.pkl"  # 3 / 0
# motion_file="${script_dir}/assets/MAPO_demo/chicken04_poses.pkl"  # 3 / 0
# motion_file="${script_dir}/assets/MAPO_demo/step_over_gap08_poses.pkl"  # 5 / 0
# motion_file="${script_dir}/assets/MAPO_demo/Subject_17_F_2_poses.pkl"  # 5 / 0
# motion_file="${script_dir}/assets/MAPO_demo/Subject_44_F_12_poses.pkl"  # 1 / 0
# motion_file="${script_dir}/assets/MAPO_demo/Subject_67_F_7_poses.pkl"  # 1 / 0
# motion_file="${script_dir}/assets/MAPO_demo/Subject_71_F_13_poses.pkl"  # 1 / 0
# motion_file="${script_dir}/assets/MAPO_demo/Subject_74_F_10_poses.pkl"  # 1 / 0
# motion_file="${script_dir}/assets/MAPO_demo/throw_toss-15-pass_to_left_light-hamada_poses.pkl"  # 1 / 0
# motion_file="${script_dir}/assets/MAPO_demo/walking_run04_poses.pkl"  # 5 / 0
# motion_file="${script_dir}/assets/MAPO_demo/wave_both10_poses.pkl"  # 5 / 0
# motion_file="${script_dir}/assets/example_motions/A6_lift_box_poses.pkl"  # 5 / 0
# motion_file="${script_dir}/assets/example_motions/B3-walk1_poses.pkl"  # 5 / 0
# motion_file="${script_dir}/assets/example_motions/A6-lift_box_t2_poses.pkl"  # 5 / 0
# motion_file="${script_dir}/assets/example_motions/0014_catching_and_throwing_poses.pkl"  # 5 / 0
# motion_file="${script_dir}/assets/example_motions/Trial_upper_left_225_poses.pkl"       
# motion_file="${script_dir}/assets/example_motions/E5-hook_left_poses.pkl"       
# motion_file="${script_dir}/assets/example_motions/E4-cross_right_poses.pkl"       
# motion_file="${script_dir}/assets/example_motions/E9-body_hook_left_poses.pkl"       
# motion_file="${script_dir}/assets/example_motions/E8-bounce_poses.pkl"       
# motion_file="${script_dir}/assets/example_motions/E22-duck_right_poses.pkl"       
# motion_file="${script_dir}/assets/example_motions/E19-dodge_left_t2_poses.pkl"       
# motion_file="${script_dir}/assets/example_motions/defaults/0807_yanjie_walk_003.pkl"       



# UNSTABLE
# motion_file="${script_dir}/assets/MAPO_demo/0016_sitting2_poses.pkl"  # 9 / 1
# motion_file="${script_dir}/assets/MAPO_demo/0017_lifting_light2_poses.pkl"  # 9 / 1
# motion_file="${script_dir}/assets/MAPO_demo/0020_lifting_heavy2_poses.pkl"  # 3 / 0
# motion_file="${script_dir}/assets/MAPO_demo/84_05_poses.pkl"  # 9 / 1
# motion_file="${script_dir}/assets/MAPO_demo/dance_turntwist180_poses.pkl"  # 5 / 0
# motion_file="${script_dir}/assets/MAPO_demo/G15-roundhouse_body_right_poses.pkl"  # 3 / 0
# motion_file="${script_dir}/assets/MAPO_demo/PushBeam_10_poses.pkl"  # 8 / 2
# motion_file="${script_dir}/assets/example_motions/0022_jumping1_poses.pkl"  # 5 / 0


# Long Motions
# motion_file="${script_dir}/assets/example_motions/LAFAN1_robotdance1_subject3.pkl" 
# motion_file="${script_dir}/assets/example_motions/LAFAN1_robotdance2_subject3.pkl" 





# Change to deploy_real directory
cd deploy_real

# by default we use our own laptop as the redis server
redis_ip="localhost"
# this is my unitree g1's ip in wifi
# redis_ip="192.168.110.24"


# Run the motion server
python server_motion_lib.py \
    --motion_file ${motion_file} \
    --robot unitree_g1_with_hands \
    --vis \
    --redis_ip ${redis_ip} \
    --fix_root_pos \
    --fix_root_heading \
    --blend_in_time 3.0 \
    --playback_speed 0.5
    # --playback_speed 0.5
    # --send_start_frame_as_end_frame \
    # --use_remote_control \
