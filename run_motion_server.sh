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
    --fix_root_heading
    # --send_start_frame_as_end_frame \
    # --use_remote_control \
