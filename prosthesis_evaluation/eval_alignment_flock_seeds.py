import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
jax.config.update('jax_platform_name', 'cpu')

import re
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

import argparse

import loco_mujoco.evaluation.evaluation_prosthesis as eval_pros
postprocess_handler = eval_pros.PostProcessMetricsHandler()

import importlib

# input --path to folder with different eval runs subfolders
# Set up argument parser
parser = argparse.ArgumentParser(description='Run alignment evaluation.')
parser.add_argument('--path', type=str, required=True, help='Path to the folder containing evaluation subfolders.')
args = parser.parse_args()

# Only read joint data from file
with open("/home/nadinebadie/loco-mujoco/prosthesis_evaluation/joint_data_walk_SkeletonMuscle.pkl", "rb") as f:
    joint_data_baseline = pickle.load(f)


folder_name = args.path

all_loaded_data = {}

for subfolder in os.listdir(folder_name):
    subfolder_path = os.path.join(folder_name, subfolder)
    if not os.path.isdir(subfolder_path):
        continue
    
    # Extract info from subfolder name: eval_3000steps_pylon_socket_x_-2deg
    match = re.search(r'eval_\d+steps_(.+?)_(.+?)_(.+?)_(.+)$', subfolder)
    if not match:
        continue
    
    # Extract direction (letter before value) and value with units from the tail part
    tail = match.group(4)  # e.g., 'pylon_socket_z_-6deg'
    tail_parts = tail.rsplit('_', 2)
    if len(tail_parts) != 3:
        continue

    name_param = tail_parts[0]     # e.g., 'pylon_socket'
    direction = tail_parts[1]      # e.g., 'z'
    param_value_units = tail_parts[2]  # e.g., '-6deg'
    # Extract numerical value and units from param_value_units
    value_match = re.search(r'(-?\d+\.?\d*)([a-zA-Z]+)', param_value_units)
    if not value_match:
        continue
    
    param_value = value_match.group(1)
    units = value_match.group(2)
    
    # Create category key
    category_key = f"{direction}_{param_value}{units}"
    
    if category_key not in all_loaded_data:
        all_loaded_data[category_key] = {}
    
    # Load all seed files in this subfolder
    for fname in os.listdir(subfolder_path):
        if fname.endswith(".pkl") and fname.startswith("seed"):
            seed_match = re.search(r'seed(\d+)', fname)
            if not seed_match:
                continue
            seed_num = seed_match.group(1)
            
            file_path = os.path.join(subfolder_path, fname)
            with open(file_path, "rb") as f:
                loaded_data = pickle.load(f)
                all_loaded_data[category_key][f"seed{seed_num}"] = loaded_data
                print(f"Loaded {category_key}/seed{seed_num}")


# FINAL CODE 
# Get steps from GRF instead of Contact 
min_walk_step_length_left = 30
min_walk_step_length = 40 #70 #60
run_step_data={}

for run_key, seeds_dict in all_loaded_data.items():
    for seed_key, run_dict in seeds_dict.items():
        all_grf_l = np.array(run_dict.get("all_grf_l"))
        all_grf_r = np.array(run_dict.get("all_grf_r"))
        z_grf_l = all_grf_l[:, 5]
        z_grf_r = all_grf_r[:, 5]

        step_start_left = postprocess_handler.get_start_steps_from_grfZ(z_grf_l, min_walk_step_length=min_walk_step_length_left)
        step_start_right = postprocess_handler.get_start_steps_from_grfZ(z_grf_r, min_walk_step_length=min_walk_step_length)
        contact_length_left = postprocess_handler.get_contact_lengths_from_grfZ(z_grf_l, min_walk_step_length=min_walk_step_length_left, threshold=0.5)
        contact_length_right = postprocess_handler.get_contact_lengths_from_grfZ(z_grf_r, min_walk_step_length=min_walk_step_length, threshold=0.5)
        all_contact_left = postprocess_handler.get_contact_from_grfZ(z_grf_l)
        all_contact_right = postprocess_handler.get_contact_from_grfZ(z_grf_r)
        single_support_right_time_per_step, single_support_left_time_per_step = postprocess_handler.compute_single_support_per_step(
            all_contact_right, all_contact_left, step_start_right, step_start_left
        )

        if run_key not in run_step_data:
            run_step_data[run_key] = {}
        
        run_step_data[run_key][seed_key] = {#[run_step_data_key] = {
            "step_start_left": step_start_left,
            "step_start_right": step_start_right,
            "contact_length_left": contact_length_left,
            "contact_length_right": contact_length_right,
            "all_contact_left": all_contact_left,
            "all_contact_right": all_contact_right,
            "single_support_right_per_step": single_support_right_time_per_step,
            "single_support_left_per_step": single_support_left_time_per_step
        }


# Take out 1st step in right & left side for all runs in run_step_data
for run_key in run_step_data.keys():
    for seed_key in run_step_data[run_key].keys():
        run_step_data[run_key][seed_key]['step_start_right'] = run_step_data[run_key][seed_key]['step_start_right'][2:]
        run_step_data[run_key][seed_key]['step_start_left'] = run_step_data[run_key][seed_key]['step_start_left'][2:]



left_switches, right_switches = postprocess_handler.plot_grf_Fz_switch_all_dirs(
        #grf_components,
        all_loaded_data,
        run_step_data,
        interp_len=100,
        body_weight_to_normalize = 86.6*9.81,
    )


sensor_body_names = ['pylon'] #, 'socket']

knee_alignment_baseline_file = "/home/nadinebadie/loco-mujoco/prosthesis_evaluation/Alignment_Data_Schmalz_2002.csv"
knee_alignment_data_mass = 85#*9.81 # std 15

pylon_alignment_baseline_file = "/home/nadinebadie/loco-mujoco/prosthesis_evaluation/Kobayashi_SACH_optimalAlignment_socketMoment_2016.csv"

direction = "z"
postprocess_handler.plot_sensor_force_pylon_all_dirs_stance_sorted_flock_each_seed(all_loaded_data, run_step_data, direction, left_switches, right_switches,folder_name, interp_len=100, 
                                                                         body_weight_to_normalize=86.6, 
                                                                         smooth_data=True, 
                                                                         body_in_sensor_names = sensor_body_names,
                                                                         knee_alignment_baseline_file = knee_alignment_baseline_file, 
                                                                         knee_alignment_data_mass = knee_alignment_data_mass,
                                                                         prosthesis_side = 'left',#'left', 
                                                                         pylon_optimal_alignment_baseline_file = pylon_alignment_baseline_file
                                                                         )

direction = "x"
postprocess_handler.plot_sensor_force_pylon_all_dirs_stance_sorted_flock_each_seed(all_loaded_data, run_step_data, direction, left_switches, right_switches,folder_name, interp_len=100, 
                                                                         body_weight_to_normalize=86.6, 
                                                                         smooth_data=True, 
                                                                         body_in_sensor_names = sensor_body_names,
                                                                         knee_alignment_baseline_file = knee_alignment_baseline_file, 
                                                                         knee_alignment_data_mass = knee_alignment_data_mass,
                                                                         prosthesis_side = 'left',#'left', 
                                                                         pylon_optimal_alignment_baseline_file = pylon_alignment_baseline_file
                                                                         )


sensor_body_names = ['knee'] #['pylon', 'socket']
knee_alignment_baseline_file = "/home/nadinebadie/loco-mujoco/prosthesis_evaluation/Alignment_Data_Schmalz_2002.csv"
knee_alignment_data_mass = 85#*9.81 # std 15
pylon_alignment_baseline_file = "/home/nadinebadie/loco-mujoco/prosthesis_evaluation/Kobayashi_SACH_optimalAlignment_socketMoment_2016.csv"
direction = "z"
postprocess_handler.plot_sensor_force_pylon_all_dirs_stance_sorted_flock(all_loaded_data, run_step_data, direction, left_switches, right_switches,folder_name, interp_len=100, 
                                                                         body_weight_to_normalize=86.6, 
                                                                         smooth_data=True, 
                                                                         body_in_sensor_names = sensor_body_names,
                                                                         knee_alignment_baseline_file = knee_alignment_baseline_file, 
                                                                         knee_alignment_data_mass = knee_alignment_data_mass,
                                                                         prosthesis_side = 'left',#'left', 
                                                                         pylon_optimal_alignment_baseline_file = pylon_alignment_baseline_file
                                                                         )





sensor_body_names = ['pylon', 'socket']
postprocess_handler.plot_sensor_force_pylon_all_dirs_stance_sorted_flock(all_loaded_data, run_step_data, direction, left_switches, right_switches,folder_name, interp_len=100, 
                                                                         body_weight_to_normalize=86.6, 
                                                                         smooth_data=True, 
                                                                         body_in_sensor_names = sensor_body_names,
                                                                         knee_alignment_baseline_file = knee_alignment_baseline_file, 
                                                                         knee_alignment_data_mass = knee_alignment_data_mass,
                                                                         prosthesis_side = 'left',#'left', 
                                                                         pylon_optimal_alignment_baseline_file = pylon_alignment_baseline_file
                                                                         )


direction = "x"
postprocess_handler.plot_sensor_force_pylon_all_dirs_stance_sorted_flock(all_loaded_data, run_step_data, direction, left_switches, right_switches,folder_name, interp_len=100, 
                                                                         body_weight_to_normalize=86.6, 
                                                                         smooth_data=True, 
                                                                         body_in_sensor_names = sensor_body_names,
                                                                         knee_alignment_baseline_file = knee_alignment_baseline_file, 
                                                                         knee_alignment_data_mass = knee_alignment_data_mass,
                                                                         prosthesis_side = 'left',#'left', 
                                                                         pylon_optimal_alignment_baseline_file = pylon_alignment_baseline_file
                                                                         )
