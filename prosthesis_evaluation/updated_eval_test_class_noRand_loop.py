import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"

import jax 
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp

import pickle


import argparse

from loco_mujoco.core.wrappers import VecEnv


from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.environments.humanoids.skeleton_prosthesis import MjxSkeletonMuscleProsthesis


from loco_mujoco.algorithms.ppo_jax import PPOAgentConf, PPOAgentState
from omegaconf import OmegaConf

from loco_mujoco.evaluation.evaluation_prosthesis import ProsthesisMetricsHandler

from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction

import mujoco
from datetime import datetime

import timeit 

os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering, which is more compatible with headless environments
# os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_triton_gemm_any=True ')

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--folder_path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--config_file', type=int, default=0, help='Configuration file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

# Use the path from command line arguments
folder_path = args.folder_path
# iterate through all files in the folder and run each pkl file in sequence 

config_file = args.config_file
agent_conf= PPOJax.load_agent_conf(config_file)
config = agent_conf.config

dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# if 'checkpoints' in folder_path:
#     i = path.rfind("/")
#     tail = path[i+1:]
#     tail = tail[:-4]
#     video_name = f'{tail}'
# else: 
#     video_name = f'lastCkpt'

# # viewer params must match MujocoViewer signature
# viewer_params = {
#     "default_camera_mode": "follow",
#     "recorder_params": {
#         "path": os.path.dirname(path),
#         "video_name": video_name,
#     },
# }



# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = True #False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic
config.experiment.env_params["add_sensors"] = True

# # Ensure reward_params exists and add missing defaults
if "reward_params" not in config.experiment.env_params or config.experiment.env_params["reward_params"] is None:
    config.experiment.env_params["reward_params"] = OmegaConf.create({})

# # Convert existing reward_params to a plain dict, merge defaults into it, then recreate a DictConfig
# rp_existing = config.experiment.env_params["reward_params"]
# rp_container = OmegaConf.to_container(rp_existing, resolve=True) if rp_existing is not None else {}
# _defaults = {
#     # "qpos_w_sum": 2*1.6, 
#     # "qvel_w_sum": 2*0.8,
#     # "rpos_w_sum": 2*2.0,
#     # "rquat_w_sum": 2*1.2,
#     # "rvel_w_sum": 2*0.4,
#     # "action_coeff": 0.002, #0.005, #0.015, #0.02, 
#     "action_muscle_coeff": 0.01,
#     "action_motor_coeff": 0.2, #0.3,


#     "grf_coeff": 0.1, #0.07281
#     "grf_threshold": 1.4,
#     "torque_at_limit_coeff": 0.01, #0.02,#0.04, #0.01, #0.1, #0.1307
#     "action_rate_coeff": 0.4, #0.6, #0.1, # 0.097
#     "action_threshold": 0.15, 
#     "action_out_of_bounds_coeff": 0.2, #0.05, # 1.57929
    
#     "lateral_range_coeff": 0, #0.3,
#     "lateral_pos_reward_range": 0.5,
    
#     "joint_limit_threshold_slide": 0.003,
#     "joint_limit_threshold_hinge": 0.1, #0.14,
    
#     "target_body": "pelvis",
#     "target_velocity": 1.2,
#     "vel_coeff": 1.0, #1.3,

#     "joint_torque_vel_arm_coeff": 0.0008,

#     "target_velocity_z": 0.0,
#     "vel_z_coeff": 0.2, 
#     "target_body_z": "pelvis",
# }
# # # vel: 10.0 
# # # CLIP ACTIONS? 

# # Merge defaults with existing values (existing values override defaults)
# merged = {**_defaults, **(rp_container or {})}
# # Re-create a DictConfig from the merged dict to avoid modifying a structured DictConfig in-place
# rp = OmegaConf.create(merged)
# config.experiment.env_params["reward_params"] = rp
# config.experiment.env_params["reward_type"] = "MimicRewardEmergenceNatural"

# config.experiment.env_params["reward_params"]["qpos_w_sum"] = 2.8
# config.experiment.env_params["reward_params"]["qvel_w_sum"] = 1.4
# config.experiment.env_params["reward_params"]["rpos_w_sum"] = 3.5
# config.experiment.env_params["reward_params"]["rquat_w_sum"] = 2.1
# config.experiment.env_params["reward_params"]["rvel_w_sum"] = 0.7
# config.experiment.env_params["reward_params"]["action_out_of_bounds_coeff"] = 0.07
# config.experiment.env_params["reward_params"]["action_coeff"] = 0.12
# config.experiment.env_params["reward_params"]["lateral_range_coeff"] = 0.7
# config.experiment.env_params["reward_params"]["lateral_pos_reward_range"] = 0.5

# config.experiment.env_params["reward_params"]["action_coeff"] = 0
# config.experiment.env_params["reward_params"]["joint_torque_vel_nonarm_coeff"] = 0.00003 #0.0003
# config.experiment.env_params["reward_params"]["joint_torque_vel_arm_coeff"] = 0.0001 #0.0002





# config.experiment.env_params["reward_params"]["action_coeff"] = 0.03
# config.experiment.env_params["joint_stiffness"] = {'ankle_angle': 1200, 'mtp_angle': 2000}
# config.experiment.env_params["contact_geom_type"] = 'sphere_scone_exact' #"2box"  #"box"  #"sphere"

# # # config.experiment.env_params['limit_knee_extension'] = True
# config.experiment.env_params["knee_extension_limit"] = {'ankle_angle_r': [-13,7], 'ankle_angle_l': [-7,10], 'hip_flexion_r': [-18,25],'hip_flexion_l': [-18,35], 'knee_angle_r': [-70,0]} #{'ankle_angle_r': [-13,7]}
# # # # config.experiment.env_params["joint_stiffness"] = {'ankle_angle': 700} #1200}

# config.experiment.env_params["joint_stiffness_each_joint"] = {'ankle_angle_l': 700} #{'ankle_angle_r': 40, 'ankle_angle_l': 700}
# # config.experiment.env_params["joint_damping_each_joint"] = {'ankle_angle_r': 10}
config.experiment.env_params["horizon"] = 3000  #2000 #1000 # Use MuJoCo for evaluation if specified
# config.experiment.env_params["socket_ty_slack"] = False
# config.experiment.env_params["socket_type"] = 'Auto'
# config.experiment.socket_ty_joint_range = [-0.5, 0.5] #[-0.09, 0.09]
# config.experiment.socket_ty_joint_stiffness = 100000 #43500

# config.experiment.env_params["contact_solref"] = [-800, -300] #[-1000,-300] # [0.02, 1] #[-800, -600]  #[0.02, 1] # [-1000, -400]  
# config.experiment.env_params["socket_ty_joint_stiffness"] = 263500
# config.experiment.env_params["socket_ty_joint_damping"] = 10
# config.experiment.env_params["socket_ty_joint_range"] = [-0.001,0.001]
# delete 'knee_extension_limit'' from config
# del config.experiment.env_params["knee_extension_limit'"]
# config.experiment.env_params["knee_extension_limit"] = - 10 
# config.experiment.env_params["socket_ty_joint_stiffness"] = 41500
# config.experiment.env_params["joint_stiffness"] = {'ankle_angle': 1500}
# config.experiment.env_params["socket_ty_joint"] = False
# config.experiment.env_params["tibia_socket_overlap"] = 0.1 
# config.experiment.env_params["amputated_tibia_length"] = 0.1  # Use MuJoCo for evaluation if specified


# randomization_params["randomize_prosthesis_body_position"] = True
# randomization_params["prosthesis_body_position_range"] = {'pylon_socket': {'x': [0.05,0.05]}} #[-0.01,-0.01]}} 
env = factory.make(
    **config.experiment.env_params,
    **config.experiment.task_factory.params,
    # terrain_type="RoughTerrain", terrain_params=dict(random_min_height=-0.05, random_max_height=0.05),
    # domain_randomization_type=randomization_type, domain_randomization_params=randomization_params,
    # **viewer_params,
)
env.th.to_jax()
env = VecEnv(env)
jit_step  = jax.jit(jax.vmap(env.mjx_step))  #env.step)
jit_reset  = jax.jit(jax.vmap(env.mjx_reset)) #env.reset)
model = env.get_model()

def sample_actions_uncompiled(ts, obs, _rng):
    y, updates = agent_conf.network.apply({'params': ts.params,
                                        'run_stats': ts.run_stats},
                                        obs, mutable=["run_stats"])
    ts = ts.replace(run_stats=updates['run_stats'])
    pi, _ = y
    a = pi.sample(seed=_rng) 
    return a, ts

sample_actions = jax.jit(sample_actions_uncompiled)

prosthesis_metrics_handler = ProsthesisMetricsHandler(env) #(config, env)

n_steps = 3000 #1000 #3000 #3000 #3000 #1000 #200 #400 #1000 #00 #1000
n_envs = 1 #1  # <--- Make sure this matches your training batch size
rng = jax.random.key(0)
train_state_seed = 0 #0  # Take first seed 

keys = jax.random.split(rng, n_envs + 1)
rng, env_keys = keys[0], keys[1:]
env_state = jit_reset(env_keys) #env.reset(env_keys)
obs = env_state.observation

muscle_skeleton_control_activation = SkeletonMuscleControlFunction(env)

# Pre-compile common metadata lookups (do once, not per file)
body_names = []
for i in range(model.nbody):
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    body_names.append(body_name)

joint_names = []
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    joint_names.append(joint_name)

all_actuator_names = []
for a in range(model.nu):
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    all_actuator_names.append(actuator_name)

# Pre-identify which actuators are muscles
muscle_indices = []
for i in range(model.nu):
    if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE:
        muscle_indices.append(i)

# Get muscle evaluation groups once
evaluation_muscle_groups = ["back_muscles", "torso_muscles", "vasti_muscles", "rectus_muscles", "medial_muscles", "leg_muscles", "all_muscles"]
evaluation_muscle_names = {}
for n in evaluation_muscle_groups:
    evaluation_muscle_names[n] = prosthesis_metrics_handler.get_muscle_group(n)

# Sort files for deterministic processing
pkl_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pkl")])

for file in pkl_files:
    if file.endswith(".pkl"):
        path = os.path.join(folder_path, file)
        print(f"Found agent file: {path}")
        agent_conf, agent_state = PPOJax.load_agent(path)
        config = agent_conf.config

        # def sample_actions_uncompiled(ts, obs, _rng):
        #     y, updates = agent_conf.network.apply({'params': ts.params,
        #                                         'run_stats': ts.run_stats},
        #                                         obs, mutable=["run_stats"])
        #     ts = ts.replace(run_stats=updates['run_stats'])
        #     pi, _ = y
        #     a = pi.sample(seed=_rng) 
        #     return a, ts

        # sample_actions = jax.jit(sample_actions_uncompiled)

        step_total = 0

        if config.experiment.n_seeds > 1:
            assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
                                                    "train_state_seed for replay.")
            train_state = jax.tree.map(lambda x: x[train_state_seed], agent_state.train_state)
        else: 
            train_state = agent_state.train_state

        # Initialize data containers for this checkpoint
        all_foot_ground_contact_left = []
        all_foot_ground_contact_right = []       
        
        body_xposes = {name: [] for name in body_names}
        body_cvels = {name: [] for name in body_names}
        
        joint_data = {}
        for joint_name in joint_names:
            joint_data[joint_name] = {
                "angle": [],
                "velocity": [],
                "forces_constraint": [],
                "forces_smooth": [],
                "forces_applied": [],
                "torque": [],
                "energy_exp": [],
            }
        
        all_sensor_force = {}
        
        # Initialize muscle activation tracking using a dict instead of locals()
        muscle_activations = {}
        for muscle_group in evaluation_muscle_groups:
            muscle_activations[f"{muscle_group}_left"] = []
            muscle_activations[f"{muscle_group}_right"] = []

        all_grf_l = []
        all_grf_r = []
        all_actions = []

        # Metrics collection
        time_all = []
        total_reward = 0.0
        time_all.append(timeit.default_timer())
        # Body name with possible contact to ground 
        foot_name = "toes"
        calcn_name = "calcn"
        left_side = "left_side"
        right_side = "right_side"

        for i in range(n_steps):
            rng, _rng = jax.random.split(rng)
            action, train_state = sample_actions(train_state, obs, _rng)
            action = jnp.atleast_2d(action)

            env_state = jit_step(env_state, action)
            obs = env_state.observation
            total_reward += env_state.reward

            if step_total % 500 == 0:
                print(f"Step {step_total}")

            # Collect metrics efficiently in one pass
            contact_left, contact_right = prosthesis_metrics_handler.get_contact_steps(env_state.data, i)
            all_foot_ground_contact_left.append(contact_left)
            all_foot_ground_contact_right.append(contact_right)

            # GRF collection
            grf_foot_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_l")
            grf_foot_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_r")
            grf_calcn_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{calcn_name}_l")
            grf_calcn_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{calcn_name}_r")

            all_grf_l.append(grf_foot_l + grf_calcn_l)
            all_grf_r.append(grf_foot_r + grf_calcn_r)

            # Body positions and velocities (batch collect)
            body_xpos = prosthesis_metrics_handler.get_xpos(env_state.data)
            body_cvel = prosthesis_metrics_handler.get_cvel(env_state.data)
            for name in body_names: 
                body_xposes[name].append(body_xpos[name])
                body_cvels[name].append(body_cvel[name])

            # Joint data (collect all at once)
            joint_angles = prosthesis_metrics_handler.get_joint_angles(env_state.data)
            joint_velocities = prosthesis_metrics_handler.get_joint_vels(env_state.data)
            joint_forces_constraint, joint_forces_smooth, joint_forces_applied = prosthesis_metrics_handler.get_joint_frces(env_state.data)
            joint_torques = prosthesis_metrics_handler.get_joint_trques(env_state.data)
            joint_energy_exp = prosthesis_metrics_handler.calc_joint_energy_exp(joint_torques, joint_velocities)
            
            for joint_name in joint_names:
                joint_data[joint_name]["angle"].append(joint_angles[joint_name])
                joint_data[joint_name]["velocity"].append(joint_velocities[joint_name])
                joint_data[joint_name]["forces_constraint"].append(joint_forces_constraint[joint_name])
                joint_data[joint_name]["forces_smooth"].append(joint_forces_smooth[joint_name])
                joint_data[joint_name]["forces_applied"].append(joint_forces_applied[joint_name])
                joint_data[joint_name]["torque"].append(joint_torques[joint_name])
                joint_data[joint_name]["energy_exp"].append(joint_energy_exp[joint_name])

            # Sensor data
            sensor_force, sensor_force_names = prosthesis_metrics_handler.get_sensor_data(env_state.data)
            for sensor_name, force in sensor_force.items():
                if sensor_name not in all_sensor_force:
                    all_sensor_force[sensor_name] = []
                all_sensor_force[sensor_name].append(force)

            # Action processing - apply sigmoid to muscle actuators
            action_processed = action.copy()
            for muscle_idx in muscle_indices:
                action_processed = action_processed.at[..., muscle_idx].set(
                    muscle_skeleton_control_activation.adapted_sigmoid(action_processed[..., muscle_idx])
                )
            all_actions.append(action_processed)

            # Muscle activations (efficient dict-based collection)
            for muscle_group, muscle_names in evaluation_muscle_names.items():
                left_activation = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, left_side, action_processed)
                right_activation = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, right_side, action_processed)
                muscle_activations[f"{muscle_group}_left"].append(left_activation)
                muscle_activations[f"{muscle_group}_right"].append(right_activation)

            step_total += n_envs 
            env.mjx_render(env_state, record=True)
        print('TOTAL REWARD: ', total_reward)
        print('TOTAL REWARD/Step_total: ', total_reward/step_total)
        time_all.append(timeit.default_timer())

        env.stop()

        # ACTIONS - Compute activations efficiently
        sum_step_activations = {}
        step_norm_activations = {}
        step_num_norm_activations = {}
        sum_muscles_activations = {}
        step_muscles_norm_activations = {}

        for muscle_group in evaluation_muscle_groups:
            left_key = f"{muscle_group}_left"
            right_key = f"{muscle_group}_right"
            
            # Convert lists to arrays once
            left_array = jnp.array(muscle_activations[left_key])
            right_array = jnp.array(muscle_activations[right_key])
            
            # Sum across steps
            sum_step_activations[left_key] = jnp.sum(left_array, axis=0)
            sum_step_activations[right_key] = jnp.sum(right_array, axis=0)

            # Normalize by steps
            step_norm_activations[left_key] = sum_step_activations[left_key] / n_steps
            step_norm_activations[right_key] = sum_step_activations[right_key] / n_steps

            # Sum across muscles
            n_musc = len(evaluation_muscle_names[muscle_group])
            sum_muscles_activations[left_key] = jnp.sum(sum_step_activations[left_key], axis=0)
            sum_muscles_activations[right_key] = jnp.sum(sum_step_activations[right_key], axis=0)
            
            step_muscles_norm_activations[left_key] = sum_muscles_activations[left_key] / n_steps
            step_muscles_norm_activations[right_key] = sum_muscles_activations[right_key] / n_steps
            
            step_num_norm_activations[left_key] = jnp.sum(step_muscles_norm_activations[left_key] / n_musc)
            step_num_norm_activations[right_key] = jnp.sum(step_muscles_norm_activations[right_key] / n_musc)

        # Collect all relevant data into a dictionary
        all_relevant_data = {
            "total_steps": step_total,
            "n_envs": n_envs,
            "all_grf_l": all_grf_l,
            "all_grf_r": all_grf_r,
            "all_foot_ground_contact_left": all_foot_ground_contact_left,
            "all_foot_ground_contact_right": all_foot_ground_contact_right,
            "all_sensor_force": all_sensor_force,
            "sensor_force_names": sensor_force_names,
            "sum_step_activations": sum_step_activations,
            "step_norm_activations": step_norm_activations,
            "sum_muscles_activations": sum_muscles_activations,
            "step_muscles_norm_activations": step_muscles_norm_activations,
            "step_num_norm_activations": step_num_norm_activations,
            "evaluation_muscle_groups": evaluation_muscle_groups,
            "evaluation_joint_names": joint_names,
            "evaluation_muscle_names": evaluation_muscle_names,
            "all_actions": all_actions,
            "all_actuator_names": all_actuator_names,
            "all_body_poses": body_xposes,
            "all_body_vels": body_cvels,
            "evaluation_body_names": body_names,
        }

        # Add joint data from joint_data dictionary
        for joint_name, joint_dict in joint_data.items():
            for key, value in joint_dict.items():
                all_relevant_data[f"{joint_name}_{key}"] = value

        # Add muscle activations
        for muscle_group in evaluation_muscle_groups:
            all_relevant_data[f"run_{muscle_group}_activation_left"] = muscle_activations.get(f"{muscle_group}_left", [])
            all_relevant_data[f"run_{muscle_group}_activation_right"] = muscle_activations.get(f"{muscle_group}_right", [])

        # Save to file
        tail = file[:-4]  # Remove .pkl extension
        output_path = os.path.join(os.path.dirname(folder_path), f"{dt_str}_{tail}_evaluation_results_{n_steps}steps_{train_state_seed}seed.pkl")
        
        with open(output_path, "wb") as f:
            pickle.dump(all_relevant_data, f)
        print(f"Saved evaluation data to {output_path}")


        #### Testing reading the data back from the file
        # # read the data back from the file
        # with open(output_path, "rb") as f:
        #     loaded_data = pickle.load(f)
        #     # print(f"Loaded data: {loaded_data.keys()}")  # Print the keys of the loaded data to verify



        time_all.append(timeit.default_timer())

        print(f"Total time taken for {n_steps} steps: {time_all[-1] - time_all[0]} seconds")
        print(f"Compilation time: {time_all[1] - time_all[0]} seconds")
        print(f"Execution time taken: {time_all[-1] - time_all[1]} seconds")
        print(f"Average time per step: {(time_all[-1] - time_all[1]) / (n_steps-1)} seconds")