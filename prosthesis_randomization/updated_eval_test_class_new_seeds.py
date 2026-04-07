# CORRECT FILE TO RUN SEEDS IN PARALLEL FOR DIFFERENT RANDOMIZATION SETTINGS
# path has different agents (diff seeds) in pkl file

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"

import jax 
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import pickle
import numpy as np
import argparse
import sys
from loco_mujoco.core.wrappers import VecEnv
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from omegaconf import OmegaConf
from loco_mujoco.evaluation.evaluation_prosthesis import ProsthesisMetricsHandler
from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction
import mujoco
from datetime import datetime
import timeit 

os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering, which is more compatible with headless environments


# --- Setup Argument Parser ---
parser = argparse.ArgumentParser(description='Run parallel seed evaluation.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
args = parser.parse_args()

# --- Load Agent ---
path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config
n_seeds = config.experiment.n_seeds  # Get total seeds from config


subfolder_name = 'parallel_eval_pylon_socket_ori_z'

randomization_params_names = ["prosthesis_dof_damping", "prosthesis_joint_stiffness", "prosthesis_body_position", "prosthesis_body_orientation"]
# --- Environment Setup ---
randomization_params_eval = {
    "prosthesis_side": "left_side",
    "randomize_prosthesis_dof_damping": False,
    "prosthesis_dof_damping_range": {'ankle_angle': [2, 10]},
    "randomize_prosthesis_joint_stiffness": False,
    "prosthesis_joint_stiffness_range": {'ankle_angle': [1000, 1300]},
    "randomize_prosthesis_body_position": False,
    "prosthesis_body_position_range": {'pylon_socket': {'z': [-0.010,0.010]}},
    # "prosthesis_body_position_range": {'pylon_socket': {'x': [-0.010,-0.005]}},
    "randomize_prosthesis_body_orientation": True,
    # "prosthesis_body_orientation_range": {'pylon_socket': {'x': [-np.deg2rad(6), -np.deg2rad(3)]}} 
    "prosthesis_body_orientation_range": {'pylon_socket': {'z': [-np.deg2rad(6), np.deg2rad(6)]}} 
}

randomization_increments = {
    "prosthesis_joint_stiffness": 100, #10,
    "prosthesis_dof_damping": 5,
    "prosthesis_body_position": 0.005, #0.05,
    "prosthesis_body_orientation":  np.deg2rad(6)/2, #0.35 #0.001 #0.2
}

talus_ori_unequal_0 = False

factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

OmegaConf.set_struct(config, False)
config.experiment.env_params["headless"] = True #False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic
config.experiment.env_params["add_sensors"] = True

randomization_type = config.randomization_config["randomization_type"]
randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)
config.experiment.env_params["horizon"] = 3000 
# update values in randommization_params based on randomization_params_eval
for key, value in randomization_params_eval.items():
    if key in randomization_params:
        # If the key exists, update it
        randomization_params[key] = value
    else:
        # If the key does not exist, add it
        randomization_params[key] = value

# add prosthesis side to randomization params if it exists in config.experiment.env_params
        if "prosthesis_side" in config.experiment.env_params:
            randomization_params["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]
# print('randomization_params: ', randomization_params)
env = factory.make(domain_randomization_type=randomization_type, 
                   domain_randomization_params=randomization_params, #_eval,
                   **config.experiment.env_params, **config.experiment.task_factory.params)

env.th.to_jax()
env = VecEnv(env)
model = env.get_model()
prosthesis_metrics_handler = ProsthesisMetricsHandler(env)
muscle_skeleton_control_activation = SkeletonMuscleControlFunction(env)



# --- Vectorized Functions ---
# We vmap over the first axis (the seed dimension)
v_reset = jax.jit(jax.vmap(env.mjx_reset))
v_step = jax.jit(jax.vmap(env.mjx_step))

# rngs = [jax.random.PRNGKey(i) for i in range(n_seeds+1)]  # create rngs from seed
# rng, rng_seeds = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:])) 

# seed = 0
# rng = jax.random.key(seed)
# keys = jax.random.split(rng, n_seeds + 1)
# rng, rng_seeds = keys[0], keys[1:]

rng_seeds = jnp.array([jax.random.PRNGKey(i) for i in range(1,n_seeds+1)])
rng = jax.random.PRNGKey(n_seeds)

def sample_actions_uncompiled(ts, obs, _rng):
    y, updates = agent_conf.network.apply({'params': ts.params, 'run_stats': ts.run_stats},
                                           obs, mutable=["run_stats"])
    ts = ts.replace(run_stats=updates['run_stats'])
    pi, _ = y
    return pi.sample(seed=_rng), ts

v_sample_actions = jax.jit(jax.vmap(sample_actions_uncompiled))

# --- Parallel Evaluation Logic ---
def run_evaluation_parallel(n_steps,n_seeds, env_states,agent_state,rng, save_dir, param_info=None):#subfolder, dt_init, param_info=None):
    

    prosthesis_side_suffix = "_l" if env.prosthesis_side == "_left_side" else "_r"
    foot_name = "toes" #+ prosthesis_side_suffix
    calcn_name = "calcn" #+ prosthesis_side_suffix
    all_grf_l = []
    all_grf_r = []
    body_names = []
    for i in range(model.nbody):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        body_names.append(joint_name)        

    body_xposes = {}
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        body_xposes[body_name] = []

    body_cvels = body_xposes.copy()

    joint_names = []
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_names.append(joint_name)

    joint_data = {}
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_data[joint_name] = {
            "angle": [],
            "velocity": [],
            "forces_constraint": [],
            "forces_smooth": [],
            "forces_applied": [] ,
            "torques": [],
            "energy_exp": [],
        }
        # Per-step data
        if "_l" in joint_name or "_r" in joint_name:
            joint_data[joint_name].update({
                "angle_per_step": [],
                "velocity_per_step": [],
                "forces_constraint_per_step": [],
                "forces_smooth_per_step": [],
                "forces_applied_per_step": [],
                "torques_per_step": [],
                "energy_exp_per_step": [],
            })
        else:
            joint_data[joint_name].update({
                "angle_per_step_left": [],
                "angle_per_step_right": [],
                "velocity_per_step_left": [],
                "velocity_per_step_right": [],
                "forces_constraint_per_step_left": [],
                "forces_constraint_per_step_right": [],
                "forces_smooth_per_step_left": [],
                "forces_smooth_per_step_right": [],
                "forces_applied_per_step_left": [],
                "forces_applied_per_step_right": [],
                "torques_per_step_left": [],
                "torques_per_step_right": [],
                "energy_exp_per_step_left": [],
                "energy_exp_per_step_right": [],
            })

    # all_tibia_sensor_data = []
    all_sensor_force = {}


    evaluation_muscle_groups = ["back_muscles", "torso_muscles", "vasti_muscles", "rectus_muscles", "medial_muscles", "leg_muscles", "all_muscles"]


    evaluation_muscle_names = {}
    for n in evaluation_muscle_groups:
        evaluation_muscle_names[n] = prosthesis_metrics_handler.get_muscle_group(n)


    for muscle_group, muscle_names in evaluation_muscle_names.items():
        locals()[f"run_{muscle_group}_activation_left"] = []
        locals()[f"run_{muscle_group}_activation_right"] = []
    
    all_actions = []
    all_actuator_names = []
    left_side = 'left_side'
    right_side = 'right_side'

    # Pre-identify muscle actuators
    muscle_indices = []
    for i in range(model.nu):
        if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE:
            muscle_indices.append(i)
    muscle_activations = {}
    for muscle_group in evaluation_muscle_groups:
        muscle_activations[f"{muscle_group}_left"] = []
        muscle_activations[f"{muscle_group}_right"] = []
    # Prepare all seeds
    # rng = jax.random.key(n_seeds)
    # rng_seeds = jax.random.split(rng, n_envs + 1)
   
    # rngs = [jax.random.PRNGKey(i) for i in range(n_seeds+1)]  # create rngs from seed
    # rng, action_rngs = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
    
    # Reset all seeds in parallel
    # env_states = v_reset(rng_seeds)
    step_rngs = rng
    
    
    # Current train states for all seeds
    curr_train_states = agent_state.train_state

    print(f"Executing {n_steps} steps for {n_seeds} seeds in parallel...")
    
    for t in range(n_steps):
        step_rngs_split = jax.vmap(lambda r: jax.random.split(r, 2))(step_rngs)
        step_rngs = step_rngs_split[:, 0]  # Keep for next iteration
        action_rngs = step_rngs_split[:, 1]

        # action_rngs = jax.random.split(rng, n_seeds)
        #  # # # rng, action_rngs = step_rngs[0], step_rngs[1:]
        
        # 1. Get actions for all seeds
        obs = env_states.observation
        actions, curr_train_states = v_sample_actions(curr_train_states, obs, action_rngs)
        
        # 2. Apply Sigmoid for muscles (vectorized)
        # # Note: In a real vmap, we index with [..., i]
        # for i in range(model.nu):
        #     if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE:
        #         actions = actions.at[..., i].set(muscle_ctrl.adapted_sigmoid(actions[..., i]))

        # 3. Step physics for all seeds
        env_states = v_step(env_states, actions)

        if (t % 500) == 0:
            print(f"  Step {t}/{n_steps} completed.")

        # GRF collection
        grf_foot_l = prosthesis_metrics_handler.get_grf_batched(env_states.data, f"{foot_name}_l")
        grf_foot_r = prosthesis_metrics_handler.get_grf_batched(env_states.data, f"{foot_name}_r")
        grf_calcn_l = prosthesis_metrics_handler.get_grf_batched(env_states.data, f"{calcn_name}_l")
        grf_calcn_r = prosthesis_metrics_handler.get_grf_batched(env_states.data, f"{calcn_name}_r")

        all_grf_l.append(grf_foot_l + grf_calcn_l)
        all_grf_r.append(grf_foot_r + grf_calcn_r)

        # Body positions and velocities
        body_xpos = prosthesis_metrics_handler.get_xpos_batched(env_states.data)
        body_cvel = prosthesis_metrics_handler.get_cvel_batched(env_states.data)
        for name in body_names: 
            body_xposes[name].append(body_xpos[name])
            body_cvels[name].append(body_cvel[name])

        # Joint data
        joint_angles = prosthesis_metrics_handler.get_joint_angles(env_states.data)
        joint_velocities = prosthesis_metrics_handler.get_joint_vels(env_states.data)
        joint_forces_constraint, joint_forces_smooth, joint_forces_applied = prosthesis_metrics_handler.get_joint_frces(env_states.data)
        joint_torques = prosthesis_metrics_handler.get_joint_trques(env_states.data)
        joint_energy_exp = prosthesis_metrics_handler.calc_joint_energy_exp(joint_torques, joint_velocities)
        
        for joint_name in joint_names:
            joint_data[joint_name]["angle"].append(joint_angles[joint_name])
            joint_data[joint_name]["velocity"].append(joint_velocities[joint_name])
            joint_data[joint_name]["forces_constraint"].append(joint_forces_constraint[joint_name])
            joint_data[joint_name]["forces_smooth"].append(joint_forces_smooth[joint_name])
            joint_data[joint_name]["forces_applied"].append(joint_forces_applied[joint_name])
            joint_data[joint_name]["torques"].append(joint_torques[joint_name])
            joint_data[joint_name]["energy_exp"].append(joint_energy_exp[joint_name])


        sensor_force, sensor_force_names = prosthesis_metrics_handler.get_sensor_data_batched_2(env_states.data)
        for sensor_name, force in sensor_force.items():
            if sensor_name not in all_sensor_force:
                all_sensor_force[sensor_name] = []
            all_sensor_force[sensor_name].append(force)

        # Action processing - apply sigmoid to muscles
        action_processed = actions.copy()
        for muscle_idx in muscle_indices:
            action_processed = action_processed.at[..., muscle_idx].set(
                muscle_skeleton_control_activation.adapted_sigmoid(action_processed[..., muscle_idx])
            )
        all_actions.append(action_processed)

        # Muscle activations
        for muscle_group, muscle_names in evaluation_muscle_names.items():
            left_activation = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, left_side, action_processed)
            right_activation = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, right_side, action_processed)
            muscle_activations[f"{muscle_group}_left"].append(left_activation)
            muscle_activations[f"{muscle_group}_right"].append(right_activation)



    # Post-processing: compute muscle activations
    sum_step_activations = {}
    step_norm_activations = {}
    step_num_norm_activations = {}
    sum_muscles_activations = {}
    step_muscles_norm_activations = {}

    for muscle_group in evaluation_muscle_groups:
        left_key = f"{muscle_group}_left"
        right_key = f"{muscle_group}_right"
        
        left_array = jnp.array(muscle_activations[left_key])
        right_array = jnp.array(muscle_activations[right_key])
        # both_array = jnp.concatenate([left_array, right_array], axis=-1)
        
        sum_step_activations[left_key] = jnp.sum(left_array, axis=0)
        sum_step_activations[right_key] = jnp.sum(right_array, axis=0)
        # sum_step_activations[f"{muscle_group}_both"] = sum_step_activations[left_key] + sum_step_activations[right_key]

        step_norm_activations[left_key] = sum_step_activations[left_key] / n_steps
        step_norm_activations[right_key] = sum_step_activations[right_key] / n_steps
        # step_norm_activations[f"{muscle_group}_both"] = sum_step_activations[f"{muscle_group}_both"] / n_steps

        n_musc = len(evaluation_muscle_names[muscle_group])
        sum_muscles_activations[left_key] = jnp.sum(sum_step_activations[left_key], axis=0)
        sum_muscles_activations[right_key] = jnp.sum(sum_step_activations[right_key], axis=0)
        sum_muscles_activations[f"{muscle_group}_both"] = sum_muscles_activations[left_key] + sum_muscles_activations[right_key] #jnp.sum(sum_step_activations[f"{muscle_group}_both"], axis=0)
        
        step_muscles_norm_activations[left_key] = sum_muscles_activations[left_key] / n_steps
        step_muscles_norm_activations[right_key] = sum_muscles_activations[right_key] / n_steps
        step_muscles_norm_activations[f"{muscle_group}_both"] = sum_muscles_activations[f"{muscle_group}_both"] / n_steps
        
        # step_num_norm_activations[left_key] = jnp.sum(step_muscles_norm_activations[left_key] / n_musc)
        # step_num_norm_activations[right_key] = jnp.sum(step_muscles_norm_activations[right_key] / n_musc)
        # step_num_norm_activations[f"{muscle_group}_both"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_both"] / (2 * n_musc))

        step_num_norm_activations[left_key] = step_muscles_norm_activations[left_key] / n_musc
        step_num_norm_activations[right_key] = step_muscles_norm_activations[right_key] / n_musc
        step_num_norm_activations[f"{muscle_group}_both"] = step_muscles_norm_activations[f"{muscle_group}_both"] / (2 * n_musc)
   

    # Helper function to extract checkpoint-specific data from nested structures
    def extract_checkpoint_data(data, checkpoint_idx):
        """
        Extract the checkpoint_idx slice from all arrays in nested structures.
        For body_xposes: {body: [step0_array(num_ckpt, 3), step1_array(num_ckpt, 3), ...]}
        Result: {body: [step0_array(3,), step1_array(3,), ...]}  - all steps for one checkpoint
        """
        if isinstance(data, dict):
            # For dicts, recursively extract from all values
            return {k: extract_checkpoint_data(v, checkpoint_idx) for k, v in data.items()}
        elif isinstance(data, list) and len(data) > 0:
            # For lists of arrays (like steps), extract checkpoint_idx from each array in the list
            result = []
            for step_array in data:
                if isinstance(step_array, (jnp.ndarray, list)):
                    try:
                        # Extract checkpoint_idx from this step's array
                        result.append(step_array[checkpoint_idx])
                    except (IndexError, TypeError):
                        result.append(step_array)
                else:
                    # Scalar or other type - keep as is
                    result.append(step_array)
            return result
        elif isinstance(data, (jnp.ndarray, list)) and hasattr(data, '__getitem__'):
            # Single array: try to extract
            try:
                return data[checkpoint_idx]
            except (IndexError, TypeError):
                return data
        else:
            return data
    
    # Save results for each checkpoint
    for seed_idx in range(n_seeds):
        # checkpoint_file = checkpoint_files[batch_idx]
        print(f"Processing {seed_idx+1}") #/{num_checkpoints}]: {checkpoint_file}")
        
        # Extract all per-checkpoint data using the helper function
        all_relevant_data = {
            "total_steps": n_steps,
            # "n_envs": n_envs,
            # "total_reward": reward_scalars[batch_idx],
            # "avg_reward_per_step": reward_scalars[batch_idx] / step_count if step_count > 0 else 0.0,
            "all_grf_l": extract_checkpoint_data(all_grf_l, seed_idx),
            "all_grf_r": extract_checkpoint_data(all_grf_r, seed_idx),
            # "all_foot_ground_contact_left": extract_checkpoint_data(all_foot_ground_contact_left, seed_idx),
            # "all_foot_ground_contact_right": extract_checkpoint_data(all_foot_ground_contact_right, seed_idx),
            "all_sensor_force": extract_checkpoint_data(all_sensor_force, seed_idx),
            "sensor_force_names": sensor_force_names,
            "evaluation_muscle_groups": evaluation_muscle_groups,
            "evaluation_joint_names": joint_names,
            "evaluation_muscle_names": evaluation_muscle_names,
            "all_actions": extract_checkpoint_data(all_actions, seed_idx),
            "all_actuator_names": all_actuator_names,
            "all_body_poses": extract_checkpoint_data(body_xposes, seed_idx),
            "all_body_vels": extract_checkpoint_data(body_cvels, seed_idx),
            "evaluation_body_names": body_names,
        }


        # Add muscle activation summaries for each checkpoint to all_relevant_data
        all_relevant_data["sum_step_activations"] = {k: v[seed_idx] for k, v in sum_step_activations.items()}
        all_relevant_data["step_norm_activations"] = {k: v[seed_idx] for k, v in step_norm_activations.items()}
        all_relevant_data["sum_muscles_activations"] = {k: v[seed_idx] for k, v in sum_muscles_activations.items()}
        all_relevant_data["step_muscles_norm_activations"] = {k: v[seed_idx] for k, v in step_muscles_norm_activations.items()}
        all_relevant_data["step_num_norm_activations"] = {k: v[seed_idx] for k, v in step_num_norm_activations.items()}
        
       
        # Add joint data for this checkpoint
        for joint_name in joint_names:
            for key in joint_data[joint_name].keys():
                all_relevant_data[f"{joint_name}_{key}"] = extract_checkpoint_data(joint_data[joint_name][key], seed_idx)

        # Add muscle activations for this checkpoint
        for muscle_group in evaluation_muscle_groups:
            all_relevant_data[f"run_{muscle_group}_activation_left"] = extract_checkpoint_data(muscle_activations.get(f"{muscle_group}_left", []), seed_idx)
            all_relevant_data[f"run_{muscle_group}_activation_right"] = extract_checkpoint_data(muscle_activations.get(f"{muscle_group}_right", []), seed_idx)

        if 'orientation' in param_info['name']:
            name_param_value = np.rad2deg(param_info['val'])
            name_units = 'deg'
        elif 'position' in param_info['name']:
            name_param_value = param_info['val']*1000
            name_units = 'mm'
        else: 
            name_param_value = param_info['val']
            name_units = ''

        tag = f"eval_{n_steps}steps_{param_info['name']}_{param_info['body']}_{param_info['direction']}_{int(np.round(name_param_value,0))}{name_units}" if param_info else "base"
        filename = f"seed{seed_idx}.pkl"
        # out_path = os.path.join(save_dir, filename)

        # Create folder save_dir/tag/ if it doesn't exist
        out_folder = os.path.join(save_dir, tag)
        os.makedirs(out_folder, exist_ok=True)
        out_path =  os.path.join(out_folder, filename)
        
        with open(out_path, "wb") as f:
            pickle.dump(all_relevant_data, f)
        print(f"Saved Seed {seed_idx} -> {filename}")

# --- Main Randomization Loop ---
# subfolder = "parallel_eval"
dt_init = datetime.now().strftime("%Y%m%d_%H%M%S")

save_dir = os.path.join(os.path.dirname(path), f'{dt_init}_{subfolder_name}')
os.makedirs(save_dir, exist_ok=True)

# Iterate over params in randomization_params_names 
for param_name in randomization_params_names:
    # Set all randomization flags to False initially to isolate the current parameter
    for name in randomization_params_names:
        env._domain_randomizer.rand_conf[f"randomize_{name}"] = False

    # Only proceed if the current parameter is set to be randomized in the eval config
    if randomization_params_eval[f"randomize_{param_name}"]:

        # Set the current parameter's randomization flag to True
        env._domain_randomizer.rand_conf[f"randomize_{param_name}"] = True

        if talus_ori_unequal_0 and param_name == "prosthesis_body_position":
            # Manually set talus orientation to 0 deg when position is being evaluated
            env._domain_randomizer.rand_conf["randomize_prosthesis_body_orientation"] = True
            env._domain_randomizer.rand_conf["prosthesis_body_orientation_range"] = randomization_params_eval["prosthesis_body_orientation_range"]
        
        if "stiffness" in param_name or "damping" in param_name:
            # The range is now a dictionary, not a simple list
            joint_name = list(randomization_params_eval[f"{param_name}_range"].keys())[0] # ONLY 1. body_name in dictionary is incrementaly changed
            min_val, max_val = randomization_params_eval[f"{param_name}_range"][joint_name]
            increment = randomization_increments[param_name]

            # Iterate through the range with the specified increment
            current_value = min_val
            while current_value <= max_val:
                # Update the randomization range for the specific joint to be a fixed value
                env._domain_randomizer.rand_conf[f"{param_name}_range"][joint_name] = [current_value, current_value]
                
                # env_keys = jax.random.split(rng, 2)
                v_reset  = jax.jit(jax.vmap(env.mjx_reset)) #env.reset)
                env_states = v_reset(rng_seeds) 
                # obs = env_state.observation
                print("Running evaluation for ", param_name, " with value: ", current_value)
                run_evaluation_parallel(n_steps=2000,
                                            n_seeds = n_seeds,
                                            env_states=env_states,
                                            agent_state=agent_state,
                                            rng=rng_seeds, #rng,
                                            save_dir=save_dir,
                                            # subfolder=subfolder_name,
                                            # dt_init=dt_init,
                                            param_info={"name": param_name, "val": current_value, "direction": axis, "body": body_name})
                                            # param_info={"name": param_name, "val": current_value})
                #run_evaluation_loop(env_state, n_steps, train_state, rng, prosthesis_metrics_handler,param_name=param_name, param_value = current_value, subfolder_name = subfolder_name,dt_str_init=dt_str_init) #min_val + increment)
                current_value += increment

        elif "position" in param_name or "orientation" in param_name:
            body_range = randomization_params_eval[f"{param_name}_range"]
            body_name = list(body_range.keys())[0] # ONLY 1. body_name in dictionary is incrementaly changed if randomization flag is true 
            directions = list(body_range[body_name].keys())

            for axis in directions:
                min_val = body_range[body_name][axis][0]
                max_val = body_range[body_name][axis][1]
                inc = randomization_increments[param_name]

                # Use a small tolerance for floating-point comparison
                current_value = min_val
                while current_value <= max_val + 1e-6:
                    # Reset all axes to a zero range first
                    for other_axis in directions:
                        env._domain_randomizer.rand_conf[f"{param_name}_range"][body_name][other_axis] = [0, 0]

                    # Set the current axis to a fixed value for evaluation
                    env._domain_randomizer.rand_conf[f"{param_name}_range"][body_name][axis] = [current_value, current_value]
                    
                    # Reset to new domain randomization state
                    # Re jit reset for correct update of env_state?
                    
                    v_reset  = jax.jit(jax.vmap(env.mjx_reset))
                    # env_keys = jax.random.split(rng, 2)

                    # rngs = [jax.random.PRNGKey(i) for i in range(n_seeds+1)]  # create rngs from seed
                    # rng, rng_seeds = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:])) 
                    env_states = v_reset(rng_seeds)
                    # obs = env_state.observation
                    print("Running evaluation for ", param_name," for axis: ", axis, " with value: ",current_value)
                    run_evaluation_parallel(n_steps=2000,
                                            n_seeds = n_seeds,
                                            env_states=env_states,
                                            agent_state=agent_state,
                                            rng=rng_seeds, #rng,
                                            save_dir=save_dir,
                                            # subfolder=subfolder_name,
                                            # dt_init=dt_init,
                                            param_info={"name": param_name, "val": current_value, "direction": axis, "body": body_name})
                                            #train_state, rng, prosthesis_metrics_handler,param_name=param_name, param_value = current_value,direction=axis, subfolder_name = subfolder_name, dt_str_init=dt_str_init)
                    # env_state, n_steps, train_state, rng, prosthesis_metrics_handler,param_name=param_name, param_value = current_value,direction=axis, subfolder_name = subfolder_name, dt_str_init=dt_str_init) #min_val + inc)
                    current_value += inc

            # # Set values and call parallel runner
            # run_evaluation_parallel(
            #     n_steps=2000, 
            #     agent_state=agent_state, 
            #     subfolder=subfolder, 
            #     dt_init=dt_init,
            #     param_info={"name": param_name, "val": current_val}
            # )