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
import sys


subfolder_name = 'pylon_socket_pos_z' #_TalZ5'

dt_str_init = datetime.now().strftime("%Y%m%d_%H%M%S")

talus_ori_ang = np.deg2rad(5)

set_fixed_randomized_targets = True
fixed_randomized_targets = {"prosthesis_body_orientation": {"pylon_socket": {"x": talus_ori_ang}, "talus": {"z": talus_ori_ang}}}

ori_ang_test= np.deg2rad(4)
ori_ang_test_2 = np.deg2rad(2)
ori_ang = np.deg2rad(6)
talus_ori_ang = np.deg2rad(5)

randomization_params_names = ["prosthesis_dof_damping", "prosthesis_joint_stiffness", "prosthesis_body_position", "prosthesis_body_orientation"]

randomization_params_eval = {
    "prosthesis_side": "left_side",
    "randomize_prosthesis_dof_damping": False,
    "prosthesis_dof_damping_range": {'ankle_angle': [2, 10]},
    "randomize_prosthesis_joint_stiffness": False,
    "prosthesis_joint_stiffness_range": {'ankle_angle': [1000, 1300]},
    "randomize_prosthesis_body_position": False,
    "prosthesis_body_position_range": {'pylon_socket': {'x': [-0.010,0.010]}},
    "randomize_prosthesis_body_orientation": True,
    "prosthesis_body_orientation_range": {'pylon_socket': {'z': [-ori_ang, ori_ang]}} 
}

randomization_increments = {
    "prosthesis_joint_stiffness": 100,
    "prosthesis_dof_damping": 5,
    "prosthesis_body_position": 0.005,
    "prosthesis_body_orientation": ori_ang/2,
}

os.environ["MUJOCO_GL"] = "egl"

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
parser.add_argument('--batch_size', type=int, default=4, help='Number of configurations to run in parallel')
args = parser.parse_args()

# Use the path from command line arguments
path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config

randomization_type = config.randomization_config["randomization_type"]
randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)

# update values in randomization_params based on randomization_params_eval
for key, value in randomization_params_eval.items():
    if key in randomization_params:
        randomization_params[key] = value
    else:
        randomization_params[key] = value

if "prosthesis_side" in config.experiment.env_params:
    randomization_params["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]


# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# create env
OmegaConf.set_struct(config, False)
config.experiment.env_params["headless"] = True
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"
config.experiment.env_params["add_sensors"] = True

env = factory.make(domain_randomization_type=randomization_type, domain_randomization_params=randomization_params,
                   **config.experiment.env_params, **config.experiment.task_factory.params)
env.th.to_jax()
env = VecEnv(env)

model = env.get_model()

prosthesis_metrics_handler = ProsthesisMetricsHandler(env)

n_steps = 1000
train_state_seed = 0

muscle_skeleton_control_activation = SkeletonMuscleControlFunction(env)

def sample_actions_uncompiled(ts, obs, _rng):
    y, updates = agent_conf.network.apply({'params': ts.params,
                                           'run_stats': ts.run_stats},
                                           obs, mutable=["run_stats"])
    ts = ts.replace(run_stats=updates['run_stats'])
    pi, _ = y
    a = pi.sample(seed=_rng)
    return a, ts

sample_actions = jax.jit(sample_actions_uncompiled)

if config.experiment.n_seeds > 1:
    train_state = jax.tree.map(lambda x: x[train_state_seed], agent_state.train_state)
else: 
    train_state = agent_state.train_state


# Build a list of all configurations to evaluate
eval_configs = []

for param_name in randomization_params_names:
    if randomization_params_eval[f"randomize_{param_name}"]:
        if "stiffness" in param_name or "damping" in param_name:
            joint_name = list(randomization_params_eval[f"{param_name}_range"].keys())[0]
            min_val, max_val = randomization_params_eval[f"{param_name}_range"][joint_name]
            increment = randomization_increments[param_name]

            current_value = min_val
            while current_value <= max_val:
                eval_configs.append({
                    'param_name': param_name,
                    'param_value': current_value,
                    'direction': None,
                    'body_or_joint_name': joint_name
                })
                current_value += increment

        elif "position" in param_name or "orientation" in param_name:
            body_range = randomization_params_eval[f"{param_name}_range"]
            body_name = list(body_range.keys())[0]
            directions = list(body_range[body_name].keys())

            for axis in directions:
                min_val = body_range[body_name][axis][0]
                max_val = body_range[body_name][axis][1]
                inc = randomization_increments[param_name]
                
                if set_fixed_randomized_targets:
                    if fixed_randomized_targets.get(param_name):
                        if body_name in fixed_randomized_targets[f"{param_name}"] and axis in fixed_randomized_targets[f"{param_name}"][body_name]:
                            raise RuntimeError(f"{param_name}_range[{body_name}][{axis}] is already defined to be fixed.")

                current_value = min_val
                while current_value <= max_val + 1e-6:
                    eval_configs.append({
                        'param_name': param_name,
                        'param_value': current_value,
                        'direction': axis,
                        'body_or_joint_name': body_name
                    })
                    current_value += inc

print(f"Total configurations to evaluate: {len(eval_configs)}")

# Get body names and joint names
body_names = []
for i in range(model.nbody):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    body_names.append(joint_name)

joint_names = []
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    joint_names.append(joint_name)

evaluation_muscle_groups = ["back_muscles", "torso_muscles", "vasti_muscles", "rectus_muscles", "medial_muscles", "leg_muscles", "all_muscles"]
evaluation_muscle_names = {}
for n in evaluation_muscle_groups:
    evaluation_muscle_names[n] = prosthesis_metrics_handler.get_muscle_group(n)

foot_name = "toes"
calcn_name = "calcn"
left_side = "left_side"
right_side = "right_side"

all_actuator_names = []
for a in range(model.nu):
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    all_actuator_names.append(actuator_name)


# Function to run evaluation for a batch of configurations in parallel
def run_batch_parallel(batch_configs):
    """Run a batch of configurations in parallel using vmap."""
    batch_size = len(batch_configs)
    
    # Create separate environment states for each config in the batch
    all_results = []
    
    for config_idx, config in enumerate(batch_configs):
        param_name = config['param_name']
        param_value = config['param_value']
        direction = config['direction']
        body_or_joint_name = config['body_or_joint_name']
        
        # Set all randomization flags to False initially
        for name in randomization_params_names:
            env._domain_randomizer.rand_conf[f"randomize_{name}"] = False
        
        # Set the current parameter's randomization flag to True
        env._domain_randomizer.rand_conf[f"randomize_{param_name}"] = True
        
        # Set fixed randomized targets if needed
        if set_fixed_randomized_targets:
            fixed_param_names = list(fixed_randomized_targets.keys())
            for fixed_param_name in fixed_param_names:
                env._domain_randomizer.rand_conf[f"{fixed_param_name}_range"] = {}
                if fixed_randomized_targets.get(fixed_param_name) is not None:
                    env._domain_randomizer.rand_conf[f"randomize_{fixed_param_name}"] = True
                    for fixed_body, fixed_axes in fixed_randomized_targets[fixed_param_name].items():
                        for fixed_axis, fixed_value in fixed_axes.items():
                            if not env._domain_randomizer.rand_conf[f"{fixed_param_name}_range"]:
                                env._domain_randomizer.rand_conf[f"{fixed_param_name}_range"] = {fixed_body: {fixed_axis: [fixed_value, fixed_value]}}
                            else:
                                if fixed_body not in env._domain_randomizer.rand_conf[f"{fixed_param_name}_range"]:
                                    env._domain_randomizer.rand_conf[f"{fixed_param_name}_range"][fixed_body] = {}
                                env._domain_randomizer.rand_conf[f"{fixed_param_name}_range"][fixed_body][fixed_axis] = [fixed_value, fixed_value]
        
        # Set the specific parameter value
        if "stiffness" in param_name or "damping" in param_name:
            env._domain_randomizer.rand_conf[f"{param_name}_range"][body_or_joint_name] = [param_value, param_value]
        elif "position" in param_name or "orientation" in param_name:
            if set_fixed_randomized_targets:
                if param_name in fixed_param_names:
                    if body_or_joint_name not in env._domain_randomizer.rand_conf[f"{param_name}_range"]:
                        env._domain_randomizer.rand_conf[f"{param_name}_range"][body_or_joint_name] = {}
                    env._domain_randomizer.rand_conf[f"{param_name}_range"][body_or_joint_name][direction] = [param_value, param_value]
            else:
                env._domain_randomizer.rand_conf[f"{param_name}_range"][body_or_joint_name] = {direction: [param_value, param_value]}
        
        # Run evaluation for this config
        result = run_single_eval(config_idx, param_name, param_value, direction, body_or_joint_name)
        all_results.append(result)
    
    return all_results


def run_single_eval(config_idx, param_name, param_value, direction, body_or_joint_name):
    """Run evaluation for a single configuration."""
    rng = jax.random.key(config_idx)
    keys = jax.random.split(rng, 2)
    rng, env_keys = keys[0], jax.random.split(keys[1], 1)
    
    # JIT compile the functions
    jit_step = jax.jit(jax.vmap(env.mjx_step))
    jit_reset = jax.jit(jax.vmap(env.mjx_reset))
    
    env_state = jit_reset(env_keys)
    obs = env_state.observation
    
    # Initialize data structures
    all_foot_ground_contact_left = []
    all_foot_ground_contact_right = []
    all_grf_l = []
    all_grf_r = []
    body_xposes = {name: [] for name in body_names}
    joint_data = {}
    for joint_name in joint_names:
        joint_data[joint_name] = {
            "angle": [],
            "velocity": [],
            "forces_constraint": [],
            "forces_smooth": [],
            "forces_applied": [],
            "torques": [],
            "energy_exp": [],
        }
    all_sensor_force = {}
    all_actions = []
    
    # Muscle activations
    run_muscle_activations = {f"run_{g}_activation_left": [] for g in evaluation_muscle_groups}
    run_muscle_activations.update({f"run_{g}_activation_right": [] for g in evaluation_muscle_groups})
    
    step_total = 0
    
    for i in range(n_steps):
        obs = env_state.observation
        rng, _rng = jax.random.split(rng)
        action, train_state_local = sample_actions(train_state, obs, _rng)
        action = jnp.atleast_2d(action)
        
        env_state = jit_step(env_state, action)
        obs = env_state.observation
        
        # Collect metrics
        contact_left, contact_right = prosthesis_metrics_handler.get_contact_steps(env_state.data, i)
        all_foot_ground_contact_left.append(contact_left)
        all_foot_ground_contact_right.append(contact_right)
        
        grf_foot_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_l")
        grf_foot_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_r")
        grf_calcn_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{calcn_name}_l")
        grf_calcn_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{calcn_name}_r")
        grf_l = grf_foot_l + grf_calcn_l
        grf_r = grf_foot_r + grf_calcn_r
        all_grf_l.append(grf_l)
        all_grf_r.append(grf_r)
        
        body_xpos = prosthesis_metrics_handler.get_xpos(env_state.data)
        for name in body_names: 
            body_xposes[name].append(body_xpos[name])
        
        joint_angles = prosthesis_metrics_handler.get_joint_angles(env_state.data)
        joint_velocities = prosthesis_metrics_handler.get_joint_vels(env_state.data)
        joint_forces_constraint, joint_forces_smooth, joint_forces_applied = prosthesis_metrics_handler.get_joint_frces(env_state.data)
        joint_torques = prosthesis_metrics_handler.get_joint_trques(env_state.data)
        joint_energy_exp = prosthesis_metrics_handler.calc_joint_energy_exp(joint_torques, joint_velocities)
        
        for joint_name, angle in joint_angles.items():
            joint_data[joint_name]["angle"].append(angle)
        for joint_name, velocity in joint_velocities.items():
            joint_data[joint_name]["velocity"].append(velocity)
        for joint_name, force in joint_forces_constraint.items():
            joint_data[joint_name]["forces_constraint"].append(force)
        for joint_name, force in joint_forces_smooth.items():
            joint_data[joint_name]["forces_smooth"].append(force)
        for joint_name, force in joint_forces_applied.items():
            joint_data[joint_name]["forces_applied"].append(force)
        for joint_name, torque in joint_torques.items():
            joint_data[joint_name]["torques"].append(torque)
        for joint_name, energy_exp in joint_energy_exp.items():
            joint_data[joint_name]["energy_exp"].append(energy_exp)
        
        sensor_force, sensor_force_names = prosthesis_metrics_handler.get_sensor_data(env_state.data)
        for sensor_name, force in sensor_force.items():
            if sensor_name not in all_sensor_force:
                all_sensor_force[sensor_name] = []
            all_sensor_force[sensor_name].append(force)
        
        for i_act in range(model.nu):
            if model.actuator_dyntype[i_act] == mujoco.mjtDyn.mjDYN_MUSCLE:
                action = action.at[...,i_act].set(muscle_skeleton_control_activation.adapted_sigmoid(action[...,i_act]))
        
        all_actions.append(action)
        
        for muscle_group, muscle_names_list in evaluation_muscle_names.items():
            left = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names_list, left_side, action)
            right = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names_list, right_side, action)
            run_muscle_activations[f"run_{muscle_group}_activation_left"].append(left)
            run_muscle_activations[f"run_{muscle_group}_activation_right"].append(right)
        
        step_total += 1
    
    env.stop()
    
    # Post-process activations
    sum_step_activations = {}
    step_norm_activations = {}
    sum_muscles_activations = {}
    step_muscles_norm_activations = {}
    step_num_norm_activations = {}
    
    for muscle_group in evaluation_muscle_groups:
        sum_step_activations[f"{muscle_group}_left"] = jnp.sum(jnp.array(run_muscle_activations[f"run_{muscle_group}_activation_left"]), axis=0)
        sum_step_activations[f"{muscle_group}_right"] = jnp.sum(jnp.array(run_muscle_activations[f"run_{muscle_group}_activation_right"]), axis=0)
        
        step_norm_activations[f"{muscle_group}_left"] = sum_step_activations[f"{muscle_group}_left"] / n_steps
        step_norm_activations[f"{muscle_group}_right"] = sum_step_activations[f"{muscle_group}_right"] / n_steps
        
        n_musc = len(evaluation_muscle_names[muscle_group])
        sum_muscles_activations[f"{muscle_group}_left"] = jnp.sum(sum_step_activations[f"{muscle_group}_left"], axis=0)
        sum_muscles_activations[f"{muscle_group}_right"] = jnp.sum(sum_step_activations[f"{muscle_group}_right"], axis=0)
        step_muscles_norm_activations[f"{muscle_group}_left"] = sum_muscles_activations[f"{muscle_group}_left"] / n_steps
        step_muscles_norm_activations[f"{muscle_group}_right"] = sum_muscles_activations[f"{muscle_group}_right"] / n_steps
        step_num_norm_activations[f"{muscle_group}_left"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_left"] / n_musc)
        step_num_norm_activations[f"{muscle_group}_right"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_right"] / n_musc)
    
    all_relevant_data = {
        "total_steps": step_total,
        "n_envs": 1,
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
        "evaluation_body_names": body_names,
    }
    
    for joint_name, joint_dict in joint_data.items():
        for key, value in joint_dict.items():
            all_relevant_data[f"{joint_name}_{key}"] = value
    
    for muscle_group in evaluation_muscle_groups:
        all_relevant_data[f"run_{muscle_group}_activation_left"] = run_muscle_activations[f"run_{muscle_group}_activation_left"]
        all_relevant_data[f"run_{muscle_group}_activation_right"] = run_muscle_activations[f"run_{muscle_group}_activation_right"]
    
    # Save to file
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    joint_name_str = body_or_joint_name
    if 'orientation' in param_name:
        name_param_value = np.rad2deg(param_value)
        name_units = 'deg'
    elif 'position' in param_name:
        name_param_value = param_value*1000
        name_units = 'mm'
    else:
        name_param_value = param_value
        name_units = ''
    
    if not os.path.exists(os.path.join(os.path.dirname(path), f"{dt_str_init}_{subfolder_name}")):
        os.makedirs(os.path.join(os.path.dirname(path), f"{dt_str_init}_{subfolder_name}"))
    output_path = os.path.join(os.path.dirname(path), f"{dt_str_init}_{subfolder_name}", 
                              f"eval_{n_steps}steps_{joint_name_str}_{direction if direction else ''}_{int(np.round(name_param_value,0))}{name_units}_{param_name}.pkl")
    
    with open(output_path, "wb") as f:
        pickle.dump(all_relevant_data, f)
    print(f"Saved evaluation data to {output_path}")
    
    return all_relevant_data


# Main execution with batching
time_all = []
time_all.append(timeit.default_timer())

batch_size = args.batch_size
total_configs = len(eval_configs)

for batch_start in range(0, total_configs, batch_size):
    batch_end = min(batch_start + batch_size, total_configs)
    batch = eval_configs[batch_start:batch_end]
    
    print(f"\nProcessing batch {batch_start//batch_size + 1}/{(total_configs + batch_size - 1)//batch_size}")
    print(f"Configurations {batch_start+1} to {batch_end} of {total_configs}")
    
    # Run batch in parallel
    results = run_batch_parallel(batch)
    
    batch_time = timeit.default_timer() - time_all[-1]
    print(f"Batch completed in {batch_time:.2f} seconds")

time_all.append(timeit.default_timer())

print(f"\n{'='*60}")
print(f"Total time taken for {total_configs} configurations: {time_all[-1] - time_all[0]:.2f} seconds")
print(f"Average time per configuration: {(time_all[-1] - time_all[0]) / total_configs:.2f} seconds")
print(f"{'='*60}")
