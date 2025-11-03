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
from datetime import datetime
import timeit

from loco_mujoco.core.wrappers import VecEnv
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), "/home/nadinebadie/loco-mujoco/prosthesis_test"))
from loco_mujoco.evaluation.evaluation_prosthesis import ProsthesisMetricsHandler
from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction

import mujoco

# Parameter randomization initialization
randomization_params_names = ["prosthesis_dof_damping", "prosthesis_joint_stiffness", "prosthesis_body_position", "prosthesis_body_orientation"]

randomization_params_eval = {
    "prosthesis_side": "left_side",
    "randomize_prosthesis_dof_damping": False,
    "prosthesis_dof_damping_range": {'ankle_angle':[2, 10]}, #, 'subtalar_angle', 'mtp_angle'],
    # "prosthesis_dof_damping_range": [0, 10],
    "randomize_prosthesis_joint_stiffness": False,
    # "randomization_joint_names": ['ankle_angle', 'subtalar_angle', 'mtp_angle'],
    "prosthesis_joint_stiffness_range":{'ankle_angle': [50, 100]}, #[],
    "randomize_prosthesis_body_position": True,
    "prosthesis_body_position_range": {'calcn': {'x': [0.3, 0.4]}}, #,'y': [0.4, 0.5]}}, #['calcn'],
    # "prosthesis_body_position_range": {
    #     'z': [0.6, 0.7]
    # },
    "randomize_prosthesis_body_orientation": False,
    "prosthesis_body_orientation_range": {'pylon_socket': {'y': [-0.3,0.3]}}, #['calcn'],
    # "prosthesis_body_orientation_range": {}

    "randomize_prosthesis_socket_joint": False,
    "socket_joint_range": {'socket_ty': [-0.25,0.025]},
}

randomization_increments = {
    "prosthesis_joint_stiffness": 10,
    "prosthesis_dof_damping": 5,
    "prosthesis_body_position": 0.1,
    "prosthesis_body_orientation": 0.1
}

os.environ["MUJOCO_GL"] = "egl"

parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for parallel evaluation')
args = parser.parse_args()

path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config  # This is the MAIN CONFIG from the loaded agent

# Setup randomization parameters (using the loaded config, not overwriting it)
randomization_type = config.randomization_config["randomization_type"]
randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)

# Update with evaluation-specific parameters
for key, value in randomization_params_eval.items():
    if key in randomization_params:
        randomization_params[key] = value
    else:
        randomization_params[key] = value

if "prosthesis_side" in config.experiment.env_params:
    randomization_params["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]

factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

OmegaConf.set_struct(config, False)
config.experiment.env_params["headless"] = False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"
config.experiment.env_params["add_sensors"] = True

n_steps = 100
rng = jax.random.key(0)
train_state_seed = 0

def sample_actions_uncompiled(ts, obs, _rng):
    y, updates = agent_conf.network.apply({'params': ts.params, 'run_stats': ts.run_stats},
                                          obs, mutable=["run_stats"])
    ts = ts.replace(run_stats=updates['run_stats'])
    pi, _ = y
    a = pi.sample(seed=_rng)
    return a, ts

sample_actions = jax.jit(sample_actions_uncompiled)

if config.experiment.n_seeds > 1:
    assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify train_state_seed for replay.")
    train_state = jax.tree.map(lambda x: x[train_state_seed], agent_state.train_state)
else: 
    train_state = agent_state.train_state

def create_env_with_randomization(rand_params_config):
    """Create environment with specific randomization configuration"""
    local_randomization_params = randomization_params.copy()
    local_randomization_params.update(rand_params_config)
    
    env_ = factory.make(domain_randomization_type=randomization_type, 
                       domain_randomization_params=local_randomization_params,
                       **config.experiment.env_params,  # Use the MAIN config here
                       **config.experiment.task_factory.params)
    env_.th.to_jax()
    env_ = VecEnv(env_)
    return env_

def run_single_evaluation_loop(env_state, train_state, rng_eval, current_model, current_prosthesis_metrics_handler, param_name=None, param_value=None):
    """Your original comprehensive evaluation function - preserved completely"""
    
    all_foot_ground_contact_left = []
    all_foot_ground_contact_right = []
    joint_data = {}
    for i in range(current_model.njnt):
        joint_name = mujoco.mj_id2name(current_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_data[joint_name] = {
            "angle": [], "velocity": [], "forces_constraint": [],
            "forces_smooth": [], "torques": [], "energy_exp": [],
        }
        if "_l" in joint_name or "_r" in joint_name:
            joint_data[joint_name].update({
                "angle_per_step": [], "velocity_per_step": [],
                "forces_constraint_per_step": [], "forces_smooth_per_step": [],
                "torques_per_step": [], "energy_exp_per_step": [],
            })
        else:
            joint_data[joint_name].update({
                "angle_per_step_left": [], "angle_per_step_right": [],
                "velocity_per_step_left": [], "velocity_per_step_right": [],
                "forces_constraint_per_step_left": [], "forces_constraint_per_step_right": [],
                "forces_smooth_per_step_left": [], "forces_smooth_per_step_right": [],
                "torques_per_step_left": [], "torques_per_step_right": [],
                "energy_exp_per_step_left": [], "energy_exp_per_step_right": [],
            })
    
    all_sensor_force = {}
    evaluation_muscle_groups = ["back_muscles", "torso_muscles", "vasti_muscles", "rectus_muscles", "medial_muscles", "leg_muscles", "all_muscles"]
    evaluation_muscle_names = {}
    for n in evaluation_muscle_groups:
        evaluation_muscle_names[n] = current_prosthesis_metrics_handler.get_muscle_group(n)
    
    run_muscle_activations_left = {mg: [] for mg in evaluation_muscle_groups}
    run_muscle_activations_right = {mg: [] for mg in evaluation_muscle_groups}

    foot_name = "toes"
    all_grf_l = []
    all_grf_r = []
    all_actions = []
    all_actuator_names = []

    jit_step = jax.jit(current_prosthesis_metrics_handler.env.mjx_step_test)
    muscle_skeleton_control_activation_local = SkeletonMuscleControlFunction(current_prosthesis_metrics_handler.env)

    for i in range(n_steps):
        obs = env_state.observation
        rng_eval, _rng = jax.random.split(rng_eval)
        action, train_state = sample_actions(train_state, obs, _rng)
        
        # FIX: Ensure action is 1D before being used for MuJoCo control
        if action.ndim > 1:
            action = action.squeeze(axis=0) # Remove the batch dimension if it exists and is 1

        env_state, sys_data = jit_step(env_state, action)
        obs = env_state.observation

        contact_left, contact_right = current_prosthesis_metrics_handler.get_contact_steps_batched(env_state.data, i)
        all_foot_ground_contact_left.append(contact_left)
        all_foot_ground_contact_right.append(contact_right)

        grf_l = current_prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_l")
        grf_r = current_prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_r")
        all_grf_l.append(grf_l)
        all_grf_r.append(grf_r)

        joint_angles = current_prosthesis_metrics_handler.get_joint_angles(env_state.data)
        joint_velocities = current_prosthesis_metrics_handler.get_joint_vels(env_state.data)
        joint_forces_constraint, joint_forces_smooth = current_prosthesis_metrics_handler.get_joint_frces(env_state.data)
        joint_torques = current_prosthesis_metrics_handler.get_joint_trques(env_state.data)
        joint_energy_exp = current_prosthesis_metrics_handler.calc_joint_energy_exp(joint_torques, joint_velocities)
        
        for joint_name, angle in joint_angles.items():
            joint_data[joint_name]["angle"].append(angle)
        for joint_name, velocity in joint_velocities.items():
            joint_data[joint_name]["velocity"].append(velocity)
        for joint_name, force in joint_forces_constraint.items():
            joint_data[joint_name]["forces_constraint"].append(force)
        for joint_name, force in joint_forces_smooth.items():
            joint_data[joint_name]["forces_smooth"].append(force)
        for joint_name, torque in joint_torques.items():
            joint_data[joint_name]["torques"].append(torque)
        for joint_name, energy_exp in joint_energy_exp.items():
            joint_data[joint_name]["energy_exp"].append(energy_exp)

        sensor_force, sensor_force_names = current_prosthesis_metrics_handler.get_sensor_data_batched(env_state.data)
        for sensor_name, force in sensor_force.items():
            if sensor_name not in all_sensor_force:
                all_sensor_force[sensor_name] = []
            all_sensor_force[sensor_name].append(force)

        processed_action = action.copy()
        for act_idx in range(current_model.nu):
            if current_model.actuator_dyntype[act_idx] == mujoco.mjtDyn.mjDYN_MUSCLE:
                processed_action = processed_action.at[act_idx].set(muscle_skeleton_control_activation_local.adapted_sigmoid(processed_action[act_idx]))
        all_actions.append(processed_action)
        
        for muscle_group in evaluation_muscle_groups:
            # Use the new JAX-compatible method (no strings passed)
            act_left = current_prosthesis_metrics_handler.get_muscle_activations_by_indices(
                processed_action, muscle_group, "left"
            )
            act_right = current_prosthesis_metrics_handler.get_muscle_activations_by_indices(
                processed_action, muscle_group, "right"
            )
            run_muscle_activations_left[muscle_group].append(act_left)
            run_muscle_activations_right[muscle_group].append(act_right)
    
        # if not all_actuator_names:
        #     for a_idx in range(current_model.nu):
        #         all_actuator_names.append(mujoco.mj_id2name(current_model, mujoco.mjtObj.mjOBJ_ACTUATOR, a_idx))

    #     for muscle_group in evaluation_muscle_groups:
    #         muscle_names = evaluation_muscle_names[muscle_group]
    #         act_left = current_prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, "left_side", processed_action)
    #         act_right = current_prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, "right_side", processed_action)
    #         run_muscle_activations_left[muscle_group].append(act_left)
    #         run_muscle_activations_right[muscle_group].append(act_right)

    # # Calculate muscle activation statistics
    # for muscle_group in evaluation_muscle_groups:
    #     run_muscle_activations_left[muscle_group] = jnp.array(run_muscle_activations_left[muscle_group])
    #     run_muscle_activations_right[muscle_group] = jnp.array(run_muscle_activations_right[muscle_group])

    sum_step_activations = {}
    step_norm_activations = {}
    sum_muscles_activations = {}
    step_muscles_norm_activations = {}
    step_num_norm_activations = {}

    for muscle_group in evaluation_muscle_groups:
        sum_step_activations[f"{muscle_group}_right"] = jnp.sum(jnp.stack(run_muscle_activations_right[muscle_group]), axis=0)
        sum_step_activations[f"{muscle_group}_left"] = jnp.sum(jnp.stack(run_muscle_activations_left[muscle_group]), axis=0)
        # sum_step_activations[f"{muscle_group}_left"] = jnp.sum(run_muscle_activations_left[muscle_group], axis=0)
        # sum_step_activations[f"{muscle_group}_right"] = jnp.sum(run_muscle_activations_right[muscle_group], axis=0)
        step_norm_activations[f"{muscle_group}_left"] = sum_step_activations[f"{muscle_group}_left"] / n_steps
        step_norm_activations[f"{muscle_group}_right"] = sum_step_activations[f"{muscle_group}_right"] / n_steps

        n_musc = len(evaluation_muscle_names[muscle_group])
        sum_muscles_activations[f"{muscle_group}_left"] = jnp.sum(sum_step_activations[f"{muscle_group}_left"], axis=0)
        sum_muscles_activations[f"{muscle_group}_right"] = jnp.sum(sum_step_activations[f"{muscle_group}_right"], axis=0)
        step_muscles_norm_activations[f"{muscle_group}_left"] = sum_muscles_activations[f"{muscle_group}_left"] / n_steps
        step_muscles_norm_activations[f"{muscle_group}_right"] = sum_muscles_activations[f"{muscle_group}_right"] / n_steps
        step_num_norm_activations[f"{muscle_group}_left"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_left"] / n_musc)
        step_num_norm_activations[f"{muscle_group}_right"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_right"] / n_musc)

    joint_names = []
    for i in range(current_model.njnt):
        joint_name = mujoco.mj_id2name(current_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_names.append(joint_name)

    all_relevant_data = {
        "total_steps": n_steps,
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
        "param_name": param_name,
        "param_value": param_value
    }

    # Add all joint data
    for joint_name, joint_dict in joint_data.items():
        for key, value in joint_dict.items():
            all_relevant_data[f"{joint_name}_{key}"] = value

    # # Add muscle activation data
    for muscle_group in evaluation_muscle_groups:
        all_relevant_data[f"run_{muscle_group}_activation_left"] = run_muscle_activations_left[muscle_group]
        all_relevant_data[f"run_{muscle_group}_activation_right"] = run_muscle_activations_right[muscle_group]

    return all_relevant_data, env_state

def generate_randomization_configs():
    """Generate all randomization configurations to test"""
    all_rand_configs = []
    all_param_names = []
    all_param_values = []

    # Mapping param_name → corresponding "_range" key
    param_to_range_key = {
        "prosthesis_dof_damping": "prosthesis_dof_damping_range",
        "prosthesis_joint_stiffness": "prosthesis_joint_stiffness_range",
        "prosthesis_body_position": "prosthesis_body_position_range",
        "prosthesis_body_orientation": "prosthesis_body_orientation_range",
        "prosthesis_socket_joint": "socket_joint_range",
    }

    for param_name in randomization_params_names:
        if randomization_params.get(f"randomize_{param_name}", False):
            range_key = param_to_range_key[param_name]
            param_range = randomization_params[range_key]

            if param_name in ["prosthesis_dof_damping", "prosthesis_joint_stiffness", "prosthesis_socket_joint"]:
                # Type 1: Flat dict {dof_name: [min, max]}
                for dof_name, (min_val, max_val) in param_range.items():
                    inc = randomization_increments[param_name]
                    for val in np.arange(min_val, max_val + inc/2, inc):
                        rand_config = {
                            f"randomize_{param_name}": True,
                            range_key: {dof_name: [val, val]}
                        }
                        # Disable other params
                        for other_param in randomization_params_names:
                            if other_param != param_name:
                                rand_config[f"randomize_{other_param}"] = False
                        all_rand_configs.append(rand_config)
                        all_param_names.append(f"{param_name}:{dof_name}")
                        all_param_values.append(val)

            elif param_name in ["prosthesis_body_position", "prosthesis_body_orientation"]:
                # Type 2: Nested dict {body_name: {axis: [min, max]}}
                for body_name, axis_dict in param_range.items():
                    for axis, (min_val, max_val) in axis_dict.items():
                        inc = randomization_increments[param_name]
                        for val in np.arange(min_val, max_val + inc/2, inc):
                            rand_config = {
                                f"randomize_{param_name}": True,
                                range_key: {
                                    body_name: {axis: [val, val]}
                                }
                            }
                            # Disable other params
                            for other_param in randomization_params_names:
                                if other_param != param_name:
                                    rand_config[f"randomize_{other_param}"] = False
                            all_rand_configs.append(rand_config)
                            all_param_names.append(f"{param_name}:{body_name}:{axis}")
                            all_param_values.append(val)

    return all_rand_configs, all_param_names, all_param_values

def run_batch_evaluation(batch_configs, batch_param_names, batch_param_values, batch_rng_keys):
    """Run evaluation for a batch of configurations"""
    batch_results = []
    all_actuator_names = []
    
    for i, (rand_config, param_name, param_value, rng_key) in enumerate(zip(batch_configs, batch_param_names, batch_param_values, batch_rng_keys)):
        print(f"  Running config {i+1}/{len(batch_configs)}: {param_name} = {param_value}")
        
        # Create environment with specific randomization
        env_instance = create_env_with_randomization(rand_config)
        model_instance = env_instance.get_model()
        metrics_handler_instance = ProsthesisMetricsHandler(env_instance)

        evaluation_muscle_groups = ["back_muscles", "torso_muscles", "vasti_muscles", "rectus_muscles", "medial_muscles", "leg_muscles", "all_muscles"]
        evaluation_muscle_names = {}
        for n in evaluation_muscle_groups:
            evaluation_muscle_names[n] = metrics_handler_instance.get_muscle_group(n)
        

        metrics_handler_instance.setup_muscle_indices(
            model_instance, 
            evaluation_muscle_groups, 
            evaluation_muscle_names
        )

        # Get actuator names outside JIT if needed
        if not all_actuator_names:
            all_actuator_names = get_actuator_names_outside_jit(model_instance)
        
        
        # Reset environment
        env_state = jax.jit(env_instance.mjx_reset)(rng_key)
        
        # Create JIT-compiled evaluation function
        jit_single_eval = jax.jit(run_single_evaluation_loop, static_argnums=(3, 4, 5, 6)) #(1,2,3, 4, 5, 6))
        
        from jax.tree_util import tree_flatten
        leaves, _ = tree_flatten(train_state)
        for i, leaf in enumerate(leaves):
            print(f"Leaf {i}: type {type(leaf)}, shape {getattr(leaf, 'shape', 'N/A')}")

        # # Run evaluation
        results, final_env_state = jit_single_eval(
            env_state, train_state, rng_key, 
            model_instance, metrics_handler_instance, param_name, param_value
        )

        # results, final_env_state = jit_single_eval(
        #     config, param_name, param_value, rng_key
        # )

        # results, final_env_state = jit_single_eval(
        #     env_state, train_state, rng_key, 
        #     model_instance, metrics_handler_instance)
        
        batch_results.append(results)
        
        # Clean up environment
        env_instance.stop()
    
    return batch_results



def get_actuator_names_outside_jit(model):
    """Call this outside JIT to get actuator names for reporting"""
    actuator_names = []
    for a_idx in range(model.nu):
        actuator_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a_idx))
    return actuator_names








def run_single_evaluation_loop_jit_compatible(env_state, train_state, rng_eval, current_model, current_prosthesis_metrics_handler):
    """JIT-compatible evaluation function - returns only JAX arrays"""
    
    all_foot_ground_contact_left = []
    all_foot_ground_contact_right = []
    joint_data = {}
    for i in range(current_model.njnt):
        joint_name = mujoco.mj_id2name(current_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_data[joint_name] = {
            "angle": [], "velocity": [], "forces_constraint": [],
            "forces_smooth": [], "torques": [], "energy_exp": [],
        }
        if "_l" in joint_name or "_r" in joint_name:
            joint_data[joint_name].update({
                "angle_per_step": [], "velocity_per_step": [],
                "forces_constraint_per_step": [], "forces_smooth_per_step": [],
                "torques_per_step": [], "energy_exp_per_step": [],
            })
        else:
            joint_data[joint_name].update({
                "angle_per_step_left": [], "angle_per_step_right": [],
                "velocity_per_step_left": [], "velocity_per_step_right": [],
                "forces_constraint_per_step_left": [], "forces_constraint_per_step_right": [],
                "forces_smooth_per_step_left": [], "forces_smooth_per_step_right": [],
                "torques_per_step_left": [], "torques_per_step_right": [],
                "energy_exp_per_step_left": [], "energy_exp_per_step_right": [],
            })
    
    all_sensor_force = {}
    evaluation_muscle_groups = ["back_muscles", "torso_muscles", "vasti_muscles", "rectus_muscles", "medial_muscles", "leg_muscles", "all_muscles"]
    evaluation_muscle_names = {}
    for n in evaluation_muscle_groups:
        evaluation_muscle_names[n] = current_prosthesis_metrics_handler.get_muscle_group(n)
    
    run_muscle_activations_left = {mg: [] for mg in evaluation_muscle_groups}
    run_muscle_activations_right = {mg: [] for mg in evaluation_muscle_groups}

    foot_name = "toes"
    all_grf_l = []
    all_grf_r = []
    all_actions = []

    jit_step = jax.jit(current_prosthesis_metrics_handler.env.mjx_step_test)
    muscle_skeleton_control_activation_local = SkeletonMuscleControlFunction(current_prosthesis_metrics_handler.env)

    for i in range(n_steps):
        obs = env_state.observation
        rng_eval, _rng = jax.random.split(rng_eval)
        action, train_state = sample_actions(train_state, obs, _rng)
        
        # FIX: Ensure action is 1D before being used for MuJoCo control
        if action.ndim > 1:
            action = action.squeeze(axis=0)

        env_state, sys_data = jit_step(env_state, action)
        obs = env_state.observation

        contact_left, contact_right = current_prosthesis_metrics_handler.get_contact_steps_batched(env_state.data, i)
        all_foot_ground_contact_left.append(contact_left)
        all_foot_ground_contact_right.append(contact_right)

        grf_l = current_prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_l")
        grf_r = current_prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_r")
        all_grf_l.append(grf_l)
        all_grf_r.append(grf_r)

        joint_angles = current_prosthesis_metrics_handler.get_joint_angles(env_state.data)
        joint_velocities = current_prosthesis_metrics_handler.get_joint_vels(env_state.data)
        joint_forces_constraint, joint_forces_smooth = current_prosthesis_metrics_handler.get_joint_frces(env_state.data)
        joint_torques = current_prosthesis_metrics_handler.get_joint_trques(env_state.data)
        joint_energy_exp = current_prosthesis_metrics_handler.calc_joint_energy_exp(joint_torques, joint_velocities)
        
        for joint_name, angle in joint_angles.items():
            joint_data[joint_name]["angle"].append(angle)
        for joint_name, velocity in joint_velocities.items():
            joint_data[joint_name]["velocity"].append(velocity)
        for joint_name, force in joint_forces_constraint.items():
            joint_data[joint_name]["forces_constraint"].append(force)
        for joint_name, force in joint_forces_smooth.items():
            joint_data[joint_name]["forces_smooth"].append(force)
        for joint_name, torque in joint_torques.items():
            joint_data[joint_name]["torques"].append(torque)
        for joint_name, energy_exp in joint_energy_exp.items():
            joint_data[joint_name]["energy_exp"].append(energy_exp)

        sensor_force, sensor_force_names = current_prosthesis_metrics_handler.get_sensor_data_batched(env_state.data)
        for sensor_name, force in sensor_force.items():
            if sensor_name not in all_sensor_force:
                all_sensor_force[sensor_name] = []
            all_sensor_force[sensor_name].append(force)

        processed_action = action.copy()
        for act_idx in range(current_model.nu):
            if current_model.actuator_dyntype[act_idx] == mujoco.mjtDyn.mjDYN_MUSCLE:
                processed_action = processed_action.at[act_idx].set(muscle_skeleton_control_activation_local.adapted_sigmoid(processed_action[act_idx]))
        all_actions.append(processed_action)
        
        for muscle_group in evaluation_muscle_groups:
            act_left = current_prosthesis_metrics_handler.get_muscle_activations_by_indices(
                processed_action, muscle_group, "left"
            )
            act_right = current_prosthesis_metrics_handler.get_muscle_activations_by_indices(
                processed_action, muscle_group, "right"
            )
            run_muscle_activations_left[muscle_group].append(act_left)
            run_muscle_activations_right[muscle_group].append(act_right)

    # Calculate muscle activation statistics
    sum_step_activations = {}
    step_norm_activations = {}
    sum_muscles_activations = {}
    step_muscles_norm_activations = {}
    step_num_norm_activations = {}

    for muscle_group in evaluation_muscle_groups:
        sum_step_activations[f"{muscle_group}_right"] = jnp.sum(jnp.stack(run_muscle_activations_right[muscle_group]), axis=0)
        sum_step_activations[f"{muscle_group}_left"] = jnp.sum(jnp.stack(run_muscle_activations_left[muscle_group]), axis=0)
        step_norm_activations[f"{muscle_group}_left"] = sum_step_activations[f"{muscle_group}_left"] / n_steps
        step_norm_activations[f"{muscle_group}_right"] = sum_step_activations[f"{muscle_group}_right"] / n_steps

        n_musc = len(evaluation_muscle_names[muscle_group])
        sum_muscles_activations[f"{muscle_group}_left"] = jnp.sum(sum_step_activations[f"{muscle_group}_left"], axis=0)
        sum_muscles_activations[f"{muscle_group}_right"] = jnp.sum(sum_step_activations[f"{muscle_group}_right"], axis=0)
        step_muscles_norm_activations[f"{muscle_group}_left"] = sum_muscles_activations[f"{muscle_group}_left"] / n_steps
        step_muscles_norm_activations[f"{muscle_group}_right"] = sum_muscles_activations[f"{muscle_group}_right"] / n_steps
        step_num_norm_activations[f"{muscle_group}_left"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_left"] / n_musc)
        step_num_norm_activations[f"{muscle_group}_right"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_right"] / n_musc)

    # Return only JAX-compatible data
    jax_compatible_results = {
        "total_steps": n_steps,
        "all_grf_l": jnp.stack(all_grf_l),
        "all_grf_r": jnp.stack(all_grf_r),
        "all_foot_ground_contact_left": jnp.stack(all_foot_ground_contact_left),
        "all_foot_ground_contact_right": jnp.stack(all_foot_ground_contact_right),
        "sum_step_activations": sum_step_activations,
        "step_norm_activations": step_norm_activations,
        "sum_muscles_activations": sum_muscles_activations,
        "step_muscles_norm_activations": step_muscles_norm_activations,
        "step_num_norm_activations": step_num_norm_activations,
        "all_actions": jnp.stack(all_actions),
    }

    # Add joint data as JAX arrays
    for joint_name, joint_dict in joint_data.items():
        for key, value_list in joint_dict.items():
            if value_list:  # Only add if there's data
                jax_compatible_results[f"{joint_name}_{key}"] = jnp.stack(value_list)

    # Add sensor force data
    for sensor_name, force_list in all_sensor_force.items():
        jax_compatible_results[f"sensor_{sensor_name}"] = jnp.stack(force_list)

    # Add muscle activation data
    for muscle_group in evaluation_muscle_groups:
        jax_compatible_results[f"run_{muscle_group}_activation_left"] = jnp.stack(run_muscle_activations_left[muscle_group])
        jax_compatible_results[f"run_{muscle_group}_activation_right"] = jnp.stack(run_muscle_activations_right[muscle_group])

    return jax_compatible_results, env_state


def run_batch_evaluation_fixed(batch_configs, batch_param_names, batch_param_values, batch_rng_keys):
    """Fixed batch evaluation that handles parameter metadata correctly"""
    batch_results = []
    all_actuator_names = []
    
    for i, (rand_config, param_name, param_value, rng_key) in enumerate(zip(batch_configs, batch_param_names, batch_param_values, batch_rng_keys)):
        print(f"  Running config {i+1}/{len(batch_configs)}: {param_name} = {param_value}")
        
        # Create environment with specific randomization
        env_instance = create_env_with_randomization(rand_config)
        model_instance = env_instance.get_model()
        metrics_handler_instance = ProsthesisMetricsHandler(env_instance)

        evaluation_muscle_groups = ["back_muscles", "torso_muscles", "vasti_muscles", "rectus_muscles", "medial_muscles", "leg_muscles", "all_muscles"]
        evaluation_muscle_names = {}
        for n in evaluation_muscle_groups:
            evaluation_muscle_names[n] = metrics_handler_instance.get_muscle_group(n)

        metrics_handler_instance.setup_muscle_indices(
            model_instance, 
            evaluation_muscle_groups, 
            evaluation_muscle_names
        )

        # Get actuator names outside JIT
        if not all_actuator_names:
            all_actuator_names = get_actuator_names_outside_jit(model_instance)
        
        # Get joint names outside JIT
        joint_names = []
        for joint_idx in range(model_instance.njnt):
            joint_name = mujoco.mj_id2name(model_instance, mujoco.mjtObj.mjOBJ_JOINT, joint_idx)
            joint_names.append(joint_name)

        # Reset environment first to get proper MJX data for sensor names
        env_state = jax.jit(env_instance.mjx_reset)(rng_key)
        
        # Get sensor names by calling the method with actual data (this will cache the names)
        _, sensor_force_names = metrics_handler_instance.get_sensor_data_batched(env_state.data)
        
        # Environment is already reset above when getting sensor names
        # env_state = jax.jit(env_instance.mjx_reset)(rng_key)
        
        # Create JIT-compiled evaluation function (only JAX-compatible args)
        jit_single_eval = jax.jit(run_single_evaluation_loop_jit_compatible, static_argnums=(3, 4))
        
        # Run evaluation (only JAX arrays passed to JIT)
        jax_results, final_env_state = jit_single_eval(
            env_state, train_state, rng_key, 
            model_instance, metrics_handler_instance
        )

        # Add metadata AFTER JIT execution
        complete_results = dict(jax_results)  # Convert to regular dict
        complete_results.update({
            "param_name": param_name,
            "param_value": param_value,
            "evaluation_muscle_groups": evaluation_muscle_groups,
            "evaluation_joint_names": joint_names,
            "evaluation_muscle_names": evaluation_muscle_names,
            "all_actuator_names": all_actuator_names,
            "sensor_force_names": sensor_force_names,
        })
        
        batch_results.append(complete_results)
        
        # Clean up environment
        env_instance.stop()
    
    return batch_results


def main():
    time_start = timeit.default_timer()
    
    # Generate all configurations
    all_rand_configs, all_param_names, all_param_values = generate_randomization_configs()
    n_configs = len(all_rand_configs)
    
    if n_configs == 0:
        print("No randomization configurations generated. Exiting.")
        return
    
    print(f"Total number of configurations: {n_configs}")
    print(f"Using batch size: {args.batch_size}")
    
    
    # Split into batches
    batch_size = args.batch_size
    n_batches = (n_configs + batch_size - 1) // batch_size  # Ceiling division
    
    all_results = []
    rng_keys = jax.random.split(rng, n_configs)

    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_configs)
        
        print(f"Processing batch {batch_idx + 1}/{n_batches} (configs {start_idx+1}-{end_idx})")
        
        batch_configs = all_rand_configs[start_idx:end_idx]
        batch_param_names = all_param_names[start_idx:end_idx]
        batch_param_values = all_param_values[start_idx:end_idx]
        batch_rng_keys = rng_keys[start_idx:end_idx]
        
        # Run batch evaluation
        batch_results = run_batch_evaluation(batch_configs, batch_param_names, batch_param_values, batch_rng_keys)
        all_results.extend(batch_results)
        
        print(f"Completed batch {batch_idx + 1}/{n_batches}")
    
    # Save results
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(path), f"{dt_str}_evaluation_results_parallel")
    os.makedirs(output_dir, exist_ok=True)
    
    for i, results_data in enumerate(all_results):
        param_name_for_file = results_data.get("param_name", "unknown_param")
        param_value_for_file = results_data.get("param_value", "unknown_value")
        
        if isinstance(param_value_for_file, float):
            param_value_for_file_str = f"{param_value_for_file:.3f}".replace('.', '_')
        else:
            param_value_for_file_str = str(param_value_for_file)

        file_name = f"eval_results_{param_name_for_file}_{param_value_for_file_str}_{n_steps}steps.pkl"
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, "wb") as f:
            pickle.dump(results_data, f)
        print(f"Saved: {param_name_for_file}={param_value_for_file} -> {output_path}")

    total_time = timeit.default_timer() - time_start
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Average time per configuration: {total_time/n_configs:.2f} seconds")
    print(f"Speedup from batching: ~{batch_size}x theoretical")


def main_fixed():
    time_start = timeit.default_timer()
    
    # Generate all configurations
    all_rand_configs, all_param_names, all_param_values = generate_randomization_configs()
    n_configs = len(all_rand_configs)
    
    if n_configs == 0:
        print("No randomization configurations generated. Exiting.")
        return
    
    print(f"Total number of configurations: {n_configs}")
    print(f"Using batch size: {args.batch_size}")
    
    # Split into batches
    batch_size = args.batch_size
    n_batches = (n_configs + batch_size - 1) // batch_size
    
    all_results = []
    rng_keys = jax.random.split(rng, n_configs)
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_configs)
        
        print(f"Processing batch {batch_idx + 1}/{n_batches} (configs {start_idx+1}-{end_idx})")
        
        batch_configs = all_rand_configs[start_idx:end_idx]
        batch_param_names = all_param_names[start_idx:end_idx]
        batch_param_values = all_param_values[start_idx:end_idx]
        batch_rng_keys = rng_keys[start_idx:end_idx]
        
        # Use the fixed batch evaluation function
        batch_results = run_batch_evaluation_fixed(batch_configs, batch_param_names, batch_param_values, batch_rng_keys)
        all_results.extend(batch_results)
        
        print(f"Completed batch {batch_idx + 1}/{n_batches}")
    
    # Save results (same as before)
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(path), f"{dt_str}_evaluation_results_parallel")
    os.makedirs(output_dir, exist_ok=True)
    
    for i, results_data in enumerate(all_results):
        param_name_for_file = results_data.get("param_name", "unknown_param")
        param_value_for_file = results_data.get("param_value", "unknown_value")
        
        if isinstance(param_value_for_file, float):
            param_value_for_file_str = f"{param_value_for_file:.3f}".replace('.', '_')
        else:
            param_value_for_file_str = str(param_value_for_file)

        file_name = f"eval_results_{param_name_for_file}_{param_value_for_file_str}_{n_steps}steps.pkl"
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, "wb") as f:
            pickle.dump(results_data, f)
        print(f"Saved: {param_name_for_file}={param_value_for_file} -> {output_path}")

    total_time = timeit.default_timer() - time_start
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Average time per configuration: {total_time/n_configs:.2f} seconds")

if __name__ == "__main__":
    # main()
    main_fixed()