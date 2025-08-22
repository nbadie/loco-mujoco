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

sys.path.append(os.path.join(os.path.dirname(__file__), "/home/nadinebadie/loco-mujoco/prosthesis_test"))
from loco_mujoco.evaluation.evaluation_prosthesis import ProsthesisMetricsHandler
from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction

import mujoco
from datetime import datetime
import timeit 
import sys

# Parameter randomization initialization
randomization_params_names = ["prosthesis_dof_damping", "prosthesis_joint_stiffness", "prosthesis_body_position", "prosthesis_body_orientation"]

randomization_params_eval = {
    "prosthesis_side": "left_side",
    "randomize_prosthesis_dof_damping": False,
    "prosthesis_dof_damping_range": {'ankle_angle': [2, 10]},
    "randomize_prosthesis_joint_stiffness": False,
    "prosthesis_joint_stiffness_range": {'ankle_angle': [50, 100]},
    "randomize_prosthesis_body_position": True,
    "prosthesis_body_position_range": {'pylon_socket': {'x': [0.2, 0.25]}},
    "randomize_prosthesis_body_orientation": False,
    "prosthesis_body_orientation_range": {'pylon_socket': {'x': [-0.1, 0.1]}},
}

randomization_increments = {
    "prosthesis_joint_stiffness": 10,
    "prosthesis_dof_damping": 5,
    "prosthesis_body_position": 0.05,
    "prosthesis_body_orientation": 0.1}


# Initialize storage
all_rand_configs = []
all_param_names = []
all_param_values = []

os.environ["MUJOCO_GL"] = "egl"

parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config

randomization_type = config.randomization_config["randomization_type"]
randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)

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
    y, updates = agent_conf.network.apply({'params': ts.params,
                                           'run_stats': ts.run_stats},
                                           obs, mutable=["run_stats"])
    ts = ts.replace(run_stats=updates['run_stats'])
    pi, _ = y
    a = pi.sample(seed=_rng)
    return a, ts

# sample_actions = jax.jit(sample_actions_uncompiled)
sample_actions = jax.jit(jax.vmap(sample_actions_uncompiled, in_axes=(None, 0, 0)))

if config.experiment.n_seeds > 1:
    assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
                                            "train_state_seed for replay.")
    
    train_state = jax.tree.map(lambda x: x[train_state_seed], agent_state.train_state)
else: 
    train_state = agent_state.train_state

def create_env_with_randomization(rand_params_config):
    local_randomization_params = randomization_params.copy()
    local_randomization_params.update(rand_params_config)
    
    env_ = factory.make(domain_randomization_type=randomization_type, domain_randomization_params=local_randomization_params,
                       **config.experiment.env_params, **config.experiment.task_factory.params)
    env_.th.to_jax()
    env_ = VecEnv(env_)

    return env_


def adapted_sigmoid(action):
    """
    Applies a sigmoid activation function to the action, adapted to the actuator limits.
    The sigmoid has a constant for a steeper slope 
    """
    q = 5
    return 1 / (1 + jax.numpy.exp(-q * action))

# JIT-compiled core simulation step
@jax.jit
def simulation_step(env_state, train_state, action, rng_step, metrics_handler):
    """JIT-compiled simulation step with batch data collection"""
    # Environment step
    env_state, sys_data = metrics_handler.env.mjx_step_test(env_state, action)
    
    # Batch data collection (all JAX-compatible)
    joint_data = metrics_handler.get_joint_data_batch(env_state.data)
    grf_l, grf_r = metrics_handler.get_grf_batch(env_state.data)
    sensor_data = metrics_handler.get_sensor_data_batch(env_state.data)
    
    return env_state, joint_data, grf_l, grf_r, sensor_data

@jax.jit
def jit_process_muscle_actions(action, muscle_mask):
    """Applies adapted_sigmoid only to muscle actuators."""
    # Create a new action array
    processed_action = action
    
    # Use jax.where to apply the sigmoid only where the mask is True
    # Assumes adapted_sigmoid is a jax-compatible function
    processed_action = jax.numpy.where(
        muscle_mask,
        adapted_sigmoid(action),  # Apply sigmoid
        action                     # Keep the original value
    )
    return processed_action
# JIT-compiled muscle activation processing
# @jax.jit
# def process_muscle_activations(action, muscle_actuator_mask):
#     """Apply sigmoid to muscle actuators only"""
#     def apply_sigmoid(val, is_muscle):
#         return jax.lax.cond(is_muscle, 
#                            lambda x: jax.nn.sigmoid(x), 
#                            lambda x: x, 
#                            val)
    
#     processed_action = jax.vmap(apply_sigmoid)(action, muscle_actuator_mask)
#     return processed_action

# def run_single_evaluation_loop(env_state, train_state, rng_eval, current_model, current_prosthesis_metrics_handler, param_name=None, param_value=None):
#     """Main evaluation loop with optimized JAX operations"""
    
#     # Pre-allocate storage arrays
#     all_foot_ground_contact_left = []
#     all_foot_ground_contact_right = []
#     all_grf_l = []
#     all_grf_r = []
#     all_actions = []
#     all_sensor_data = []
    
#     # Pre-allocate joint data storage
#     joint_data_storage = {
#         'angles': [],
#         'velocities': [],
#         'forces_constraint': [],
#         'forces_smooth': [],
#         'torques': [],
#         'energy_exp': []
#     }
    
#     # Pre-compute muscle group information
#     evaluation_muscle_groups = ["back_muscles", "torso_muscles", "vasti_muscles", "rectus_muscles", "medial_muscles", "leg_muscles", "all_muscles"]
#     evaluation_muscle_names = {}
#     muscle_indices_left = {}
#     muscle_indices_right = {}

    
#     for group in evaluation_muscle_groups:
#         evaluation_muscle_names[group] = current_prosthesis_metrics_handler.get_muscle_group(group)
#         muscle_indices_left[group] = current_prosthesis_metrics_handler.setup_muscle_indices(current_model, evaluation_muscle_names, group, 'left')
#         muscle_indices_right[group] = current_prosthesis_metrics_handler.setup_muscle_indices(current_model, evaluation_muscle_names, group, 'right')
    
#     # Pre-allocate muscle activation storage
#     run_muscle_activations_left = {mg: [] for mg in evaluation_muscle_groups}
#     run_muscle_activations_right = {mg: [] for mg in evaluation_muscle_groups}
    
#     # Get JIT-compiled functions
#     # jit_simulation_step = jax.jit(simulation_step, static_argnums=(4,))
#     jit_contact_detection = jax.jit(current_prosthesis_metrics_handler.get_contact_steps_batched)
#     # jit_muscle_processing = jax.jit(process_muscle_activations)
    
#     # Main simulation loop
#     for i in range(n_steps):
#         obs = env_state.observation
#         rng_eval, _rng = jax.random.split(rng_eval)
#         action, train_state = sample_actions(train_state, obs, _rng)
        
#         # # Ensure action is 1D
#         # if action.ndim > 1:
#         #     action = action.squeeze(axis=0)
        
#         # Process muscle activations
#         # processed_action = jit_muscle_processing(action, current_prosthesis_metrics_handler.muscle_actuator_mask)
#         processed_action = jit_process_muscle_actions(action, muscle_actuator_mask)
#         # for i in range(current_model.nu):
#         #     if current_model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE:
#         #         action = action.at[...,i].set(muscle_skeleton_control_activation[...].adapted_sigmoid(action[...,i]))
#         # Run simulation step (JIT-compiled)
#         # env_state, joint_data, grf_l, grf_r, sensor_data = jit_simulation_step(
#         #     env_state, train_state, action, _rng, current_prosthesis_metrics_handler
#         # )

#         env_state = jit_step(env_state, processed_action)
#         joint_data = current_prosthesis_metrics_handler.get_joint_data_batch(env_state.data)
#         grf_l, grf_r = current_prosthesis_metrics_handler.get_grf_batch(env_state.data)
#         sensor_data = current_prosthesis_metrics_handler.get_sensor_data_batch(env_state.data)

#         # Joint Data: 
#         joint_angles = current_prosthesis_metrics_handler.get_joint_angles(env_state.data)
#         joint_velocities = current_prosthesis_metrics_handler.get_joint_vels(env_state.data)
#         joint_forces_constraint, joint_forces_smooth = current_prosthesis_metrics_handler.get_joint_frces(env_state.data)
#         joint_torques = current_prosthesis_metrics_handler.get_joint_trques(env_state.data)
#         joint_energy_exp = current_prosthesis_metrics_handler.calc_joint_energy_exp(joint_torques, joint_velocities)
        
#         # Append all joint data to the joint_data dictionary
#         for joint_name, angle in joint_angles.items():
#             joint_data[joint_name]["angle"].append(angle)
#         for joint_name, velocity in joint_velocities.items():
#             joint_data[joint_name]["velocity"].append(velocity)
#         for joint_name, force in joint_forces_constraint.items():
#             joint_data[joint_name]["forces_constraint"].append(force)
#         for joint_name, force in joint_forces_smooth.items():
#             joint_data[joint_name]["forces_smooth"].append(force)
#         for joint_name, torque in joint_torques.items():
#             joint_data[joint_name]["torques"].append(torque)
#         for joint_name, energy_exp in joint_energy_exp.items():
#             joint_data[joint_name]["energy_exp"].append(energy_exp)

#         # Contact detection (JIT-compiled)
#         contact_left, contact_right = jit_contact_detection(env_state.data, i)
        
#         # Store data (non-JIT operations)
#         all_foot_ground_contact_left.append(int(contact_left))
#         all_foot_ground_contact_right.append(int(contact_right))
#         all_grf_l.append(grf_l)
#         all_grf_r.append(grf_r)
#         all_actions.append(action)
#         all_sensor_data.append(sensor_data)
        
#         # Store joint data
#         for key in joint_data_storage:
#             joint_data_storage[key].append(joint_data[key])
        
#         # Extract muscle activations for each group
#         for muscle_group in evaluation_muscle_groups:
#             if len(muscle_indices_left[muscle_group]) > 0:
#                 act_left = action[muscle_indices_left[muscle_group]]
#                 run_muscle_activations_left[muscle_group].append(act_left)
#             else:
#                 run_muscle_activations_left[muscle_group].append(jnp.array([]))
                
#             if len(muscle_indices_right[muscle_group]) > 0:
#                 act_right = action[muscle_indices_right[muscle_group]]
#                 run_muscle_activations_right[muscle_group].append(act_right)
#             else:
#                 run_muscle_activations_right[muscle_group].append(jnp.array([]))
    
#     # Post-process data (non-JIT operations)
#     # Convert lists to arrays
#     for key in joint_data_storage:
#         joint_data_storage[key] = jnp.array(joint_data_storage[key])
    
#     # Process muscle activations
#     for muscle_group in evaluation_muscle_groups:
#         if len(run_muscle_activations_left[muscle_group]) > 0 and len(run_muscle_activations_left[muscle_group][0]) > 0:
#             run_muscle_activations_left[muscle_group] = jnp.array(run_muscle_activations_left[muscle_group])
#             run_muscle_activations_right[muscle_group] = jnp.array(run_muscle_activations_right[muscle_group])
#         else:
#             run_muscle_activations_left[muscle_group] = jnp.array([])
#             run_muscle_activations_right[muscle_group] = jnp.array([])
    
#     # Calculate aggregated muscle activation metrics
#     sum_step_activations = {}
#     step_norm_activations = {}
#     sum_muscles_activations = {}
#     step_muscles_norm_activations = {}
#     step_num_norm_activations = {}

#     for muscle_group in evaluation_muscle_groups:
#         if len(run_muscle_activations_left[muscle_group]) > 0:
#             sum_step_activations[f"{muscle_group}_left"] = jnp.sum(run_muscle_activations_left[muscle_group], axis=0)
#             sum_step_activations[f"{muscle_group}_right"] = jnp.sum(run_muscle_activations_right[muscle_group], axis=0)
#             step_norm_activations[f"{muscle_group}_left"] = sum_step_activations[f"{muscle_group}_left"] / n_steps
#             step_norm_activations[f"{muscle_group}_right"] = sum_step_activations[f"{muscle_group}_right"] / n_steps

#             n_musc = len(evaluation_muscle_names[muscle_group])
#             sum_muscles_activations[f"{muscle_group}_left"] = jnp.sum(sum_step_activations[f"{muscle_group}_left"])
#             sum_muscles_activations[f"{muscle_group}_right"] = jnp.sum(sum_step_activations[f"{muscle_group}_right"])
#             step_muscles_norm_activations[f"{muscle_group}_left"] = sum_muscles_activations[f"{muscle_group}_left"] / n_steps
#             step_muscles_norm_activations[f"{muscle_group}_right"] = sum_muscles_activations[f"{muscle_group}_right"] / n_steps
#             step_num_norm_activations[f"{muscle_group}_left"] = step_muscles_norm_activations[f"{muscle_group}_left"] / n_musc
#             step_num_norm_activations[f"{muscle_group}_right"] = step_muscles_norm_activations[f"{muscle_group}_right"] / n_musc
#         else:
#             # Handle empty muscle groups
#             sum_step_activations[f"{muscle_group}_left"] = jnp.array([])
#             sum_step_activations[f"{muscle_group}_right"] = jnp.array([])
#             step_norm_activations[f"{muscle_group}_left"] = jnp.array([])
#             step_norm_activations[f"{muscle_group}_right"] = jnp.array([])
#             sum_muscles_activations[f"{muscle_group}_left"] = 0.0
#             sum_muscles_activations[f"{muscle_group}_right"] = 0.0
#             step_muscles_norm_activations[f"{muscle_group}_left"] = 0.0
#             step_muscles_norm_activations[f"{muscle_group}_right"] = 0.0
#             step_num_norm_activations[f"{muscle_group}_left"] = 0.0
#             step_num_norm_activations[f"{muscle_group}_right"] = 0.0

#     # Process sensor data to dictionary format
#     all_sensor_force = {}
#     sensor_force_names = current_prosthesis_metrics_handler._simple_sensor_names #sensor_names
    
#     if len(all_sensor_data) > 0:
#         for i, sensor_name in enumerate(sensor_force_names):
#             all_sensor_force[sensor_name] = []
#             for step_data in all_sensor_data:
#                 if current_prosthesis_metrics_handler._simple_nsensor ==1 : #nsensor == 1:
#                     all_sensor_force[sensor_name].append(step_data)
#                 else:
#                     start_idx = i * 3
#                     end_idx = start_idx + 3
#                     all_sensor_force[sensor_name].append(step_data[start_idx:end_idx])

#     # Create joint data dictionary in the expected format
#     joint_data = {}
#     for i, joint_name in enumerate(current_prosthesis_metrics_handler.joint_names):
#         joint_data[joint_name] = {
#             "angle": [float(joint_data_storage['angles'][step][i]) for step in range(n_steps)],
#             "velocity": [float(joint_data_storage['velocities'][step][i]) for step in range(n_steps)],
#             "forces_constraint": [float(joint_data_storage['forces_constraint'][step][i]) for step in range(n_steps)],
#             "forces_smooth": [float(joint_data_storage['forces_smooth'][step][i]) for step in range(n_steps)],
#             "torques": [float(joint_data_storage['torques'][step][i]) for step in range(n_steps)],
#             "energy_exp": [float(joint_data_storage['energy_exp'][step][i]) for step in range(n_steps)],
#         }
        
#         # Add per-step data based on joint naming convention
#         if "_l" in joint_name or "_r" in joint_name:
#             joint_data[joint_name].update({
#                 "angle_per_step": joint_data[joint_name]["angle"],
#                 "velocity_per_step": joint_data[joint_name]["velocity"],
#                 "forces_constraint_per_step": joint_data[joint_name]["forces_constraint"],
#                 "forces_smooth_per_step": joint_data[joint_name]["forces_smooth"],
#                 "torques_per_step": joint_data[joint_name]["torques"],
#                 "energy_exp_per_step": joint_data[joint_name]["energy_exp"],
#             })
#         else:
#             joint_data[joint_name].update({
#                 "angle_per_step_left": joint_data[joint_name]["angle"],
#                 "angle_per_step_right": joint_data[joint_name]["angle"],
#                 "velocity_per_step_left": joint_data[joint_name]["velocity"],
#                 "velocity_per_step_right": joint_data[joint_name]["velocity"],
#                 "forces_constraint_per_step_left": joint_data[joint_name]["forces_constraint"],
#                 "forces_constraint_per_step_right": joint_data[joint_name]["forces_constraint"],
#                 "forces_smooth_per_step_left": joint_data[joint_name]["forces_smooth"],
#                 "forces_smooth_per_step_right": joint_data[joint_name]["forces_smooth"],
#                 "torques_per_step_left": joint_data[joint_name]["torques"],
#                 "torques_per_step_right": joint_data[joint_name]["torques"],
#                 "energy_exp_per_step_left": joint_data[joint_name]["energy_exp"],
#                 "energy_exp_per_step_right": joint_data[joint_name]["energy_exp"],
#             })

#     # Compile all results
#     all_relevant_data = {
#         "total_steps": n_steps,
#         "all_grf_l": all_grf_l,
#         "all_grf_r": all_grf_r,
#         "all_foot_ground_contact_left": all_foot_ground_contact_left,
#         "all_foot_ground_contact_right": all_foot_ground_contact_right,
#         "all_sensor_force": all_sensor_force,
#         "sensor_force_names": sensor_force_names,
#         "sum_step_activations": sum_step_activations,
#         "step_norm_activations": step_norm_activations,
#         "sum_muscles_activations": sum_muscles_activations,
#         "step_muscles_norm_activations": step_muscles_norm_activations,
#         "step_num_norm_activations": step_num_norm_activations,
#         "evaluation_muscle_groups": evaluation_muscle_groups,
#         "evaluation_joint_names": current_prosthesis_metrics_handler.joint_names,
#         "evaluation_muscle_names": evaluation_muscle_names,
#         "all_actions": all_actions,
#         "all_actuator_names": current_prosthesis_metrics_handler.actuator_names,
#         "param_name": param_name,
#         "param_value": param_value
#     }

#     # Add joint data to results
#     for joint_name, joint_dict in joint_data.items():
#         for key, value in joint_dict.items():
#             all_relevant_data[f"{joint_name}_{key}"] = value

#     # Add muscle activation data
#     for muscle_group in evaluation_muscle_groups:
#         all_relevant_data[f"run_{muscle_group}_activation_left"] = run_muscle_activations_left[muscle_group]
#         all_relevant_data[f"run_{muscle_group}_activation_right"] = run_muscle_activations_right[muscle_group]

#     return all_relevant_data, env_state

@jax.jit
def run_all_evaluations_batched(batched_env_state, train_state, rng_eval):
    """
    Runs the full evaluation for all environments in parallel using jax.lax.scan.
    This function replaces the Python `for` loop over environments and time.
    """

    step_count = 0
    
    def _eval_step_fn(carry, unused):
        """
        A single JAX step for a batch of environments.
        This function is executed repeatedly by jax.lax.scan.
        """
        env_state, train_state, rng_eval = carry

        obs = env_state.observation
        rng_eval, _rng = jax.random.split(rng_eval)
        
        # Action sampling is already vmapped
        action, train_state = sample_actions(train_state, obs, _rng)
        
        # Process actions with a vmapped function
        processed_action = jit_process_muscle_actions(action, muscle_actuator_mask)
        
        # Step the batched environment with the vmapped jit_step
        env_state = jit_step(env_state, processed_action)
        
        foot_name = "toes"  # name of the foot box in the model
        calcn_name = "calcn"
        # Collect metrics for this timestep and all environments.
        # This will create a list of JAX arrays, one for each timestep.
        # The axes will be (n_envs, ...). jax.lax.scan will stack these
        # along a new leading axis, resulting in (n_steps, n_envs, ...).
        step_metrics = {
            # 'qpos': env_state.data.qpos,
            # 'qvel': env_state.data.qvel,
            'joint_angles': jax.vmap(lambda d: prosthesis_metrics_handlers[0].get_joint_angles(d))(env_state.data),
            'joint_vels': jax.vmap(lambda d: prosthesis_metrics_handlers[0].get_joint_vels(d))(env_state.data),
            'joint_frces': jax.vmap(lambda d: prosthesis_metrics_handlers[0].get_joint_frces(d))(env_state.data),
            'joint_trques': jax.vmap(lambda d: prosthesis_metrics_handlers[0].get_joint_trques(d))(env_state.data),
            'joint_energy_exp': jax.vmap(lambda d: prosthesis_metrics_handlers[0].get_joint_energy_exp(d))(env_state.data),

            'toes_l': jax.vmap(lambda d: prosthesis_metrics_handlers[0].get_grf(env_state.data,f"{foot_name}_l" )),
            'calcn_l': jax.vmap(lambda d: prosthesis_metrics_handlers[0].get_grf(env_state.data,f"{calcn_name}_l")),
            'toes_r': jax.vmap(lambda d: prosthesis_metrics_handlers[0].get_grf(env_state.data,f"{foot_name}_r" )),
            'calcn_r': jax.vmap(lambda d: prosthesis_metrics_handlers[0].get_grf(env_state.data,f"{calcn_name}_r")),

            'sensor_data': jax.vmap(lambda d: prosthesis_metrics_handlers[0].get_sensor_data_batched(env_state.data)),
            
            'sensor_data': jax.vmap(lambda d: prosthesis_metrics_handlers[0].get_sensor_data_batched(env_state.data)),
            'contacts' : jax.vmap(lambda d: prosthesis_metrics_handlers[0].get_contact_steps_batched(env_state.data, step_count)),
        }

        step_count += 1


        # Return the new carry state and the metrics for this step
        return (env_state, train_state, rng_eval), step_metrics   


    # Initial state for the scan loop
    initial_carry = (batched_env_state, train_state, rng_eval)

    # Use jax.lax.scan to run the loop over `n_steps`
    final_carry, all_steps_metrics = jax.lax.scan(
        _eval_step_fn,
        initial_carry,
        None, # `None` because the step function doesn't use the second argument
        length=n_steps
    )
    
    # Return all the collected data and the final state
    return all_steps_metrics, final_carry[0]


# Main execution
time_all = []
time_all.append(timeit.default_timer())

all_evaluation_results = []
all_rand_configs = []
all_param_names = []
all_param_values = []

# Generate randomization configurations
for param_name in randomization_params_names:
    if randomization_params[f"randomize_{param_name}"]:
        if "stiffness" in param_name or "damping" in param_name:
            # Handle nested structure: {'joint_name': [min, max]}
            param_range = randomization_params[f"{param_name}_range"]
            
            for joint_name, (min_val, max_val) in param_range.items():
                inc = randomization_increments[param_name]
                for val in range(min_val, max_val + 1, inc):
                    rand_config = {
                        f"randomize_{param_name}": True, 
                        f"{param_name}_range": {joint_name: [val, val]}
                    }
                    # Set all other joints in this parameter to zero/default
                    for other_joint in param_range.keys():
                        if other_joint != joint_name:
                            rand_config[f"{param_name}_range"][other_joint] = [0, 0]
                    
                    # Disable other parameter types
                    for other_param in randomization_params_names:
                        if other_param != param_name:
                            rand_config[f"randomize_{other_param}"] = False
                    
                    all_rand_configs.append(rand_config)
                    all_param_names.append(f"{param_name}_{joint_name}")
                    all_param_values.append(val)

        elif "position" in param_name or "orientation" in param_name:
            # Handle nested structure: {'body_name': {'axis': [min, max]}}
            param_range = randomization_params[f"{param_name}_range"]
            
            for body_name, body_axes in param_range.items():
                for axis, (min_val, max_val) in body_axes.items():
                    inc = randomization_increments[param_name]
                    for val in np.arange(min_val, max_val + inc/2, inc):
                        rand_config = {
                            f"randomize_{param_name}": True, 
                            f"{param_name}_range": {body_name: {axis: [val, val]}}
                        }
                        
                        # Set all other axes for this body to zero/default
                        for other_axis in body_axes.keys():
                            if other_axis != axis:
                                rand_config[f"{param_name}_range"][body_name][other_axis] = [0.0, 0.0]
                        
                        # Set all other bodies in this parameter to zero/default
                        for other_body in param_range.keys():
                            if other_body != body_name:
                                rand_config[f"{param_name}_range"][other_body] = {}
                                for other_body_axis in param_range[other_body].keys():
                                    rand_config[f"{param_name}_range"][other_body][other_body_axis] = [0.0, 0.0]
                        
                        # Disable other parameter types
                        for other_param in randomization_params_names:
                            if other_param != param_name:
                                rand_config[f"randomize_{other_param}"] = False
                        
                        all_rand_configs.append(rand_config)
                        all_param_names.append(f"{param_name}_{body_name}_{axis}")
                        all_param_values.append(val)

# Print results for verification
print(f"Generated {len(all_rand_configs)} configurations:")
for i, (rand_config, name, value) in enumerate(zip(all_rand_configs, all_param_names, all_param_values)):
    print(f"Config {i+1}: {name} = {value}")
    print(f"  Configuration: {rand_config}")
    print()
    
n_envs = len(all_rand_configs)
if n_envs == 0:
    print("No randomization configurations generated. Exiting.")
    sys.exit()

print(f"Total number of environments to run in parallel: {n_envs}")

# # Create environments and handlers
# envs = [create_env_with_randomization(rand_conf) for rand_conf in all_rand_configs]
# muscle_skeleton_control_activation = [SkeletonMuscleControlFunction(env_) for env_ in envs]
# models = [env_.get_model() for env_ in envs]
# # batched_model = jax.vmap(lambda m: m)(models)

# # Initial data for each environment
# mjx_data_list = [env_.data for env_ in envs]
# # batched_data = jax.vmap(lambda d: d)(mjx_data_list)

# model_0 = models[0]
# nu = model_0.nu
# muscle_actuator_mask = jnp.zeros(nu, dtype=jnp.bool_)


# # This loop runs once, outside the JIT-compiled function
# for i in range(nu):
#     if model_0.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE:
#         muscle_actuator_mask = muscle_actuator_mask.at[i].set(True)
        

# prosthesis_metrics_handlers = [ProsthesisMetricsHandler(env_) for env_ in envs]

# env_keys = jax.random.split(rng, n_envs)


# single_mjx_step_fn = envs[0].mjx_step
# jit_step = jax.jit(jax.vmap(single_mjx_step_fn, in_axes=(0, 0)))

# # jit_step  = jax.jit(jax.vmap(envs.mjx_step))  #env.step)
# # jit_reset  = jax.jit(jax.vmap(envs.mjx_reset))

# # Run evaluations
# all_results = []
# for i, (env_instance, model_instance, metrics_handler_instance, rand_config_i, p_name, p_value) in enumerate(
#     zip(envs, models, prosthesis_metrics_handlers, all_rand_configs, all_param_names, all_param_values)
# ):
#     print(f"Running evaluation for configuration {i+1}/{n_envs}: {p_name} with value {p_value}")
    
#     current_env_state = jax.jit(env_instance.mjx_reset)(env_keys[i])
    
#     results, final_env_state = run_single_evaluation_loop(
#         current_env_state, train_state, jax.random.split(rng, n_envs)[i], 
#         model_instance, metrics_handler_instance,p_name, p_value
#     )
#     all_results.append(results)

# time_all.append(timeit.default_timer())



# Step 1: Create a batched representation of everything.
envs = [create_env_with_randomization(rand_conf) for rand_conf in all_rand_configs]
prosthesis_metrics_handlers = [ProsthesisMetricsHandler(env_) for env_ in envs]
models = [env_.get_model() for env_ in envs]

# Extract the muscle mask from a single model (they should all be the same)
model_0 = models[0]
nu = model_0.nu
muscle_actuator_mask = jnp.zeros(nu, dtype=jnp.bool_)
for i in range(nu):
    if model_0.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE:
        muscle_actuator_mask = muscle_actuator_mask.at[i].set(True)

def batch_reset(key, envs):
    """Correctly resets a batch of environments with consistent logic."""
    keys = jax.random.split(key, n_envs)
    
    # Use the mjx_reset function from the environment class
    reset_fn = jax.vmap(lambda env, k: env.mjx_reset(k))
    
    # Call the reset function on the batch of environments
    return reset_fn(envs, keys)

# Step 2: Create a single batched environment state and rng key
rng = jax.random.key(0)
env_keys = jax.random.split(rng, n_envs)
initial_env_states = [env.mjx_reset(key) for env, key in zip(envs, env_keys)]
batched_env_state = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *initial_env_states)

# Step 3: Define the vmapped step function
single_mjx_step_fn = envs[0].mjx_step
jit_step = jax.jit(jax.vmap(single_mjx_step_fn, in_axes=(0, 0)))

# Step 4: Run the single, JIT-compiled parallel evaluation loop
rng, eval_rng = jax.random.split(rng)
all_steps_metrics, final_batched_env_state = run_all_evaluations_batched(
    batched_env_state, train_state, eval_rng
)

# Step 5: Post-process and save the results
# `all_steps_metrics` now contains all your data.
# It's a PyTree of arrays. The first dimension is time, the second is the environment batch.
# For example, `all_steps_metrics['qpos'].shape` would be `(n_steps, n_envs, qpos_size)`.

# Use a Python loop to save the results, as this is I/O-bound and not a bottleneck.
# You can split the batched data and save it one by one.
for i in range(n_envs):
    param_name = all_param_names[i]
    param_value = all_param_values[i]
    
    single_env_results = jax.tree_util.tree_map(lambda x: x[:, i], all_steps_metrics)

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(path), f"{dt_str}_evaluation_results_parallel")
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"eval_results_{param_name}_{param_value}_{n_steps}steps.pkl"
    output_path = os.path.join(output_dir, file_name)

    with open(output_path, "wb") as f:
        pickle.dump(single_env_results, f)
    print(f"Saved evaluation data for {param_name}={param_value} to {output_path}")



# # Save results
# dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_dir = os.path.join(os.path.dirname(path), f"{dt_str}_evaluation_results_parallel")
# os.makedirs(output_dir, exist_ok=True)

# for i, results_data in enumerate(all_results):
#     param_name_for_file = results_data.get("param_name", "unknown_param")
#     param_value_for_file = results_data.get("param_value", "unknown_value")
    
#     if isinstance(param_value_for_file, float):
#         param_value_for_file_str = f"{param_value_for_file:.3f}".replace('.', '_')
#     else:
#         param_value_for_file_str = str(param_value_for_file)

#     file_name = f"eval_results_{param_name_for_file}_{param_value_for_file_str}_{n_steps}steps.pkl"
#     output_path = os.path.join(output_dir, file_name)
#     with open(output_path, "wb") as f:
#         pickle.dump(results_data, f)
#     print(f"Saved evaluation data for {param_name_for_file}={param_value_for_file} to {output_path}")

print(f"Total time taken for all evaluations: {time_all[-1] - time_all[0]} seconds")
print(f"Execution time taken (overall): {time_all[-1] - time_all[0]} seconds")

# Clean up
for env_to_stop in envs:
    env_to_stop.stop()