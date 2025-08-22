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



# Parameter randomization initialization (no changes)
randomization_params_names = ["prosthesis_dof_damping", "prosthesis_joint_stiffness", "prosthesis_body_position", "prosthesis_body_orientation"]
randomization_params_eval = {
    "prosthesis_side": "left_side",
    "randomize_prosthesis_dof_damping": False,
    "prosthesis_dof_damping_range": {'ankle_angle': [2, 10]},
    "randomize_prosthesis_joint_stiffness": False,
    "prosthesis_joint_stiffness_range": {'ankle_angle': [50, 100]},
    "randomize_prosthesis_body_position": True,
    "prosthesis_body_position_range": {'pylon_socket': {'x': [0.2, 1.0]}},
    "randomize_prosthesis_body_orientation": False,
    "prosthesis_body_orientation_range": {'pylon_socket': {'x': [-0.1, 0.1]}},
}
randomization_increments = {
    "prosthesis_joint_stiffness": 10,
    "prosthesis_dof_damping": 5,
    "prosthesis_body_position": 0.4,
    "prosthesis_body_orientation": 0.1
}

os.environ["MUJOCO_GL"] = "egl"

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

# Use the path from command line arguments
path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config

randomization_type = config.randomization_config["randomization_type"]
randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)

# Update values in randomization_params based on randomization_params_eval
for key, value in randomization_params_eval.items():
    if key in randomization_params:
        randomization_params[key] = value
    else:
        randomization_params[key] = value

# Add prosthesis side to randomization params if it exists in config.experiment.env_params
if "prosthesis_side" in config.experiment.env_params:
    randomization_params["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]

# Get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# Create env
OmegaConf.set_struct(config, False)
config.experiment.env_params["headless"] = True
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"
config.experiment.env_params["add_sensors"] = True
env = factory.make(domain_randomization_type=randomization_type, domain_randomization_params=randomization_params,
                   **config.experiment.env_params, **config.experiment.task_factory.params)
env.th.to_jax()
env = VecEnv(env)
# The jit_step and jit_reset functions will be re-JIT'd inside the new sweep function
jit_step = jax.jit(jax.vmap(env.mjx_step_test))

model = env.get_model()

prosthesis_metrics_handler = ProsthesisMetricsHandler(env)

n_steps = 50 #200
n_envs = 1
rng = jax.random.key(0)
train_state_seed = 0

keys = jax.random.split(rng, n_envs + 1)
rng, env_keys = keys[0], keys[1:]

def sample_actions_uncompiled(ts, obs, _rng):
    y, updates = agent_conf.network.apply({'params': ts.params, 'run_stats': ts.run_stats},
                                          obs, mutable=["run_stats"])
    ts = ts.replace(run_stats=updates['run_stats'])
    pi, _ = y
    a = pi.sample(seed=_rng)
    return a, ts

sample_actions = jax.jit(sample_actions_uncompiled)

if config.experiment.n_seeds > 1:
    assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
                                             "train_state_seed for replay.")
    train_state = jax.tree.map(lambda x: x[train_state_seed], agent_state.train_state)
else: 
    train_state = agent_state.train_state

muscle_skeleton_control_activation = SkeletonMuscleControlFunction(env)

def run_evaluation_loop(env_state, n_steps, train_state, rng, prosthesis_metrics_handler, param_name=None, param_value=None):
    step_total = 0

    all_foot_ground_contact_left =[]
    all_foot_ground_contact_right =[]

    joint_data = {}
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_data[joint_name] = {
            "angle": [],
            "velocity": [],
            "forces_constraint": [],
            "forces_smooth": [],
            "torques": [],
            "energy_exp": [],
        }
        if "_l" in joint_name or "_r" in joint_name:
            joint_data[joint_name].update({
                "angle_per_step": [],
                "velocity_per_step": [],
                "forces_constraint_per_step": [],
                "forces_smooth_per_step": [],
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
                "torques_per_step_left": [],
                "torques_per_step_right": [],
                "energy_exp_per_step_left": [],
                "energy_exp_per_step_right": [],
            })

    all_sensor_force = {}

    evaluation_muscle_groups = ["back_muscles", "torso_muscles", "vasti_muscles", "rectus_muscles", "medial_muscles", "leg_muscles", "all_muscles"]

    evaluation_muscle_names = {}
    for n in evaluation_muscle_groups:
        evaluation_muscle_names[n] = prosthesis_metrics_handler.get_muscle_group(n)

    for muscle_group, muscle_names in evaluation_muscle_names.items():
        locals()[f"run_{muscle_group}_activation_left"] = []
        locals()[f"run_{muscle_group}_activation_right"] = []

    foot_name = "toes"
    calcn_name = "calcn"
    all_grf_l = []
    all_grf_r = []

    left_side = "left_side"
    right_side = "right_side"

    all_actions = []
    all_actuator_names = []
    
    # This jit_step should be defined outside and passed to the loop if you want to avoid recompiling it.
    # However, for this example, we assume it's defined once globally.
    
    for i in range(n_steps):
        obs = env_state.observation
        rng, _rng = jax.random.split(rng)
        action, train_state = sample_actions(train_state, obs, _rng)
        action = jnp.atleast_2d(action)

        env_state, sys = jit_step(env_state, action)
        obs = env_state.observation

        if i == 0:
            print(f"sys.jnt_stiffness: {sys.jnt_stiffness}")
            print(f"sys.dof_damping: {sys.dof_damping}")
            print(f"sys.body_pos: {sys.body_pos}")
            print(f"sys.body_quat: {sys.body_quat}")
            print(f"Step {i}")

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

        joint_angles = prosthesis_metrics_handler.get_joint_angles(env_state.data)
        joint_velocities = prosthesis_metrics_handler.get_joint_vels(env_state.data)
        joint_forces_constraint, joint_forces_smooth = prosthesis_metrics_handler.get_joint_frces(env_state.data)
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
                action = action.at[..., i_act].set(muscle_skeleton_control_activation.adapted_sigmoid(action[..., i_act]))
        
        all_actions.append(action)

        all_actuator_names = []
        for a in range(model.nu):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            all_actuator_names.append(actuator_name)

        for muscle_group, muscle_names in evaluation_muscle_names.items():
            locals()[f"{muscle_group}_activation_left"] = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, left_side, action)
            locals()[f"{muscle_group}_activation_right"] = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, right_side, action)
            locals()[f"run_{muscle_group}_activation_left"].append(locals()[f"{muscle_group}_activation_left"])
            locals()[f"run_{muscle_group}_activation_right"].append(locals()[f"{muscle_group}_activation_right"])

        step_total += n_envs
        env.mjx_render_domain_randomization(env_state, record=True)

    time_all.append(timeit.default_timer())

    sum_step_activations = {}
    step_norm_activations = {}
    sum_muscles_activations = {}
    step_muscles_norm_activations = {}
    step_num_norm_activations = {}

    for muscle_group in evaluation_muscle_groups:
        sum_step_activations[f"{muscle_group}_left"] = jnp.sum(jnp.array(locals()[f"run_{muscle_group}_activation_left"]), axis=0)
        sum_step_activations[f"{muscle_group}_right"] = jnp.sum(jnp.array(locals()[f"run_{muscle_group}_activation_right"]), axis=0)

        step_norm_activations[f"{muscle_group}_left"] = sum_step_activations[f"{muscle_group}_left"] / n_steps
        step_norm_activations[f"{muscle_group}_right"] = sum_step_activations[f"{muscle_group}_right"] / n_steps

        n_musc = len(evaluation_muscle_names[muscle_group])
        sum_muscles_activations[f"{muscle_group}_left"] = jnp.sum(sum_step_activations[f"{muscle_group}_left"], axis=0)
        sum_muscles_activations[f"{muscle_group}_right"] = jnp.sum(sum_step_activations[f"{muscle_group}_right"], axis=0)
        step_muscles_norm_activations[f"{muscle_group}_left"] = sum_muscles_activations[f"{muscle_group}_left"] / n_steps
        step_muscles_norm_activations[f"{muscle_group}_right"] = sum_muscles_activations[f"{muscle_group}_right"] / n_steps
        step_num_norm_activations[f"{muscle_group}_left"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_left"] / n_musc),
        step_num_norm_activations[f"{muscle_group}_right"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_right"] / n_musc)

    joint_names = []
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_names.append(joint_name)

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
    }

    for joint_name, joint_dict in joint_data.items():
        for key, value in joint_dict.items():
            all_relevant_data[f"{joint_name}_{key}"] = value

    for muscle_group in evaluation_muscle_groups:
        all_relevant_data[f"run_{muscle_group}_activation_left"] = locals().get(f"run_{muscle_group}_activation_left", [])
        all_relevant_data[f"run_{muscle_group}_activation_right"] = locals().get(f"run_{muscle_group}_activation_right", [])

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if param_name:
        # Use a more descriptive file name that can handle floats and different types
        param_value_str = str(param_value).replace('.', '_').replace('-', 'm')
        output_path = os.path.join(os.path.dirname(path), f"{dt_str}_evaluation_results_{n_steps}steps_{param_name}_{param_value_str}.pkl")
    else:
        output_path = os.path.join(os.path.dirname(path), f"{dt_str}_evaluation_results_{n_steps}steps.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(all_relevant_data, f)
    print(f"Saved evaluation data to {output_path}")

    time_all.append(timeit.default_timer())

    print(f"Total time taken for {n_steps} steps: {time_all[-1] - time_all[0]} seconds")
    print(f"Compilation time: {time_all[1] - time_all[0]} seconds")
    print(f"Execution time taken: {time_all[-1] - time_all[1]} seconds")
    print(f"Average time per step: {(time_all[-1] - time_all[1]) / (n_steps-1)} seconds")

# --- ADD THE HELPER FUNCTIONS HERE ---

def reinitialize_prosthesis_randomizer(env, new_config_updates):
    """
    Smart solution to properly reset ProsthesisRandomizer with new configuration.
    Compatible with loco-mujoco and mujoco_mjx.
    """
    for key, value in new_config_updates.items():
        env._domain_randomizer.rand_conf[key] = value

    env._domain_randomizer._body_pos_indices = {}
    env._domain_randomizer._body_quat_indices = {}
    env._domain_randomizer._joint_indices = {}
    env._domain_randomizer._dof_indices = {}
    env._domain_randomizer._socket_joint_indices = {}

    if hasattr(env._domain_randomizer, 'stiffness_dict'):
        env._domain_randomizer.stiffness_dict = {}
    if hasattr(env._domain_randomizer, 'damping_dict'):
        env._domain_randomizer.damping_dict = {}
    if hasattr(env._domain_randomizer, 'socket_joint_range_dict'):
        env._domain_randomizer.socket_joint_range_dict = {}
    if hasattr(env._domain_randomizer, 'prosthesis_body_position_range_dict'):
        env._domain_randomizer.prosthesis_body_position_range_dict = {}
    if hasattr(env._domain_randomizer, 'prosthesis_body_quat_range_dict'):
        env._domain_randomizer.prosthesis_body_quat_range_dict = {}

    if hasattr(env._domain_randomizer, 'randomized_pos_body_names'):
        env._domain_randomizer.randomized_pos_body_names = []
    if hasattr(env._domain_randomizer, 'randomized_quat_body_names'):
        env._domain_randomizer.randomized_quat_body_names = []

    dummy_key = jax.random.PRNGKey(0)
    model = env._model
    data = env._first_data

    new_randomizer_state = env._domain_randomizer.init_state(
        env, dummy_key, model, data, jnp
    )

    new_carry = env._init_additional_carry(dummy_key, model, data, jnp)
    new_carry = new_carry.replace(domain_randomizer_state=new_randomizer_state)
    env._init_carry_template = new_carry

    return env

def run_prosthesis_parameter_sweep(env, randomization_params_eval, randomization_increments,
                                  train_state, prosthesis_metrics_handler, n_envs, n_steps, jit_step, run_evaluation_loop):
    """
    Run a parameter sweep with proper ProsthesisRandomizer reset for loco-mujoco.
    """
    original_config = env._domain_randomizer.rand_conf.copy()
    rng = jax.random.key(0) # We need a new rng for each run

    try:
        for param_name, param_enabled in randomization_params_eval.items():
            # Check if the key indicates a randomization flag
            if not param_name.startswith("randomize_") or not param_enabled:
                continue

            param_key = param_name.replace("randomize_", "")
            
            # This is a general loop for position/orientation
            if "position" in param_key or "orientation" in param_key:
                body_range = randomization_params_eval[f"{param_key}_range"]
                body_name = list(body_range.keys())[0]
                directions = list(body_range[body_name].keys())
                
                for axis in directions:
                    min_val = body_range[body_name][axis][0]
                    max_val = body_range[body_name][axis][1]
                    inc = randomization_increments[param_key]
                    
                    current_value = min_val
                    while current_value <= max_val + 1e-6:
                        new_config = {f"randomize_{param_key}": True}
                        new_config[f"{param_key}_range"] = {body_name: {}}
                        
                        for other_axis in directions:
                            new_config[f"{param_key}_range"][body_name][other_axis] = [0, 0]
                        
                        new_config[f"{param_key}_range"][body_name][axis] = [current_value, current_value]
                        
                        env = reinitialize_prosthesis_randomizer(env, new_config)
                        
                        jit_reset = jax.jit(jax.vmap(env.mjx_reset))
                        
                        keys = jax.random.split(rng, n_envs + 1)
                        rng, env_keys = keys[0], keys[1:]
                        env_state = jit_reset(env_keys)
                        
                        print(f"Running evaluation for {param_key} axis {axis} with value: {current_value}")
                        run_evaluation_loop(env_state, n_steps, train_state, rng,
                                           prosthesis_metrics_handler, param_name=f"{param_key}_{axis}",
                                           param_value=current_value)
                        
                        current_value += inc
            
            # This is a general loop for stiffness/damping
            elif "stiffness" in param_key or "damping" in param_key:
                param_dict = randomization_params_eval[f"{param_key}_range"]
                
                for joint_name, (min_val, max_val) in param_dict.items():
                    inc = randomization_increments[param_key]
                    
                    current_value = min_val
                    while current_value <= max_val + 1e-6:
                        new_config = {f"randomize_{param_key}": True}
                        new_config[f"{param_key}_range"] = {}

                        for other_joint in param_dict.keys():
                            new_config[f"{param_key}_range"][other_joint] = [0, 0]
                        
                        new_config[f"{param_key}_range"][joint_name] = [current_value, current_value]
                        
                        env = reinitialize_prosthesis_randomizer(env, new_config)
                        
                        jit_reset = jax.jit(jax.vmap(env.mjx_reset))
                        
                        keys = jax.random.split(rng, n_envs + 1)
                        rng, env_keys = keys[0], keys[1:]
                        env_state = jit_reset(env_keys)
                        
                        print(f"Running evaluation for {param_key} joint {joint_name} with value: {current_value}")
                        run_evaluation_loop(env_state, n_steps, train_state, rng,
                                          prosthesis_metrics_handler, param_name=f"{param_key}_{joint_name}",
                                          param_value=current_value)
                        
                        current_value += inc

    finally:
        if original_config:
            reinitialize_prosthesis_randomizer(env, original_config)


# --- REPLACE YOUR OLD LOOP WITH THIS ---

time_all = []
time_all.append(timeit.default_timer())

# Call the new function to run the parameter sweep
run_prosthesis_parameter_sweep(
    env,
    randomization_params_eval,
    randomization_increments,
    train_state,
    prosthesis_metrics_handler,
    n_envs,
    n_steps,
    jit_step,
    run_evaluation_loop
)

# 1. ADD THE EGL CONTEXT CLEANUP CODE HERE
try:
    if hasattr(env, 'mjx_model') and hasattr(env.mjx_model, 'ctx'):
        print("Manually freeing EGL context...")
        env.mjx_model.ctx.free()
        print("EGL context freed successfully.")
except Exception as e:
    print(f"Failed to manually free EGL context: {e}")

# 2. Add the final timing here
time_all.append(timeit.default_timer())

# 3. Print the final results
print(f"Total time taken for {n_steps} steps: {time_all[-1] - time_all[0]} seconds")
print(f"Compilation time: {time_all[1] - time_all[0]} seconds")
print(f"Execution time taken: {time_all[-1] - time_all[1]} seconds")
print(f"Average time per step: {(time_all[-1] - time_all[1]) / (n_steps-1)} seconds")
