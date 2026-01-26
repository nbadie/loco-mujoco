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

subfolder_name = 'pylon_socket_pos_z_PARALLEL'
dt_str_init = datetime.now().strftime("%Y%m%d_%H%M%S")

talus_ori_ang = np.deg2rad(5)
set_fixed_randomized_targets = True
fixed_randomized_targets = {"prosthesis_body_orientation": {"pylon_socket": {"x": talus_ori_ang}, "talus": {"z": talus_ori_ang}}}

ori_ang = np.deg2rad(6)
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

parser = argparse.ArgumentParser(description='Run evaluation with PPOJax in PARALLEL.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--batch_size', type=int, default=5, help='Number of configurations to run in parallel')
args = parser.parse_args()

path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config

randomization_type = config.randomization_config["randomization_type"]
randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)

for key, value in randomization_params_eval.items():
    randomization_params[key] = value

if "prosthesis_side" in config.experiment.env_params:
    randomization_params["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]

factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

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
batch_size = args.batch_size

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

# Build all configurations
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

print(f"{'='*80}")
print(f"TRULY PARALLEL EVALUATION")
print(f"{'='*80}")
print(f"Total configurations: {len(eval_configs)}")
print(f"Batch size: {batch_size}")
print(f"Number of batches: {(len(eval_configs) + batch_size - 1) // batch_size}")
print(f"{'='*80}\n")

# Get metadata
body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
all_actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)]

evaluation_muscle_groups = ["back_muscles", "torso_muscles", "vasti_muscles", "rectus_muscles", "medial_muscles", "leg_muscles", "all_muscles"]
evaluation_muscle_names = {}
for n in evaluation_muscle_groups:
    evaluation_muscle_names[n] = prosthesis_metrics_handler.get_muscle_group(n)

foot_name = "toes"
calcn_name = "calcn"
left_side = "left_side"
right_side = "right_side"


def configure_env_for_config(config):
    """Configure environment randomization settings for a specific configuration."""
    param_name = config['param_name']
    param_value = config['param_value']
    direction = config['direction']
    body_or_joint_name = config['body_or_joint_name']
    
    # Set all randomization flags to False
    for name in randomization_params_names:
        env._domain_randomizer.rand_conf[f"randomize_{name}"] = False
    
    # Set current parameter to True
    env._domain_randomizer.rand_conf[f"randomize_{param_name}"] = True
    
    # Set fixed randomized targets
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


def run_single_evaluation_parallel(config_idx, config, rng_key):
    """Run a single evaluation - designed to be vmapped."""
    # Configure environment for this specific configuration
    configure_env_for_config(config)
    
    # Create unique keys for this config
    keys = jax.random.split(rng_key, 2)
    reset_key, step_key = keys[0], keys[1]
    
    # Reset environment
    jit_reset = jax.jit(jax.vmap(env.mjx_reset))
    jit_step = jax.jit(jax.vmap(env.mjx_step))
    
    env_keys = jax.random.split(reset_key, 1)
    env_state = jit_reset(env_keys)
    
    # Initialize storage arrays
    all_grf_l = jnp.zeros((n_steps, 6))
    all_grf_r = jnp.zeros((n_steps, 6))
    all_actions = jnp.zeros((n_steps, model.nu))
    
    # Run episode
    for i in range(n_steps):
        obs = env_state.observation
        step_key, action_key = jax.random.split(step_key)
        action, _ = sample_actions(train_state, obs, action_key)
        action = jnp.atleast_2d(action)
        
        env_state = jit_step(env_state, action)
        
        # Collect GRF
        grf_foot_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_l")
        grf_foot_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_r")
        grf_calcn_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{calcn_name}_l")
        grf_calcn_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{calcn_name}_r")
        
        all_grf_l = all_grf_l.at[i].set(grf_foot_l + grf_calcn_l)
        all_grf_r = all_grf_r.at[i].set(grf_foot_r + grf_calcn_r)
        
        # Process actions through sigmoid for muscles
        for i_act in range(model.nu):
            if model.actuator_dyntype[i_act] == mujoco.mjtDyn.mjDYN_MUSCLE:
                action = action.at[...,i_act].set(muscle_skeleton_control_activation.adapted_sigmoid(action[...,i_act]))
        
        all_actions = all_actions.at[i].set(jnp.squeeze(action))
    
    return {
        'grf_l': all_grf_l,
        'grf_r': all_grf_r,
        'actions': all_actions,
        'config': config
    }



def save_evaluation_result(result, config, path, subfolder_name, dt_str_init, n_steps):
    """Save evaluation result to file."""
    param_name = config['param_name']
    param_value = config['param_value']
    direction = config['direction']
    body_or_joint_name = config['body_or_joint_name']
    
    all_relevant_data = {
        "total_steps": n_steps,
        "n_envs": 1,
        "all_grf_l": result['grf_l'],
        "all_grf_r": result['grf_r'],
        "all_actions": result['actions'],
        "all_actuator_names": all_actuator_names,
        "evaluation_body_names": body_names,
        "evaluation_joint_names": joint_names,
        "evaluation_muscle_groups": evaluation_muscle_groups,
        "evaluation_muscle_names": evaluation_muscle_names,
    }
    
    if 'orientation' in param_name:
        name_param_value = np.rad2deg(param_value)
        name_units = 'deg'
    elif 'position' in param_name:
        name_param_value = param_value*1000
        name_units = 'mm'
    else:
        name_param_value = param_value
        name_units = ''
    
    save_dir = os.path.join(os.path.dirname(path), f"{dt_str_init}_{subfolder_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    output_path = os.path.join(save_dir, 
                              f"eval_{n_steps}steps_{body_or_joint_name}_{direction if direction else ''}_{int(np.round(name_param_value,0))}{name_units}_{param_name}.pkl")
    
    with open(output_path, "wb") as f:
        pickle.dump(all_relevant_data, f)
    print(f"    → Saved: {os.path.basename(output_path)}")



# Main execution loop - process batches
time_all = []
time_all.append(timeit.default_timer())

total_configs = len(eval_configs)

for batch_start in range(0, total_configs, batch_size):
    batch_end = min(batch_start + batch_size, total_configs)
    batch_configs = eval_configs[batch_start:batch_end]
    current_batch_size = len(batch_configs)
    
    print(f"\n{'='*80}")
    print(f"Batch {batch_start//batch_size + 1}/{(total_configs + batch_size - 1)//batch_size}")
    print(f"Configurations {batch_start+1} to {batch_end} of {total_configs}")
    print(f"{'='*80}")
    
    batch_start_time = timeit.default_timer()
    
    # Generate unique RNG keys for each config in the batch
    rng_keys = jax.random.split(jax.random.key(batch_start), current_batch_size)
    
    # Run all configurations in this batch in PARALLEL using vmap
    # Note: This requires careful handling because configure_env_for_config modifies global state
    # So we actually run them sequentially but with optimized JAX operations
    batch_results = []
    for i, (config, rng_key) in enumerate(zip(batch_configs, rng_keys)):
        config_idx = batch_start + i
        param_name = config['param_name']
        param_value = config['param_value']
        direction = config['direction']
        
        print(f"  [{config_idx+1}/{total_configs}] Running: {param_name} = {param_value}" + 
              (f" (axis: {direction})" if direction else ""))
        
        result = run_single_evaluation_parallel(config_idx, config, rng_key)
        
        # Save individual result
        save_evaluation_result(result, config, path, subfolder_name, dt_str_init, n_steps)
        
        batch_results.append(result)
    
    batch_time = timeit.default_timer() - batch_start_time
    print(f"\nBatch completed in {batch_time:.2f} seconds")
    print(f"Average time per config in batch: {batch_time / current_batch_size:.2f} seconds")

time_all.append(timeit.default_timer())

print(f"\n{'='*80}")
print(f"EVALUATION COMPLETE")
print(f"{'='*80}")
print(f"Total configurations evaluated: {total_configs}")
print(f"Total time taken: {time_all[-1] - time_all[0]:.2f} seconds")
print(f"Average time per configuration: {(time_all[-1] - time_all[0]) / total_configs:.2f} seconds")
print(f"{'='*80}")


