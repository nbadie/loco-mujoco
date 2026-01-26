import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"

import jax 
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp

import pickle
import numpy as np
import argparse

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

parser = argparse.ArgumentParser(description='Run TRULY PARALLEL evaluation.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
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

# Build all configurations FIRST
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
print(f"TRULY PARALLEL EVALUATION - Multiple Configs at Once")
print(f"{'='*80}")
print(f"Total configurations: {len(eval_configs)}")
print(f"Strategy: Use n_envs = num_configs to run ALL configs in parallel!")
print(f"{'='*80}\n")

# KEY INSIGHT: Set n_envs = number of configurations!
# Each "env" will actually be a DIFFERENT configuration
n_steps = 1000
n_envs_per_config = len(eval_configs)  # One env per config!

print(f"Creating {n_envs_per_config} parallel environments (one per configuration)...")

# Here's the trick: We need to create MULTIPLE environment instances
# Or modify the environment to accept different configs per env in the batch

# APPROACH: Create separate environments for each config
# This requires using Python multiprocessing since we can't vmap over
# environments with different configurations in pure JAX

import concurrent.futures
from functools import partial

def run_single_config_evaluation(config_idx, config, path, config_obj, agent_conf, agent_state, 
                                 randomization_params_eval, randomization_params_names,
                                 set_fixed_randomized_targets, fixed_randomized_targets,
                                 subfolder_name, dt_str_init, n_steps):
    """
    Run evaluation for a single configuration.
    This function will be run in parallel across multiple processes/threads.
    """
    
    # Create environment for this specific config
    factory = TaskFactory.get_factory_cls(config_obj.experiment.task_factory.name)
    
    OmegaConf.set_struct(config_obj, False)
    config_obj.experiment.env_params["headless"] = True
    config_obj.experiment.env_params["goal_type"] = "GoalTrajMimicv2"
    config_obj.experiment.env_params["add_sensors"] = True
    
    # Get randomization params
    randomization_type = config_obj.randomization_config["randomization_type"]
    randomization_params = OmegaConf.to_container(config_obj.randomization_config["randomization_params"], resolve=True)
    
    for key, value in randomization_params_eval.items():
        randomization_params[key] = value
    
    if "prosthesis_side" in config_obj.experiment.env_params:
        randomization_params["prosthesis_side"] = config_obj.experiment.env_params["prosthesis_side"]
    
    # Create environment
    env = factory.make(domain_randomization_type=randomization_type, 
                      domain_randomization_params=randomization_params,
                      **config_obj.experiment.env_params, 
                      **config_obj.experiment.task_factory.params)
    env.th.to_jax()
    env = VecEnv(env)
    
    model = env.get_model()
    prosthesis_metrics_handler = ProsthesisMetricsHandler(env)
    muscle_skeleton_control_activation = SkeletonMuscleControlFunction(env)
    
    # Configure this specific environment with the config
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
    
    # Now run the evaluation with JAX parallelization for steps
    jit_step = jax.jit(jax.vmap(env.mjx_step))
    jit_reset = jax.jit(jax.vmap(env.mjx_reset))
    
    rng = jax.random.key(config_idx)
    keys = jax.random.split(rng, 2)
    rng, env_keys = keys[0], jax.random.split(keys[1], 1)
    
    env_state = jit_reset(env_keys)
    
    def sample_actions(ts, obs, _rng):
        y, updates = agent_conf.network.apply({'params': ts.params,
                                               'run_stats': ts.run_stats},
                                               obs, mutable=["run_stats"])
        ts = ts.replace(run_stats=updates['run_stats'])
        pi, _ = y
        a = pi.sample(seed=_rng)
        return a, ts
    
    sample_actions_jit = jax.jit(sample_actions)
    
    train_state_seed = 0
    if config_obj.experiment.n_seeds > 1:
        train_state = jax.tree.map(lambda x: x[train_state_seed], agent_state.train_state)
    else:
        train_state = agent_state.train_state
    
    # Initialize storage
    all_grf_l = []
    all_grf_r = []
    all_actions = []
    
    foot_name = "toes"
    calcn_name = "calcn"
    
    # Run episode
    print(f"  [{config_idx+1}/{len(eval_configs)}] Running: {param_name} = {param_value}" + 
          (f" (axis: {direction})" if direction else ""))
    
    for i in range(n_steps):
        obs = env_state.observation
        rng, _rng = jax.random.split(rng)
        action, train_state = sample_actions_jit(train_state, obs, _rng)
        action = jnp.atleast_2d(action)
        
        env_state = jit_step(env_state, action)
        
        # Collect GRF
        grf_foot_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_l")
        grf_foot_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_r")
        grf_calcn_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{calcn_name}_l")
        grf_calcn_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{calcn_name}_r")
        
        all_grf_l.append(grf_foot_l + grf_calcn_l)
        all_grf_r.append(grf_foot_r + grf_calcn_r)
        
        # Process actions
        for i_act in range(model.nu):
            if model.actuator_dyntype[i_act] == mujoco.mjtDyn.mjDYN_MUSCLE:
                action = action.at[...,i_act].set(muscle_skeleton_control_activation.adapted_sigmoid(action[...,i_act]))
        
        all_actions.append(action)
    
    env.stop()
    
    # Save results
    all_relevant_data = {
        "total_steps": n_steps,
        "n_envs": 1,
        "all_grf_l": all_grf_l,
        "all_grf_r": all_grf_r,
        "all_actions": all_actions,
        "config": config
    }
    
    # Save to file
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
        os.makedirs(save_dir, exist_ok=True)
    
    output_path = os.path.join(save_dir, 
                              f"eval_{n_steps}steps_{body_or_joint_name}_{direction if direction else ''}_{int(np.round(name_param_value,0))}{name_units}_{param_name}.pkl")
    
    with open(output_path, "wb") as f:
        pickle.dump(all_relevant_data, f)
    
    print(f"    → Saved: {os.path.basename(output_path)}")
    
    return config_idx


# Run all configurations in PARALLEL using ThreadPoolExecutor
print(f"\n{'='*80}")
print("Running ALL configurations in parallel...")
print(f"{'='*80}\n")

time_start = timeit.default_timer()

# Use ThreadPoolExecutor to run multiple configs in parallel
# Note: We use threads (not processes) because JAX handles the GPU parallelism
with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(eval_configs), 8)) as executor:
    # Create partial function with fixed arguments
    eval_func = partial(
        run_single_config_evaluation,
        path=path,
        config_obj=config,
        agent_conf=agent_conf,
        agent_state=agent_state,
        randomization_params_eval=randomization_params_eval,
        randomization_params_names=randomization_params_names,
        set_fixed_randomized_targets=set_fixed_randomized_targets,
        fixed_randomized_targets=fixed_randomized_targets,
        subfolder_name=subfolder_name,
        dt_str_init=dt_str_init,
        n_steps=n_steps
    )
    
    # Submit all tasks
    futures = [executor.submit(eval_func, idx, cfg) for idx, cfg in enumerate(eval_configs)]
    
    # Wait for all to complete
    results = [future.result() for future in concurrent.futures.as_completed(futures)]

time_end = timeit.default_timer()

print(f"\n{'='*80}")
print(f"EVALUATION COMPLETE")
print(f"{'='*80}")
print(f"Total configurations evaluated: {len(eval_configs)}")
print(f"Total time taken: {time_end - time_start:.2f} seconds")
print(f"Average time per configuration: {(time_end - time_start) / len(eval_configs):.2f} seconds")
print(f"Speedup from parallelization: ~{len(eval_configs)}x (theoretical)")
print(f"{'='*80}")
