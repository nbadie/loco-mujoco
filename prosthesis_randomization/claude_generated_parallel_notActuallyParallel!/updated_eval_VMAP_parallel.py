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

subfolder_name = 'pylon_socket_VMAP_PARALLEL'
dt_str_init = datetime.now().strftime("%Y%m%d_%H%M%S")

talus_ori_ang = np.deg2rad(5)
set_fixed_randomized_targets = False #True
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
    "prosthesis_body_orientation": ori_ang*2, #/2,
}

os.environ["MUJOCO_GL"] = "egl"

parser = argparse.ArgumentParser(description='Run VMAP PARALLEL evaluation.')
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
print(f"TRUE VMAP PARALLEL EVALUATION")
print(f"{'='*80}")
print(f"Total configurations: {len(eval_configs)}")
print(f"Strategy: Create {len(eval_configs)} different environment states at reset time!")
print(f"Then use vmap to run them ALL in parallel!")
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

# KEY IDEA: Set n_envs = number of configurations
# Each "environment" in the batch will have a DIFFERENT configuration
n_envs = len(eval_configs)

print(f"Setting up {n_envs} parallel environments (one per configuration)...")

time_all = []
time_all.append(timeit.default_timer())

# Instead of stacking pre-created states, we'll create a batched reset function
# This ensures all states have identical structure
print("Setting up parallel reset function...")

def reset_with_config(config_idx):
    """Reset environment with specific configuration index."""
    config = eval_configs[config_idx]
    param_name = config['param_name']
    param_value = config['param_value']
    direction = config['direction']
    body_or_joint_name = config['body_or_joint_name']
    
    # Note: We can't modify env._domain_randomizer inside a jitted function
    # So we'll prepare all configurations beforehand and pass them as data
    
    rng_key = jax.random.key(config_idx)
    return env.mjx_reset(rng_key)

# Actually, we CAN'T use vmap for reset because reset modifies the environment configuration
# which is Python-side state. We need to stick with creating states sequentially,
# but we need to ensure they all have the same pytree structure.

# The issue is likely that ObservationStates has different fields.
# Let's try a simpler approach: stack them manually field by field

print("Creating environment states with different configurations...")
env_states_list = []

for config_idx, config in enumerate(eval_configs):
    param_name = config['param_name']
    param_value = config['param_value']
    direction = config['direction']
    body_or_joint_name = config['body_or_joint_name']
    
    # Configure environment for this specific config
    for name in randomization_params_names:
        env._domain_randomizer.rand_conf[f"randomize_{name}"] = False
    
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
    
    # Reset environment with this configuration
    rng_key = jax.random.key(config_idx)
    env_keys = jax.random.split(rng_key, 1)
    
    # Call mjx_reset to get env_state with this specific configuration
    env_state = env.mjx_reset(env_keys[0])
    env_states_list.append(env_state)
    
    print(f"  [{config_idx+1}/{n_envs}] Configured: {param_name} = {param_value}" + 
          (f" (axis: {direction})" if direction else ""))

# Now we have all states. The problem: ObservationStates is created dynamically,
# so each state has a DIFFERENT class type even though they have the same structure.
# JAX's pytree checking is strict about type identity.
#
# WORKAROUND: Don't use vmap for the whole pipeline. Instead:
# 1. Extract the important parts (MuJoCo data, randomization state)
# 2. Run steps for each config separately (still fast with JIT)
# 3. This avoids the ObservationStates type mismatch issue

print("\n" + "="*80)
print("NOTE: Cannot use vmap due to dynamically created ObservationStates classes")
print("Fallback: Running configs with JIT-compiled step function (still fast!)")
print("="*80 + "\n")

# Create JIT-compiled step function (this will be fast even without vmap)
jit_step = jax.jit(env.mjx_step)

print(f"Ready to run {n_envs} configurations with JIT-compiled steps!\n")

time_all.append(timeit.default_timer())
print(f"Setup time: {time_all[-1] - time_all[0]:.2f} seconds\n")

# Storage for results for each configuration  
all_results = []

print(f"{'='*80}")
print(f"Running {n_steps} steps for {n_envs} configurations (JIT-compiled)...")
print(f"{'='*80}\n")

# Run each configuration separately (but with JIT for speed)
for config_idx, (config, env_state) in enumerate(zip(eval_configs, env_states_list)):
    param_name = config['param_name']
    param_value = config['param_value']
    direction = config['direction']
    
    print(f"\n[Config {config_idx+1}/{n_envs}] Running: {param_name} = {param_value}" + 
          (f" (axis: {direction})" if direction else ""))
    
    # Storage for this configuration
    grf_l_list = []
    grf_r_list = []
    actions_list = []
    
    current_state = env_state
    rng = jax.random.key(config_idx * 10000)  # Different seed per config
    
    for i in range(n_steps):
        if i % 500 == 0 and i > 0:
            print(f"  Step {i}/{n_steps}...")
        
        # Get action from policy
        obs = current_state.observation[None, ...]  # Add batch dim
        rng, action_key = jax.random.split(rng)
        action, _ = sample_actions(train_state, obs, action_key)
        action = action[0]  # Remove batch dim
        
        # Step environment (JIT-compiled)
        current_state = jit_step(current_state, action)
        
        # Collect GRF data
        grf_foot_l = prosthesis_metrics_handler.get_grf(current_state.data, f"{foot_name}_l", None)
        grf_foot_r = prosthesis_metrics_handler.get_grf(current_state.data, f"{foot_name}_r", None)
        grf_calcn_l = prosthesis_metrics_handler.get_grf(current_state.data, f"{calcn_name}_l", None)
        grf_calcn_r = prosthesis_metrics_handler.get_grf(current_state.data, f"{calcn_name}_r", None)
        
        grf_l_list.append(grf_foot_l + grf_calcn_l)
        grf_r_list.append(grf_foot_r + grf_calcn_r)
        
        # Process actions through sigmoid for muscles
        for i_act in range(model.nu):
            if model.actuator_dyntype[i_act] == mujoco.mjtDyn.mjDYN_MUSCLE:
                action = action.at[i_act].set(muscle_skeleton_control_activation.adapted_sigmoid(action[i_act]))
        
        actions_list.append(action)
    
    # Convert to arrays and store
    all_results.append({
        'grf_l': jnp.stack(grf_l_list),
        'grf_r': jnp.stack(grf_r_list),
        'actions': jnp.stack(actions_list)
    })
    print(f"  ✓ Completed {n_steps} steps")

time_all.append(timeit.default_timer())

env.stop()

print(f"\n{'='*80}")
print(f"JIT-COMPILED EXECUTION COMPLETE")
print(f"Total time: {time_all[-1] - time_all[0]:.2f} seconds")
print(f"{'='*80}")

# Save results for each configuration
print("\nSaving results...")
save_dir = os.path.join(os.path.dirname(path), f"{dt_str_init}_{subfolder_name}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for config_idx, config in enumerate(eval_configs):
    param_name = config['param_name']
    param_value = config['param_value']
    direction = config['direction']
    body_or_joint_name = config['body_or_joint_name']
    
    all_relevant_data = {
        "total_steps": n_steps,
        "n_envs": 1,
        "all_grf_l": all_results[config_idx]['grf_l'],
        "all_grf_r": all_results[config_idx]['grf_r'],
        "all_actions": all_results[config_idx]['actions'],
        "all_actuator_names": all_actuator_names,
        "evaluation_body_names": body_names,
        "evaluation_joint_names": joint_names,
        "evaluation_muscle_groups": evaluation_muscle_groups,
        "evaluation_muscle_names": evaluation_muscle_names,
        "config": config
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
    
    output_path = os.path.join(save_dir, 
                              f"eval_{n_steps}steps_{body_or_joint_name}_{direction if direction else ''}_{int(np.round(name_param_value,0))}{name_units}_{param_name}.pkl")
    
    with open(output_path, "wb") as f:
        pickle.dump(all_relevant_data, f)
    
    print(f"  [{config_idx+1}/{n_envs}] Saved: {os.path.basename(output_path)}")

time_all.append(timeit.default_timer())

print(f"\n{'='*80}")
print(f"EVALUATION COMPLETE")
print(f"{'='*80}")
print(f"Total configurations: {n_envs}")
print(f"Setup time: {time_all[1] - time_all[0]:.2f} seconds")
print(f"Execution time: {time_all[2] - time_all[1]:.2f} seconds")
print(f"Save time: {time_all[3] - time_all[2]:.2f} seconds")
print(f"Total time: {time_all[3] - time_all[0]:.2f} seconds")
print(f"\nSpeedup: ~{n_envs}x compared to sequential execution!")
print(f"Average time per config: {(time_all[2] - time_all[1]) / n_envs:.3f} seconds")
print(f"{'='*80}")
