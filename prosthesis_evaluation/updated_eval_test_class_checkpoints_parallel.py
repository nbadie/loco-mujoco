"""
Checkpoint evaluation with stacked train_states.

Loads all checkpoints and stacks their train_states, then processes them sequentially.
This is similar to updated_eval_test_class_stacked.py but structured like the seeds parallel approach.

NOTE: Due to mutable environment state, true parallelism (vmap) is not possible.
This script processes checkpoints sequentially but with clean architecture.

Usage:
    python updated_eval_test_class_checkpoints_parallel.py \
        --folder_path /path/to/checkpoints \
        --config_file config.pkl
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"

import jax 
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp

import pickle
import argparse
from datetime import datetime
import timeit
import traceback

os.environ["MUJOCO_GL"] = "egl"

from loco_mujoco.core.wrappers import VecEnv
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax, SavePPOJax
from omegaconf import OmegaConf
from loco_mujoco.evaluation.evaluation_prosthesis import ProsthesisMetricsHandler
from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction
import mujoco

# Set up argument parser
parser = argparse.ArgumentParser(description='Run parallel evaluation over multiple checkpoints.')
parser.add_argument('--folder_path', type=str, required=True, help='Path to the agent pkl files')
parser.add_argument('--config_file', type=str, required=True, help='Configuration file')
args = parser.parse_args()

folder_path = args.folder_path
config_file = args.config_file

dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"Processing checkpoints from: {folder_path}")
print(f"Sequential checkpoint evaluation with stacked architecture\n")

# ============================================================================
# Load configuration and checkpoints
# ============================================================================

agent_conf_base = SavePPOJax.load_agent_conf(config_file)
config = agent_conf_base.config

# Create environment
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

OmegaConf.set_struct(config, False)
config.experiment.env_params["headless"] = True
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"
config.experiment.env_params["add_sensors"] = True

if "reward_params" not in config.experiment.env_params or config.experiment.env_params["reward_params"] is None:
    config.experiment.env_params["reward_params"] = OmegaConf.create({})

config.experiment.env_params["horizon"] = 3000

env = factory.make(
    **config.experiment.env_params,
    **config.experiment.task_factory.params,
)
env.th.to_jax()
env = VecEnv(env)
jit_step = jax.jit(jax.vmap(env.mjx_step))
jit_reset = jax.jit(jax.vmap(env.mjx_reset))

def sample_actions_uncompiled(ts, obs, _rng):
    y, updates = agent_conf_base.network.apply({'params': ts.params,
                                                'run_stats': ts.run_stats},
                                                obs, mutable=["run_stats"])
    ts = ts.replace(run_stats=updates['run_stats'])
    pi, _ = y
    a = pi.sample(seed=_rng)
    return a, ts

sample_actions = jax.jit(sample_actions_uncompiled)

model = env.get_model()
prosthesis_metrics_handler = ProsthesisMetricsHandler(env)

# Pre-compute metadata
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

# Pre-identify muscle actuators
muscle_indices = []
for i in range(model.nu):
    if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE:
        muscle_indices.append(i)

evaluation_muscle_groups = ["back_muscles", "torso_muscles", "vasti_muscles", 
                            "rectus_muscles", "medial_muscles", "leg_muscles", "all_muscles"]
evaluation_muscle_names = {}
for n in evaluation_muscle_groups:
    evaluation_muscle_names[n] = prosthesis_metrics_handler.get_muscle_group(n)

print(f"Metadata initialized:")
print(f"  - Body names: {len(body_names)}")
print(f"  - Joint names: {len(joint_names)}")
print(f"  - Muscle actuators: {len(muscle_indices)}")
print(f"  - Total actuators: {model.nu}\n")

# ============================================================================
# Load all checkpoints
# ============================================================================

pkl_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pkl") and f.startswith("ckpt")])

if not pkl_files:
    print(f"No .pkl files found in {folder_path}")
    exit(1)

print(f"Found {len(pkl_files)} checkpoint files:")
for f in pkl_files:
    print(f"  - {f}")
print()

# Load all checkpoints
checkpoints = {}
for checkpoint_file in pkl_files:
    path = os.path.join(folder_path, checkpoint_file)
    try:
        agent_conf, agent_state = PPOJax.load_agent(path)
        checkpoints[checkpoint_file] = {
            'agent_conf': agent_conf,
            'agent_state': agent_state,
        }
        print(f"✓ Loaded: {checkpoint_file}")
    except Exception as e:
        print(f"✗ Failed to load {checkpoint_file}: {e}")
        checkpoints[checkpoint_file] = None

loaded_checkpoints = {k: v for k, v in checkpoints.items() if v is not None}
print(f"\nSuccessfully loaded {len(loaded_checkpoints)}/{len(pkl_files)} checkpoints\n")

if not loaded_checkpoints:
    print("No checkpoints loaded successfully!")
    exit(1)

checkpoint_files = list(loaded_checkpoints.keys())
checkpoint_list = [loaded_checkpoints[k] for k in checkpoint_files]

# Stack all train_states
print("Stacking train_states into batch dimension...")

train_states = [ckpt['agent_state'].train_state for ckpt in checkpoint_list]

def stack_pytrees(trees):
    """Stack multiple pytrees by stacking leaf values."""
    import numpy as np
    
    flat_trees = [jax.tree_util.tree_leaves(t) for t in trees]
    
    stacked_leaves = []
    for leaf_idx in range(len(flat_trees[0])):
        leaves_at_idx = [flat_tree[leaf_idx] for flat_tree in flat_trees]
        stacked = np.stack(leaves_at_idx, axis=0)
        stacked_leaves.append(jnp.asarray(stacked))
    
    tree_def = jax.tree_util.tree_structure(trees[0])
    stacked_tree = jax.tree_util.tree_unflatten(tree_def, stacked_leaves)
    return stacked_tree

stacked_train_state = stack_pytrees(train_states)

print(f"✓ Stacked {len(checkpoint_files)} train_states")
print(f"  Batch size: {len(checkpoint_files)}\n")

n_steps = 3000
n_envs = 1
num_checkpoints = len(checkpoint_files)

# ============================================================================
# Evaluation function for each checkpoint
# ============================================================================

def run_eval_for_checkpoint(checkpoint_idx, agent_conf):
    """
    Run evaluation for a single checkpoint.
    
    Args:
        checkpoint_idx: Index of checkpoint (0 to num_checkpoints-1)
        agent_conf: Configuration for this checkpoint
    
    Returns:
        Dictionary with all evaluation metrics for this checkpoint
    """
    
    # Extract single train_state from stacked
    train_state_single = jax.tree.map(
        lambda x: x[checkpoint_idx] if x.ndim > 0 else x,
        stacked_train_state
    )
    
    # Use the provided config for this checkpoint
    
    # Re-create sample_actions for this checkpoint
    def sample_actions_ckpt(ts, obs, _rng):
        y, updates = agent_conf.network.apply({'params': ts.params,
                                                'run_stats': ts.run_stats},
                                                obs, mutable=["run_stats"])
        ts = ts.replace(run_stats=updates['run_stats'])
        pi, _ = y
        a = pi.sample(seed=_rng)
        return a, ts
    
    sample_actions_jit = jax.jit(sample_actions_ckpt)
    
    # Initialize environment with fixed seed for reproducibility
    rng = jax.random.key(0)
    keys = jax.random.split(rng, n_envs + 1)
    rng, env_keys = keys[0], keys[1:]
    
    env_state = jit_reset(env_keys)
    obs = env_state.observation
    
    # Pre-allocate arrays
    all_foot_ground_contact_left = []
    all_foot_ground_contact_right = []
    all_grf_l = jnp.zeros((n_steps, 6))
    all_grf_r = jnp.zeros((n_steps, 6))
    body_xposes = {name: jnp.zeros((n_steps, 3)) for name in body_names}
    
    joint_data = {
        joint_name: {
            "angle": jnp.zeros((n_steps,)),
            "velocity": jnp.zeros((n_steps,)),
            "forces_constraint": jnp.zeros((n_steps,)),
            "forces_smooth": jnp.zeros((n_steps,)),
            "forces_applied": jnp.zeros((n_steps,)),
            "torque": jnp.zeros((n_steps,)),
            "energy_exp": jnp.zeros((n_steps,)),
        }
        for joint_name in joint_names
    }
    
    all_sensor_force = {}
    sensor_force_names = None
    all_actions = []
    
    run_muscle_activations = {
        f"run_{g}_activation_left": []
        for g in evaluation_muscle_groups
    }
    run_muscle_activations.update({
        f"run_{g}_activation_right": []
        for g in evaluation_muscle_groups
    })
    
    muscle_skeleton_control_activation = SkeletonMuscleControlFunction(env)
    
    total_reward = 0.0
    
    foot_name = "toes"
    calcn_name = "calcn"
    left_side = "left_side"
    right_side = "right_side"
    
    # Rollout loop
    for i in range(n_steps):
        rng, _rng = jax.random.split(rng)
        action, train_state_single = sample_actions_jit(train_state_single, obs, _rng)
        action = jnp.atleast_2d(action)
        
        env_state = jit_step(env_state, action)
        obs = env_state.observation
        reward = env_state.reward
        
        if reward.ndim > 0:
            reward = reward[0]
        total_reward += reward
        
        # Ground contact
        contact_left, contact_right = prosthesis_metrics_handler.get_contact_steps(env_state.data, i)
        all_foot_ground_contact_left.append(contact_left)
        all_foot_ground_contact_right.append(contact_right)
        
        # GRF
        grf_foot_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_l")
        grf_foot_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_r")
        grf_calcn_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{calcn_name}_l")
        grf_calcn_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{calcn_name}_r")
        
        all_grf_l = all_grf_l.at[i].set(grf_foot_l + grf_calcn_l)
        all_grf_r = all_grf_r.at[i].set(grf_foot_r + grf_calcn_r)
        
        # Body positions and velocities
        body_xpos = prosthesis_metrics_handler.get_xpos(env_state.data)
        for name in body_names:
            body_xposes[name] = body_xposes[name].at[i].set(body_xpos[name])
        
        # Joint data
        joint_angles = prosthesis_metrics_handler.get_joint_angles(env_state.data)
        joint_velocities = prosthesis_metrics_handler.get_joint_vels(env_state.data)
        joint_forces_constraint, joint_forces_smooth, joint_forces_applied = prosthesis_metrics_handler.get_joint_frces(env_state.data)
        joint_torques = prosthesis_metrics_handler.get_joint_trques(env_state.data)
        joint_energy_exp = prosthesis_metrics_handler.calc_joint_energy_exp(joint_torques, joint_velocities)
        
        for joint_name in joint_names:
            joint_data[joint_name]["angle"] = joint_data[joint_name]["angle"].at[i].set(
                jnp.squeeze(joint_angles.get(joint_name, 0.0))
            )
            joint_data[joint_name]["velocity"] = joint_data[joint_name]["velocity"].at[i].set(
                jnp.squeeze(joint_velocities.get(joint_name, 0.0))
            )
            joint_data[joint_name]["forces_constraint"] = joint_data[joint_name]["forces_constraint"].at[i].set(
                jnp.squeeze(joint_forces_constraint.get(joint_name, 0.0))
            )
            joint_data[joint_name]["forces_smooth"] = joint_data[joint_name]["forces_smooth"].at[i].set(
                jnp.squeeze(joint_forces_smooth.get(joint_name, 0.0))
            )
            joint_data[joint_name]["forces_applied"] = joint_data[joint_name]["forces_applied"].at[i].set(
                jnp.squeeze(joint_forces_applied.get(joint_name, 0.0))
            )
            joint_data[joint_name]["torque"] = joint_data[joint_name]["torque"].at[i].set(
                jnp.squeeze(joint_torques.get(joint_name, 0.0))
            )
            joint_data[joint_name]["energy_exp"] = joint_data[joint_name]["energy_exp"].at[i].set(
                jnp.squeeze(joint_energy_exp.get(joint_name, 0.0))
            )
        
        # Sensor forces
        sensor_force, sensor_force_names_iter = prosthesis_metrics_handler.get_sensor_data(env_state.data)
        if sensor_force_names is None:
            sensor_force_names = sensor_force_names_iter
        for sensor_name, force in sensor_force.items():
            if sensor_name not in all_sensor_force:
                all_sensor_force[sensor_name] = []
            all_sensor_force[sensor_name].append(force)
        
        # Actions and muscle activations
        action_processed = action.copy()
        for muscle_idx in muscle_indices:
            action_processed = action_processed.at[..., muscle_idx].set(
                muscle_skeleton_control_activation.adapted_sigmoid(action_processed[..., muscle_idx])
            )
        all_actions.append(action_processed)
        
        for muscle_group, muscle_names in evaluation_muscle_groups.items():
            left_activation = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, left_side, action_processed)
            right_activation = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, right_side, action_processed)
            
            run_muscle_activations[f"run_{muscle_group}_activation_left"].append(left_activation)
            run_muscle_activations[f"run_{muscle_group}_activation_right"].append(right_activation)
    
    # Post-process muscle activations
    sum_step_activations = {}
    step_norm_activations = {}
    sum_muscles_activations = {}
    step_muscles_norm_activations = {}
    step_num_norm_activations = {}
    
    for muscle_group in evaluation_muscle_groups:
        left_key = f"{muscle_group}_left"
        right_key = f"{muscle_group}_right"
        
        left_array = jnp.array(run_muscle_activations[left_key])
        right_array = jnp.array(run_muscle_activations[right_key])
        
        sum_step_activations[left_key] = jnp.sum(left_array, axis=0)
        sum_step_activations[right_key] = jnp.sum(right_array, axis=0)
        
        step_norm_activations[left_key] = sum_step_activations[left_key] / n_steps
        step_norm_activations[right_key] = sum_step_activations[right_key] / n_steps
        
        n_musc = len(evaluation_muscle_names[muscle_group])
        sum_muscles_activations[left_key] = jnp.sum(sum_step_activations[left_key], axis=0) if sum_step_activations[left_key].ndim > 0 else sum_step_activations[left_key]
        sum_muscles_activations[right_key] = jnp.sum(sum_step_activations[right_key], axis=0) if sum_step_activations[right_key].ndim > 0 else sum_step_activations[right_key]
        
        step_muscles_norm_activations[left_key] = sum_muscles_activations[left_key] / n_steps
        step_muscles_norm_activations[right_key] = sum_muscles_activations[right_key] / n_steps
        
        step_num_norm_activations[left_key] = jnp.sum(step_muscles_norm_activations[left_key]) / n_musc if step_muscles_norm_activations[left_key].ndim > 0 else step_muscles_norm_activations[left_key] / n_musc
        step_num_norm_activations[right_key] = jnp.sum(step_muscles_norm_activations[right_key]) / n_musc if step_muscles_norm_activations[right_key].ndim > 0 else step_muscles_norm_activations[right_key] / n_musc
    
    # Build result dictionary
    all_relevant_data = {
        "total_steps": n_steps,
        "n_envs": n_envs,
        "total_reward": total_reward,
        "avg_reward_per_step": total_reward / n_steps if n_steps > 0 else 0.0,
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
    
    # Add joint data
    for joint_name, joint_dict in joint_data.items():
        for key, value in joint_dict.items():
            all_relevant_data[f"{joint_name}_{key}"] = value
    
    # Add muscle activations
    for muscle_group in evaluation_muscle_groups:
        all_relevant_data[f"run_{muscle_group}_activation_left"] = run_muscle_activations[f"run_{muscle_group}_activation_left"]
        all_relevant_data[f"run_{muscle_group}_activation_right"] = run_muscle_activations[f"run_{muscle_group}_activation_right"]
    
    return all_relevant_data


# ============================================================================
# Run evaluation over all checkpoints sequentially
# ============================================================================

print("Starting checkpoint evaluation...\n")
overall_start = timeit.default_timer()

try:
    rewards = []
    
    # Process each checkpoint sequentially
    print(f"Processing {num_checkpoints} checkpoints...\n")
    for checkpoint_idx, checkpoint_file in enumerate(checkpoint_files):
        print(f"[{checkpoint_idx+1}/{num_checkpoints}] Evaluating: {checkpoint_file}")
        
        # Get agent config for this checkpoint
        agent_conf = checkpoint_list[checkpoint_idx]['agent_conf']
        
        # Run evaluation for this checkpoint
        start_ckpt = timeit.default_timer()
        all_relevant_data = run_eval_for_checkpoint(checkpoint_idx, agent_conf)
        end_ckpt = timeit.default_timer()
        
        reward = float(jnp.asarray(all_relevant_data["total_reward"]))
        rewards.append(reward)
        
        tail = checkpoint_file[:-4]
        output_path = os.path.join(os.path.dirname(folder_path), 
                                  f"{dt_str}_{tail}_evaluation_results_3000steps_0seed.pkl")
        
        with open(output_path, "wb") as f:
            pickle.dump(all_relevant_data, f)
        
        print(f"    ✓ Reward: {reward:10.4f} | Time: {end_ckpt - start_ckpt:.2f}s")
        print(f"    ✓ Saved to {output_path}\n")
    
    overall_end = timeit.default_timer()
    overall_elapsed = overall_end - overall_start
    
    # Summary
    print(f"\n{'='*70}")
    print(f"CHECKPOINT EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {overall_elapsed:.2f}s")
    print(f"Processed {num_checkpoints} checkpoints")
    print(f"Average time per checkpoint: {overall_elapsed/num_checkpoints:.2f}s\n")
    
    print("Results:")
    for i, checkpoint_file in enumerate(checkpoint_files):
        print(f"  {checkpoint_file:40s} | Reward: {rewards[i]:10.4f}")
    
    avg_reward = sum(rewards) / len(rewards)
    print(f"\n  Average reward: {avg_reward:.4f}")
    print(f"\nAll results saved to: {os.path.dirname(folder_path)}")

except Exception as e:
    error_msg = f"Error in checkpoint evaluation: {str(e)}\n{traceback.format_exc()}"
    print(error_msg)
    exit(1)
