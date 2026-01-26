"""
Checkpoint evaluation - parallel evaluation of multiple checkpoints with multiple seeds.

Loads all checkpoints, stacks their train_states, then evaluates each checkpoint
with 10 different seeds in parallel using JAX vmap. This provides comprehensive 
evaluation metrics across multiple stochastic rollouts.

PARALLELIZATION STRATEGY:
- Vmap over checkpoints (first dimension)
- Vmap over seeds (second dimension)
- All checkpoint-seed combinations run in parallel
- Results organized in checkpoint-specific folders

For N checkpoints with M seeds: ~30s total time (not N*M*30s)
Results saved to: checkpoint_folder/ckpt_N/ with seed_0.pkl, seed_1.pkl, etc.

Usage:
    python updated_eval_test_class_stacked.py --folder_path /path/to/checkpoints --config_file config.pkl --num_seeds 10
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
parser = argparse.ArgumentParser(description='Run parallel evaluation for multiple checkpoints with multiple seeds.')
parser.add_argument('--folder_path', type=str, required=True, help='Path to the agent pkl files')
parser.add_argument('--config_file', type=str, required=True, help='Configuration file')
parser.add_argument('--num_seeds', type=int, default=10, help='Number of seeds per checkpoint')
args = parser.parse_args()

folder_path = args.folder_path
config_file = args.config_file
num_seeds = args.num_seeds

dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"Processing checkpoints from: {folder_path}")
print(f"Parallel evaluation: {num_seeds} seeds per checkpoint")
print(f"All checkpoint-seed combinations will run in parallel\n")
overall_initial_time = timeit.default_timer()

# ============================================================================
# Load configuration
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
# Don't vmap yet - we'll vmap over both environments and checkpoints
mjx_step_base = env.mjx_step
mjx_reset_base = env.mjx_reset
# jit_step_env = jax.jit(jax.vmap(mjx_step_base))
# jit_reset_env = jax.jit(jax.vmap(mjx_reset_base))
model = env.get_model()

muscle_skeleton_control_activation = SkeletonMuscleControlFunction(env)

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
    print(f"No checkpoint .pkl files found in {folder_path}")
    
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

# ============================================================================
# Stack all train_states manually (avoid tree_map comparison issues)
# ============================================================================

print("Stacking train_states into batch dimensions...")
print(f"  Checkpoints: {len(checkpoint_files)}")
print(f"  Seeds per checkpoint: {num_seeds}")
print(f"  Total batches: {len(checkpoint_files) * num_seeds}\n")

# Extract all train_states (replicate each checkpoint num_seeds times)
train_states = []
checkpoint_files_expanded = []
for ckpt_file in checkpoint_files:
    for seed_idx in range(num_seeds):
        train_states.append(loaded_checkpoints[ckpt_file]['agent_state'].train_state)
        checkpoint_files_expanded.append((ckpt_file, seed_idx))

# Manually stack using numpy's stack (convert to numpy, then back to jax)
def stack_pytrees(trees):
    """Stack multiple pytrees by stacking leaf values."""
    import numpy as np
    
    # Flatten all trees
    flat_trees = [jax.tree_util.tree_leaves(t) for t in trees]
    
    # Stack corresponding leaves
    stacked_leaves = []
    for leaf_idx in range(len(flat_trees[0])):
        leaves_at_idx = [flat_tree[leaf_idx] for flat_tree in flat_trees]
        # Use numpy stack to avoid jnp issues
        stacked = np.stack(leaves_at_idx, axis=0)
        stacked_leaves.append(jnp.asarray(stacked))
    
    # Unflatten back to tree structure
    tree_def = jax.tree_util.tree_structure(trees[0])
    stacked_tree = jax.tree_util.tree_unflatten(tree_def, stacked_leaves)
    return stacked_tree

stacked_train_state = stack_pytrees(train_states)

print(f"✓ Stacked {len(train_states)} train_states (checkpoints × seeds)")
print(f"  Batch size: {len(train_states)}\n")

# ============================================================================
# Vectorized rollout function using vmap
# ============================================================================

# ============================================================================
# Vectorized rollout - process all checkpoints with stacked state
# ============================================================================

def rollout_with_stacked_state(stacked_train_state, agent_conf, checkpoint_files_expanded, num_seeds, n_steps=3000, n_envs=1):
    """
    Process all checkpoints and seeds in parallel using vmap.
    
    Args:
        stacked_train_state: train_states stacked as (num_checkpoints * num_seeds,) 
        agent_conf: agent configuration
        checkpoint_files_expanded: list of (checkpoint_file, seed_idx) tuples
        num_seeds: number of seeds per checkpoint
        n_steps: rollout length
        n_envs: number of environments per batch
    """
    num_batches = len(checkpoint_files_expanded)  # checkpoints * seeds
    num_checkpoints = num_batches // num_seeds
    all_results = []
    
    # Initialize environment for ONE instance
    rng = jax.random.key(0)
    keys = jax.random.split(rng, n_envs + 1)
    rng, env_keys = keys[0], keys[1:]
    env_state_single = mjx_reset_base(env_keys[0])  # Single environment reset
    obs_single = env_state_single.observation
    
    # Replicate env_state for all checkpoint-seed combinations
    def replicate_for_batches(x):
        """Replicate a leaf value for all batches."""
        x = jnp.asarray(x)
        if x.ndim == 0:  # scalar
            return jnp.repeat(x, num_batches)
        else:  # array
            return jnp.repeat(x[jnp.newaxis, ...], num_batches, axis=0)
    
    env_state = jax.tree.map(replicate_for_batches, env_state_single)
    obs = jax.tree.map(replicate_for_batches, obs_single)
    
    train_state = stacked_train_state
    total_reward = jnp.zeros(num_batches)
    
    # Create vmapped action sampling function
    def sample_actions_single(ts, obs_single, _rng):
        """Sample actions for a single checkpoint."""
        y, updates = agent_conf.network.apply({'params': ts.params,
                                                'run_stats': ts.run_stats},
                                                obs_single, mutable=["run_stats"])
        ts = ts.replace(run_stats=updates['run_stats'])
        pi, _ = y
        a = pi.sample(seed=_rng) 
        return a, ts
    
    # Vmap over checkpoints for action sampling
    sample_actions_vmapped = jax.vmap(sample_actions_single, 
                                       in_axes=(0, 0, 0),  # vmap over all dimensions
                                       out_axes=(0, 0))
    sample_actions_jit = jax.jit(sample_actions_vmapped)
    
    # Vmap step over checkpoints (each checkpoint has its own env_state)
    step_jit = jax.jit(jax.vmap(mjx_step_base))
    
    # Initialize data collection containers
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
            "torques": [],
            "energy_exp": [],
        }
    
    all_sensor_force = {}
    
    muscle_activations = {}
    for muscle_group in evaluation_muscle_groups:
        muscle_activations[f"{muscle_group}_left"] = []
        muscle_activations[f"{muscle_group}_right"] = []

    all_grf_l = []
    all_grf_r = []
    all_actions = []
    
    # Rollout loop
    step_count = 0
    foot_name = "toes"
    calcn_name = "calcn"
    left_side = "left_side"
    right_side = "right_side"
    
    for i in range(n_steps):
        rng, _rng = jax.random.split(rng)
        # Generate independent RNG for each checkpoint-seed combination
        rngs_batches = jax.random.split(_rng, num_batches)
        action, train_state = sample_actions_jit(train_state, obs, rngs_batches)
        
        env_state = step_jit(env_state, action)
        obs = env_state.observation
        
        # Reward is shape (num_checkpoints,) from vmapped step
        reward = env_state.reward
        total_reward = total_reward + reward
        step_count += 1
        
        if step_count % 500 == 0:
            print(f"  Step {step_count}/{n_steps}")
        
        # # Collect all metrics
        # contact_left, contact_right = prosthesis_metrics_handler.get_contact_steps_batched(env_state.data, i)
        # all_foot_ground_contact_left.append(contact_left)
        # all_foot_ground_contact_right.append(contact_right)

        # GRF collection
        grf_foot_l = prosthesis_metrics_handler.get_grf_batched(env_state.data, f"{foot_name}_l")
        grf_foot_r = prosthesis_metrics_handler.get_grf_batched(env_state.data, f"{foot_name}_r")
        grf_calcn_l = prosthesis_metrics_handler.get_grf_batched(env_state.data, f"{calcn_name}_l")
        grf_calcn_r = prosthesis_metrics_handler.get_grf_batched(env_state.data, f"{calcn_name}_r")

        all_grf_l.append(grf_foot_l + grf_calcn_l)
        all_grf_r.append(grf_foot_r + grf_calcn_r)

        # Body positions and velocities
        body_xpos = prosthesis_metrics_handler.get_xpos_batched(env_state.data)
        body_cvel = prosthesis_metrics_handler.get_cvel_batched(env_state.data)
        for name in body_names: 
            body_xposes[name].append(body_xpos[name])
            body_cvels[name].append(body_cvel[name])

        # Joint data
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
            joint_data[joint_name]["torques"].append(joint_torques[joint_name])
            joint_data[joint_name]["energy_exp"].append(joint_energy_exp[joint_name])

        # Sensor data
        # sensor_force, sensor_force_names = prosthesis_metrics_handler.get_sensor_data_batched(env_state.data)
        # for sensor_name, force in sensor_force.items():
        #     if sensor_name not in all_sensor_force:
        #         all_sensor_force[sensor_name] = []
        #     all_sensor_force[sensor_name].append(force)

        sensor_force, sensor_force_names = prosthesis_metrics_handler.get_sensor_data_batched_2(env_state.data)
        for sensor_name, force in sensor_force.items():
            if sensor_name not in all_sensor_force:
                all_sensor_force[sensor_name] = []
            all_sensor_force[sensor_name].append(force)

        # Action processing - apply sigmoid to muscles
        action_processed = action.copy()
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

        # env.mjx_render(env_state, record=True)
    
    # Convert rewards (shape: num_batches) to scalars
    reward_scalars = [float(jnp.asarray(total_reward[i])) for i in range(num_batches)]
    
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
        
        sum_step_activations[left_key] = jnp.sum(left_array, axis=0)
        sum_step_activations[right_key] = jnp.sum(right_array, axis=0)

        step_norm_activations[left_key] = sum_step_activations[left_key] / n_steps
        step_norm_activations[right_key] = sum_step_activations[right_key] / n_steps

        n_musc = len(evaluation_muscle_names[muscle_group])
        sum_muscles_activations[left_key] = jnp.sum(sum_step_activations[left_key], axis=0)
        sum_muscles_activations[right_key] = jnp.sum(sum_step_activations[right_key], axis=0)
        sum_muscles_activations[f"{muscle_group}_both"] = sum_muscles_activations[left_key] + sum_muscles_activations[right_key]
        
        step_muscles_norm_activations[left_key] = sum_muscles_activations[left_key] / n_steps
        step_muscles_norm_activations[right_key] = sum_muscles_activations[right_key] / n_steps
        step_muscles_norm_activations[f"{muscle_group}_both"] = sum_muscles_activations[f"{muscle_group}_both"] / n_steps

        step_num_norm_activations[left_key] = step_muscles_norm_activations[left_key] / n_musc
        step_num_norm_activations[right_key] = step_muscles_norm_activations[right_key] / n_musc
        step_num_norm_activations[f"{muscle_group}_both"] = step_muscles_norm_activations[f"{muscle_group}_both"] / (2 * n_musc)

    # Helper function to extract batch-specific data from nested structures
    def extract_batch_data(data, batch_idx):
        """
        Extract the batch_idx slice from all arrays in nested structures.
        For body_xposes: {body: [step0_array(num_batches, 3), step1_array(num_batches, 3), ...]}
        Result: {body: [step0_array(3,), step1_array(3,), ...]}  - all steps for one batch
        """
        if isinstance(data, dict):
            return {k: extract_batch_data(v, batch_idx) for k, v in data.items()}
        elif isinstance(data, list) and len(data) > 0:
            result = []
            for step_array in data:
                if isinstance(step_array, (jnp.ndarray, list)):
                    try:
                        result.append(step_array[batch_idx])
                    except (IndexError, TypeError):
                        result.append(step_array)
                else:
                    result.append(step_array)
            return result
        elif isinstance(data, (jnp.ndarray, list)) and hasattr(data, '__getitem__'):
            try:
                return data[batch_idx]
            except (IndexError, TypeError):
                return data
        else:
            return data
    
    # Save results for each checkpoint-seed combination
    for batch_idx in range(num_batches):
        checkpoint_file, seed_idx = checkpoint_files_expanded[batch_idx]
        checkpoint_name = checkpoint_file[:-4]  # Remove .pkl extension
        
        # Create checkpoint-specific folder
        checkpoint_folder = os.path.join(os.path.dirname(folder_path), 
                                        f"{dt_str}_{checkpoint_name}")
        os.makedirs(checkpoint_folder, exist_ok=True)
        
        print(f"Processing [{batch_idx+1}/{num_batches}]: {checkpoint_file} (seed {seed_idx})")
        
        # Extract all per-batch data
        all_relevant_data = {
            "checkpoint": checkpoint_file,
            "seed": seed_idx,
            "total_steps": step_count,
            "n_envs": n_envs,
            "total_reward": reward_scalars[batch_idx],
            "avg_reward_per_step": reward_scalars[batch_idx] / step_count if step_count > 0 else 0.0,
            "all_grf_l": extract_batch_data(all_grf_l, batch_idx),
            "all_grf_r": extract_batch_data(all_grf_r, batch_idx),
            "all_sensor_force": extract_batch_data(all_sensor_force, batch_idx),
            "sensor_force_names": sensor_force_names,
            "evaluation_muscle_groups": evaluation_muscle_groups,
            "evaluation_joint_names": joint_names,
            "evaluation_muscle_names": evaluation_muscle_names,
            "all_actions": extract_batch_data(all_actions, batch_idx),
            "all_actuator_names": all_actuator_names,
            "all_body_poses": extract_batch_data(body_xposes, batch_idx),
            "all_body_vels": extract_batch_data(body_cvels, batch_idx),
            "evaluation_body_names": body_names,
        }

        # Add muscle activation summaries for this batch
        all_relevant_data["sum_step_activations"] = {k: v[batch_idx] for k, v in sum_step_activations.items()}
        all_relevant_data["step_norm_activations"] = {k: v[batch_idx] for k, v in step_norm_activations.items()}
        all_relevant_data["sum_muscles_activations"] = {k: v[batch_idx] for k, v in sum_muscles_activations.items()}
        all_relevant_data["step_muscles_norm_activations"] = {k: v[batch_idx] for k, v in step_muscles_norm_activations.items()}
        all_relevant_data["step_num_norm_activations"] = {k: v[batch_idx] for k, v in step_num_norm_activations.items()}
        
        # Add joint data for this batch
        for joint_name in joint_names:
            for key in joint_data[joint_name].keys():
                all_relevant_data[f"{joint_name}_{key}"] = extract_batch_data(joint_data[joint_name][key], batch_idx)

        # Add muscle activations for this batch
        for muscle_group in evaluation_muscle_groups:
            all_relevant_data[f"run_{muscle_group}_activation_left"] = extract_batch_data(muscle_activations.get(f"{muscle_group}_left", []), batch_idx)
            all_relevant_data[f"run_{muscle_group}_activation_right"] = extract_batch_data(muscle_activations.get(f"{muscle_group}_right", []), batch_idx)

        # Save results for this batch in checkpoint-specific folder
        output_path = os.path.join(checkpoint_folder, f"seed_{seed_idx}_evaluation_results_{n_steps}steps.pkl")
        
        with open(output_path, "wb") as f:
            pickle.dump(all_relevant_data, f)
        
        print(f"  ✓ Reward: {reward_scalars[batch_idx]:.4f}")
        print(f"  ✓ Saved to {output_path}\n")
        
        all_results.append({
            "checkpoint": checkpoint_file,
            "seed": seed_idx,
            "reward": reward_scalars[batch_idx],
            "output_path": output_path,
        })

    return all_results

# ============================================================================
# Run vmapped rollout
# ============================================================================

print("Starting parallel rollout with stacked states...")
overall_start = timeit.default_timer()

print('Initial-Start Time:', overall_start - overall_initial_time)

try:
    # Run all checkpoints and seeds in parallel
    all_results = rollout_with_stacked_state(
        stacked_train_state,
        checkpoint_list[0]['agent_conf'],  # Same config for all
        checkpoint_files_expanded,
        num_seeds,
        n_steps=1500,
        n_envs=1,
    )
    
    # Extract rewards for summary
    rewards = jnp.array([r["reward"] for r in all_results])
    
    overall_end = timeit.default_timer()
    overall_elapsed = overall_end - overall_start
    
    print(f"\n{'='*70}")
    print(f"PARALLEL CHECKPOINT-SEED EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {overall_elapsed:.2f}s")
    print(f"Total Total Time: {overall_end - overall_initial_time:.2f}s")
    print(f"Processed {len(checkpoint_files)} checkpoints × {num_seeds} seeds = {len(checkpoint_files) * num_seeds} total batches")
    print(f"All combinations run in parallel\n")
    
    # Print results by checkpoint
    print("Results by Checkpoint:")
    checkpoints_unique = sorted(set(r['checkpoint'] for r in all_results))
    for ckpt in checkpoints_unique:
        ckpt_results = [r for r in all_results if r['checkpoint'] == ckpt]
        ckpt_rewards = [r['reward'] for r in ckpt_results]
        ckpt_mean = float(jnp.mean(jnp.array(ckpt_rewards)))
        ckpt_std = float(jnp.std(jnp.array(ckpt_rewards)))
        print(f"  {ckpt:40s} | Mean: {ckpt_mean:10.4f} ± {ckpt_std:8.4f}")
        for seed_result in ckpt_results:
            print(f"    Seed {seed_result['seed']}: {seed_result['reward']:.4f}")
    
    avg_reward = float(jnp.mean(rewards))
    print(f"\n  Overall average reward: {avg_reward:.4f}")
    print(f"\nAll results organized in checkpoint-specific folders under: {os.path.dirname(folder_path)}")
    
except Exception as e:
    error_msg = f"Error in parallel stacked state rollout: {str(e)}\n{traceback.format_exc()}"
    print(error_msg)
    exit(1)
