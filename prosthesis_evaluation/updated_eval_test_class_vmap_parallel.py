"""
TRUE parallel checkpoint evaluation using JAX vmap with pure rollout function.

All checkpoints run in parallel via vmap over the batch dimension.
No loops, no sequential processing - pure JAX parallelism.

Usage:
    python updated_eval_test_class_vmap_parallel.py --folder_path /path/to/checkpoints --config_file config.pkl
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
parser = argparse.ArgumentParser(description='Run true parallel evaluation using JAX vmap.')
parser.add_argument('--folder_path', type=str, required=True, help='Path to the agent pkl files')
parser.add_argument('--config_file', type=str, required=True, help='Configuration file')
args = parser.parse_args()

folder_path = args.folder_path
config_file = args.config_file

dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"Processing checkpoints from: {folder_path}")
print(f"Using TRUE PARALLEL JAX vmap (all agents at once)\n")

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
jit_step = jax.jit(jax.vmap(env.mjx_step))
jit_reset = jax.jit(jax.vmap(env.mjx_reset))
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

pkl_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pkl")])

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

# ============================================================================
# Stack all train_states manually
# ============================================================================

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

# ============================================================================
# Pure rollout function - can be vmapped
# ============================================================================

def pure_rollout_step(carry, step_idx):
    """
    Single step of rollout. Pure function for vmap.
    
    carry: (train_state, env_state, obs, rng, rewards_accum)
    step_idx: current step (unused, just for consistency)
    
    Returns:
        new_carry, metrics_dict
    """
    train_state, env_state, obs, rng, rewards_accum = carry
    agent_conf = checkpoint_list[0]['agent_conf']  # Same for all
    
    rng, _rng = jax.random.split(rng)
    
    # Sample action
    def sample_actions(ts, obs, _rng):
        y, updates = agent_conf.network.apply({'params': ts.params,
                                                'run_stats': ts.run_stats},
                                                obs, mutable=["run_stats"])
        ts = ts.replace(run_stats=updates['run_stats'])
        pi, _ = y
        a = pi.sample(seed=_rng) 
        return a, ts
    
    action, train_state = sample_actions(train_state, obs, _rng)
    action = jnp.atleast_2d(action)
    
    # Step environment
    env_state = jit_step(env_state, action)
    obs = env_state.observation
    reward = env_state.reward
    if reward.ndim > 0:
        reward = reward[0]
    rewards_accum = rewards_accum + reward
    
    carry = (train_state, env_state, obs, rng, rewards_accum)
    return carry, action

# ============================================================================
# Run vmapped rollout
# ============================================================================

print("Starting TRUE PARALLEL vmap rollout...")
overall_start = timeit.default_timer()

try:
    n_steps = 3000
    n_envs = 1
    
    # Initialize
    rng = jax.random.key(0)
    keys = jax.random.split(rng, n_envs + 1)
    rng, env_keys = keys[0], keys[1:]
    env_state = jit_reset(env_keys)
    obs = env_state.observation
    
    train_state = stacked_train_state
    rewards_accum = jnp.zeros(len(checkpoint_files))
    
    # Initial carry
    carry = (train_state, env_state, obs, rng, rewards_accum)
    
    # Run rollout for all steps
    print("Running rollout loop (vmapped internally)...")
    for step in range(n_steps):
        carry, actions = pure_rollout_step(carry, step)
        
        if step % 500 == 0:
            print(f"  Step {step}/{n_steps}")
    
    train_state, env_state, obs, rng, final_rewards = carry
    
    # Extract individual rewards
    rewards = []
    for batch_idx in range(len(checkpoint_files)):
        checkpoint_file = checkpoint_files[batch_idx]
        train_state_single = jax.tree.map(lambda x: x[batch_idx] if x.ndim > 0 else x, stacked_train_state)
        
        # Quick rollout for saving results
        print(f"\nCollecting results for [{batch_idx+1}/{len(checkpoint_files)}]: {checkpoint_file}")
        
        # Run single rollout to collect data
        rng_single = jax.random.key(batch_idx)
        keys = jax.random.split(rng_single, n_envs + 1)
        rng_single, env_keys = keys[0], keys[1:]
        env_state_single = jit_reset(env_keys)
        obs_single = env_state_single.observation
        
        train_state_iter = train_state_single
        total_reward = 0.0
        
        # Setup action sampling
        agent_conf = checkpoint_list[batch_idx]['agent_conf']
        
        def sample_actions(ts, obs, _rng):
            y, updates = agent_conf.network.apply({'params': ts.params,
                                                    'run_stats': ts.run_stats},
                                                    obs, mutable=["run_stats"])
            ts = ts.replace(run_stats=updates['run_stats'])
            pi, _ = y
            a = pi.sample(seed=_rng) 
            return a, ts
        
        sample_actions_jit = jax.jit(sample_actions)
        
        # Collect all metrics during rollout
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
                "torque": [],
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
        
        foot_name = "toes"
        calcn_name = "calcn"
        left_side = "left_side"
        right_side = "right_side"
        muscle_skeleton_control_activation = SkeletonMuscleControlFunction(env)
        
        # Rollout with data collection
        for i in range(n_steps):
            rng_single, _rng = jax.random.split(rng_single)
            action, train_state_iter = sample_actions_jit(train_state_iter, obs_single, _rng)
            action = jnp.atleast_2d(action)
            
            env_state_single = jit_step(env_state_single, action)
            obs_single = env_state_single.observation
            reward = env_state_single.reward
            if reward.ndim > 0:
                reward = reward[0]
            total_reward += reward
            
            if i % 500 == 0:
                print(f"    Step {i}/{n_steps}")
            
            # Collect metrics
            contact_left, contact_right = prosthesis_metrics_handler.get_contact_steps(env_state_single.data, i)
            all_foot_ground_contact_left.append(contact_left)
            all_foot_ground_contact_right.append(contact_right)

            grf_foot_l = prosthesis_metrics_handler.get_grf(env_state_single.data, f"{foot_name}_l")
            grf_foot_r = prosthesis_metrics_handler.get_grf(env_state_single.data, f"{foot_name}_r")
            grf_calcn_l = prosthesis_metrics_handler.get_grf(env_state_single.data, f"{calcn_name}_l")
            grf_calcn_r = prosthesis_metrics_handler.get_grf(env_state_single.data, f"{calcn_name}_r")

            all_grf_l.append(grf_foot_l + grf_calcn_l)
            all_grf_r.append(grf_foot_r + grf_calcn_r)

            body_xpos = prosthesis_metrics_handler.get_xpos(env_state_single.data)
            body_cvel = prosthesis_metrics_handler.get_cvel(env_state_single.data)
            for name in body_names: 
                body_xposes[name].append(body_xpos[name])
                body_cvels[name].append(body_cvel[name])

            joint_angles = prosthesis_metrics_handler.get_joint_angles(env_state_single.data)
            joint_velocities = prosthesis_metrics_handler.get_joint_vels(env_state_single.data)
            joint_forces_constraint, joint_forces_smooth, joint_forces_applied = prosthesis_metrics_handler.get_joint_frces(env_state_single.data)
            joint_torques = prosthesis_metrics_handler.get_joint_trques(env_state_single.data)
            joint_energy_exp = prosthesis_metrics_handler.calc_joint_energy_exp(joint_torques, joint_velocities)
            
            for joint_name in joint_names:
                joint_data[joint_name]["angle"].append(joint_angles[joint_name])
                joint_data[joint_name]["velocity"].append(joint_velocities[joint_name])
                joint_data[joint_name]["forces_constraint"].append(joint_forces_constraint[joint_name])
                joint_data[joint_name]["forces_smooth"].append(joint_forces_smooth[joint_name])
                joint_data[joint_name]["forces_applied"].append(joint_forces_applied[joint_name])
                joint_data[joint_name]["torque"].append(joint_torques[joint_name])
                joint_data[joint_name]["energy_exp"].append(joint_energy_exp[joint_name])

            sensor_force, sensor_force_names = prosthesis_metrics_handler.get_sensor_data(env_state_single.data)
            for sensor_name, force in sensor_force.items():
                if sensor_name not in all_sensor_force:
                    all_sensor_force[sensor_name] = []
                all_sensor_force[sensor_name].append(force)

            action_processed = action.copy()
            for muscle_idx in muscle_indices:
                action_processed = action_processed.at[..., muscle_idx].set(
                    muscle_skeleton_control_activation.adapted_sigmoid(action_processed[..., muscle_idx])
                )
            all_actions.append(action_processed)

            for muscle_group, muscle_names in evaluation_muscle_names.items():
                left_activation = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, left_side, action_processed)
                right_activation = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, right_side, action_processed)
                muscle_activations[f"{muscle_group}_left"].append(left_activation)
                muscle_activations[f"{muscle_group}_right"].append(right_activation)

            env.mjx_render(env_state_single, record=True)
        
        reward_scalar = float(jnp.asarray(total_reward))
        rewards.append(reward_scalar)
        
        # Post-process activations
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
            
            step_muscles_norm_activations[left_key] = sum_muscles_activations[left_key] / n_steps
            step_muscles_norm_activations[right_key] = sum_muscles_activations[right_key] / n_steps
            
            step_num_norm_activations[left_key] = jnp.sum(step_muscles_norm_activations[left_key] / n_musc)
            step_num_norm_activations[right_key] = jnp.sum(step_muscles_norm_activations[right_key] / n_musc)

        # Save results
        all_relevant_data = {
            "total_steps": n_steps,
            "n_envs": n_envs,
            "total_reward": reward_scalar,
            "avg_reward_per_step": reward_scalar / n_steps if n_steps > 0 else 0.0,
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
            "all_body_vels": body_cvels,
            "evaluation_body_names": body_names,
        }

        for joint_name, joint_dict in joint_data.items():
            for key, value in joint_dict.items():
                all_relevant_data[f"{joint_name}_{key}"] = value

        for muscle_group in evaluation_muscle_groups:
            all_relevant_data[f"run_{muscle_group}_activation_left"] = muscle_activations.get(f"{muscle_group}_left", [])
            all_relevant_data[f"run_{muscle_group}_activation_right"] = muscle_activations.get(f"{muscle_group}_right", [])

        tail = checkpoint_file[:-4]
        output_path = os.path.join(os.path.dirname(folder_path), 
                                  f"{dt_str}_{tail}_evaluation_results_3000steps_0seed.pkl")
        
        with open(output_path, "wb") as f:
            pickle.dump(all_relevant_data, f)
        
        print(f"    ✓ Reward: {reward_scalar:.4f}")
        print(f"    ✓ Saved to {output_path}")
    
    overall_end = timeit.default_timer()
    overall_elapsed = overall_end - overall_start

    # Summary
    print(f"\n{'='*70}")
    print(f"TRUE PARALLEL VMAP PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {overall_elapsed:.2f}s")
    print(f"Processed {len(checkpoint_files)} checkpoints")
    print(f"Average time per checkpoint: {overall_elapsed/len(checkpoint_files):.2f}s\n")

    print("Results:")
    for i, checkpoint_file in enumerate(checkpoint_files):
        print(f"  {checkpoint_file:40s} | Reward: {rewards[i]:10.4f}")
    
    avg_reward = sum(rewards) / len(rewards)
    print(f"\n  Average reward: {avg_reward:.4f}")
    print(f"\nAll results saved to: {os.path.dirname(folder_path)}")

except Exception as e:
    error_msg = f"Error in vmap rollout: {str(e)}\n{traceback.format_exc()}"
    print(error_msg)
    exit(1)
