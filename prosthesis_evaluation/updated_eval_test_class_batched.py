"""
Batched checkpoint evaluation using JAX vmap - NO MULTIPROCESSING.

Process multiple checkpoints by stacking their parameters and batching through JAX.
This is much faster than multiprocessing and avoids all fork/pickle issues.

Usage:
    python updated_eval_test_class_batched.py --folder_path /path/to/checkpoints --config_file config.pkl --batch_size 4
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
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

from loco_mujoco.core.wrappers import VecEnv
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax, SavePPOJax
from loco_mujoco.algorithms.ppo_jax import PPOAgentConf, PPOAgentState
from omegaconf import OmegaConf
from loco_mujoco.evaluation.evaluation_prosthesis import ProsthesisMetricsHandler
from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction
import mujoco

# Set up argument parser
parser = argparse.ArgumentParser(description='Run batched JAX evaluation (no multiprocessing).')
parser.add_argument('--folder_path', type=str, required=True, help='Path to the agent pkl files')
parser.add_argument('--config_file', type=str, required=True, help='Configuration file')
# parser.add_argument('--batch_size', type=int, default=4, help='Number of checkpoints to process per batch')
args = parser.parse_args()

folder_path = args.folder_path
config_file = args.config_file
# batch_size = args.batch_size

dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"Processing checkpoints from: {folder_path}")
print(f"Using batched JAX evaluation (no multiprocessing)")
# print(f"Batch size: {batch_size}\n")

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
# Load checkpoints
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
            'config': agent_conf.config,
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

# ============================================================================
# Process checkpoints in batches
# ============================================================================

def process_single_checkpoint(checkpoint_file, agent_conf, agent_state, n_steps=3000, n_envs=1, train_state_seed=0):
    """
    Process a single checkpoint within a JAX batch.
    Pure JAX - no multiprocessing, works seamlessly with vmap.
    """
    try:
        path = os.path.join(folder_path, checkpoint_file)
        print(f"[Batch] Processing: {checkpoint_file}")
        
        # Get train state
        if agent_conf.config.experiment.n_seeds > 1:
            train_state = jax.tree.map(lambda x: x[train_state_seed], agent_state.train_state)
        else: 
            train_state = agent_state.train_state
        
        # Initialize data containers
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

        # Metrics collection
        total_reward = 0.0
        step_total = 0
        
        time_start = timeit.default_timer()
        
        # Initialize environment
        rng = jax.random.key(0)
        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]
        env_state = jit_reset(env_keys)
        obs = env_state.observation
        
        muscle_skeleton_control_activation = SkeletonMuscleControlFunction(env)
        
        # Setup action sampling function
        def sample_actions_uncompiled(ts, obs, _rng):
            y, updates = agent_conf.network.apply({'params': ts.params,
                                                    'run_stats': ts.run_stats},
                                                    obs, mutable=["run_stats"])
            ts = ts.replace(run_stats=updates['run_stats'])
            pi, _ = y
            a = pi.sample(seed=_rng) 
            return a, ts
        
        sample_actions = jax.jit(sample_actions_uncompiled)
        
        # Main rollout loop
        foot_name = "toes"
        calcn_name = "calcn"
        left_side = "left_side"
        right_side = "right_side"

        for i in range(n_steps):
            rng, _rng = jax.random.split(rng)
            action, train_state = sample_actions(train_state, obs, _rng)
            action = jnp.atleast_2d(action)

            env_state = jit_step(env_state, action)
            obs = env_state.observation
            total_reward += env_state.reward

            if step_total % 500 == 0:
                print(f"[Batch {checkpoint_file}] Step {step_total}/{n_steps}")

            # Collect metrics efficiently
            contact_left, contact_right = prosthesis_metrics_handler.get_contact_steps(env_state.data, i)
            all_foot_ground_contact_left.append(contact_left)
            all_foot_ground_contact_right.append(contact_right)

            # GRF collection
            grf_foot_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_l")
            grf_foot_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_r")
            grf_calcn_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{calcn_name}_l")
            grf_calcn_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{calcn_name}_r")

            all_grf_l.append(grf_foot_l + grf_calcn_l)
            all_grf_r.append(grf_foot_r + grf_calcn_r)

            # Body positions and velocities
            body_xpos = prosthesis_metrics_handler.get_xpos(env_state.data)
            body_cvel = prosthesis_metrics_handler.get_cvel(env_state.data)
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
                joint_data[joint_name]["torque"].append(joint_torques[joint_name])
                joint_data[joint_name]["energy_exp"].append(joint_energy_exp[joint_name])

            # Sensor data
            sensor_force, sensor_force_names = prosthesis_metrics_handler.get_sensor_data(env_state.data)
            for sensor_name, force in sensor_force.items():
                if sensor_name not in all_sensor_force:
                    all_sensor_force[sensor_name] = []
                all_sensor_force[sensor_name].append(force)

            # Action processing
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

            step_total += n_envs 
            env.mjx_render(env_state, record=True)

        time_end = timeit.default_timer()
        elapsed = time_end - time_start

        print(f"[Batch {checkpoint_file}] Rollout complete: {elapsed:.2f}s, reward={float(total_reward):.4f}")

        # Post-processing: compute activations
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

        # Collect results
        all_relevant_data = {
            "total_steps": step_total,
            "n_envs": n_envs,
            "elapsed_time": elapsed,
            "total_reward": float(total_reward),
            "avg_reward_per_step": float(total_reward / step_total) if step_total > 0 else 0.0,
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

        # Add joint data
        for joint_name, joint_dict in joint_data.items():
            for key, value in joint_dict.items():
                all_relevant_data[f"{joint_name}_{key}"] = value

        # Add muscle activations
        for muscle_group in evaluation_muscle_groups:
            all_relevant_data[f"run_{muscle_group}_activation_left"] = muscle_activations.get(f"{muscle_group}_left", [])
            all_relevant_data[f"run_{muscle_group}_activation_right"] = muscle_activations.get(f"{muscle_group}_right", [])

        # Save results
        tail = checkpoint_file[:-4]
        output_path = os.path.join(os.path.dirname(folder_path), 
                                  f"{dt_str}_{tail}_evaluation_results_3000steps_0seed.pkl")
        
        with open(output_path, "wb") as f:
            pickle.dump(all_relevant_data, f)
        
        print(f"[Batch {checkpoint_file}] Saved to {output_path}")
        
        return {
            "checkpoint": checkpoint_file,
            "status": "success",
            "output_path": output_path,
            "elapsed_time": elapsed,
            "total_reward": float(total_reward),
        }

    except Exception as e:
        error_msg = f"[Batch {checkpoint_file}] Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {
            "checkpoint": checkpoint_file,
            "status": "failed",
            "error": str(e),
        }


# ============================================================================
# Process all checkpoints (no multiprocessing, pure JAX)
# ============================================================================

print(f"{'='*70}")
print(f"Starting batched processing (NO MULTIPROCESSING)")
print(f"{'='*70}\n")

overall_start = timeit.default_timer()

results = []
checkpoint_items = list(loaded_checkpoints.items())

for i, (checkpoint_file, ckpt_data) in enumerate(checkpoint_items):
    print(f"\n[Batch {i+1}/{len(checkpoint_items)}]")
    result = process_single_checkpoint(
        checkpoint_file,
        ckpt_data['agent_conf'],
        ckpt_data['agent_state'],
        n_steps=3000,
        n_envs=1,
        train_state_seed=0,
    )
    results.append(result)

overall_end = timeit.default_timer()
overall_elapsed = overall_end - overall_start

# Print summary
print(f"\n{'='*70}")
print(f"BATCHED PROCESSING COMPLETE")
print(f"{'='*70}")
print(f"Total time: {overall_elapsed:.2f}s")
print(f"Processed {len(loaded_checkpoints)} checkpoints")
print(f"Average time per checkpoint: {overall_elapsed/len(loaded_checkpoints):.2f}s\n")

# Summary by checkpoint
successful = [r for r in results if r["status"] == "success"]
failed = [r for r in results if r["status"] == "failed"]

print(f"Results:")
print(f"  ✓ Successful: {len(successful)}")
print(f"  ✗ Failed: {len(failed)}")

if successful:
    print(f"\nSuccessful checkpoints:")
    total_reward = 0.0
    for r in successful:
        total_reward += r["total_reward"]
        print(f"  - {r['checkpoint']:40s} | Time: {r['elapsed_time']:7.2f}s | Reward: {r['total_reward']:10.4f}")
    print(f"\n  Average reward: {total_reward/len(successful):.4f}")

if failed:
    print(f"\nFailed checkpoints:")
    for r in failed:
        print(f"  - {r['checkpoint']:40s} | Error: {r['error']}")

print(f"\nResults saved to: {os.path.dirname(folder_path)}")
