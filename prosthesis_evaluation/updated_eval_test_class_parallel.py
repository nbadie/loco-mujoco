"""
Parallel checkpoint evaluation script for faster processing.
Processes multiple checkpoints concurrently using multiprocessing.

Usage:
    python updated_eval_test_class_parallel.py --folder_path /path/to/checkpoints --num_workers 4
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
from functools import partial
from multiprocessing import cpu_count, get_context
from concurrent.futures import ProcessPoolExecutor
import traceback
import sys

from loco_mujoco.core.wrappers import VecEnv
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax, SavePPOJax
from loco_mujoco.environments.humanoids.skeleton_prosthesis import MjxSkeletonMuscleProsthesis
from loco_mujoco.algorithms.ppo_jax import PPOAgentConf, PPOAgentState
from omegaconf import OmegaConf
from loco_mujoco.evaluation.evaluation_prosthesis import ProsthesisMetricsHandler
from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction

import mujoco

os.environ["MUJOCO_GL"] = "egl"

# Set up argument parser
parser = argparse.ArgumentParser(description='Run parallel evaluation with PPOJax.')
parser.add_argument('--folder_path', type=str, required=True, help='Path to the agent pkl files')
parser.add_argument('--config_file', type=str, default=0, help='Configuration file')
parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (default: CPU count - 1)')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

folder_path = args.folder_path
config_file = args.config_file
agent_conf= SavePPOJax.load_agent_conf(config_file)
config = agent_conf.config
num_workers = args.num_workers or max(1, cpu_count() - 1)

dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"Processing checkpoints from: {folder_path}")
print(f"Using {num_workers} parallel workers")
print(f"Total CPU cores available: {cpu_count()}")

# ============================================================================
# Setup (run once, before parallel processing)
# ============================================================================

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

# Pre-compute metadata (once for all checkpoints)
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
print(f"  - Total actuators: {model.nu}")


# ============================================================================
# Worker Function - Runs in parallel for each checkpoint
# ============================================================================

def process_checkpoint(checkpoint_file, config_dict, dt_str, train_state_seed=0, n_steps=3000, n_envs=1):
    """
    Process a single checkpoint file in parallel.
    Each worker recreates its own environment (not inherited from main process).
    
    Args:
        checkpoint_file: filename of checkpoint (not full path)
        config_dict: dictionary with all picklable configuration
        dt_str: datetime string for file naming
        train_state_seed: which seed to use
        n_steps: number of rollout steps
        n_envs: number of parallel environments
        
    Returns:
        dict with evaluation results
    """
    try:
        # Extract config from pickle-safe dict
        folder_path = config_dict['folder_path']
        config = config_dict['config']
        agent_conf = config_dict['agent_conf']
        body_names = config_dict['body_names']
        joint_names = config_dict['joint_names']
        muscle_indices = config_dict['muscle_indices']
        all_actuator_names = config_dict['all_actuator_names']
        evaluation_muscle_groups = config_dict['evaluation_muscle_groups']
        evaluation_muscle_names = config_dict['evaluation_muscle_names']
        
        path = os.path.join(folder_path, checkpoint_file)
        print(f"[Worker] Processing: {checkpoint_file}")
        
        # =====================================================================
        # Create fresh environment in THIS worker process (NOT inherited)
        # =====================================================================
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
        
        OmegaConf.set_struct(config, False)
        config.experiment.env_params["headless"] = True
        config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"
        config.experiment.env_params["add_sensors"] = True
        
        if "reward_params" not in config.experiment.env_params or config.experiment.env_params["reward_params"] is None:
            config.experiment.env_params["reward_params"] = OmegaConf.create({})
        
        # config.experiment.env_params["horizon"] = 3000
        
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
        
        # Setup action sampling function for this worker
        def sample_actions_uncompiled(ts, obs, _rng):
            y, updates = agent_conf.network.apply({'params': ts.params,
                                                    'run_stats': ts.run_stats},
                                                    obs, mutable=["run_stats"])
            ts = ts.replace(run_stats=updates['run_stats'])
            pi, _ = y
            a = pi.sample(seed=_rng) 
            return a, ts
        
        sample_actions = jax.jit(sample_actions_uncompiled)
        
        # =====================================================================
        # Load checkpoint and agent
        # =====================================================================
        agent_conf_ckpt, agent_state = PPOJax.load_agent(path)
        
        # Get train state
        if config.experiment.n_seeds > 1:
            assert train_state_seed is not None
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
                print(f"[Worker {checkpoint_file}] Step {step_total}")

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

        env.stop()
        
        time_end = timeit.default_timer()
        elapsed = time_end - time_start

        print(f"[Worker {checkpoint_file}] Rollout complete: {elapsed:.2f}s, reward={float(total_reward):.4f}")

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
                                  f"{dt_str}_{tail}_evaluation_results_{n_steps}steps_{train_state_seed}seed.pkl")
        
        with open(output_path, "wb") as f:
            pickle.dump(all_relevant_data, f)
        
        print(f"[Worker {checkpoint_file}] Saved to {output_path}")
        
        return {
            "checkpoint": checkpoint_file,
            "status": "success",
            "output_path": output_path,
            "elapsed_time": elapsed,
            "total_reward": float(total_reward),
        }

    except Exception as e:
        error_msg = f"[Worker {checkpoint_file}] Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {
            "checkpoint": checkpoint_file,
            "status": "failed",
            "error": str(e),
        }


# ============================================================================
# Main - Orchestrate parallel processing
# ============================================================================

if __name__ == '__main__':
    # Get list of checkpoint files
    pkl_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pkl")])
    
    if not pkl_files:
        print(f"No .pkl files found in {folder_path}")
        exit(1)
    
    print(f"\nFound {len(pkl_files)} checkpoint files:")
    for f in pkl_files:
        print(f"  - {f}")
    print()
    
    # Create picklable config dict (don't pass env/jit/model - workers recreate them)
    config_dict = {
        'folder_path': folder_path,
        'config': config,
        'agent_conf': agent_conf,
        'body_names': body_names,
        'joint_names': joint_names,
        'muscle_indices': muscle_indices,
        'all_actuator_names': all_actuator_names,
        'evaluation_muscle_groups': evaluation_muscle_groups,
        'evaluation_muscle_names': evaluation_muscle_names,
    }
    
    # Prepare arguments for workers
    worker_args = [
        (file, config_dict, dt_str, 0, 3000, 1)
        for file in pkl_files
    ]
    
    # Process checkpoints in parallel using ProcessPoolExecutor (spawn-safe)
    print(f"Starting parallel processing with {num_workers} workers...\n")
    overall_start = timeit.default_timer()
    
    from concurrent.futures import as_completed
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_checkpoint, *args): args[0] for args in worker_args}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                checkpoint_file = futures[future]
                print(f"Error processing {checkpoint_file}: {e}")
                results.append({
                    "checkpoint": checkpoint_file,
                    "status": "failed",
                    "error": str(e),
                })
    
    overall_end = timeit.default_timer()
    overall_elapsed = overall_end - overall_start
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"PARALLEL PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {overall_elapsed:.2f}s")
    print(f"Processed {len(pkl_files)} checkpoints")
    print(f"Average time per checkpoint: {overall_elapsed/len(pkl_files):.2f}s\n")
    
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
