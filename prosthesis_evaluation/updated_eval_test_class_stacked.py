"""
Checkpoint evaluation - processes checkpoints sequentially with full metrics collection.

Loads all checkpoints, stacks their train_states, then processes each checkpoint
sequentially to collect comprehensive evaluation metrics. This is the most efficient
approach for complex, multi-metric rollouts with mutable environment state.

WHY NOT VMAP OVER CHECKPOINTS?
- JAX vmap is designed for pure functions operating on a data batch dimension
- Environment rollouts are inherently sequential (each step depends on previous state)
- Can't vmap over time steps - vmap only works on single-point-in-time batch dimensions
- Mutable state (env_state, train_state) cannot be vmapped across sequential iterations

RESULT: Sequential processing is actually optimal for this use case. Each checkpoint
takes ~30s for 3000 steps. For N checkpoints: ~30s * N total time.

Usage:
    python updated_eval_test_class_stacked.py --folder_path /path/to/checkpoints --config_file config.pkl
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
parser = argparse.ArgumentParser(description='Run stacked parallel evaluation using JAX vmap.')
parser.add_argument('--folder_path', type=str, required=True, help='Path to the agent pkl files')
parser.add_argument('--config_file', type=str, required=True, help='Configuration file')
args = parser.parse_args()

folder_path = args.folder_path
config_file = args.config_file

dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"Processing checkpoints from: {folder_path}")
print(f"Sequential checkpoint processing with stacked train_states\n")
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

randomization_type = config.randomization_config["randomization_type"]
# Convert to plain dict to allow adding new keys
randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)

# add prosthesis side to randomization params if it exists in config.experiment.env_params
if "prosthesis_side" in config.experiment.env_params:
    randomization_params["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]


env = factory.make(
    **config.experiment.env_params,
    **config.experiment.task_factory.params,
    # domain_randomization_type=randomization_type, domain_randomization_params=randomization_params,
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

print("Stacking train_states into batch dimension...")

# Extract all train_states
train_states = [ckpt['agent_state'].train_state for ckpt in checkpoint_list]

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

print(f"✓ Stacked {len(checkpoint_files)} train_states")
print(f"  Batch size: {len(checkpoint_files)}\n")

# ============================================================================
# Vectorized rollout function using vmap
# ============================================================================

# ============================================================================
# Vectorized rollout - process all checkpoints with stacked state
# ============================================================================

def rollout_with_stacked_state(stacked_train_state, agent_conf, checkpoint_files, checkpoint_list, n_steps=3000, n_envs=1):
    """
    Process all checkpoints in parallel using vmap over the checkpoint batch dimension.
    Each checkpoint gets its own environment state (replicated), all run in parallel.
    """
    num_checkpoints = len(checkpoint_files)
    all_results = []
    
    # Initialize environment for ONE instance
    seed = 5
    rng = jax.random.key(seed) #0
    keys = jax.random.split(rng, n_envs + 1)
    rng, env_keys = keys[0], keys[1:]
    env_state_single = mjx_reset_base(env_keys[0])  # Single environment reset
    obs_single = env_state_single.observation
    
    # Replicate env_state for all checkpoints
    # Use tree_map to expand first dimension from 1 to num_checkpoints
    def replicate_for_checkpoints(x):
        """Replicate a leaf value for all checkpoints."""
        # Convert to jax array if needed
        x = jnp.asarray(x)
        # If scalar, just repeat it; if array, add batch dimension then repeat
        if x.ndim == 0:  # scalar
            return jnp.repeat(x, num_checkpoints)
        else:  # array
            return jnp.repeat(x[jnp.newaxis, ...], num_checkpoints, axis=0)
    
    env_state = jax.tree.map(replicate_for_checkpoints, env_state_single)
    obs = jax.tree.map(replicate_for_checkpoints, obs_single)
    
    train_state = stacked_train_state
    total_reward = jnp.zeros(num_checkpoints)
    
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
        # When evaluating single checkpoint, must use same RNG sequence as noRand
        # When evaluating multiple, split RNG so each checkpoint gets independent randomness
        rngs_checkpoints = jax.random.split(_rng, num_checkpoints)
        action, train_state = sample_actions_jit(train_state, obs, rngs_checkpoints)

        # rngs_checkpoints = jnp.tile(_rng, (num_checkpoints, 1))  # Replicate _rng for all checkpoints
        # action, train_state = sample_actions_jit(train_state, obs, rngs_checkpoints)
        
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
    
    # Convert rewards (shape: num_checkpoints) to scalars
    reward_scalars = [float(jnp.asarray(total_reward[i])) for i in range(num_checkpoints)]
    
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
        # both_array = jnp.concatenate([left_array, right_array], axis=-1)
        
        sum_step_activations[left_key] = jnp.sum(left_array, axis=0)
        sum_step_activations[right_key] = jnp.sum(right_array, axis=0)
        # sum_step_activations[f"{muscle_group}_both"] = sum_step_activations[left_key] + sum_step_activations[right_key]

        step_norm_activations[left_key] = sum_step_activations[left_key] / n_steps
        step_norm_activations[right_key] = sum_step_activations[right_key] / n_steps
        # step_norm_activations[f"{muscle_group}_both"] = sum_step_activations[f"{muscle_group}_both"] / n_steps

        n_musc = len(evaluation_muscle_names[muscle_group])
        sum_muscles_activations[left_key] = jnp.sum(sum_step_activations[left_key], axis=0)
        sum_muscles_activations[right_key] = jnp.sum(sum_step_activations[right_key], axis=0)
        sum_muscles_activations[f"{muscle_group}_both"] = sum_muscles_activations[left_key] + sum_muscles_activations[right_key] #jnp.sum(sum_step_activations[f"{muscle_group}_both"], axis=0)
        
        step_muscles_norm_activations[left_key] = sum_muscles_activations[left_key] / n_steps
        step_muscles_norm_activations[right_key] = sum_muscles_activations[right_key] / n_steps
        step_muscles_norm_activations[f"{muscle_group}_both"] = sum_muscles_activations[f"{muscle_group}_both"] / n_steps
        
        # step_num_norm_activations[left_key] = jnp.sum(step_muscles_norm_activations[left_key] / n_musc)
        # step_num_norm_activations[right_key] = jnp.sum(step_muscles_norm_activations[right_key] / n_musc)
        # step_num_norm_activations[f"{muscle_group}_both"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_both"] / (2 * n_musc))

        step_num_norm_activations[left_key] = step_muscles_norm_activations[left_key] / n_musc
        step_num_norm_activations[right_key] = step_muscles_norm_activations[right_key] / n_musc
        step_num_norm_activations[f"{muscle_group}_both"] = step_muscles_norm_activations[f"{muscle_group}_both"] / (2 * n_musc)

        # print('step_muscles_norm_activations[left_key]:', step_muscles_norm_activations[left_key])
        # print('step_num_norm_activations[left_key]:', step_num_norm_activations[left_key])
        # print('step_num_norm_activations[f"{muscle_group}_both"]: ', step_num_norm_activations[f"{muscle_group}_both"])


    # Helper function to extract checkpoint-specific data from nested structures
    def extract_checkpoint_data(data, checkpoint_idx):
        """
        Extract the checkpoint_idx slice from all arrays in nested structures.
        For body_xposes: {body: [step0_array(num_ckpt, 3), step1_array(num_ckpt, 3), ...]}
        Result: {body: [step0_array(3,), step1_array(3,), ...]}  - all steps for one checkpoint
        """
        if isinstance(data, dict):
            # For dicts, recursively extract from all values
            return {k: extract_checkpoint_data(v, checkpoint_idx) for k, v in data.items()}
        elif isinstance(data, list) and len(data) > 0:
            # For lists of arrays (like steps), extract checkpoint_idx from each array in the list
            result = []
            for step_array in data:
                if isinstance(step_array, (jnp.ndarray, list)):
                    try:
                        # Extract checkpoint_idx from this step's array
                        result.append(step_array[checkpoint_idx])
                    except (IndexError, TypeError):
                        result.append(step_array)
                else:
                    # Scalar or other type - keep as is
                    result.append(step_array)
            return result
        elif isinstance(data, (jnp.ndarray, list)) and hasattr(data, '__getitem__'):
            # Single array: try to extract
            try:
                return data[checkpoint_idx]
            except (IndexError, TypeError):
                return data
        else:
            return data
    
    # Save results for each checkpoint
    for batch_idx in range(num_checkpoints):
        checkpoint_file = checkpoint_files[batch_idx]
        print(f"Processing [{batch_idx+1}/{num_checkpoints}]: {checkpoint_file}")
        
        # Extract all per-checkpoint data using the helper function
        all_relevant_data = {
            "total_steps": step_count,
            "n_envs": n_envs,
            "total_reward": reward_scalars[batch_idx],
            "avg_reward_per_step": reward_scalars[batch_idx] / step_count if step_count > 0 else 0.0,
            "all_grf_l": extract_checkpoint_data(all_grf_l, batch_idx),
            "all_grf_r": extract_checkpoint_data(all_grf_r, batch_idx),
            # "all_foot_ground_contact_left": extract_checkpoint_data(all_foot_ground_contact_left, batch_idx),
            # "all_foot_ground_contact_right": extract_checkpoint_data(all_foot_ground_contact_right, batch_idx),
            "all_sensor_force": extract_checkpoint_data(all_sensor_force, batch_idx),
            "sensor_force_names": sensor_force_names,
            "evaluation_muscle_groups": evaluation_muscle_groups,
            "evaluation_joint_names": joint_names,
            "evaluation_muscle_names": evaluation_muscle_names,
            "all_actions": extract_checkpoint_data(all_actions, batch_idx),
            "all_actuator_names": all_actuator_names,
            "all_body_poses": extract_checkpoint_data(body_xposes, batch_idx),
            "all_body_vels": extract_checkpoint_data(body_cvels, batch_idx),
            "evaluation_body_names": body_names,
        }


        # Add muscle activation summaries for each checkpoint to all_relevant_data
        all_relevant_data["sum_step_activations"] = {k: v[batch_idx] for k, v in sum_step_activations.items()}
        all_relevant_data["step_norm_activations"] = {k: v[batch_idx] for k, v in step_norm_activations.items()}
        all_relevant_data["sum_muscles_activations"] = {k: v[batch_idx] for k, v in sum_muscles_activations.items()}
        all_relevant_data["step_muscles_norm_activations"] = {k: v[batch_idx] for k, v in step_muscles_norm_activations.items()}
        all_relevant_data["step_num_norm_activations"] = {k: v[batch_idx] for k, v in step_num_norm_activations.items()}
        
        # # Compute muscle activations for this checkpoint
        # sum_step_activations_checkpoint = {}
        # step_norm_activations_checkpoint = {}
        # sum_muscles_activations_checkpoint = {}
        # step_muscles_norm_activations_checkpoint = {}
        # step_num_norm_activations_checkpoint = {}

        # for muscle_group in evaluation_muscle_groups:
        #     left_key = f"{muscle_group}_left"
        #     right_key = f"{muscle_group}_right"
            
        #     left_array = jnp.array(muscle_activations[left_key])
        #     right_array = jnp.array(muscle_activations[right_key])
            
        #     # Extract checkpoint data
        #     left_checkpoint = left_array[:, batch_idx] if left_array.ndim > 1 else left_array[batch_idx]
        #     right_checkpoint = right_array[:, batch_idx] if right_array.ndim > 1 else right_array[batch_idx]
            
        #     sum_step_activations_checkpoint[left_key] = jnp.sum(left_checkpoint, axis=0)
        #     sum_step_activations_checkpoint[right_key] = jnp.sum(right_checkpoint, axis=0)

        #     step_norm_activations_checkpoint[left_key] = sum_step_activations_checkpoint[left_key] / n_steps
        #     step_norm_activations_checkpoint[right_key] = sum_step_activations_checkpoint[right_key] / n_steps

        #     n_musc = len(evaluation_muscle_names[muscle_group])
        #     sum_muscles_activations_checkpoint[left_key] = jnp.sum(sum_step_activations_checkpoint[left_key], axis=0)
        #     sum_muscles_activations_checkpoint[right_key] = jnp.sum(sum_step_activations_checkpoint[right_key], axis=0)
            
        #     step_muscles_norm_activations_checkpoint[left_key] = sum_muscles_activations_checkpoint[left_key] / n_steps
        #     step_muscles_norm_activations_checkpoint[right_key] = sum_muscles_activations_checkpoint[right_key] / n_steps
            
        #     step_num_norm_activations_checkpoint[left_key] = jnp.sum(step_muscles_norm_activations_checkpoint[left_key] / n_musc)
        #     step_num_norm_activations_checkpoint[right_key] = jnp.sum(step_muscles_norm_activations_checkpoint[right_key] / n_musc)
        
        # all_relevant_data["sum_step_activations"] = sum_step_activations_checkpoint
        # all_relevant_data["step_norm_activations"] = step_norm_activations_checkpoint
        # all_relevant_data["sum_muscles_activations"] = sum_muscles_activations_checkpoint
        # all_relevant_data["step_muscles_norm_activations"] = step_muscles_norm_activations_checkpoint
        # all_relevant_data["step_num_norm_activations"] = step_num_norm_activations_checkpoint
        
        # Add joint data for this checkpoint
        for joint_name in joint_names:
            for key in joint_data[joint_name].keys():
                all_relevant_data[f"{joint_name}_{key}"] = extract_checkpoint_data(joint_data[joint_name][key], batch_idx)

        # Add muscle activations for this checkpoint
        for muscle_group in evaluation_muscle_groups:
            all_relevant_data[f"run_{muscle_group}_activation_left"] = extract_checkpoint_data(muscle_activations.get(f"{muscle_group}_left", []), batch_idx)
            all_relevant_data[f"run_{muscle_group}_activation_right"] = extract_checkpoint_data(muscle_activations.get(f"{muscle_group}_right", []), batch_idx)

        # Save results for this checkpoint
        tail = checkpoint_file[:-4]
        output_path = os.path.join(os.path.dirname(folder_path), 
                                    f"{dt_str}_{tail}_evaluation_results_{n_steps}steps_{seed}seed.pkl")
        
        with open(output_path, "wb") as f:
            pickle.dump(all_relevant_data, f)
        
        print(f"  ✓ Reward: {reward_scalars[batch_idx]:.4f}")
        print(f"  ✓ Saved to {output_path}\n")
        
        all_results.append({
            "checkpoint": checkpoint_file,
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
    # Run all checkpoints by slicing the stacked state
    all_results = rollout_with_stacked_state(
        stacked_train_state,
        checkpoint_list[0]['agent_conf'],  # Same config for all
        checkpoint_files,
        checkpoint_list,
        n_steps=2000,#3000,
        n_envs=1,
    )
    
    # Extract rewards for summary
    rewards = jnp.array([r["reward"] for r in all_results])
    
    overall_end = timeit.default_timer()
    overall_elapsed = overall_end - overall_start
    
    print(f"\n{'='*70}")
    print(f"CHECKPOINT EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {overall_elapsed:.2f}s")
    print(f"Total Total Time: {overall_end - overall_initial_time:.2f}s")
    print(f"Processed {len(checkpoint_files)} checkpoints (sequentially)")
    print(f"Average time per checkpoint: {overall_elapsed/len(checkpoint_files):.2f}s\n")
    
    # Print results
    print("Results:")
    for result in all_results:
        print(f"  {result['checkpoint']:40s} | Reward: {result['reward']:10.4f}")
    
    avg_reward = float(jnp.mean(rewards))
    print(f"\n  Average reward: {avg_reward:.4f}")
    print(f"\nAll results saved to: {os.path.dirname(folder_path)}")
    
except Exception as e:
    error_msg = f"Error in stacked state rollout: {str(e)}\n{traceback.format_exc()}"
    print(error_msg)
    exit(1)
