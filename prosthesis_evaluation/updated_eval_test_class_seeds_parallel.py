import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"

import jax 
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp

import pickle


import argparse

from loco_mujoco.core.wrappers import VecEnv


from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.environments.humanoids.skeleton_prosthesis import MjxSkeletonMuscleProsthesis


from loco_mujoco.algorithms.ppo_jax import PPOAgentConf, PPOAgentState
from omegaconf import OmegaConf

from loco_mujoco.evaluation.evaluation_prosthesis import ProsthesisMetricsHandler

from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction

import mujoco
from datetime import datetime

import timeit 

os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering, which is more compatible with headless environments
# os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_triton_gemm_any=True ')

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

# Use the path from command line arguments
path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config

start_time = timeit.default_timer()
# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = True #False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic
config.experiment.env_params["add_sensors"] = True
# config.experiment.env_params["socket_ty_slack"] = False

randomization_type = config.randomization_config["randomization_type"]
# Convert to plain dict to allow adding new keys
randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)

# add prosthesis side to randomization params if it exists in config.experiment.env_params
if "prosthesis_side" in config.experiment.env_params:
    randomization_params["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]

# randomization_params["randomize_prosthesis_body_position"] = True
# randomization_params["prosthesis_body_position_range"] = {'pylon_socket': {'x': [0.05,0.05]}} #[-0.01,-0.01]}} 
env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params,
                #    terrain_type="RoughTerrain", terrain_params=dict(random_min_height=-0.00005,random_max_height=0.00005),
                #    domain_randomization_type=randomization_type, domain_randomization_params=randomization_params
                )
env.th.to_jax()
env = VecEnv(env)
jit_step  = jax.jit(jax.vmap(env.mjx_step))  #env.step)
jit_reset  = jax.jit(jax.vmap(env.mjx_reset)) #env.reset)

def sample_actions_uncompiled(ts, obs, _rng): # Renamed for clarity
        y, updates = agent_conf.network.apply({'params': ts.params,
                                            'run_stats': ts.run_stats},
                                            obs, mutable=["run_stats"])
        ts = ts.replace(run_stats=updates['run_stats'])  # update stats
        pi, _ = y
        a = pi.sample(seed=_rng)
        return a, ts

# JIT compile the function
# sample_actions = jax.jit(sample_actions_uncompiled) # <--- ADD THIS LINE
sample_actions = jax.jit(sample_actions_uncompiled)

model = env.get_model()

prosthesis_metrics_handler = ProsthesisMetricsHandler(env) #(config, env)

n_steps = 1000 #400 #2 #400 #400 #1000 #10 #1000 #00 #1000
n_envs = 1 #1  # <--- Make sure this matches your training batch size

# seed = 0
seeds = 10 #4 #5 #10
# Start a new folder in the same directory as the agent pkl file to save all evaluation results for all seeds
save_eval_folder_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_eval"
eval_seeds_dir = os.path.join(os.path.dirname(path), save_eval_folder_name)
os.makedirs(eval_seeds_dir, exist_ok=True)
print("Evaluation directory for seeds:", eval_seeds_dir)

# Initilaize some parameters
evaluation_muscle_groups = ["back_muscles", "torso_muscles", "vasti_muscles", "rectus_muscles", "medial_muscles", "leg_muscles", "all_muscles"]
evaluation_muscle_names = {}
for n in evaluation_muscle_groups:
    evaluation_muscle_names[n] = prosthesis_metrics_handler.get_muscle_group(n)

body_names = []
for i in range(model.nbody):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    body_names.append(joint_name)

# Body name with possible contact to ground 
foot_name = "toes"  # name of the foot box in the model
calcn_name = "calcn"
all_grf_l = []
all_grf_r = []

left_side = "left_side"
right_side = "right_side"

# all_actions = []
# all_actuator_names = []

# # Pre-allocate actions and muscle activations
# all_actions = jnp.zeros((n_steps, model.nu))
# run_muscle_activations = {
#     f"run_{g}_activation_left": jnp.zeros((n_steps, len(evaluation_muscle_names[g])))
#     for g in evaluation_muscle_groups
# }
# run_muscle_activations.update({
#     f"run_{g}_activation_right": jnp.zeros((n_steps, len(evaluation_muscle_names[g])))
#     for g in evaluation_muscle_groups
# })

muscle_skeleton_control_activation = SkeletonMuscleControlFunction(env)

def filter_muscles(muscle_names, side, actuators_removed):
    return [m for m in muscle_names if m+side not in actuators_removed]

# Assuming you have access to self.prosthesis_side and self.actuators_removed
prosthesis_side = config.experiment.env_params["prosthesis_side"]
actuators_removed = env.actuators_removed

body_names_eval = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
all_actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)]
sensor_force, sensor_force_names = prosthesis_metrics_handler.get_sensor_data_batched(env.data)
joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]

# Parallel evaluation over all seeds using JAX vmap

def run_eval_for_seed(seed):
    # All per-seed data must be initialized inside this function for parallelism
    rng = jax.random.key(seed)
    train_state_seed = 0  # Take first seed

    keys = jax.random.split(rng, n_envs + 1)
    rng, env_keys = keys[0], keys[1:]

    env_state = jit_reset(env_keys)
    obs = env_state.observation

    # Per-seed metrics containers
    all_foot_ground_contact_left = []
    all_foot_ground_contact_right = []
    all_grf_l = jnp.zeros((n_steps, 6))
    all_grf_r = jnp.zeros((n_steps, 6))
    # all_actions = []
    # all_actuator_names = []
    body_xposes = {name: jnp.zeros((n_steps,3)) for name in body_names}
    # joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    joint_metrics = ["angle", "velocity", "forces_constraint", "forces_smooth", "forces_applied", "torques", "energy_exp"]
    # Pre-allocate arrays for each joint and metric
    joint_data = {
        joint_name: {
            metric: jnp.zeros((n_steps,))  # shape can be (n_steps,) or (n_steps, dim) if metric is vector
            for metric in joint_metrics
        }
        for joint_name in joint_names
    }
    all_sensor_force = {}

    # Muscle activations
    # Pre-allocate actions and muscle activations
    all_actions = jnp.zeros((n_steps, model.nu))
    # run_muscle_activations = {
    #     f"run_{g}_activation_left": jnp.zeros((n_steps, len(evaluation_muscle_names[g])))
    #     for g in evaluation_muscle_groups
    # }
    # run_muscle_activations.update({
    #     f"run_{g}_activation_right": jnp.zeros((n_steps, len(evaluation_muscle_names[g])))
    #     for g in evaluation_muscle_groups
    # })
    run_muscle_activations = {
    f"run_{g}_activation_left": jnp.zeros(
        (n_steps, len(filter_muscles(evaluation_muscle_names[g], '_l', actuators_removed if prosthesis_side == "left_side" else [])))
    )
    for g in evaluation_muscle_groups
    }
    run_muscle_activations.update({
        f"run_{g}_activation_right": jnp.zeros(
            (n_steps, len(filter_muscles(evaluation_muscle_names[g], '_r', actuators_removed if prosthesis_side == "right_side" else [])))
        )
        for g in evaluation_muscle_groups
    })
    # run_muscle_activations = {f"run_{g}_activation_left": [] for g in evaluation_muscle_groups}
    # run_muscle_activations.update({f"run_{g}_activation_right": [] for g in evaluation_muscle_groups})

    step_total = 0
    if config.experiment.n_seeds > 1:
        train_state = jax.tree_map(lambda x: x[train_state_seed], agent_state.train_state)
    else:
        train_state = agent_state.train_state

    for i in range(n_steps):
        rng, _rng = jax.random.split(rng)
        action, train_state = sample_actions(train_state, obs, _rng)
        action = jnp.atleast_2d(action)
        env_state = jit_step(env_state, action)
        obs = env_state.observation

        contact_left, contact_right = prosthesis_metrics_handler.get_contact_steps_batched(env_state.data, i) #prosthesis_metrics_handler.get_contact_steps(env_state.data, i)
        all_foot_ground_contact_left.append(contact_left)
        all_foot_ground_contact_right.append(contact_right)

        grf_foot_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_l")
        grf_foot_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_r")
        grf_calcn_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{calcn_name}_l")
        grf_calcn_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{calcn_name}_r")
        grf_l = grf_foot_l + grf_calcn_l
        grf_r = grf_foot_r + grf_calcn_r
        all_grf_l = all_grf_l.at[i].set(grf_l)
        all_grf_r = all_grf_r.at[i].set(grf_r)

        body_xpos = prosthesis_metrics_handler.get_xpos_batched(env_state.data)
        for name in body_names:
            # body_xposes[name]= body_xpos[name].at[i].set(body_xpos[name][0])
            body_xposes[name]= body_xposes[name].at[i].set(jnp.squeeze(body_xpos[name])) 
            # body_xposes[name]= body_xpos[name].at[i].set(jnp.squeeze(body_xpos.get(name, jnp.zeros((3,)))))
            # body_xposes[name].append(body_xpos[name])

        joint_angles = prosthesis_metrics_handler.get_joint_angles(env_state.data)
        joint_velocities = prosthesis_metrics_handler.get_joint_vels(env_state.data)
        joint_forces_constraint, joint_forces_smooth, joint_forces_applied = prosthesis_metrics_handler.get_joint_frces(env_state.data)
        joint_torques = prosthesis_metrics_handler.get_joint_trques(env_state.data)
        joint_energy_exp = prosthesis_metrics_handler.calc_joint_energy_exp(joint_torques, joint_velocities)
        for joint_name in joint_data:
            joint_data[joint_name]["angle"] = joint_data[joint_name]["angle"].at[i].set(jnp.squeeze(joint_angles.get(joint_name)))
            joint_data[joint_name]["velocity"] = joint_data[joint_name]["velocity"].at[i].set(jnp.squeeze(joint_velocities.get(joint_name)))
            joint_data[joint_name]["forces_constraint"] = joint_data[joint_name]["forces_constraint"].at[i].set(jnp.squeeze(joint_forces_constraint.get(joint_name)))
            joint_data[joint_name]["forces_smooth"] = joint_data[joint_name]["forces_smooth"].at[i].set(jnp.squeeze(joint_forces_smooth.get(joint_name)))
            joint_data[joint_name]["forces_applied"] = joint_data[joint_name]["forces_applied"].at[i].set(jnp.squeeze(joint_forces_applied.get(joint_name)))
            joint_data[joint_name]["torques"] = joint_data[joint_name]["torques"].at[i].set(jnp.squeeze(joint_torques.get(joint_name)))
            joint_data[joint_name]["energy_exp"] = joint_data[joint_name]["energy_exp"].at[i].set(jnp.squeeze(joint_energy_exp.get(joint_name)))
            
            # joint_data[joint_name]["angle"] = joint_data[joint_name]["angle"].at[...,i].set(
            #     jnp.array(joint_angles.get(joint_name, 0))
            # )
            # joint_data[joint_name]["velocity"] = joint_data[joint_name]["velocity"].at[i].set(
            #     jnp.array(joint_velocities.get(joint_name, 0))
            # )
            # joint_data[joint_name]["forces_constraint"] = joint_data[joint_name]["forces_constraint"].at[i].set(
            #     jnp.array(joint_forces_constraint.get(joint_name, 0))
            # )
            # joint_data[joint_name]["forces_smooth"] = joint_data[joint_name]["forces_smooth"].at[i].set(
            #     jnp.array(joint_forces_smooth.get(joint_name, 0))
            # )
            # joint_data[joint_name]["forces_applied"] = joint_data[joint_name]["forces_applied"].at[i].set(
            #     jnp.array(joint_forces_applied.get(joint_name, 0))
            # )
            # joint_data[joint_name]["torques"] = joint_data[joint_name]["torques"].at[i].set(
            #     jnp.array(joint_torques.get(joint_name, 0))
            # )
            # joint_data[joint_name]["energy_exp"] = joint_data[joint_name]["energy_exp"].at[i].set(
            #     jnp.array(joint_energy_exp.get(joint_name, 0))
            # )
        sensor_force, sensor_force_names = prosthesis_metrics_handler.get_sensor_data_batched(env_state.data)
        for sensor_name, force in sensor_force.items():
            if sensor_name not in all_sensor_force:
                all_sensor_force[sensor_name] = jnp.zeros((n_steps,3)) # + force.shape)
            all_sensor_force[sensor_name] = all_sensor_force[sensor_name].at[i].set(force)

        # for sensor_name, force in sensor_force.items():
        #     if sensor_name not in all_sensor_force:
        #         all_sensor_force[sensor_name] = []
        #     all_sensor_force[sensor_name].append(force)

        # for i_act in range(model.nu):
        #     if model.actuator_dyntype[i_act] == mujoco.mjtDyn.mjDYN_MUSCLE:
        #         action = action.at[..., i_act].set(muscle_skeleton_control_activation.adapted_sigmoid(action[..., i_act]))
        # all_actions.append(action)

        # all_actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)]

        # for muscle_group, muscle_names in evaluation_muscle_names.items():
        #     left = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, left_side, action)
        #     right = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, right_side, action)
        #     run_muscle_activations[f"run_{muscle_group}_activation_left"].append(left)
        #     run_muscle_activations[f"run_{muscle_group}_activation_right"].append(right)

        for i_act in range(model.nu):
            if model.actuator_dyntype[i_act] == mujoco.mjtDyn.mjDYN_MUSCLE:
                action = action.at[..., i_act].set(muscle_skeleton_control_activation.adapted_sigmoid(action[..., i_act]))
        all_actions = all_actions.at[i].set(jnp.squeeze(action))

        # all_actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)]

        for muscle_group, muscle_names in evaluation_muscle_names.items():
            left = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, left_side, action)
            right = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, right_side, action)
            # flatten to [batch]
            left = left.reshape(left.shape[0],)
            right = right.reshape(right.shape[0],)

            run_muscle_activations[f"run_{muscle_group}_activation_left"] = (
                run_muscle_activations[f"run_{muscle_group}_activation_left"].at[i, :].set(left)
            )
            run_muscle_activations[f"run_{muscle_group}_activation_right"] = (
                run_muscle_activations[f"run_{muscle_group}_activation_right"].at[i, :].set(right)
            )
            # run_muscle_activations[f"run_{muscle_group}_activation_left"] = run_muscle_activations[f"run_{muscle_group}_activation_left"].at[i].set(jnp.squeeze(left))
            # run_muscle_activations[f"run_{muscle_group}_activation_right"] = run_muscle_activations[f"run_{muscle_group}_activation_right"].at[i].set(jnp.squeeze(right))


        step_total += n_envs

        # env.mjx_render_domain_randomization(env_state, record=True)

    env.stop()

    # Post-processing muscle activations
    sum_step_activations = {}
    step_norm_activations = {}
    sum_muscles_activations = {}
    step_muscles_norm_activations = {}
    step_num_norm_activations = {}

    for muscle_group in evaluation_muscle_groups:
        left_arr = jnp.array(run_muscle_activations[f"run_{muscle_group}_activation_left"])
        right_arr = jnp.array(run_muscle_activations[f"run_{muscle_group}_activation_right"])
        sum_step_activations[f"{muscle_group}_left"] = jnp.sum(left_arr, axis=1)
        sum_step_activations[f"{muscle_group}_right"] = jnp.sum(right_arr, axis=1)
        step_norm_activations[f"{muscle_group}_left"] = sum_step_activations[f"{muscle_group}_left"] / n_steps
        step_norm_activations[f"{muscle_group}_right"] = sum_step_activations[f"{muscle_group}_right"] / n_steps
        # n_musc = len(evaluation_muscle_names[muscle_group])
        # sum_muscles_activations[f"{muscle_group}_left"] = jnp.sum(sum_step_activations[f"{muscle_group}_left"], axis=1)
        # sum_muscles_activations[f"{muscle_group}_right"] = jnp.sum(sum_step_activations[f"{muscle_group}_right"], axis=1)
        # step_muscles_norm_activations[f"{muscle_group}_left"] = sum_muscles_activations[f"{muscle_group}_left"] / n_steps
        # step_muscles_norm_activations[f"{muscle_group}_right"] = sum_muscles_activations[f"{muscle_group}_right"] / n_steps
        # step_num_norm_activations[f"{muscle_group}_left"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_left"] / n_musc)
        # step_num_norm_activations[f"{muscle_group}_right"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_right"] / n_musc)

    # joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    # # body_names_eval = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]

    all_relevant_data = {
        "total_steps": step_total,
        "n_envs": n_envs,
        "all_grf_l": all_grf_l,
        "all_grf_r": all_grf_r,
        "all_foot_ground_contact_left": all_foot_ground_contact_left,
        "all_foot_ground_contact_right": all_foot_ground_contact_right,
        "all_sensor_force": all_sensor_force,
        # # "sensor_force_names": sensor_force_names,
        "sum_step_activations": sum_step_activations,
        "step_norm_activations": step_norm_activations,
        # "sum_muscles_activations": sum_muscles_activations,
        # "step_muscles_norm_activations": step_muscles_norm_activations,
        # "step_num_norm_activations": step_num_norm_activations,
        # # "evaluation_muscle_groups": evaluation_muscle_groups,
        # # "evaluation_joint_names": joint_names,
        # # "evaluation_muscle_names": evaluation_muscle_names,
        "all_actions": all_actions,
        # "all_actuator_names": all_actuator_names,
        "all_body_poses": body_xposes,
        # "evaluation_body_names": body_names_eval,
    }
    for joint_name, joint_dict in joint_data.items():
        for key, value in joint_dict.items():
            all_relevant_data[f"{joint_name}_{key}"] = value
    for muscle_group in evaluation_muscle_groups:
        all_relevant_data[f"run_{muscle_group}_activation_left"] = run_muscle_activations[f"run_{muscle_group}_activation_left"]
        all_relevant_data[f"run_{muscle_group}_activation_right"] = run_muscle_activations[f"run_{muscle_group}_activation_right"]

    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return all_relevant_data

# Run parallel evaluation over all seeds
seed_values = jnp.arange(0, seeds)
start_seed_time = timeit.default_timer()
results = jax.vmap(run_eval_for_seed)(seed_values)
end_seed_time = timeit.default_timer()
print(f"Time taken for parallel evaluation over {seeds} seeds: {end_seed_time - start_seed_time:.2f} seconds")

def extract_seed_data(nested_dict, seed_idx):
    """Extract the i-th seed from all arrays in a nested dict."""
    extracted = {}
    for subkey, array in nested_dict.items():
        # Only index if it's an array/list with __getitem__
        if isinstance(array, (jnp.ndarray, list)) and hasattr(array, '__getitem__'):
            extracted[subkey] = array[seed_idx]
        else:
            extracted[subkey] = array
    return extracted
# def extract_seed_data(tree, seed_idx):
#     """
#     Recursively extract the i-th seed from a nested dict of arrays/lists.
#     """
#     if isinstance(tree, dict):
#         return {k: extract_seed_data(v, seed_idx) for k, v in tree.items()}
#     elif isinstance(tree, (jnp.ndarray, list)):
#         return tree[seed_idx]
#     else:
#         # Scalars or other objects
#         return tree
for i, seed in enumerate(seed_values):
    all_relevant_data = {
        k: extract_seed_data(v, i) if k in ["all_sensor_force", "all_body_poses"] else
           (v[i] if isinstance(v, (jnp.ndarray, list)) and hasattr(v, '__getitem__') and len(v) > i else v)
        for k, v in results.items()
    }    
# for i, seed in enumerate(seed_values):
#     all_relevant_data = {}
    # for k, v in results.items():
    #     if k in ["all_sensor_force", "all_body_poses"]:
    #         # Keep all sub-keys but extract the seed dimension
    #         all_relevant_data[k] = extract_seed_data(v, i)
    #     else:
    #         # Just take i-th seed if possible
    #         if isinstance(v, (jnp.ndarray, list)) and hasattr(v, '__getitem__'):
    #             all_relevant_data[k] = v[i]
    #         else:
    #             all_relevant_data[k] = v    
# # Save results for each seed
# for i, seed in enumerate(seed_values):
#     # all_relevant_data = results[i]

#     all_relevant_data = {}
#     for k, v in results.items():
#         if k == "all_sensor_force" or k == "all_body_poses":
#             # Extract nested array: first index is outer (e.g., step or env), second index is seed
#             all_relevant_data[k] = v[0][i]
#         elif isinstance(v, (jnp.ndarray, list)) and hasattr(v, '__getitem__'):
#             # Normal per-seed extraction
#             all_relevant_data[k] = v[i]
#         else:
#             # Scalar or non-array values, keep as-is
#             all_relevant_data[k] = v
    
#     # # Extract the i-th seed's data from the dict of arrays
#     # all_relevant_data = {k: v[i] if isinstance(v, (jnp.ndarray, list)) and hasattr(v, '__getitem__') else v
#     #                     for k, v in results.items()}

#     # if 'all_sensor_force' then [0][0] to get first seed and [0][1] to get second seed
#     # if  "all_body_poses" then [0][0] to get first seed and [0][1] to get second seed
    
    sum_muscles_activations={}
    step_muscles_norm_activations={}
    step_num_norm_activations={}
    sum_step_activations = all_relevant_data["sum_step_activations"]
    for muscle_group in evaluation_muscle_groups:
        n_musc = len(evaluation_muscle_names[muscle_group])
        # sum_step_activations = all_relevant_data["sum_step_activations"] #"step_norm_activations"]
        # sum_muscles_activations[f"{muscle_group}_left"] = jnp.sum(sum_step_activations[f"{muscle_group}_left"], axis=1)
        # sum_muscles_activations[f"{muscle_group}_right"] = jnp.sum(sum_step_activations[f"{muscle_group}_right"], axis=1)
        # step_muscles_norm_activations[f"{muscle_group}_left"] = sum_muscles_activations[f"{muscle_group}_left"] / n_steps
        # step_muscles_norm_activations[f"{muscle_group}_right"] = sum_muscles_activations[f"{muscle_group}_right"] / n_steps
        # step_num_norm_activations[f"{muscle_group}_left"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_left"] / n_musc)
        # step_num_norm_activations[f"{muscle_group}_right"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_right"] / n_musc)
        sum_muscles_activations[f"{muscle_group}_left"] = jnp.sum(sum_step_activations[f"{muscle_group}_left"], axis=1)[i]
        sum_muscles_activations[f"{muscle_group}_right"] = jnp.sum(sum_step_activations[f"{muscle_group}_right"], axis=1)[i]
        step_muscles_norm_activations[f"{muscle_group}_left"] = sum_muscles_activations[f"{muscle_group}_left"] / n_steps
        step_muscles_norm_activations[f"{muscle_group}_right"] = sum_muscles_activations[f"{muscle_group}_right"] / n_steps
        step_num_norm_activations[f"{muscle_group}_left"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_left"] / n_musc)
        step_num_norm_activations[f"{muscle_group}_right"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_right"] / n_musc)

        # Uncomment if you want to keep the commented code
        # sum_step_activations[f"{muscle_group}_left"] = jnp.sum(run_muscle_activations[f"run_{mus
        # sum_muscles_activations[f"{muscle_group}_left"] = jnp.sum(sum_step_activations[f"{muscle_group}_left"], axis=1)
        # sum_muscles_activations[f"{muscle_group}_right"] = jnp.sum(sum_step_activations[f"{muscle_group}_right"], axis=1)
        # step_muscles_norm_activations[f"{muscle_group}_left"] = sum_muscles_activations[f"{muscle_group}_left"] / n_steps
        # step_muscles_norm_activations[f"{muscle_group}_right"] = sum_muscles_activations[f"{muscle_group}_right"] / n_steps
        # step_num_norm_activations[f"{muscle_group}_left"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_left"] / n_musc)
        # step_num_norm_activations[f"{muscle_group}_right"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_right"] / n_musc)
    all_relevant_data["sum_muscles_activations"] = sum_muscles_activations
    all_relevant_data["step_muscles_norm_activations"] = step_muscles_norm_activations
    all_relevant_data["step_num_norm_activations"] = step_num_norm_activations
    # Append names to all_relevant_data
    all_relevant_data["evaluation_muscle_names"] = evaluation_muscle_names
    all_relevant_data["all_actuator_names"] = all_actuator_names
    all_relevant_data["evaluation_body_names"] = body_names_eval
    all_relevant_data["sensor_force_names"] = sensor_force_names
    all_relevant_data["evaluation_joint_names"]= joint_names
    all_relevant_data["evaluation_muscle_groups"] = evaluation_muscle_groups
    output_path = os.path.join(eval_seeds_dir, f"{n_steps}steps_{int(seed)}seed.pkl")
    print('Full path', output_path)
    with open(output_path, "wb") as f:
        pickle.dump(all_relevant_data, f)
    print(f"Saved evaluation data to {output_path}")

# Print the total time taken for evaluation
total_time = timeit.default_timer() - start_time
print(f"Total evaluation time: {total_time:.2f} seconds")
print(f"Average time per seed: {total_time / seeds:.2f} seconds")

#print time when script finishes
print(f"Script finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
