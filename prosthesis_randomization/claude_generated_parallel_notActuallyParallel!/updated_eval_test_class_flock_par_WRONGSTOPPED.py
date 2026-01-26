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
# from loco_mujoco.environments.humanoids.skeleton_prosthesis import MjxSkeletonMuscleProsthesis


# from loco_mujoco.algorithms.ppo_jax import PPOAgentConf, PPOAgentState
from omegaconf import OmegaConf


# sys.path.append(os.path.join(os.path.dirname(__file__), "/home/nadinebadie/loco-mujoco/prosthesis_test"))
from loco_mujoco.evaluation.evaluation_prosthesis import ProsthesisMetricsHandler

from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction

import mujoco
from datetime import datetime

import timeit 
import sys

import copy
from functools import partial


subfolder_name = 'pylon_socket_pos_z' #_TalZ5'
# subfolder_name = 'talus_ori_z'

dt_str_init = datetime.now().strftime("%Y%m%d_%H%M%S")

talus_ori_ang = np.deg2rad(5)

# Test False set_fixed & ori rand 
# Set with postiion 

# eval_targets = {"prosthesis_body_orientation": {"pylon_socket": ["x"]}}
set_fixed_randomized_targets = True #True #False
fixed_randomized_targets = {"prosthesis_body_orientation": {"pylon_socket": {"x": talus_ori_ang}, "talus": {"z": talus_ori_ang}}}
# fixed_bodies = []
# fixed_all_axis = []


# talus_ori_unequal_0 = False #True


ori_ang_test= np.deg2rad(4)
ori_ang_test_2 = np.deg2rad(2)
# ori_ang = np.deg2rad(10)
ori_ang = np.deg2rad(6)
talus_ori_ang = np.deg2rad(5)
# ori_ang_t = np.deg2rad(4)
# Parameter randomization initialization (no changes)
randomization_params_names = ["prosthesis_dof_damping", "prosthesis_joint_stiffness", "prosthesis_body_position", "prosthesis_body_orientation"]
##### ONLY 1. Name in dictionary is incrementaly changed if randomization flag is true 
randomization_params_eval = {
    "prosthesis_side": "left_side",
    "randomize_prosthesis_dof_damping": False,
    "prosthesis_dof_damping_range": {'ankle_angle': [2, 10]},
    "randomize_prosthesis_joint_stiffness": False,
    "prosthesis_joint_stiffness_range": {'ankle_angle': [1000, 1300]}, #[14924, 24924]}, #{'ankle_angle': [50, 100]},
    "randomize_prosthesis_body_position": False, #False, #True, #False, #True, #False, #True,
    "prosthesis_body_position_range": {'pylon_socket': {'x': [-0.010,0.010]}}, #{'z': [-0.0292, 0.0292]}}, #{'pylon_socket': {'x': [-0.1, 0.1]}},
    # "prosthesis_body_position_range": {'pylon_socket': {'x': [-0.0147, 0.0147]}}, #{'z': [-0.0292, 0.0292]}}, #{'pylon_socket': {'x': [-0.1, 0.1]}},
    # "prosthesis_body_position_range": {'talus': {'z': [-0.015, 0.015]}}, #{'pylon_socket': {'x': [-0.1, 0.1]}},
    "randomize_prosthesis_body_orientation": True, #False, #True, #False, #True, #False, #True, #False,
    # "prosthesis_body_orientation_range": {'talus': {'z': [-talus_ori_ang, talus_ori_ang]}},
    # "prosthesis_body_orientation_range": {'talus': {'z': [-talus_ori_ang, -talus_ori_ang]}}
    # "prosthesis_body_orientation_range": {'pylon_socket': {'x': [-ori_ang, ori_ang]}, 'talus': {'z': [talus_ori_ang, talus_ori_ang]}}
    "prosthesis_body_orientation_range": {'pylon_socket': {'z': [-ori_ang, ori_ang]}} 
}
randomization_increments = {
    "prosthesis_joint_stiffness": 100, #10,
    "prosthesis_dof_damping": 5,
    # "prosthesis_body_position": 0.01, #
    "prosthesis_body_position": 0.005, #0.04, #0.05,
    # "prosthesis_body_orientation": talus_ori_ang*2, #0.175, #0.35 #0.001 #0.2
    # "prosthesis_body_orientation": ori_ang/3, #0.175, #0.35 #0.001 #0.2
    "prosthesis_body_orientation": ori_ang/2, #0.175, #0.35 #0.001 #0.2
}

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

randomization_type = config.randomization_config["randomization_type"]
randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)

# update values in randommization_params based on randomization_params_eval
for key, value in randomization_params_eval.items():
    if key in randomization_params:
        # If the key exists, update it
        randomization_params[key] = value
    else:
        # If the key does not exist, add it
        randomization_params[key] = value

# add prosthesis side to randomization params if it exists in config.experiment.env_params
        if "prosthesis_side" in config.experiment.env_params:
            randomization_params["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]


# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = True #False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic
config.experiment.env_params["add_sensors"] = True

# delete 'knee_extension_limit'' from config
# del config.experiment.env_params["knee_extension_limit'"]

# config.experiment.env_params['socket_ty_slack'] = False
# config.experiment.env_params['limit_hip_joints'] = True
# config.experiment.env_params["socket_ty_joint"] = False
env = factory.make(domain_randomization_type=randomization_type, domain_randomization_params=randomization_params,
                   **config.experiment.env_params, **config.experiment.task_factory.params)
env.th.to_jax()
env = VecEnv(env)
# jit_step  = jax.jit(jax.vmap(env.mjx_step_test))  #env.step)
jit_step  = jax.jit(jax.vmap(env.mjx_step)) #_test))  #env.step)
jit_reset  = jax.jit(jax.vmap(env.mjx_reset)) #env.reset)
model = env.get_model()

prosthesis_metrics_handler = ProsthesisMetricsHandler(env)

n_steps = 1000
n_envs = 1  # <--- Make sure this matches your training batch size
rng = jax.random.key(0)
train_state_seed = 0  # Take first seed 

keys = jax.random.split(rng, n_envs + 1)
rng, env_keys = keys[0], keys[1:]

def sample_actions_uncompiled(ts, obs, _rng): # Renamed for clarity
    y, updates = agent_conf.network.apply({'params': ts.params,
                                           'run_stats': ts.run_stats},
                                           obs, mutable=["run_stats"])
    ts = ts.replace(run_stats=updates['run_stats'])  # update stats
    pi, _ = y
    a = pi.sample(seed=_rng)
    return a, ts

# JIT compile the function
sample_actions = jax.jit(sample_actions_uncompiled)


if config.experiment.n_seeds > 1:
    assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
                                            "train_state_seed for replay.")
    
    train_state = jax.tree.map(lambda x: x[train_state_seed], agent_state.train_state)
else: 
    # obs, env_state = jit_reset(env_keys) #env.reset(env_keys)
    train_state = agent_state.train_state


muscle_skeleton_control_activation = SkeletonMuscleControlFunction(env)

def run_eval_batched(env_vec, param_name, n_steps, agent_conf, agent_state, prosthesis_metrics_handler, values, directions=None, subfolder_name=None, dt_str_init=None):
    """
    Run parallel evaluations for a list of `values`. env_vec is a VecEnv wrapping len(values) envs.
    Saves one pickle per run/value (same format as run_evaluation_loop).
    """
    n_runs = len(values)
    # prepare random keys: one per env/run
    keys_local = jax.random.split(rng, n_runs + 1)
    rng_local, env_keys_batch = keys_local[0], keys_local[1:]

    # reset batched env
    jit_reset_batched = jax.jit(jax.vmap(env_vec.mjx_reset))
    jit_step_batched  = jax.jit(jax.vmap(env_vec.mjx_step))

    env_state_batch = jit_reset_batched(env_keys_batch)
    obs = env_state_batch.observation

    # prepare train_state for usage (same logic as single-run)
    if config.experiment.n_seeds > 1:
        train_state_local = jax.tree_map(lambda x: x[train_state_seed], agent_state.train_state)
    else:
        train_state_local = agent_state.train_state

    model_local = env_vec.get_model()

    # pre-allocate arrays with batch dim first
    all_grf_l = jnp.zeros((n_runs, n_steps, 6))
    all_grf_r = jnp.zeros((n_runs, n_steps, 6))

    # body poses per body name
    body_names = [mujoco.mj_id2name(model_local, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model_local.nbody)]
    body_xposes = {name: jnp.zeros((n_runs, n_steps, 3)) for name in body_names}

    # joint metrics
    joint_names = [mujoco.mj_id2name(model_local, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model_local.njnt)]
    joint_metrics = ["angle", "velocity", "forces_constraint", "forces_smooth", "forces_applied", "torques", "energy_exp"]
    joint_data = {
        jn: {m: jnp.zeros((n_runs, n_steps)) for m in joint_metrics}
        for jn in joint_names
    }

    all_sensor_force = {}
    # muscle activations & actions
    all_actions = jnp.zeros((n_runs, n_steps, model_local.nu))

    evaluation_muscle_groups = ["back_muscles", "torso_muscles", "vasti_muscles", "rectus_muscles", "medial_muscles", "leg_muscles", "all_muscles"]
    evaluation_muscle_names = {n: prosthesis_metrics_handler.get_muscle_group(n) for n in evaluation_muscle_groups}

    # filter muscles (reuse helper logic from seeds parallel)
    def filter_muscles(muscle_names, side, actuators_removed):
        return [m for m in muscle_names if m+side not in actuators_removed]

    prosthesis_side = config.experiment.env_params.get("prosthesis_side", None)
    actuators_removed = env_vec.actuators_removed if hasattr(env_vec, "actuators_removed") else []

    run_muscle_activations = {}
    for g in evaluation_muscle_groups:
        left_len = len(filter_muscles(evaluation_muscle_names[g], '_l', actuators_removed if prosthesis_side == "left_side" else []))
        right_len = len(filter_muscles(evaluation_muscle_names[g], '_r', actuators_removed if prosthesis_side == "right_side" else []))
        run_muscle_activations[f"run_{g}_activation_left"] = jnp.zeros((n_runs, n_steps, left_len))
        run_muscle_activations[f"run_{g}_activation_right"] = jnp.zeros((n_runs, n_steps, right_len))

    # main loop
    for t in range(n_steps):
        # split keys: one per run
        rng_local, step_key = jax.random.split(rng_local)
        step_keys = jax.random.split(step_key, n_runs)

        # sample actions for all runs (vmapped)
        # sample_actions expects (train_state, obs, rng) -> (action, train_state)
        # vmapping with in_axes=(None, 0, 0) reuses the same train_state_local for each run
        actions_batched, train_state_local = jax.vmap(lambda ts, o, k: sample_actions(ts, o, k), in_axes=(None, 0, 0))(train_state_local, env_state_batch.observation, step_keys)
        actions_batched = jnp.atleast_3d(actions_batched)  # (batch, 1?, act_dim) -> ensure shape

        # apply muscle sigmoid per-actuator for muscle actuators (vectorized)
        # convert to mutable array
        acts = actions_batched
        for a_idx in range(model_local.nu):
            if model_local.actuator_dyntype[a_idx] == mujoco.mjtDyn.mjDYN_MUSCLE:
                # adapted_sigmoid expects shape consistent with per-run action slice
                acts = acts.at[..., a_idx].set(muscle_skeleton_control_activation.adapted_sigmoid(acts[..., a_idx]))

        env_state_batch = jit_step_batched(env_state_batch, acts)
        obs = env_state_batch.observation

        # batched metric extraction (use batched helpers where available)
        contact_left_b, contact_right_b = prosthesis_metrics_handler.get_contact_steps_batched(env_state_batch.data, t)
        # GRF
        grf_foot_l = prosthesis_metrics_handler.get_grf(env_state_batch.data, f"toes_l")
        grf_foot_r = prosthesis_metrics_handler.get_grf(env_state_batch.data, f"toes_r")
        grf_calcn_l = prosthesis_metrics_handler.get_grf(env_state_batch.data, f"calcn_l")
        grf_calcn_r = prosthesis_metrics_handler.get_grf(env_state_batch.data, f"calcn_r")
        grf_l = grf_foot_l + grf_calcn_l
        grf_r = grf_foot_r + grf_calcn_r
        all_grf_l = all_grf_l.at[:, t].set(grf_l)
        all_grf_r = all_grf_r.at[:, t].set(grf_r)

        # body xpos batched
        body_xpos_batched = prosthesis_metrics_handler.get_xpos_batched(env_state_batch.data)
        for name in body_names:
            body_xposes[name] = body_xposes[name].at[:, t, :].set(jnp.squeeze(body_xpos_batched.get(name, jnp.zeros((n_runs,3)))))

        # joint data (some helpers return non-batched dicts, handle accordingly)
        joint_angles = prosthesis_metrics_handler.get_joint_angles(env_state_batch.data)
        joint_velocities = prosthesis_metrics_handler.get_joint_vels(env_state_batch.data)
        joint_forces_constraint, joint_forces_smooth, joint_forces_applied = prosthesis_metrics_handler.get_joint_frces(env_state_batch.data)
        joint_torques = prosthesis_metrics_handler.get_joint_trques(env_state_batch.data)
        joint_energy_exp = prosthesis_metrics_handler.calc_joint_energy_exp(joint_torques, joint_velocities)

        for jn in joint_names:
            # each of these should be shape (n_runs, ...) or broadcastable
            joint_data[jn]["angle"] = joint_data[jn]["angle"].at[:, t].set(jnp.squeeze(joint_angles.get(jn)))
            joint_data[jn]["velocity"] = joint_data[jn]["velocity"].at[:, t].set(jnp.squeeze(joint_velocities.get(jn)))
            joint_data[jn]["forces_constraint"] = joint_data[jn]["forces_constraint"].at[:, t].set(jnp.squeeze(joint_forces_constraint.get(jn)))
            joint_data[jn]["forces_smooth"] = joint_data[jn]["forces_smooth"].at[:, t].set(jnp.squeeze(joint_forces_smooth.get(jn)))
            joint_data[jn]["forces_applied"] = joint_data[jn]["forces_applied"].at[:, t].set(jnp.squeeze(joint_forces_applied.get(jn)))
            joint_data[jn]["torques"] = joint_data[jn]["torques"].at[:, t].set(jnp.squeeze(joint_torques.get(jn)))
            joint_data[jn]["energy_exp"] = joint_data[jn]["energy_exp"].at[:, t].set(jnp.squeeze(joint_energy_exp.get(jn)))

        # sensor forces batched
        sensor_force_b, sensor_force_names = prosthesis_metrics_handler.get_sensor_data_batched(env_state_batch.data)
        for sname, force in sensor_force_b.items():
            if sname not in all_sensor_force:
                all_sensor_force[sname] = jnp.zeros((n_runs, n_steps, force.shape[-1]))
            all_sensor_force[sname] = all_sensor_force[sname].at[:, t, :].set(force)

        # store actions
        all_actions = all_actions.at[:, t, :].set(jnp.squeeze(acts))

        # muscle activations per group
        for mg, muscle_names in evaluation_muscle_names.items():
            left = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, "left_side", acts)
            right = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, "right_side", acts)
            # ensure shapes (n_runs, n_musc)
            left = left.reshape((n_runs, -1))
            right = right.reshape((n_runs, -1))
            run_muscle_activations[f"run_{mg}_activation_left"] = run_muscle_activations[f"run_{mg}_activation_left"].at[:, t, :].set(left)
            run_muscle_activations[f"run_{mg}_activation_right"] = run_muscle_activations[f"run_{mg}_activation_right"].at[:, t, :].set(right)

    # stop envs
    env_vec.stop()

    # Post-process and save per-run files
    for idx in range(n_runs):
        val = values[idx]
        direction = directions[idx] if directions is not None else None

        # per-run assembly (mirror run_evaluation_loop keys)
        sum_step_activations = {}
        step_norm_activations = {}
        sum_muscles_activations = {}
        step_muscles_norm_activations = {}
        step_num_norm_activations = {}
        for mg in evaluation_muscle_groups:
            left_arr = run_muscle_activations[f"run_{mg}_activation_left"][idx]  # (n_steps, n_musc)
            right_arr = run_muscle_activations[f"run_{mg}_activation_right"][idx]
            sum_step_activations[f"{mg}_left"] = jnp.sum(left_arr, axis=0)
            sum_step_activations[f"{mg}_right"] = jnp.sum(right_arr, axis=0)
            step_norm_activations[f"{mg}_left"] = sum_step_activations[f"{mg}_left"] / n_steps
            step_norm_activations[f"{mg}_right"] = sum_step_activations[f"{mg}_right"] / n_steps
            n_musc = len(evaluation_muscle_names[mg])
            sum_muscles_activations[f"{mg}_left"] = jnp.sum(sum_step_activations[f"{mg}_left"])
            sum_muscles_activations[f"{mg}_right"] = jnp.sum(sum_step_activations[f"{mg}_right"])
            step_muscles_norm_activations[f"{mg}_left"] = sum_muscles_activations[f"{mg}_left"] / n_steps
            step_muscles_norm_activations[f"{mg}_right"] = sum_muscles_activations[f"{mg}_right"] / n_steps
            step_num_norm_activations[f"{mg}_left"] = jnp.sum(step_muscles_norm_activations[f"{mg}_left"] / max(1, n_musc))
            step_num_norm_activations[f"{mg}_right"] = jnp.sum(step_muscles_norm_activations[f"{mg}_right"] / max(1, n_musc))

        # build all_relevant_data
        all_relevant_data = {
            "total_steps": n_steps,
            "n_envs": 1,
            "all_grf_l": all_grf_l[idx].tolist(),
            "all_grf_r": all_grf_r[idx].tolist(),
            "all_sensor_force": {k: v[idx].tolist() for k, v in all_sensor_force.items()},
            "sum_step_activations": sum_step_activations,
            "step_norm_activations": step_norm_activations,
            "sum_muscles_activations": sum_muscles_activations,
            "step_muscles_norm_activations": step_muscles_norm_activations,
            "step_num_norm_activations": step_num_norm_activations,
            "evaluation_muscle_groups": evaluation_muscle_groups,
            "evaluation_joint_names": joint_names,
            "evaluation_muscle_names": evaluation_muscle_names,
            "all_actions": all_actions[idx].tolist(),
            "all_actuator_names": [mujoco.mj_id2name(model_local, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model_local.nu)],
            "all_body_poses": {k: v[idx].tolist() for k, v in body_xposes.items()},
            "evaluation_body_names": body_names,
        }

        for jn, jdict in joint_data.items():
            for key, arr in jdict.items():
                all_relevant_data[f"{jn}_{key}"] = arr[idx].tolist()

        for mg in evaluation_muscle_groups:
            all_relevant_data[f"run_{mg}_activation_left"] = run_muscle_activations[f"run_{mg}_activation_left"][idx].tolist()
            all_relevant_data[f"run_{mg}_activation_right"] = run_muscle_activations[f"run_{mg}_activation_right"][idx].tolist()

        # name and save
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        joint_name = list(randomization_params_eval[f"{param_name}_range"].keys())[0] if param_name in randomization_params_eval else "param"
        if 'orientation' in param_name:
            name_param_value = np.rad2deg(val)
            name_units = 'deg'
        elif 'position' in param_name:
            name_param_value = val*1000
            name_units = 'mm'
        else:
            name_param_value = val
            name_units = ''
        if subfolder_name is not None:
            os.makedirs(os.path.join(os.path.dirname(path), f"{dt_str_init}_{subfolder_name}"), exist_ok=True)
            output_path = os.path.join(os.path.dirname(path), f"{dt_str_init}_{subfolder_name}", f"eval_{n_steps}steps_{joint_name}_{direction}_{int(np.round(name_param_value,0))}{name_units}_{param_name}_{idx}.pkl")
        else:
            output_path = os.path.join(os.path.dirname(path), f"{dt_str}_eval_{n_steps}steps_{joint_name}_{direction}_{int(np.round(name_param_value,0))}{name_units}_{param_name}_{idx}.pkl")

        with open(output_path, "wb") as f:
            pickle.dump(all_relevant_data, f)
        print(f"Saved evaluation data to {output_path}")






###########
time_all = []
time_all.append(timeit.default_timer())  # Start timer


# Iterate over params in randomization_params_names 
for param_name in randomization_params_names:
    # Set all randomization flags to False initially to isolate the current parameter
    for name in randomization_params_names:
        env._domain_randomizer.rand_conf[f"randomize_{name}"] = False

    # Only proceed if the current parameter is set to be randomized in the eval config
    if randomization_params_eval[f"randomize_{param_name}"]:

        # Set the current parameter's randomization flag to True
        env._domain_randomizer.rand_conf[f"randomize_{param_name}"] = True

        # if talus_ori_unequal_0 and param_name == "prosthesis_body_position":
        #     # Manually set talus orientation to 0 deg when position is being evaluated
        #     env._domain_randomizer.rand_conf["randomize_prosthesis_body_orientation"] = True
        #     env._domain_randomizer.rand_conf["prosthesis_body_orientation_range"] = randomization_params_eval["prosthesis_body_orientation_range"]
        

        if set_fixed_randomized_targets: 
            fixed_param_names = list(fixed_randomized_targets.keys())
            for fixed_param_name in fixed_param_names:
                env._domain_randomizer.rand_conf[f"{fixed_param_name}_range"] = {}
                if fixed_randomized_targets.get(fixed_param_name) is not None:
                    # set randomize_param_name to True 
                    env._domain_randomizer.rand_conf[f"randomize_{fixed_param_name}"] = True
                    for fixed_body, fixed_axes in fixed_randomized_targets[fixed_param_name].items():
                        # fixed_bodies = fixed_bodies.append(fixed_body)
                        for fixed_axis, fixed_value in fixed_axes.items():
                            # fixed_all_axis = fixed_all_axis.append(fixed_axis)
                            if not env._domain_randomizer.rand_conf[f"{fixed_param_name}_range"]:
                                env._domain_randomizer.rand_conf[f"{fixed_param_name}_range"]= {fixed_body : {fixed_axis : [fixed_value, fixed_value]}}
                            else: 
                                if fixed_body not in env._domain_randomizer.rand_conf[f"{fixed_param_name}_range"]:
                                    env._domain_randomizer.rand_conf[f"{fixed_param_name}_range"][fixed_body] = {}
                                env._domain_randomizer.rand_conf[f"{fixed_param_name}_range"][fixed_body][fixed_axis] = [fixed_value, fixed_value]

        if "stiffness" in param_name or "damping" in param_name:
            # The range is now a dictionary, not a simple list
            joint_name = list(randomization_params_eval[f"{param_name}_range"].keys())[0] # ONLY 1. body_name in dictionary is incrementaly changed
            min_val, max_val = randomization_params_eval[f"{param_name}_range"][joint_name]
            increment = randomization_increments[param_name]

            # Iterate through the range with the specified increment
            current_value = min_val
            while current_value <= max_val:
                # Update the randomization range for the specific joint to be a fixed value
                env._domain_randomizer.rand_conf[f"{param_name}_range"][joint_name] = [current_value, current_value]
                
                # env_keys = jax.random.split(rng, 2)
                jit_reset  = jax.jit(jax.vmap(env.mjx_reset)) #env.reset)
                env_state = jit_reset(env_keys) 
                obs = env_state.observation
                print("Running evaluation for ", param_name, " with value: ", current_value)
                run_evaluation_loop(env_state, n_steps, train_state, rng, prosthesis_metrics_handler,param_name=param_name, param_value = current_value, subfolder_name = subfolder_name,dt_str_init=dt_str_init) #min_val + increment)
                current_value += increment

        elif "position" in param_name or "orientation" in param_name:
            body_range = randomization_params_eval[f"{param_name}_range"]
            body_name = list(body_range.keys())[0] # ONLY 1. body_name in dictionary is incrementaly changed if randomization flag is true 
            directions = list(body_range[body_name].keys())

            for axis in directions:
                min_val = body_range[body_name][axis][0]
                max_val = body_range[body_name][axis][1]
                inc = randomization_increments[param_name]
                values = np.arange(min_val, max_val + inc/2, inc).tolist()
                if not values: 
                    values = [min_val]

                env_list = []
                dirs = []
                for v in values: 
                    rnd_params_copy = copy.deepcopy(randomization_params)
                    rnd_params_copy[f"{param_name}_range"] = {body_name: {axis: [float(v), float(v)]}}
                    rnd_params_copy[f"randomize_{param_name}"] = True
                    if set_fixed_randomized_targets:
                        for k_f, v_f in fixed_randomized_targets.items():
                            rnd_params_copy[f"{k_f}_range"] = copy.deepcopy(v_f)
                            rnd_params_copy[f"randomize_{k_f}"] = True
                    env_i = factory.make(domain_randomization_type=randomization_type, domain_randomization_params=rnd_params_copy,
                                        **config.experiment.env_params, **config.experiment.task_factory.params)
                    env_i.th.to_jax()
                    env_list.append(env_i)
                    dirs.append(axis)
            batched_env = VecEnv(env_list)
            run_eval_batched(batched_env, param_name, n_steps, agent_conf, agent_state, prosthesis_metrics_handler, values, directions=dirs, subfolder_name=subfolder_name, dt_str_init=dt_str_init)
                        
                # if set_fixed_randomized_targets:
                #     if fixed_randomized_targets.get(param_name):
                #         # if axis and body_name already exist, error stop code that they should be fixed already or whatever
                #         if body_name in fixed_randomized_targets[f"{param_name}"] and axis in fixed_randomized_targets[f"{param_name}"][body_name]:
                #             raise RuntimeError(f"{param_name}_range[{body_name}][{axis}] is already defined to be fixed.")

                # # Use a small tolerance for floating-point comparison
                # current_value = min_val
                # while current_value <= max_val + 1e-6:
                #     # # Reset all axes to a zero range first
                #     # for other_axis in directions:
                #     #     env._domain_randomizer.rand_conf[f"{param_name}_range"][body_name][other_axis] = [0, 0]
                    
                #     # if set_fixed_randomized_targets: 
                #     #     fixed_param_names = list(fixed_randomized_targets.keys())
                #     #     for fixed_param_name in fixed_param_names:
                #     #         env._domain_randomizer.rand_conf[f"{fixed_param_name}_range"] = {}
                #     #         if fixed_randomized_targets.get(fixed_param_name) is not None:
                #     #             # set randomize_param_name to True 
                #     #             env._domain_randomizer.rand_conf[f"randomize_{fixed_param_name}"] = True
                #     #             for fixed_body, fixed_axes in fixed_randomized_targets[fixed_param_name].items():
                #     #                 for fixed_axis, fixed_value in fixed_axes.items():
                #     #                     env._domain_randomizer.rand_conf[f"{fixed_param_name}_range"]= {fixed_body : {fixed_axis : [fixed_value, fixed_value]}}
                #     #                     # env._domain_randomizer.rand_conf[f"{fixed_param_name}_range"][fixed_body][fixed_axis] = [fixed_value, fixed_value]

                #     # Set the current axis to a fixed value for evaluation
                #     # env._domain_randomizer.rand_conf[f"{param_name}_range"][body_name][axis] = [current_value, current_value]

                #     if set_fixed_randomized_targets:
                #         if param_name in fixed_param_names: 
                #             # expand the dictionary rather than overwriting it. Problem body name might not be in the dict yet or axis might not be in dict yet 
                #             if body_name not in env._domain_randomizer.rand_conf[f"{param_name}_range"]:
                #                 env._domain_randomizer.rand_conf[f"{param_name}_range"][body_name] = {}
                #             # if axis not in env._domain_randomizer.rand_conf[f"{param_name}_range"][body_name]:
                #             env._domain_randomizer.rand_conf[f"{param_name}_range"][body_name][axis] = [current_value, current_value]
                #             # else:
                #                 # env._domain_randomizer.rand_conf[f"{param_name}_range"][body_name][axis] = [current_value, current_value]
                #     else:
                #         env._domain_randomizer.rand_conf[f"{param_name}_range"][body_name] = {axis: [current_value, current_value]}

                    # Reset to new domain randomization state
                    # Re jit reset for correct update of env_state?
                    
                    # jit_reset  = jax.jit(jax.vmap(env.mjx_reset))
                    # # env_keys = jax.random.split(rng, 2)
                    # env_state = jit_reset(env_keys)
                    # obs = env_state.observation
                    # print("Running evaluation for ", param_name," for axis: ", axis, " with value: ",current_value)
                    # run_evaluation_loop(env_state, n_steps, train_state, rng, prosthesis_metrics_handler,param_name=param_name, param_value = current_value,direction=axis, subfolder_name = subfolder_name, dt_str_init=dt_str_init) #min_val + inc)
                    # current_value += inc

                


    


