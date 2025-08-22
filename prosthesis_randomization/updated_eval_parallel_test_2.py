import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from functools import partial

import pickle
import numpy as np
import argparse
import sys
from omegaconf import OmegaConf
from typing import Dict, Any

from loco_mujoco import TaskFactory, LocoEnv
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction
from loco_mujoco.evaluation.evaluation_prosthesis import ProsthesisMetricsHandler
from loco_mujoco.core.mujoco_mjx import MjxAdditionalCarry, MjxState, AdditionalCarry#, MjvScene
from jax.tree_util import tree_map, tree_leaves, tree_unflatten

import mujoco
from datetime import datetime
import timeit

# Parameter randomization initialization (no changes)
randomization_params_names = ["prosthesis_dof_damping", "prosthesis_joint_stiffness", "prosthesis_body_position", "prosthesis_body_orientation"]
randomization_params_eval = {
    "prosthesis_side": "left_side",
    "randomize_prosthesis_dof_damping": False,
    "prosthesis_dof_damping_range": {'ankle_angle': [2, 10]},
    "randomize_prosthesis_joint_stiffness": False,
    "prosthesis_joint_stiffness_range": {'ankle_angle': [50, 100]},
    "randomize_prosthesis_body_position": True,
    "prosthesis_body_position_range": {'pylon_socket': {'x': [0.2, 0.25]}},
    "randomize_prosthesis_body_orientation": False,
    "prosthesis_body_orientation_range": {'pylon_socket': {'x': [-0.1, 0.1]}},
}
randomization_increments = {
    "prosthesis_joint_stiffness": 10,
    "prosthesis_dof_damping": 5,
    "prosthesis_body_position": 0.05,
    "prosthesis_body_orientation": 0.1
}

# Parse arguments (no changes)
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config

# Set up randomization parameters (no changes)
randomization_type = config.randomization_config["randomization_type"]
randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)
for key, value in randomization_params_eval.items():
    randomization_params[key] = value
if "prosthesis_side" in config.experiment.env_params:
    randomization_params["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]

factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

OmegaConf.set_struct(config, False)
config.experiment.env_params["headless"] = False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"
config.experiment.env_params["add_sensors"] = True

n_steps = 100
rng = jax.random.key(0)
train_state_seed = 0

def adapted_sigmoid(action):
    q = 5
    return 1 / (1 + jnp.exp(-q * action))

@jax.jit
def sample_actions_uncompiled(ts, obs, _rng):
    y, updates = agent_conf.network.apply({'params': ts.params,
                                           'run_stats': ts.run_stats},
                                           obs, mutable=["run_stats"])
    ts = ts.replace(run_stats=updates['run_stats'])
    pi, _ = y
    a = pi.sample(seed=_rng)
    return a, ts

sample_actions = jax.jit(jax.vmap(sample_actions_uncompiled, in_axes=(None, 0, 0)))

if config.experiment.n_seeds > 1:
    assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
                                          "train_state_seed for replay.")
    train_state = jax.tree.map(lambda x: x[train_state_seed], agent_state.train_state)
else:
    train_state = agent_state.train_state
# --- START OF REFACTORED CODE ---

def generate_randomization_configs():
    # This part of your code is fine as it generates the configurations
    # we need to set up the environments dynamically.
    all_rand_configs = []
    all_param_names = []
    all_param_values = []


    for param_name in randomization_params_names:
        if randomization_params[f"randomize_{param_name}"]:
            if "stiffness" in param_name or "damping" in param_name:
                param_range = randomization_params[f"{param_name}_range"]
                for joint_name, (min_val, max_val) in param_range.items():
                    inc = randomization_increments[param_name]
                    for val in range(min_val, max_val + 1, inc):
                        rand_config = {
                            f"randomize_{param_name}": True,
                            f"{param_name}_range": {joint_name: [val, val]}
                        }

                        for other_joint in param_range.keys():
                            if other_joint != joint_name:
                                rand_config[f"{param_name}_range"][other_joint] = [0, 0]

                        for other_param in randomization_params_names:
                            if other_param != param_name:
                                rand_config[f"randomize_{other_param}"] = False

                        all_rand_configs.append(rand_config)
                        all_param_names.append(f"{param_name}_{joint_name}")
                        all_param_values.append(val)

            elif "position" in param_name or "orientation" in param_name:
                param_range = randomization_params[f"{param_name}_range"]
                for body_name, body_axes in param_range.items():
                    for axis, (min_val, max_val) in body_axes.items():
                        inc = randomization_increments[param_name]
                        for val in np.arange(min_val, max_val + inc/2, inc):
                            rand_config = {
                                f"randomize_{param_name}": True,
                                f"{param_name}_range": {body_name: {axis: [val, val]}}
                            }

                            for other_axis in body_axes.keys():
                                if other_axis != axis:
                                    rand_config[f"{param_name}_range"][body_name][other_axis] = [0.0, 0.0]

                            for other_body in param_range.keys():
                                if other_body != body_name:
                                    rand_config[f"{param_name}_range"][other_body] = {}
                                    for other_body_axis in param_range[other_body].keys():
                                        rand_config[f"{param_name}_range"][other_body][other_body_axis] = [0.0, 0.0]

                            for other_param in randomization_params_names:
                                if other_param != param_name:
                                    rand_config[f"randomize_{other_param}"] = False

                            all_rand_configs.append(rand_config)
                            all_param_names.append(f"{param_name}_{body_name}_{axis}")
                            all_param_values.append(val)

    return all_rand_configs, all_param_names, all_param_values


def batch_pytree(pytrees):
    return jax.tree_util.tree_map(lambda *args: jnp.stack(args), *pytrees)

main_env = factory.make(
    domain_randomization_type=randomization_type,
    domain_randomization_params=randomization_params,
    **config.experiment.env_params,
    **config.experiment.task_factory.params
)
main_env.th.to_jax()

def make_batched_reset_fn(env):
    @partial(jax.jit, static_argnums=(0,))
    def batched_reset(env, key, batched_rand_configs):

        def single_reset_with_rand(key, rand_config):
            key, subkey = jax.random.split(key)
            
            # Initialize carry, data, and model. These are not modified here.
            data = env._first_data
            model = env._model
            backend = jnp

            # The _init_additional_carry method needs the model and data as inputs.
            carry = env._init_additional_carry(key, model, data, backend)

            # Pass the specific rand_config to the randomizer's reset method.
            # This method returns the updated data and carry objects.
            env._domain_randomizer.rand_conf= rand_config
            data, carry = env._domain_randomizer.reset(env, model, data, carry, backend)#, rand_conf=rand_config)

            # Continue with the rest of the mjx_reset logic using the updated data and carry.
            data, carry = env.obs_container.reset_state(env, env._model, data, carry, jnp)
            obs, carry = env._mjx_create_observation(env._model, data, carry)
            reward = 0.0
            absorbing = jnp.array(False, dtype=bool)
            done = jnp.array(False, dtype=bool)
            info = env._mjx_reset_info_dictionary(obs, data, subkey)
            
            return MjxState(data=data, observation=obs, reward=reward, absorbing=absorbing, done=done, info=info, additional_carry=carry)

        # Vectorize the single-environment reset function
        vmapped_reset_fn = jax.vmap(single_reset_with_rand, in_axes=(0, 0))
        
        # Split a single key into a batched key for each environment.
        n_envs = jax.tree_util.tree_leaves(batched_rand_configs)[0].shape[0]
        batched_keys = jax.random.split(key, n_envs)
        
        return vmapped_reset_fn(batched_keys, batched_rand_configs)
    
    return batched_reset

def make_step_fn(env):
    vmapped_step_fn = jax.vmap(env.mjx_step, in_axes=(None, 0, 0))
    @jax.jit
    def batched_step(state, action):
        return vmapped_step_fn(env, state, action)
    return batched_step

def process_muscle_actions(actions, muscle_actuator_mask):
    return actions

nu = main_env.get_model().nu
global muscle_actuator_mask
muscle_actuator_mask = jnp.zeros(nu, dtype=jnp.bool_)
for i in range(nu):
    if main_env.get_model().actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE:
        muscle_actuator_mask = muscle_actuator_mask.at[i].set(True)

step_fn = make_step_fn(main_env)

@partial(jax.jit, static_argnums=(1,))
def parallel_evaluation_step(carry, unused_input, n_envs):
    env_state, train_state, rng_eval = carry
    obs = env_state.observation
    rng_eval, action_rng = jax.random.split(rng_eval)
    action_rngs = jax.random.split(action_rng, n_envs)
    actions, train_state = sample_actions(train_state, obs, action_rngs)
    processed_actions = process_muscle_actions(actions, muscle_actuator_mask)
    env_state = step_fn(env_state, processed_actions)
    new_carry = (env_state, train_state, rng_eval)
    step_metrics = {
        'actions': actions,
        'processed_actions': processed_actions,
        'qpos': env_state.data.qpos,
        'qvel': env_state.data.qvel,
        'contact_force': env_state.data.contact.force,
    }
    return new_carry, step_metrics

def run_parallel_evaluation(batched_env_state, train_state, rng_eval, n_envs):
    initial_carry = (batched_env_state, train_state, rng_eval)
    step_fn_partial = partial(parallel_evaluation_step, n_envs=n_envs)
    final_carry, all_step_metrics = jax.lax.scan(
        step_fn_partial,
        initial_carry,
        None,
        length=n_steps
    )
    return all_step_metrics, final_carry[0]

def process_and_save_results(all_step_metrics, all_param_names, all_param_values, path):
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(path), f"{dt_str}_evaluation_results_parallel")
    os.makedirs(output_dir, exist_ok=True)
    n_envs = len(all_param_names)
    for i in range(n_envs):
        param_name = all_param_names[i]
        param_value = all_param_values[i]
        single_env_results = jax.tree_util.tree_map(lambda x: x[:, i], all_step_metrics)
        processed_results = {
            'param_name': param_name,
            'param_value': param_value,
            'total_steps': n_steps,
            'metrics': single_env_results
        }
        if isinstance(param_value, float):
            param_value_str = f"{param_value:.3f}".replace('.', '_')
        else:
            param_value_str = str(param_value)
        file_name = f"eval_results_{param_name}_{param_value_str}_{n_steps}steps.pkl"
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, "wb") as f:
            pickle.dump(processed_results, f)
        print(f"Saved evaluation data for {param_name}={param_value} to {output_path}")

def main():
    time_start = timeit.default_timer()
    
    all_rand_configs, all_param_names, all_param_values = generate_randomization_configs()
    n_envs = len(all_rand_configs)
    
    if n_envs == 0:
        print("No randomization configurations generated. Exiting.")
        sys.exit()

    print(f"Total number of environments to run in parallel: {n_envs}")
    
    # We use a batch of configurations as input to the vmapped function.
    batched_rand_configs = batch_pytree(all_rand_configs)
    
    print("Initializing batched environment state...")
    rng = jax.random.key(0)
    rng, reset_key = jax.random.split(rng)
    
    # The factory.make call already sets the domain_randomization_params,
    # which is a static property of the environment object. We do not need
    # to manually set `rand_conf` here as it's passed dynamically.
    reset_fn = make_batched_reset_fn(main_env)
    
    # Pass the batched configs to the reset function.
    batched_env_state = reset_fn(main_env, reset_key, batched_rand_configs)
    
    # This is the call to run the simulation loop
    print("Starting parallel evaluation...")
    all_step_metrics, final_batched_env_state = run_parallel_evaluation(batched_env_state, train_state, rng, n_envs)
    
    print("Evaluation completed!")
    
    # The following line is needed if you want to save the results.
    # process_and_save_results(all_step_metrics, all_param_names, all_param_values, path)
    
    main_env.stop()
    
    time_end = timeit.default_timer()
    print(f"Total execution time: {time_end - time_start:.2f} seconds")

if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "egl"
    main()