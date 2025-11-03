import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"

import jax 
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import tree_util

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

from loco_mujoco.algorithms import SavePPOJax

from flax.training.train_state import TrainState # Make sure this import is correct
from loco_mujoco.algorithms.common.dataclasses import TrainState #as YourActualTrainStateClass # Or your actual class


os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering, which is more compatible with headless environments
# os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_triton_gemm_any=True ')

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--conf_path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--ckpt_folder', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

config_checkpoint_path = args.conf_path
# agent_conf, agent_state = SavePPOJax.load_agent(config_checkpoint_path)
agent_conf= SavePPOJax.load_agent_conf(config_checkpoint_path)
config = agent_conf.config
network = agent_conf.network
tx = agent_conf.tx 
checkpoint_folder = args.ckpt_folder
checkpoint_path = checkpoint_folder


last_checkpoint_index = checkpoint_folder.rfind('/checkpoints/')
if last_checkpoint_index != -1:
    checkpoint_base_path = checkpoint_folder[:last_checkpoint_index]
else:
    # Handle the case where the substring is not found, 
    # perhaps by setting it to the whole folder or raising an error.
    checkpoint_base_path = checkpoint_folder

checkpoint_base_path = checkpoint_base_path + '/'


# Loop over all checkpoints

# Assuming checkpoint number is in the folder name like ckpt_20, ckpt_50
checkpoint_number = checkpoint_folder.split('_')[-1]
print(f"Evaluating checkpoint: {checkpoint_number}")

# Load agent and config from orbax

# # get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# # create env
# OmegaConf.set_struct(config, False)  # Allow modifications
# config.experiment.env_params["headless"] = True #False
# config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic
# config.experiment.env_params["add_sensors"] = True
# # add prosthesis side to randomization params if it exists in config.experiment.env_params
# # randomization_params_dict = OmegaConf.to_container(config.randomization_config.randomization_params, resolve=True)

# # if "prosthesis_side" in config.experiment.env_params:
# #     randomization_params_dict["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]

env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params,
                #    domain_randomization_type=config.randomization_config.randomization_type, domain_randomization_params=randomization_params_dict)
                #    domain_randomization_type=config.randomization_config.randomization_type, domain_randomization_params=config.randomization_config.randomization_params)
)
# env.th.to_jax()
# env = VecEnv(env)
# jit_step  = jax.jit(jax.vmap(env.mjx_step))  #env.step)
# jit_reset  = jax.jit(jax.vmap(env.mjx_reset)) #env.reset)
# model = env.get_model()

# prosthesis_metrics_handler = ProsthesisMetricsHandler(env) #(config, env)

n_steps = 1000 #1000 #1000
n_envs = 1 #1  # <--- Make sure this matches your training batch size
rng = jax.random.key(0)
train_state_seed = 0  # Take first seed 


keys = jax.random.split(rng, n_envs + 1)
rng, env_keys = keys[0], keys[1:]

rng = jax.random.PRNGKey(seed=0)
rng, _rng1, _rng2 = jax.random.split(rng, 3)
init_x = jnp.zeros(env.info.observation_space.shape)
network_params = network.init(_rng1, init_x)
train_state=None
# init new train states from old params
train_state_template = TrainState.create(
    apply_fn=network.apply,
    params=network_params["params"] if train_state is None else train_state.params,
    run_stats=network_params["run_stats"] if train_state is None else train_state.run_stats,
    tx=tx,
)
# To continue training from your saved checkpoint:
loaded_state = SavePPOJax.load_checkpoint_with_device_fix(checkpoint_path, train_state_template) #load_checkpoint_callback(ckpt_path)

# Extract the components
params = loaded_state['params']
run_stats = loaded_state['run_stats'] 
current_step = loaded_state['step']  # This should be 3073
opt_state = loaded_state['opt_state']


train_state= TrainState.create(
    apply_fn=network.apply,
    params=params,
    run_stats=run_stats,
    tx=tx,
)

# # def _leaf_info(x):
# #     t = type(x)
# #     shp = getattr(x, "shape", None)
# #     return (t, shp)

# # infos = tree_util.tree_map(_leaf_info, train_state)
# # print(infos)            # structured map
# # leaves = tree_util.tree_leaves(train_state)
# # for i, leaf in enumerate(leaves):
# #     print(i, type(leaf), getattr(leaf, "shape", None))


# def sample_actions_uncompiled(ts, obs, _rng): # Renamed for clarity
#     y, updates = agent_conf.network.apply({'params': ts.params,
#                                         'run_stats': ts.run_stats},
#                                         obs, mutable=["run_stats"])
#     ts = ts.replace(run_stats=updates['run_stats'])  # update stats
#     pi, _ = y
#     a = pi.sample(seed=_rng)
#     return a, ts

# # JIT compile the function
# # sample_actions = jax.jit(sample_actions_uncompiled) # <--- ADD THIS LINE
# sample_actions = jax.jit(sample_actions_uncompiled)
# env_state = jit_reset(env_keys) #env.reset(env_keys)
# obs = env_state.observation
# # obs, env_state = jit_reset(env_keys) #env.reset(env_keys)
# # train_state = agent_state.train_state

step_total = 0

leaves, treedef = tree_util.tree_flatten(train_state)
print("num leaves:", len(leaves))
for i, leaf in enumerate(leaves):
    print(i, type(leaf), getattr(leaf, "shape", repr(leaf)))

if config.experiment.n_seeds > 1:
    assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
                                            "train_state_seed for replay.")

    # Safe selector: only index array-like leaves that have a leading seed dimension.
    def _select_seed_leaf(x):
        try:
            shape = getattr(x, 'shape', None)
            if shape is not None and len(shape) > 0:
                # If first dim larger than train_state_seed, index it
                if shape[0] > train_state_seed:
                    return x[train_state_seed]
        except Exception:
            pass
        return x

    train_state = jax.tree_util.tree_map(_select_seed_leaf, train_state)
    # train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)
else: 
    # obs, env_state = jit_reset(env_keys) #env.reset(env_keys)
    train_state = train_state


