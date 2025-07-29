import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"
import orbax.checkpoint as ocp
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from flax.training.train_state import TrainState # Make sure this import is correct
from loco_mujoco.algorithms.common.dataclasses import TrainState #as YourActualTrainStateClass # Or your actual class
import argparse
import hydra 

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import SavePPOJax
from loco_mujoco.utils import MetricsHandler

from omegaconf import OmegaConf
import wandb

from dataclasses import fields
from loco_mujoco.utils.metrics import QuantityContainer

os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering, which is more compatible with headless environments
# os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_triton_gemm_any=True ')

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--conf_path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--state_path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

# # SOLUTION for your specific error:
# def load_checkpoint_with_device_fix(ckpt_path, train_state_template):
#     """
#     Load checkpoint with proper device handling and target tree
#     This fixes the TFRT_CPU_0 device error and missing target tree warning
#     """
#     import jax
    
#     # Create proper restore arguments with target template
#     template = {
#         'params': train_state_template.params,
#         'run_stats': train_state_template.run_stats, 
#         'step': train_state_template.step,
#         'opt_state': train_state_template.opt_state,
#     }
    
#     check_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=False)
    
#     with ocp.CheckpointManager(ckpt_path, options=check_options, item_names=('agent_state',)) as mngr:
#         latest_step = mngr.latest_step()
#         print(f"Loading checkpoint from step: {latest_step}")
        
#         # Use StandardRestore with template to avoid device mismatch
#         restored = mngr.restore(
#             latest_step,
#             args=ocp.args.Composite(
#                 agent_state=ocp.args.StandardRestore(template),
#             )
#         )
        
#         # Transfer to current devices if needed
#         loaded_state = restored['agent_state']
        
#         # Ensure all arrays are on current devices
#         loaded_state = jax.tree_util.tree_map( #jax.tree_map(
#             lambda x: jax.device_put(x) if hasattr(x, 'device') else x,
#             loaded_state
#         )
        
#         return loaded_state
    

config_checkpoint_path = args.conf_path #"/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-07-25/03-32-34/SavePPOJaxagent_conf_saved.pkl"


agent_conf = SavePPOJax.load_agent_conf(config_checkpoint_path)
config = agent_conf.config
network = agent_conf.network
tx = agent_conf.tx 

factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic

randomization_type = config.randomization_config["randomization_type"]
# Convert to plain dict to allow adding new keys
randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)

# add prosthesis side to randomization params if it exists in config.experiment.env_params
if "prosthesis_side" in config.experiment.env_params:
    randomization_params["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]


env = factory.make(domain_randomization_type=randomization_type, domain_randomization_params=randomization_params,
                   **config.experiment.env_params, **config.experiment.task_factory.params)


# To continue training from your saved checkpoint:
ckpt_path = args.state_path
#"/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-07-24/19-20-30/checkpoints/ckpt_257"#"/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-07-25/03-32-34/checkpoints/ckpt_819200"
#args.state_path #'/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-07-24/19-20-30/checkpoints/ckpt_3073/'
#"ckpt_3073"  # Base path to your checkpoint

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

loaded_state = SavePPOJax.load_checkpoint_with_device_fix(ckpt_path, train_state_template) #load_checkpoint_callback(ckpt_path)

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
# Determine which evaluation environment to run
SavePPOJax.play_policy(env, agent_conf, train_state, deterministic=False, n_steps=1000, n_envs=1, record=True,
                       train_state_seed=0)


