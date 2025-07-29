import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 
# os.environ["JAX_PLATFORMS"] = "cpu"
import orbax.checkpoint as ocp
import jax
# jax.config.update('jax_platform_name', 'cpu')
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


# def load_checkpoint_callback(ckpt_path, train_state_template=None):
#     """Load checkpoint for continuing training or testing"""
    
#     # Method 1: Load with CheckpointManager (recommended)
#     check_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=False)
    
#     with ocp.CheckpointManager(ckpt_path, options=check_options, item_names=('agent_state',)) as mngr:
#         # Get the latest step or specify a particular step
#         latest_step = mngr.latest_step()
#         print(f"Loading checkpoint from step: {latest_step}")
        
#         # Load the checkpoint with device transfer and restore options
#         restore_args = ocp.args.StandardRestore()
        
#         # Handle device mismatch by transferring to current devices
#         if train_state_template is not None:
#             restore_args = ocp.args.StandardRestore(train_state_template)
        
#         restored = mngr.restore(
#             latest_step,
#             args=ocp.args.Composite(
#                 agent_state=restore_args,
#             ),
#             # Add restore kwargs to handle device mismatch
#             restore_kwargs={'restore_args': restore_args}
#         )
        
#         return restored['agent_state']

# def load_specific_checkpoint(ckpt_path, step):
#     """Load a specific checkpoint by step number"""
    
#     check_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=False)
    
#     with ocp.CheckpointManager(ckpt_path, options=check_options, item_names=('agent_state',)) as mngr:
#         print(f"Loading checkpoint from step: {step}")
        
#         restored = mngr.restore(
#             step,
#             args=ocp.args.Composite(
#                 agent_state=ocp.args.StandardRestore(),
#             )
#         )
        
#         return restored['agent_state']

# def load_checkpoint_with_template(ckpt_path, train_state_template):
#     """Load checkpoint with a template train_state for shape/structure reference"""
    
#     check_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=False)
    
#     # Create template for restoration
#     template = {
#         'params': train_state_template.params,
#         'run_stats': train_state_template.run_stats, 
#         'step': train_state_template.step,
#         'opt_state': train_state_template.opt_state,
#     }
    
#     with ocp.CheckpointManager(ckpt_path, options=check_options, item_names=('agent_state',)) as mngr:
#         latest_step = mngr.latest_step()
#         print(f"Loading checkpoint from step: {latest_step}")
        
#         restored = mngr.restore(
#             latest_step,
#             args=ocp.args.Composite(
#                 agent_state=ocp.args.StandardRestore(template),
#             )
#         )
        
#         return restored['agent_state']

# # RECOMMENDED USAGE for your case:
# def your_loading_solution():
#     """
#     This is what you should use based on your error
#     You MUST provide a train_state_template to avoid the device error
#     """
    
#     # 1. First, create a fresh train_state template with the same structure
#     # This should match exactly what you used during training
#     # Example (adapt to your actual train_state structure):
    
#     # Initialize your model, optimizer, etc. exactly as in training
#     # model = YourModel(...)  
#     # optimizer = YourOptimizer(...)
#     # dummy_params = model.init(...)
    
#     # Create template train_state (adapt this to your TrainState class)
#     # train_state_template = TrainState(
#     #     params=dummy_params,
#     #     run_stats=dummy_run_stats,  
#     #     step=0,
#     #     opt_state=optimizer.init(dummy_params)
#     # )
    
#     ckpt_path = "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-07-24/19-20-30/checkpoints/ckpt_3073"
    
#     # 2. Load using the template (THIS FIXES YOUR ERROR)
#     # loaded_state = load_checkpoint_with_device_fix(ckpt_path, train_state_template)
    
#     # 3. OR use the PyTree method
#     # loaded_state = load_checkpoint_pytree_method(ckpt_path, train_state_template)
    
#     print("Checkpoint loaded successfully!")
#     # return loaded_state

# # Debug function to check what's in your checkpoint
# def debug_checkpoint_contents(ckpt_path):
#     """Debug function to see what's actually saved in your checkpoint"""
#     import os
    
#     print("Checkpoint directory contents:")
#     for root, dirs, files in os.walk(ckpt_path):
#         level = root.replace(ckpt_path, '').count(os.sep)
#         indent = ' ' * 2 * level
#         print(f"{indent}{os.path.basename(root)}/")
#         subindent = ' ' * 2 * (level + 1)
#         for file in files:
#             print(f"{subindent}{file}")
    
#     # Try to get checkpoint info
#     try:
#         check_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=False)
#         with ocp.CheckpointManager(ckpt_path, options=check_options, item_names=('agent_state',)) as mngr:
#             print(f"Available steps: {mngr.all_steps()}")
#             print(f"Latest step: {mngr.latest_step()}")
#     except Exception as e:
#         print(f"Error reading checkpoint metadata: {e}")

# SOLUTION for your specific error:
def load_checkpoint_with_device_fix(ckpt_path, train_state_template):
    """
    Load checkpoint with proper device handling and target tree
    This fixes the TFRT_CPU_0 device error and missing target tree warning
    """
    import jax
    
    # Create proper restore arguments with target template
    template = {
        'params': train_state_template.params,
        'run_stats': train_state_template.run_stats, 
        'step': train_state_template.step,
        'opt_state': train_state_template.opt_state,
    }
    
    check_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=False)
    
    with ocp.CheckpointManager(ckpt_path, options=check_options, item_names=('agent_state',)) as mngr:
        latest_step = mngr.latest_step()
        print(f"Loading checkpoint from step: {latest_step}")
        
        # Use StandardRestore with template to avoid device mismatch
        restored = mngr.restore(
            latest_step,
            args=ocp.args.Composite(
                agent_state=ocp.args.StandardRestore(template),
            )
        )
        
        # Transfer to current devices if needed
        loaded_state = restored['agent_state']
        
        # Ensure all arrays are on current devices
        loaded_state = jax.tree_util.tree_map( #jax.tree_map(
            lambda x: jax.device_put(x) if hasattr(x, 'device') else x,
            loaded_state
        )
        
        return loaded_state


# def load_checkpoint_with_device_fix_state_conf(ckpt_path, train_state_template, agent_conf_template):
#     """
#     Load checkpoint with proper device handling and target tree
#     This fixes the TFRT_CPU_0 device error and missing target tree warning
#     """
#     import jax
    
#     # Create proper restore arguments with target template
#     template_state = {
#         'params': train_state_template.params,
#         'run_stats': train_state_template.run_stats, 
#         'step': train_state_template.step,
#         'opt_state': train_state_template.opt_state,
#         'network': train_state_template.network,
#     }

#     template_conf = {
#         'experiment': agent_conf_template.experiment,
#         'control_config': agent_conf_template.control_config,
#         'randomization_config': agent_conf_template.randomization_config,
#     }


    
#     check_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=False)
    
#     with ocp.CheckpointManager(ckpt_path, options=check_options, item_names=('agent_state',)) as mngr:
#         latest_step = mngr.latest_step()
#         print(f"Loading checkpoint from step: {latest_step}")
        
#         # Use StandardRestore with template to avoid device mismatch
#         restored = mngr.restore(
#             latest_step,
#             args=ocp.args.Composite(
#                 agent_state=ocp.args.StandardRestore(template_state),
#                 agent_conf=ocp.args.StandardRestore(template_conf),
#             )
#         )
        
#         # Transfer to current devices if needed
#         loaded_state = restored['agent_state']
#         loaded_conf = restored['agent_conf']
        
#         # Ensure all arrays are on current devices
#         loaded_state = jax.tree_util.tree_map( #jax.tree_map(
#             lambda x: jax.device_put(x) if hasattr(x, 'device') else x,
#             loaded_state
#         )

#         loaded_conf = jax.tree_util.tree_map( #jax.tree_map(
#             lambda x: jax.device_put(x) if hasattr(x, 'device') else x,
#             loaded_conf
#         )
        
#         return loaded_state, loaded_conf



# # Alternative method using PyTreeCheckpointer directly
# def load_checkpoint_pytree_method(ckpt_path, train_state_template):
#     """Alternative method using PyTreeCheckpointer"""
    
#     # Create the exact template structure
#     template = {
#         'params': train_state_template.params,
#         'run_stats': train_state_template.run_stats, 
#         'step': train_state_template.step,
#         'opt_state': train_state_template.opt_state,
#     }
    
#     # Use PyTreeCheckpointer directly
#     checkpointer = ocp.PyTreeCheckpointer()
    
#     # Construct the full path to the agent_state
#     agent_state_path = f"{ckpt_path}/ckpt_3073/agent_state"
    
#     # Restore with template
#     loaded_state = checkpointer.restore(
#         agent_state_path,
#         item=template
#     )
    
#     return loaded_state




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

# Use the path from command line arguments
path = args.conf_path
agent_conf, agent_state = SavePPOJax.load_agent(path)
config = agent_conf.config
# config.experiment.env_params.reward_params.joint_torque_coeff=0.003 #5 #2 #0.001 #0.01
# config.experiment.env_params.reward_params.sites_for_mimic = "upper_body_mimic"
# get task factory
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



# extract static agent info
config, network, tx =\
    (agent_conf.config.experiment, agent_conf.network, agent_conf.tx)

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
ckpt_path = args.state_path #'/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-07-24/19-20-30/checkpoints/ckpt_3073/'
#"ckpt_3073"  # Base path to your checkpoint
loaded_state = SavePPOJax.load_checkpoint_with_device_fix(ckpt_path, train_state_template) #load_checkpoint_callback(ckpt_path)

# Extract the components
params = loaded_state['params']
run_stats = loaded_state['run_stats'] 
current_step = loaded_state['step']  # This should be 3073
opt_state = loaded_state['opt_state']


# # Reconstruct your train_state (adapt this to your TrainState structure)
# train_state = TrainState(
#     params=params,
#     run_stats=run_stats,
#     step=current_step,
#     opt_state=opt_state
# )

train_state= TrainState.create(
    apply_fn=network.apply,
    params=params,
    run_stats=run_stats,
    tx=tx,
)
# Determine which evaluation environment to run
if args.use_mujoco:
    # run eval mujoco
    SavePPOJax.play_policy_mujoco(env, agent_conf, train_state, deterministic=False, n_steps=1000, record=True,
                              train_state_seed=0)
else:
    # run eval mjx
    SavePPOJax.play_policy(env, agent_conf, train_state, deterministic=False, n_steps=1000, n_envs=1, record=True,
                       train_state_seed=0)



