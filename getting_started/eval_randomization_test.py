import os
import argparse

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax

from omegaconf import OmegaConf

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

# Use the path from command line arguments
path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config

# Print the expected observation shape
print(f"Agent expects observation shape: {agent_conf.network.actor_obs_ind.shape}")


# randomization_config = {
#     'randomize_prosthesis_dof_damping': False, #True,
#     'prosthesis_dof_damping_range': {'ankle_angle': [2, 10]}, 

#     'randomize_prosthesis_joint_stiffness': False, #True, 

#     'prosthesis_joint_stiffness_range': {'ankle_angle': [50, 100]}, 

#     'randomize_prosthesis_body_position': False, #True, #True, #True,
#     'prosthesis_body_position_range': {'pylon_socket': {'x': [-0,-0]}},
#     # 'prosthesis_body_position_range': {'pylon_socket': {'x': [-0.01,0.01],'z': [-0.01,0.01]},'talus': {'x': [-0.01,0.01],'z': [-0.01,0.01]}},

#     'randomize_prosthesis_body_orientation': False, #True, #False, #True,
#     'prosthesis_body_orientation_range': {'pylon_socket': {'z': [-0.5,-0.5]}},
#     # 'prosthesis_body_orientation_range': {'pylon_socket': {'x': [-0.12,0.12],'y': [-0.2,0.2],'z': [-0.12,0.12]},'talus': {'x': [-0.1,0.1],'z': [-0.1,0.1]}},
      
# }


env_params = config.experiment.env_params
# config.experiment.env_params = env_params
config.experiment.n_seeds = 1


randomization_type = "ProsthesisRandomizer"
task_factory_params = {
    "default_dataset_conf": {
        "task": "walk"
    }
}


# get task factory
# factory = TaskFactory.get_factory_cls("ImitationFactory")
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
# factory = TaskFactory.get_factory_cls("ImitationFactory")

# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic


env = factory.make(#domain_randomization_type=randomization_type, domain_randomization_params=randomization_config,
                # terrain_type="RoughTerrain", terrain_params=dict(random_min_height=-0.05, random_max_height=0.05,), 
            **env_params, **task_factory_params)

print(f"Environment provides observation shape: {env.info.observation_space.shape}")

# env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

# Determine which evaluation environment to run
if args.use_mujoco:
    # run eval mujoco
    PPOJax.play_policy_mujoco(env, agent_conf, agent_state, deterministic=False, n_steps=10000, record=True,
                              train_state_seed=0)
else:
    # run eval mjx
    PPOJax.play_policy(env, agent_conf, agent_state, deterministic=False, n_steps=10000, n_envs=1, record=True,
                       train_state_seed=0)
