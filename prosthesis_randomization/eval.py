import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"

import jax 
jax.config.update('jax_platform_name', 'cpu')

import argparse


from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax

from omegaconf import OmegaConf

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
# JOINT TORQUE TEST 
# config.experiment.env_params.reward_params.joint_torque_coeff=0.003 #5 #2 #0.001 #0.01
config.experiment.env_params.reward_params.sites_for_mimic = "upper_body_mimic"

# config.experiment.env_params.reward_type="MimicRewardCross"
# config.experiment.env_params.reward_params.foot_cross_coeff=0.002

# config.experiment.env_params.reward_type="MimicRewardVel" 
# config.experiment.env_params.reward_params.joint_torque_vel_coeff=0.003 #0.002

config.experiment.env_params.reward_type="MimicRewardVelArm" 
# config.experiment.env_params.reward_params.joint_torque_arm_coeff=0.003 
# config.experiment.env_params.reward_params.joint_torque_nonarm_coeff=0.002 
# config.experiment.env_params.reward_params.joint_torque_vel_arm_coeff=0.004 #0.003 
# config.experiment.env_params.reward_params.joint_torque_vel_nonarm_coeff=0.0015 #0.002 
config.experiment.env_params.reward_params.action_coeff=0.03 #0.002
config.experiment.env_params.reward_params.lateral_range_coeff = 0.5 
config.experiment.env_params.reward_params.lateral_pos_reward_range = 0.3 


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


# Determine which evaluation environment to run
if args.use_mujoco:
    # run eval mujoco
    PPOJax.play_policy_mujoco(env, agent_conf, agent_state, deterministic=False, n_steps=1000, record=True,
                              train_state_seed=0)
else:
    # run eval mjx
    PPOJax.play_policy(env, agent_conf, agent_state, deterministic=False, n_steps=1000, n_envs=1, record=True,
                       train_state_seed=0)

# # save the video
# if env.video_recorder is not None:
#     video_path = os.path.join(os.path.dirname(path), "eval_video.mp4")
#     env.video_recorder.save(video_path)
#     print(f"Video saved to {video_path}")