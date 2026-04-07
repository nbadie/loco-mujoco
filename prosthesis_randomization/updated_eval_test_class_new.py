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


# # Parameter randomization initialization
# randomization_params_names = ["prosthesis_dof_damping", "prosthesis_joint_stiffness", "prosthesis_body_position", "prosthesis_body_orientation"]

# randomization_params_eval = {
#     "prosthesis_side": "left_side",
#     "randomize_prosthesis_dof_damping": False,
#     "prosthesis_dof_damping_range": {'ankle_angle':[2, 10]}, #, 'subtalar_angle', 'mtp_angle'],
#     # "prosthesis_dof_damping_range": [0, 10],
#     "randomize_prosthesis_joint_stiffness": False,
#     # "randomization_joint_names": ['ankle_angle', 'subtalar_angle', 'mtp_angle'],
#     "prosthesis_joint_stiffness_range":{'ankle_angle': [50, 100]}, #[],
#     "randomize_prosthesis_body_position": True,
#     "prosthesis_body_position_range": {'calcn': {'x': [0.0, 0.1]}}, #,'y': [0.4, 0.5]}}, #['calcn'],
#     # "prosthesis_body_position_range": {
#     #     'z': [0.6, 0.7]
#     # },
#     "randomize_prosthesis_body_orientation": False,
#     "prosthesis_body_orientation_range": {'pylon_socket': {'y': [-0.3,0.3]}}, #['calcn'],
#     # "prosthesis_body_orientation_range": {}

#     "randomize_prosthesis_socket_joint": False,
#     "socket_joint_range": {'socket_ty': [-0.25,0.025]},
# }

# randomization_increments = {
#     "prosthesis_joint_stiffness": 10,
#     "prosthesis_dof_damping": 5,
#     "prosthesis_body_position": 0.05, #0.002,
#     "prosthesis_body_orientation": 0.1
# }

subfolder_name = 'ckpt_199884800_pylon_socket_ori_x'# _TalZ5'
# subfolder_name = 'talus_ori_z' #_test_knee_lim' #_3000steps'

dt_str_init = datetime.now().strftime("%Y%m%d_%H%M%S")

talus_ori_unequal_0 = False #True #False #True


ori_ang_test= np.deg2rad(4)
ori_ang_test_2 = np.deg2rad(3)
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
    # "prosthesis_body_position_range": {'pylon_socket': {'z': [-0.010,0.010]}}, #{'z': [-0.0292, 0.0292]}}, #{'pylon_socket': {'x': [-0.1, 0.1]}},
    "prosthesis_body_position_range": {'pylon_socket': {'z': [0.005,0.010]}}, #{'z': [-0.0292, 0.0292]}}, #{'pylon_socket': {'x': [-0.1, 0.1]}},
    # "prosthesis_body_position_range": {'pylon_socket': {'x': [-0.010,-0.005]}}, #{'z': [-0.0292, 0.0292]}}, #{'pylon_socket': {'x': [-0.1, 0.1]}},
    # "prosthesis_body_position_range": {'pylon_socket': {'x': [-0.0147, 0.0147]}}, #{'z': [-0.0292, 0.0292]}}, #{'pylon_socket': {'x': [-0.1, 0.1]}},
    # "prosthesis_body_position_range": {'talus': {'z': [-0.015, 0.015]}}, #{'pylon_socket': {'x': [-0.1, 0.1]}},
    "randomize_prosthesis_body_orientation": True, #False, #True, #False, #True, #False, #True, #False,
    # "prosthesis_body_orientation_range": {'talus': {'z': [-talus_ori_ang, talus_ori_ang]}},
    # "prosthesis_body_orientation_range": {'talus': {'z': [talus_ori_ang, talus_ori_ang]}}
    # "prosthesis_body_orientation_range": {'pylon_socket': {'z': [-ori_ang, ori_ang]}, 'talus': {'z': [talus_ori_ang, talus_ori_ang]}}
    "prosthesis_body_orientation_range": {'pylon_socket': {'x': [ori_ang_test_2,ori_ang]}} 
}
randomization_increments = {
    "prosthesis_joint_stiffness": 100, #10,
    "prosthesis_dof_damping": 5,
    # "prosthesis_body_position": 0.01, #
    "prosthesis_body_position":0.005, #0.04, #0.05,
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

# config.experiment.env_params["knee_extension_limit"] = {'ankle_angle_r': [-13,7], 'ankle_angle_l': [-7,10], 'hip_flexion_r': [-18,25],'hip_flexion_l': [-18,35], 'knee_angle_r': [-120,0]}

randomization_type = config.randomization_config["randomization_type"]
randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)
config.experiment.env_params["horizon"] = 3000 
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

n_steps = 2000 #3000
n_envs = 1  # <--- Make sure this matches your training batch size
seed =  1 #2 #0 #0
rng = jax.random.key(seed)
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


# if config.experiment.n_seeds > 1:
#     assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
#                                             "train_state_seed for replay.")
    
#     train_state = jax.tree.map(lambda x: x[train_state_seed], agent_state.train_state)
# else: 
#     # obs, env_state = jit_reset(env_keys) #env.reset(env_keys)
#     train_state = agent_state.train_state

train_state = agent_state.train_state


# ###### Some params for evaluation 
# all_foot_ground_contact_left =[]
# all_foot_ground_contact_right =[]            


# joint_data = {}
# for i in range(model.njnt):
#     joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
#     joint_data[joint_name] = {
#         "angle": [],
#         "velocity": [],
#         "forces_constraint": [],
#         "forces_smooth": [],
#         "torques": [],
#         "energy_exp": [],
#     }
#     # Per-step data
#     if "_l" in joint_name or "_r" in joint_name:
#         joint_data[joint_name].update({
#             "angle_per_step": [],
#             "velocity_per_step": [],
#             "forces_constraint_per_step": [],
#             "forces_smooth_per_step": [],
#             "torques_per_step": [],
#             "energy_exp_per_step": [],
#         })
#     else:
#         joint_data[joint_name].update({
#             "angle_per_step_left": [],
#             "angle_per_step_right": [],
#             "velocity_per_step_left": [],
#             "velocity_per_step_right": [],
#             "forces_constraint_per_step_left": [],
#             "forces_constraint_per_step_right": [],
#             "forces_smooth_per_step_left": [],
#             "forces_smooth_per_step_right": [],
#             "torques_per_step_left": [],
#             "torques_per_step_right": [],
#             "energy_exp_per_step_left": [],
#             "energy_exp_per_step_right": [],
#         })


# # all_tibia_sensor_data = []
# all_sensor_force = {}


# evaluation_muscle_groups = ["back_muscles", "torso_muscles", "vasti_muscles", "rectus_muscles", "medial_muscles", "leg_muscles", "all_muscles"]


# evaluation_muscle_names = {}
# for n in evaluation_muscle_groups:
#     evaluation_muscle_names[n] = prosthesis_metrics_handler.get_muscle_group(n)


# for muscle_group, muscle_names in evaluation_muscle_names.items():
#     locals()[f"run_{muscle_group}_activation_left"] = []
#     locals()[f"run_{muscle_group}_activation_right"] = []


# # Body name with possible contact to ground 
# foot_name = "toes"  # name of the foot box in the model
# all_grf_l = []
# all_grf_r = []

# left_side = "left_side"
# right_side = "right_side"

# all_actions = []
# all_actuator_names = []


muscle_skeleton_control_activation = SkeletonMuscleControlFunction(env)

def run_evaluation_loop(env_state, n_steps, train_state, rng,prosthesis_metrics_handler, param_name=None, param_value=None, direction=None, subfolder_name=None, dt_str_init=None):
    step_total = 0

    ###### Some params for evaluation 
    all_foot_ground_contact_left =[]
    all_foot_ground_contact_right =[]   

    body_names = []
    for i in range(model.nbody):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        body_names.append(joint_name)        

    body_xposes = {}
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        body_xposes[body_name] = [] 


    joint_data = {}
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_data[joint_name] = {
            "angle": [],
            "velocity": [],
            "forces_constraint": [],
            "forces_smooth": [],
            "forces_applied": [] ,
            "torques": [],
            "energy_exp": [],
        }
        # Per-step data
        if "_l" in joint_name or "_r" in joint_name:
            joint_data[joint_name].update({
                "angle_per_step": [],
                "velocity_per_step": [],
                "forces_constraint_per_step": [],
                "forces_smooth_per_step": [],
                "forces_applied_per_step": [],
                "torques_per_step": [],
                "energy_exp_per_step": [],
            })
        else:
            joint_data[joint_name].update({
                "angle_per_step_left": [],
                "angle_per_step_right": [],
                "velocity_per_step_left": [],
                "velocity_per_step_right": [],
                "forces_constraint_per_step_left": [],
                "forces_constraint_per_step_right": [],
                "forces_smooth_per_step_left": [],
                "forces_smooth_per_step_right": [],
                "forces_applied_per_step_left": [],
                "forces_applied_per_step_right": [],
                "torques_per_step_left": [],
                "torques_per_step_right": [],
                "energy_exp_per_step_left": [],
                "energy_exp_per_step_right": [],
            })


    # all_tibia_sensor_data = []
    all_sensor_force = {}


    evaluation_muscle_groups = ["back_muscles", "torso_muscles", "vasti_muscles", "rectus_muscles", "medial_muscles", "leg_muscles", "all_muscles"]


    evaluation_muscle_names = {}
    for n in evaluation_muscle_groups:
        evaluation_muscle_names[n] = prosthesis_metrics_handler.get_muscle_group(n)


    for muscle_group, muscle_names in evaluation_muscle_names.items():
        locals()[f"run_{muscle_group}_activation_left"] = []
        locals()[f"run_{muscle_group}_activation_right"] = []


    # Body name with possible contact to ground 
    foot_name = "toes"  # name of the foot box in the model
    calcn_name = "calcn"
    all_grf_l = []
    all_grf_r = []

    left_side = "left_side"
    right_side = "right_side"

    all_actions = []
    all_actuator_names = []


    # muscle_skeleton_control_activation = SkeletonMuscleControlFunction(env)

    for i in range(n_steps):
        obs = env_state.observation
        rng, _rng = jax.random.split(rng)
        action, train_state = sample_actions(train_state, obs, _rng) # Now calls the JITted version
        action = jnp.atleast_2d(action)


        # env_state, sys = jit_step(env_state, action)  #env.step(env_state, action)
        env_state = jit_step(env_state, action)  #env.step(env_state, action)
        obs = env_state.observation

        # print('pylon_socket xpos: ', env_state.data.body('pylon_socket_l').xpos )
        # print('pylon_socket xipos: ', env_state.data.body('pylon_socket_l').xipos)
        # print('pylon_socket xquat: ', env_state.data.body('pylon_socket_l').xquat)
        # print('pylon_socket xmat: ', env_state.data.body('pylon_socket_l').xmat)

        # print('torso xpos: ', env_state.data.body('torso').xpos )
        # print('torso xipos: ', env_state.data.body('torso').xipos)
        # print('torso xquat: ', env_state.data.body('torso').xquat)
        # print('torso xmat: ', env_state.data.body('torso').xmat)

        # if i == 0:
            # print(f"sys.jnt_stiffness: {sys.jnt_stiffness}")
            # print(f"sys.dof_damping: {sys.dof_damping}")
            # print(f"sys.body_pos: {sys.body_pos}")
            # print(f"sys.body_quat: {sys.body_quat}")
            # print(f"Step {i}")
        # env.mjx_render_domain_randomization(env_state)#, record=True)

        # if step_total % 100 == 0:
        #     print(f"Step {step_total}")


        # # # # # # # # Collect metrics 
        
        # Foot ground contact indices 
        contact_left, contact_right = prosthesis_metrics_handler.get_contact_steps(env_state.data, i)
        #print("contact_left: ", contact_left)
        #print("contact_right: ", contact_right)
        # if contact_left not equal [] then append to list

        all_foot_ground_contact_left.append(contact_left)
        all_foot_ground_contact_right.append(contact_right)

        # GRF 
        grf_foot_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_l")
        grf_foot_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_r")
    
        grf_calcn_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{calcn_name}_l")
        grf_calcn_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{calcn_name}_r")
        
        grf_l = grf_foot_l + grf_calcn_l
        grf_r = grf_foot_r + grf_calcn_r

        all_grf_l.append(grf_l)
        all_grf_r.append(grf_r)

        # Body position
        body_xpos = prosthesis_metrics_handler.get_xpos(env_state.data)
        for name in body_names: 
            # for body_name, pos in body_xpos[name]:
            body_xposes[name].append(body_xpos[name])


        # Joint data
        joint_angles = prosthesis_metrics_handler.get_joint_angles(env_state.data)
        joint_velocities = prosthesis_metrics_handler.get_joint_vels(env_state.data)
        joint_forces_constraint, joint_forces_smooth, joint_forces_applied  = prosthesis_metrics_handler.get_joint_frces(env_state.data)
        joint_torques = prosthesis_metrics_handler.get_joint_trques(env_state.data)
        joint_energy_exp = prosthesis_metrics_handler.calc_joint_energy_exp(joint_torques, joint_velocities)
        # Append all joint data to the joint_data dictionary
        for joint_name, angle in joint_angles.items():
            joint_data[joint_name]["angle"].append(angle)
        for joint_name, velocity in joint_velocities.items():
            joint_data[joint_name]["velocity"].append(velocity)
        for joint_name, force in joint_forces_constraint.items():
            joint_data[joint_name]["forces_constraint"].append(force)
        for joint_name, force in joint_forces_smooth.items():
            joint_data[joint_name]["forces_smooth"].append(force)
        for joint_name, force in joint_forces_applied.items():
            joint_data[joint_name]["forces_applied"].append(force)
        for joint_name, torque in joint_torques.items():
            joint_data[joint_name]["torques"].append(torque)
        for joint_name, energy_exp in joint_energy_exp.items():
            joint_data[joint_name]["energy_exp"].append(energy_exp)



        #  Tibia sensor data 
        sensor_force, sensor_force_names = prosthesis_metrics_handler.get_sensor_data(env_state.data)
        for sensor_name, force in sensor_force.items():
            # Initialize the list if it doesn't exist
            if sensor_name not in all_sensor_force:
                all_sensor_force[sensor_name] = []
            # Append the sensor force data to the corresponding list
            all_sensor_force[sensor_name].append(force)



        # # Action data 
        # print(f"Step: {i}, Action: {action}")

        for i in range(model.nu):
            if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE: #Muscle
                # jax.debug.print("Actuator {i} is a Muscle", i=i)
                # apply sigmoid activation function for muscle control
                # action = action.at[i].set(jax.nn.sigmoid(action[i]))
                action = action.at[...,i].set(muscle_skeleton_control_activation.adapted_sigmoid(action[...,i]))

        # print(f"Step: {i}, Action after sigmoid: {action}")

        all_actions.append(action)

        all_actuator_names = []
        for a in range(model.nu):
            # get all actuator_names
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            all_actuator_names.append(actuator_name)


        for muscle_group, muscle_names in evaluation_muscle_names.items():
            locals()[f"{muscle_group}_activation_left"] = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, left_side, action)
            locals()[f"{muscle_group}_activation_right"] = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, right_side, action)
            locals()[f"run_{muscle_group}_activation_left"].append(locals()[f"{muscle_group}_activation_left"])
            locals()[f"run_{muscle_group}_activation_right"].append(locals()[f"{muscle_group}_activation_right"])



        step_total += n_envs 

        # env.mjx_render_domain_randomization(env_state, record=True)

    time_all.append(timeit.default_timer())  # End timer

    env.stop()



    # ACTIONS 
    sum_step_activations = {}
    step_norm_activations = {}
    step_num_norm_activations = {}
    sum_muscles_activations = {}
    step_muscles_norm_activations = {}
    step_num_norm_activations = {}

    for muscle_group in evaluation_muscle_groups:
        sum_step_activations[f"{muscle_group}_left"] = jnp.sum(jnp.array(locals()[f"run_{muscle_group}_activation_left"]), axis=0)
        sum_step_activations[f"{muscle_group}_right"] = jnp.sum(jnp.array(locals()[f"run_{muscle_group}_activation_right"]), axis=0)

        step_norm_activations[f"{muscle_group}_left"] = sum_step_activations[f"{muscle_group}_left"] / n_steps
        step_norm_activations[f"{muscle_group}_right"] = sum_step_activations[f"{muscle_group}_right"] / n_steps

        # print(f"Sum of {muscle_group.replace('_', ' ').title()} Activation Left Per Muscle: {sum_step_activations[f'{muscle_group}_left']}")
        # print(f"Sum of {muscle_group.replace('_', ' ').title()} Activation Right Per Muscle: {sum_step_activations[f'{muscle_group}_right']}")

        # print(f"Step normalized {muscle_group.replace('_', ' ').title()} Activation Left Per Muscle: {step_norm_activations[f'{muscle_group}_left']}")
        # print(f"Step normalized {muscle_group.replace('_', ' ').title()} Activation Right Per Muscle: {step_norm_activations[f'{muscle_group}_right']}")

        # Get amount of muscles in the group
        n_musc = len(evaluation_muscle_names[muscle_group])
        # get sum along axis=0 and 1    
        sum_muscles_activations[f"{muscle_group}_left"] = jnp.sum(sum_step_activations[f"{muscle_group}_left"], axis=0)
        sum_muscles_activations[f"{muscle_group}_right"] = jnp.sum(sum_step_activations[f"{muscle_group}_right"], axis=0)
        step_muscles_norm_activations[f"{muscle_group}_left"] = sum_muscles_activations[f"{muscle_group}_left"] / n_steps
        step_muscles_norm_activations[f"{muscle_group}_right"] = sum_muscles_activations[f"{muscle_group}_right"] / n_steps
        step_num_norm_activations[f"{muscle_group}_left"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_left"]/ n_musc), #axis=0)
        step_num_norm_activations[f"{muscle_group}_right"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_right"]/ n_musc) #, axis=0)

        # print(f"Sum of {muscle_group.replace('_', ' ').title()} Muscles Activation Left: {sum_muscles_activations[f'{muscle_group}_left']}")
        # print(f"Sum of {muscle_group.replace('_', ' ').title()} Muscles Activation Right: {sum_muscles_activations[f'{muscle_group}_right']}")
        # print(f"Step normalized {muscle_group.replace('_', ' ').title()} Muscles Activation Left: {step_muscles_norm_activations[f'{muscle_group}_left']}")
        # print(f"Step normalized {muscle_group.replace('_', ' ').title()} Muscles Activation Right: {step_muscles_norm_activations[f'{muscle_group}_right']}")




    joint_names = []
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_names.append(joint_name)

    body_names = []
    for i in range(model.nbody):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        body_names.append(joint_name)

    # Collect all relevant data into a dictionary
    all_relevant_data = {
        "total_steps": step_total,
        "n_envs": n_envs,
        "all_grf_l": all_grf_l,
        "all_grf_r": all_grf_r,
        "all_foot_ground_contact_left": all_foot_ground_contact_left,
        "all_foot_ground_contact_right": all_foot_ground_contact_right,
        "all_sensor_force": all_sensor_force,
        "sensor_force_names": sensor_force_names,
        # "all_tibia_sensor_data": all_tibia_sensor_data,
        # "filtered_step_start_left": filtered_step_start_left,
        # "filtered_step_start_right": filtered_step_start_right,
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
        "evaluation_body_names": body_names,
    }


    # Add joint data from joint_data dictionary
    for joint_name, joint_dict in joint_data.items():
        for key, value in joint_dict.items():
            all_relevant_data[f"{joint_name}_{key}"] = value



    # Add muscle activations
    for muscle_group in evaluation_muscle_groups:
        all_relevant_data[f"run_{muscle_group}_activation_left"] = locals().get(f"run_{muscle_group}_activation_left", [])
        all_relevant_data[f"run_{muscle_group}_activation_right"] = locals().get(f"run_{muscle_group}_activation_right", [])

    # Save to file
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "param_name":
        joint_name = list(randomization_params_eval[f"{param_name}_range"].keys())[0]
        if 'orientation' in param_name:
            name_param_value = np.rad2deg(param_value)
            name_units = 'deg'
        elif 'position' in param_name:
            name_param_value = param_value*1000
            name_units = 'mm'
        else:
            name_param_value = param_value
            name_units = ''
        if subfolder_name is not None: 
            if not os.path.exists(os.path.join(os.path.dirname(path), f"{dt_str_init}_{subfolder_name}")):
                os.makedirs(os.path.join(os.path.dirname(path), f"{dt_str_init}_{subfolder_name}"))
            output_path = os.path.join(os.path.dirname(path), f"{dt_str_init}_{subfolder_name}", f"eval_{n_steps}steps_{joint_name}_{direction}_{int(np.round(name_param_value,0))}{name_units}_{param_name}_{seed}seed.pkl")
        else: 
            output_path = os.path.join(os.path.dirname(path), f"{dt_str}_eval_{n_steps}steps_{joint_name}_{direction}_{int(np.round(name_param_value,0))}{name_units}_{param_name}_{seed}seed.pkl")
        # output_path = os.path.join(os.path.dirname(path), f"{dt_str}_evaluation_results_{n_steps}steps_{int(param_value*1000)}{param_name}.pkl")
    else:   
        output_path = os.path.join(os.path.dirname(path), f"{dt_str}_evaluation_results_{n_steps}steps_{seed}seed.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(all_relevant_data, f)
    print(f"Saved evaluation data to {output_path}")


    #### Testing reading the data back from the file
    # # read the data back from the file
    # with open(output_path, "rb") as f:
    #     loaded_data = pickle.load(f)
    #     # print(f"Loaded data: {loaded_data.keys()}")  # Print the keys of the loaded data to verify


    # # # Get step from the loaded data
    # # test 
    # total_steps = loaded_data.get("total_steps")
    # loaded_data.get("lumbar_bending_angle")[0] # get first angle of lumbar_bending_angle
    # print(f"Loaded Step: {total_steps}")
    # print(f"Loaded Lumbar bending angle: {loaded_data.get('lumbar_bending_angle')[0]}")  # Example of accessing a specific joint angle


    time_all.append(timeit.default_timer())  # End timer

    print(f"Total time taken for {n_steps} steps: {time_all[-1] - time_all[0]} seconds")
    print(f"Compilation time: {time_all[1] - time_all[0]} seconds")
    print(f"Execution time taken: {time_all[-1] - time_all[1]} seconds")
    print(f"Average time per step: {(time_all[-1] - time_all[1]) / (n_steps-1)} seconds")







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

        if talus_ori_unequal_0 and param_name == "prosthesis_body_position":
            # Manually set talus orientation to 0 deg when position is being evaluated
            env._domain_randomizer.rand_conf["randomize_prosthesis_body_orientation"] = True
            env._domain_randomizer.rand_conf["prosthesis_body_orientation_range"] = randomization_params_eval["prosthesis_body_orientation_range"]
        
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

                # Use a small tolerance for floating-point comparison
                current_value = min_val
                while current_value <= max_val + 1e-6:
                    # Reset all axes to a zero range first
                    for other_axis in directions:
                        env._domain_randomizer.rand_conf[f"{param_name}_range"][body_name][other_axis] = [0, 0]

                    # Set the current axis to a fixed value for evaluation
                    env._domain_randomizer.rand_conf[f"{param_name}_range"][body_name][axis] = [current_value, current_value]
                    
                    # Reset to new domain randomization state
                    # Re jit reset for correct update of env_state?
                    
                    jit_reset  = jax.jit(jax.vmap(env.mjx_reset))
                    # env_keys = jax.random.split(rng, 2)
                    env_state = jit_reset(env_keys)
                    obs = env_state.observation
                    print("Running evaluation for ", param_name," for axis: ", axis, " with value: ",current_value)
                    run_evaluation_loop(env_state, n_steps, train_state, rng, prosthesis_metrics_handler,param_name=param_name, param_value = current_value,direction=axis, subfolder_name = subfolder_name, dt_str_init=dt_str_init) #min_val + inc)
                    current_value += inc

        # jit_reset  = jax.jit(jax.vmap(env.mjx_reset))
        # # env_keys = jax.random.split(rng, 2)
        # env_state = jit_reset(env_keys)
        # obs = env_state.observation
        # print("Running evaluation for ", param_name," for axis: ", axis, " with value: ",min_val + n/1000)
        # run_evaluation_loop(env_state, n_steps, train_state, rng, prosthesis_metrics_handler,param_name=param_name, param_value = min_val*1000 + n)


                


    


