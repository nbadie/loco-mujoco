import os
import jax
import numpy as np
import time

from loco_mujoco import ImitationFactory


# can increase the speed by ~30% on some GPUs
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')


randomization_params = {
    "prosthesis_side": "left_side",

    "randomize_prosthesis_dof_damping": False, #True,
    # "randomization_dof_names": ['ankle_angle', 'subtalar_angle', 'mtp_angle'],
    "prosthesis_dof_damping_range": {'ankle_angle': [0.09, 0.1], 'subtalar_angle': [0,0.01],'mtp_angle': [0.02,0.03]}, #[0, 10],
    # "prosthesis_dof_damping_range": {'ankle_angle': [0, 0.1], 'subtalar_angle': [0,0.01],'mtp_angle': [0,0.01]}, #[0, 10],

    "randomize_prosthesis_joint_stiffness": False, #True, # False #True,
    # "randomization_joint_names": ['ankle_angle', 'subtalar_angle', 'mtp_angle'],
    "prosthesis_joint_stiffness_range":{'ankle_angle': [30, 100], 'subtalar_angle': [10,20],'mtp_angle': [20,30]}, # [0, 100],


    # "randomize_prosthesis_dof_damping": True,
    # "randomization_dof_names": ['ankle_angle', 'subtalar_angle', 'mtp_angle'],
    # "prosthesis_dof_damping_range": [0, 10],

    # "randomize_prosthesis_joint_stiffness": False, #True,
    # "randomization_joint_names": ['ankle_angle', 'subtalar_angle', 'mtp_angle'],
    # "prosthesis_joint_stiffness_range":[], # [0, 100],

    "randomize_prosthesis_body_position": False,
    # "randomization_body_position_names": ['calcn'],
    "prosthesis_body_position_range": {
        'calcn': {'x': [0.3, 0.4],'y': [0.4, 0.5]}
        # 'y': [0.0, 0.1],
        # 'z': [0.5, 0.9]
    },
    "randomize_prosthesis_body_orientation": False, #True, #False, #True,
    # "randomization_body_orientation_names": ['calcn'], #['toe'], #['calcn'], #['foot_box'], #['calcn'],
    "prosthesis_body_orientation_range": {
        # 'x': [0.1, 0.2],
        'calcn': {'y': [1.5, 1.51], 'z': [1.1, 1.11]} 
        #'y': [0.8, 0.9],
        # 'z': [0.1, 0.2]
        #'x': [-0.01, 0.01],
        # 'y': [0.8, 0.9]#, #[0.15, 0.3], #(9-17°) #[0.8, 0.9],
        #'z': [-0.01, 0.01]
    },


    "randomize_prosthesis_socket_joint": True, #True, #True, 
    "socket_joint_range": {
        # 'socket_flexion': [],
        # 'socket_adduction': [],
        # 'socket_rotation': [],
        # 'socket_tx': [1,1.5], 
        'socket_ty':[-0.2,-0.21],  
        # 'socket_tz': []
    },

    "adapt_tibia_socket_parameters": False, # True TO ADAPT MASS, INERTIA & CENTER OF MASS dependig on socket_ty
    # "randomize_prosthesis_socket_pos": True, 
    # "socket_position_range": {
    #     'x':[0.1,0.2],
    #     'y': [0.2,0.3],
    #     'z': [0.3,0.4]
    # }
}


# create env
env = ImitationFactory.make(
    "MjxSkeletonMuscleProsthesis",
    prosthesis_side = "left_side", prosthesis_type= "transtibial", prosthesis_subtype= "SACH",
    delete_joints=False, remove_joint_names=['subtalar_angle'],
    tibia_socket_offset_z = 0, tibia_socket_offset_x = 0, tibia_socket_overlap= 0.2, amputated_tibia_length= 0.2,
    # joint_stiffness = 50, joint_damping = 10,
    joint_stiffness = {'ankle_angle': 50, 'mtp_angle': 10}, joint_damping = {'ankle_angle': 0.1, 'mtp_angle': 0.01},
    # reattach_gas_muscle = True, reattach_muscles_offset = {'med_gas': [0,0.05,0], 'lat_gas':[0,0.05,0]},# relative position above amputation
    use_2_box_per_foot = True,
    default_dataset_conf=dict(task="walk"),
    # reward_type = 'LocomotionReward'
    #domain_randomization_type="ProsthesisRandomizer",
    #domain_randomization_params=randomization_params
)

action_dim = env.info.action_space.shape[0]

env.reset()

env.render()
absorbing = False
i = 0

while True:
    if i == 1000 or absorbing:
        env.reset()
        i = 0
    action = np.random.randn(action_dim)
    nstate, reward, absorbing, done, info = env.step(action)

    # env.render()
    i += 1

