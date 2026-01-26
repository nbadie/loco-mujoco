# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 
# os.environ["JAX_PLATFORMS"] = "cpu"
import time 
import matplotlib
from scipy.ndimage import gaussian_filter1d

import jax 
import jax.numpy as jnp
# jax.config.update('jax_platform_name', 'cpu')

# import argparse
import mujoco

import numpy as np
import matplotlib.pyplot as plt

from mujoco import mjx

from datetime import datetime
from matplotlib import colors as mcolors
import matplotlib.cm as cm

# from omegaconf import OmegaConf, DictConfig


# from loco_mujoco import TaskFactory
# from loco_mujoco.algorithms import PPOJax
# from loco_mujoco.environments.humanoids.skeleton_prosthesis import MjxSkeletonMuscleProsthesis
# from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction


# from omegaconf import OmegaConf

# os.environ["MUJOCO_GL"] = "egl"

from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction
from loco_mujoco.utils import MetricsHandler
import numpy as np
import itertools

import re
import os




class ProsthesisMetricsHandler(): #MetricsHandler): #MetricsHandler):
    """
    Metrics handler for prosthesis evaluation.
    This class extends the MetricsHandler to include specific metrics for prosthesis evaluation.
    """

    def __init__(self, env):
        """
        Initialize the ProsthesisMetricsHandler.
        Args:
        config (DictConfig): The configuration dictionary.
        env (MjxSkeletonMuscleProsthesis): The environment instance.
        """
        #self.config = config
        self.env = env
        self.model = env.get_model()  # Get the model from the environment
        self.muscle_skeleton_control_activation = SkeletonMuscleControlFunction(self.env)

        # self.all_step_contact_right = []  # List to store start steps for right foot
        # self.all_step_contact_left = []  # List to store start steps for left foot
        # self.all_step_start_right = []  # List to store start steps for right foot
        # self.all_step_start_left = []  # List to store start steps for left foot
        # self.all_grf_left = []  # List to store ground reaction forces for left foot
        # self.all_grf_right = []  # List to store ground reaction forces for right foot
        # super().__init__(config, env)


    # def get_joint_group(self, which_joints):
    #     """
    #     Get the joint angles based on the specified joint group.
        
    #     Args:
    #     which_joints (str): The joint group to extract. Options: "all_joints", "left_side", "right_side".
        
    #     Returns:
    #     list: A list of joint names corresponding to the specified group.
    #     """
    #     joint_groups = {
    #         "all_joints": [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)],
    #         "leg_joints": ["hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle", "toe_angle", "mtp_angle"],
    #         "hip_joints": ["hip_flexion", "hip_adduction", "hip_rotation"],
    #         "hip_flexion": ["hip_flexion_l"],
    #         "hip_abduction": ["hip_adduction"],
    #         "knee_joint": ["knee_angle"],
    #         "ankle_joint": ["ankle_angle"],
    #         "talus_joint": ["talus_angle"],
    #         "mtp_angle": ["mtp_angle"],
    #     }


    def get_xpos(self, mjx_data):
        body_xpos  = {}
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            body_pos = mjx_data.xpos[...][0, i]
            body_xpos[body_name] = body_pos
        return body_xpos
    
    def get_xpos_batched(self, mjx_data):
        body_xpos = {}
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            # Get all seeds for body i
            body_pos = mjx_data.xpos[:, i]  # shape: [1,n_bodies,n_seeds]
            body_xpos[body_name] = body_pos
            # jax.debug.print("Body: {b}, Pos shape all: {p}, Body_pos_all: {body_pos}", b=body_name, p=mjx_data.xpos.shape, body_pos=mjx_data.xpos)
        return body_xpos

    def get_cvel(self, mjx_data):
        body_cvel  = {}
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            body_vel = mjx_data.cvel[...][0,i] #mjx_data.cvel[..., i]
            body_cvel[body_name] = body_vel
        return body_cvel
    
    def get_cvel_batched(self, mjx_data):
        body_cvel = {}
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            # Get all seeds for body i
            body_vel = mjx_data.cvel[:, i]  # shape: [1,n_bodies,n_seeds]
            body_cvel[body_name] = body_vel
            # jax.debug.print("Body: {b}, Vel shape all: {p}, Body Vel all: {body_pos}", b=body_name, p=mjx_data.cvel.shape, body_pos=mjx_data.cvel)
        return body_cvel

    def get_joint_angles(self,mjx_data): 
        """ Get all joint angles and save in dictionary with joint name as key and angle as value.
        """
        joint_angles = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            qpos_address = self.model.jnt_qposadr[i]
            # joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_angle = mjx_data.qpos[...,qpos_address]
            joint_angles[joint_name] = joint_angle
            # jax.debug.print("Joint: {j}, Angle shape all: {p}, Joint Angle all: {joint_angle}", j=joint_name, p=mjx_data.qpos.shape, joint_angle=mjx_data.qpos)
        return joint_angles
    

    def get_joint_vels(self, mjx_data):
        """ Get all joint velocities and save in dictionary with joint name as key and velocity as value.
        """
        joint_velocities = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            dof_address = self.model.jnt_dofadr[i]
            joint_velocity = mjx_data.qvel[...,dof_address]
            joint_velocities[joint_name] = joint_velocity
        return joint_velocities
    
    def get_joint_frces(self, mjx_data):
        """ Get all joint forces and save in dictionary with joint name as key and force as value.
        """
        joint_forces_constraint = {}
        joint_forces_smooth = {}
        joint_forces_applied = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            dof_address = self.model.jnt_dofadr[i]
            joint_force_constraint = mjx_data.qfrc_constraint[...,dof_address] # constraint force; joint limits and contacts etc. 
            joint_force_smooth = mjx_data.qfrc_smooth[...,dof_address] # net unconstrained force; Sum of all forces, e.g., gravity, applied torques etc,
            joint_force_applied = mjx_data.qfrc_applied[...,dof_address]
            joint_forces_constraint[joint_name] = joint_force_constraint
            joint_forces_smooth[joint_name] = joint_force_smooth
            joint_forces_applied[joint_name] = joint_force_applied
        return joint_forces_constraint, joint_forces_smooth, joint_forces_applied
    
    def get_joint_trques(self, mjx_data):
        """ 
        Get all joint torques and save in dictionary with joint name as key and torque as value.
        qfrc_actuator in actuator force --> As we are using hinge joints, this is the torque applied by the actuator? (As mjx tutorial?)
        """
        joint_torques = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            dof_address = self.model.jnt_dofadr[i]
            joint_torque = mjx_data.qfrc_actuator[...,dof_address]
            joint_torques[joint_name] = joint_torque
        return joint_torques
    
    

    def calc_joint_energy_exp(self, joint_torques, joint_vels):
        """Energy expenditure is calculated as joint torques multiplied by joint velocities."""
        joint_energy_expenditure = {}
        for joint_name in joint_torques.keys():
            if joint_name in joint_vels.keys():
                energy_expenditure = joint_torques[joint_name] * joint_vels[joint_name]
                joint_energy_expenditure[joint_name] = energy_expenditure
            else:
                raise ValueError(f"Joint velocity for {joint_name} not found.")
        return joint_energy_expenditure



    def get_muscle_group(self, which_muscles): 
        muscle_groups = {
        "back_muscles": ['ercspn'],
        "torso_muscles": ['intobl', 'extobl'],
        "vasti_muscles": ['vas_int', 'vas_lat', 'vas_med'],
        "rectus_muscles": ['rect_fem'],
        "medial_muscles": ['add_mag1', 'add_mag2', 'add_mag3', 'add_brev', 'add_long', 'grac'],
        "leg_muscles": [
            "glut_med1", "glut_med2", "glut_med3", "glut_min1", "glut_min2", "glut_min3",
            "semimem", "semiten", "bifemlh", "bifemsh", "sar", "add_long", "add_brev",
            "add_mag1", "add_mag2", "add_mag3", "tfl", "pect", "grac", "glut_max1",
            "glut_max2", "glut_max3", "iliacus", "psoas", "quad_fem", "gem", "peri",
            "rect_fem", "vas_med", "vas_int", "vas_lat", "med_gas", "lat_gas", "soleus",
            "tib_post", "flex_dig", "flex_hal", "tib_ant", "per_brev", "per_long",
            "per_tert", "ext_dig", "ext_hal"
        ],
        "all_muscles": [
            "glut_med1", "glut_med2", "glut_med3", "glut_min1", "glut_min2", "glut_min3",
            "semimem", "semiten", "bifemlh", "bifemsh", "sar", "add_long", "add_brev",
            "add_mag1", "add_mag2", "add_mag3", "tfl", "pect", "grac", "glut_max1",
            "glut_max2", "glut_max3", "iliacus", "psoas", "quad_fem", "gem", "peri",
            "rect_fem", "vas_med", "vas_int", "vas_lat", "med_gas", "lat_gas", "soleus",
            "tib_post", "flex_dig", "flex_hal", "tib_ant", "per_brev", "per_long",
            "per_tert", "ext_dig", "ext_hal", "ercspn", "intobl", "extobl"
        ]
        }

        # Determine muscle names based on group and side
        if which_muscles in muscle_groups:
            muscle_name = muscle_groups[which_muscles]
        else:
            raise ValueError(f"Invalid muscle group: {which_muscles}")
        
        return muscle_name



    def get_relevant_ctrl(self, muscle_name, muscle_side, action):
        """
        Extract actions for specific muscle groups and sides.

        Args:
        which_muscles (str): The muscle group to extract actions for. 
                        Options: "back_muscles", "torso_muscles", "leg_muscles", "all_muscles", or specific muscle names.
        muscle_side (str): The side of the body. Options: "left_side", "right_side".
        action (jnp.ndarray): The action array to extract from.

        Returns:
        jnp.ndarray: The extracted actions corresponding to the specified muscle group and side.
        """
    
        # elif any(which_muscles in a.name for a in env.model.actuators):
        #     muscle_name = [a.name for a in env.model.actuators if which_muscles in a.name]
        if muscle_side == 'left_side':
            muscle_name = [name+'_l' for name in muscle_name]
        elif muscle_side == 'right_side':
            muscle_name = [name+'_r' for name in muscle_name]
        # elif which_muscles == "all_muscles":
        #     suffix = '_l' if muscle_side == 'left_side' else '_r'
        #     muscle_name = [name for name in env.info.action_space.names if name.endswith(suffix)]
        else:
            raise ValueError(f"Invalid muscle side: {muscle_side}")
        action_indices = []
        extracted_actions = []

        # Get indices for the specified muscles
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name in muscle_name:
                # print(f"Actuator {i}: {name}")
                action_indices.append(i)
        # action_indices = [i for i, name in enumerate(env.info.action_space.names) if name in muscle_name]

        # Extract the actions corresponding to the specified indices

        # Ensure action_indices is not empty
        if not action_indices:
            raise ValueError(f"No matching muscles found for group '{muscle_name}' and side '{muscle_side}'.")

        # Extract the actions corresponding to the specified indices
        extracted_actions = [action[..., k] for k in action_indices]

        # muscle_skeleton_control_activation = SkeletonMuscleControlFunction(self.env)
        
        action_processed = extracted_actions.copy()
        # action_processed = self.muscle_skeleton_control_activation.adapted_sigmoid(jnp.array(extracted_actions)) #action_processed.at[...].set(self.adapted_sigmoid(action_processed))
        #self.muscle_skeleton_control_activation.adapted_sigmoid(jnp.array(extracted_actions)) #extracted_actions.set(self.muscle_skeleton_control_activation.adapted_sigmoid(action_processed))

        # run sigmoid on extracted actions

        # extracted_action = action[jnp.array(action_indices)]

        return action_processed # , #jnp.array(extracted_actions)


    # def get_relevant_ctrl_sigmoid_batched(self, muscle_name, muscle_side, action):
    #     """
    #     Extract actions for specific muscle groups and sides.

    #     Args:
    #     which_muscles (str): The muscle group to extract actions for. 
    #                     Options: "back_muscles", "torso_muscles", "leg_muscles", "all_muscles", or specific muscle names.
    #     muscle_side (str): The side of the body. Options: "left_side", "right_side".
    #     action (jnp.ndarray): The action array to extract from.

    #     Returns:
    #     jnp.ndarray: The extracted actions corresponding to the specified muscle group and side.
    #     """
    
    #     # elif any(which_muscles in a.name for a in env.model.actuators):
    #     #     muscle_name = [a.name for a in env.model.actuators if which_muscles in a.name]
    #     if muscle_side == 'left_side':
    #         muscle_name = [name+'_l' for name in muscle_name]
    #     elif muscle_side == 'right_side':
    #         muscle_name = [name+'_r' for name in muscle_name]
    #     # elif which_muscles == "all_muscles":
    #     #     suffix = '_l' if muscle_side == 'left_side' else '_r'
    #     #     muscle_name = [name for name in env.info.action_space.names if name.endswith(suffix)]
    #     else:
    #         raise ValueError(f"Invalid muscle side: {muscle_side}")
    #     action_indices = []
    #     extracted_actions = []

    #     # Get indices for the specified muscles
    #     for i in range(self.model.nu):
    #         name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    #         if name in muscle_name:
    #             # print(f"Actuator {i}: {name}")
    #             action_indices.append(i)
    #     # action_indices = [i for i, name in enumerate(env.info.action_space.names) if name in muscle_name]

    #     # Extract the actions corresponding to the specified indices

    #     # Ensure action_indices is not empty
    #     if not action_indices:
    #         raise ValueError(f"No matching muscles found for group '{muscle_name}' and side '{muscle_side}'.")

    #     # Extract the actions corresponding to the specified indices
    #     extracted_actions = [action[..., k] for k in action_indices]

    #     # muscle_skeleton_control_activation = SkeletonMuscleControlFunction(self.env)
        
    #     # action_processed = extracted_actions.copy()
    #     action_processed = self.muscle_skeleton_control_activation.adapted_sigmoid(jnp.array(extracted_actions)) #extracted_actions.set(self.muscle_skeleton_control_activation.adapted_sigmoid(action_processed))

    #     # run sigmoid on extracted actions

    #     # extracted_action = action[jnp.array(action_indices)]

    #     return action_processed #jnp.array(extracted_actions)
    

    def get_relevant_ctrl_batched(self, muscle_indices, side, processed_action):
        """
        Fixed version that takes pre-computed indices instead of muscle names and side strings
        """
        return processed_action[muscle_indices]


    def setup_muscle_indices(self, model, evaluation_muscle_names, muscle_group, side):
        """
        Computes muscle indices for a specific muscle group and side.

        Args:
            model (mujoco.MjModel): The MuJoCo model.
            evaluation_muscle_names (dict): A dictionary mapping muscle group names to a list of muscle names.
            muscle_group (str): The name of the muscle group to process.
            side (str): The side of the body ('left' or 'right').

        Returns:
            jnp.array: A JAX array of muscle indices for the specified group and side.
        """
        # Build actuator name to index mapping
        name_to_idx = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)
        }

        indices = []
        muscle_names = evaluation_muscle_names.get(muscle_group, [])

        for muscle_name in muscle_names:
            if muscle_name in name_to_idx:
                idx = name_to_idx[muscle_name]
                
                # Determine left/right based on naming convention
                is_left = "_l" in muscle_name or "left" in muscle_name.lower()
                
                if (side == 'left' and is_left) or (side == 'right' and not is_left):
                    indices.append(idx)

        return jnp.array(indices) if indices else jnp.array([])


    # def setup_muscle_indices(self, model, evaluation_muscle_groups, evaluation_muscle_names):
    #     """
    #     Call this ONCE before JIT compilation to pre-compute muscle indices
    #     Add this method to your ProsthesisMetricsHandler class
    #     """
    #     if not hasattr(self, '_muscle_indices_cache'):
    #         self._muscle_indices_cache = {}
            
    #         # Build actuator name to index mapping
    #         name_to_idx = {}
    #         for i in range(model.nu):
    #             name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    #             name_to_idx[name] = i
            
    #         # Convert muscle names to indices for each group
    #         for muscle_group in evaluation_muscle_groups:
    #             muscle_names = evaluation_muscle_names[muscle_group]
                
    #             left_indices = []
    #             right_indices = []
                
    #             for muscle_name in muscle_names:
    #                 if muscle_name in name_to_idx:
    #                     idx = name_to_idx[muscle_name]
    #                     # Determine left/right based on naming convention
    #                     if "_l" in muscle_name or "left" in muscle_name.lower():
    #                         left_indices.append(idx)
    #                     else:
    #                         right_indices.append(idx)
                
    #             self._muscle_indices_cache[f"{muscle_group}_left"] = jnp.array(left_indices) if left_indices else jnp.array([])
    #             self._muscle_indices_cache[f"{muscle_group}_right"] = jnp.array(right_indices) if right_indices else jnp.array([])


    def get_muscle_activations_by_indices(self, processed_action, muscle_group, side):
        """
        JAX-compatible muscle activation extraction using pre-computed indices
        Add this method to your ProsthesisMetricsHandler class
        """
        key = f"{muscle_group}_{side}"
        if hasattr(self, '_muscle_indices_cache') and key in self._muscle_indices_cache:
            indices = self._muscle_indices_cache[key]
            return processed_action[indices] if len(indices) > 0 else jnp.array([])
        else:
            return jnp.array([])


    def calc_mean_grf(self, data):
        f_contact_frame_r = np.zeros(3)
        f_contact_frame_l = np.zeros(3)
        geom1_id = data.contact.geom1
        geom2_id = data.contact.geom2

        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        foot_box_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_box_r')
        foot_box_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_box_l')
        toes_box_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'toes_box_r')
        toes_box_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'toes_box_l')

        mj_data = mjx.get_data(self.model, data)
        geom1_id_is_floor = geom1_id == floor_id
        if (geom1_id_is_floor).all():
            for n in range(mj_data[0].ncon):
                # Calculate contact force
                # mj_contactForce takes (model, data, contact_id, force_array)
                # The force_array is a 6-element array: (f_x, f_y, f_z, t_x, t_y, t_z)
                contact_force_raw = np.zeros((6,1), dtype=np.float64)
                mujoco.mj_contactForce(self.model, mj_data[0], n, contact_force_raw)

                frX = mj_data[0].contact.frame[0][0:3] #[0][n][0]
                frY = mj_data[0].contact.frame[0][3:6] #[0][n][1]
                frZ = mj_data[0].contact.frame[0][6:9] #[0][n][2]

                # `contact_force_raw[0:3]` are the force components in the contact frame
                
                if mj_data[0].contact.geom2[n] == foot_box_r_id:
                    # f_contact_frame_r += contact_force_raw[0:3]
                    f_contact_frame_r += frX * contact_force_raw[0] + frY * contact_force_raw[1] + frZ * contact_force_raw[2]
                elif mj_data[0].contact.geom2[n] == foot_box_l_id:
                    # f_contact_frame_l += contact_force_raw[0:3]
                    f_contact_frame_l += frX * contact_force_raw[0] + frY * contact_force_raw[1] + frZ * contact_force_raw[2]
                elif mj_data[0].contact.geom2[n] == toes_box_r_id:
                    # f_contact_frame_r += contact_force_raw[0:3]
                    f_contact_frame_r += frX * contact_force_raw[0] + frY * contact_force_raw[1] + frZ * contact_force_raw[2]
                elif mj_data[0].contact.geom2[n] == toes_box_l_id:
                    # f_contact_frame_l += contact_force_raw[0:3]
                    f_contact_frame_l += frX * contact_force_raw[0] + frY * contact_force_raw[1] + frZ * contact_force_raw[2]

            # force_in_world_frame_l = frX * f_contact_frame_l[0] + frY * f_contact_frame_l[1] + frZ * f_contact_frame_l[2]
            # force_in_world_frame_r = frX * f_contact_frame_r[0] + frY * f_contact_frame_r[1] + frZ * f_contact_frame_r[2]

        return f_contact_frame_l, f_contact_frame_r



    # def evaluate_action_symmetry(self, all_data, muscle_name, left_indices, right_indices):
    #     """
    #     Evaluate the symmetry of the actions in the environment states.
    #     This function calculates the symmetry of the actions for left and right sides.
    #     """

    #     muscle_name_left = [name+'_l' for name in muscle_name]
    #     muscle_name_right = [name+'_r' for name in muscle_name]

    #     action_indices_left = []
    #     action_indices_right = []
    #     all_actions_left = []
    #     all_actions_right = []

    #     for n in range(len(all_data)):
    #         data = all_data[n]

    #         # Get indices for the specified muscles
    #         for i in range(self.model.nu):
    #             name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    #             if name in muscle_name_left:
    #                 print(f"Actuator {i}: {name}")
    #                 action_indices_left.append(i)
    #             if name in muscle_name_right:
    #                 print(f"Actuator {i}: {name}")
    #                 action_indices_right.append(i)


    #         for i in range(len(left_indices)):
    #             for a in action_indices_left:
    #                 actions_left = data[i].ctrl[a][left_indices[i][0]:left_indices[i][1]]
    #                 all_actions_left.append(actions_left)
    #                 print(f"Left Action {a} at step {i}: {actions_left}")

    #         for i in range(len(right_indices)):
    #             for a in action_indices_right:
    #                 actions_right = data[i].ctrl[a][right_indices[i][0]:right_indices[i][1]]
    #                 all_actions_right.append(actions_right)
    #                 print(f"Right Action {a} at step {i}: {actions_right}")

    #         # Calculate the symmetry of the actions
    #         symmetry_scores = []
    #         for i in range(len(all_actions_left)):
    #             left_action = all_actions_left[i]
    #             right_action = all_actions_right[i]
    #             if len(left_action) != len(right_action):
    #                 raise ValueError("Left and right actions must have the same length for symmetry evaluation.")
    #             # Calculate symmetry score as the absolute difference between left and right actions
    #             symmetry_score = jnp.abs(left_action - right_action).mean()
    #             symmetry_scores.append(symmetry_score)
    #             print(f"Symmetry Score for step {i}: {symmetry_score}")


    def get_contact_steps(self, data, step):
        contact_index_r = []  # List to store steps where left foot contacts the ground
        contact_index_l = []  # List to store steps where right foot contacts the ground
       
        geom_distance = data.contact.dist
        geom1_id = data.contact.geom1
        geom2_id = data.contact.geom2

        # check that geom1_index is all 0 and geom_name is floor 
        # get from geom name to index 
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        foot_box_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_box_r')
        foot_box_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_box_l')
        toes_box_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'toes_box_l')
        toes_box_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'toes_box_r')

        geom1_id_is_floor = geom1_id == floor_id
        # if all geom1_id_is lfoor true: get indices wheren dist is smaller= 0 
        if (geom1_id_is_floor).all(): 
            penetration = geom_distance <= 0
            # get indices where geom_distance is smaller than 0
            penetration_indices_geom = jnp.where(penetration[0])[0]
            # check which floot_box_r_id or floot_box_l_id are at penetraction_indices_geom 
            for n in penetration_indices_geom:
                if geom2_id[...,n] == foot_box_r_id: 
                    contact_index_r = step
                elif geom2_id[...,n] == toes_box_r_id: 
                    contact_index_r = step
                    
                if geom2_id[...,n] == toes_box_l_id: 
                    contact_index_l = step
                elif geom2_id[...,n] == foot_box_l_id: 
                    contact_index_l = step

        return contact_index_l, contact_index_r 
    


    def get_contact_steps_batched(self, data, step):
        """JAX-compatible contact detection"""
        # Pre-compute IDs
        if not hasattr(self, '_floor_id'):
            self._floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
            self._foot_box_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_box_r') 
            self._foot_box_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_box_l')
            self._toes_box_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'toes_box_l')
            self._toes_box_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'toes_box_r')
        
        geom_distance = data.contact.dist
        geom1_id = data.contact.geom1
        geom2_id = data.contact.geom2
        
        geom1_id_is_floor = geom1_id == self._floor_id
        all_floor = jnp.all(geom1_id_is_floor)
        
        # Initialize contact indices
        contact_index_r = -1  # Default value when no contact
        contact_index_l = -1  # Default value when no contact
        
        # Only process if all geom1 are floor
        def process_contacts():
            penetration = geom_distance <= 0.0
            valid_contacts = penetration  # Since we know all are floor
            
            left_foot_contact = jnp.any(valid_contacts & (geom2_id == self._foot_box_l_id))
            right_foot_contact = jnp.any(valid_contacts & (geom2_id == self._foot_box_r_id))

            left_toes_contact = jnp.any(valid_contacts & (geom2_id == self._toes_box_l_id))
            right_toes_contact = jnp.any(valid_contacts & (geom2_id == self._toes_box_r_id))

            # Combine left foot and toes contact
            left_contact = left_foot_contact | left_toes_contact
            right_contact = right_foot_contact | right_toes_contact
            
            contact_l = jnp.where(left_contact, step, -1)
            contact_r = jnp.where(right_contact, step, -1)
            
            return contact_l, contact_r
        
        def no_contacts():
            return -1, -1
        
        # Use jax.lax.cond for conditional execution
        contact_index_l, contact_index_r = jax.lax.cond(
            all_floor,
            process_contacts,
            no_contacts
        )
        
        return contact_index_l, contact_index_r
        
        # # Pre-compute IDs once (store as class attributes)
        # if not hasattr(self, '_floor_id'):
        #     self._floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        #     self._foot_box_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_box_r') 
        #     self._foot_box_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_box_l')
        
        # # Vectorized contact detection
        # dist = data.contact.dist
        # geom1 = data.contact.geom1
        # geom2 = data.contact.geom2
        
        # floor_contacts = (geom1 == self._floor_id)
        # penetrations = (dist <= 0.0)
        # valid_contacts = floor_contacts & penetrations
        
        # left_contact = jnp.any(valid_contacts & (geom2 == self._foot_box_l_id))
        # right_contact = jnp.any(valid_contacts & (geom2 == self._foot_box_r_id))
        
        # return left_contact, right_contact


    # def get_contact_steps_batched(self, data, step):
    #     contact_index_r = []  # List to store steps where left foot contacts the ground
    #     contact_index_l = []  # List to store steps where right foot contacts the ground
       
    #     geom_distance = data.contact.dist
    #     geom1_id = data.contact.geom1
    #     geom2_id = data.contact.geom2

    #     # check that geom1_index is all 0 and geom_name is floor 
    #     # get from geom name to index 
    #     floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    #     foot_box_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_box_r')
    #     foot_box_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_box_l')

    #     geom1_id_is_floor = geom1_id == floor_id
    #     # if all geom1_id_is lfoor true: get indices wheren dist is smaller= 0 

    #     def true_fn():
    #         penetration = geom_distance <= 0
    #         # get indices where geom_distance is smaller than 0
    #         penetration_indices_geom = jnp.where(jnp.atleast_1d(penetration).nonzero()) #penetration[0])[0]
    #         # check which floot_box_r_id or floot_box_l_id are at penetraction_indices_geom 
    #         for n in penetration_indices_geom:
    #             if geom2_id[...,n] == foot_box_r_id: 
    #                 contact_index_r = step
    #             elif geom2_id[...,n] == foot_box_l_id: 
    #                 contact_index_l = step

    #         return contact_index_l, contact_index_r

    #     def false_fn():
    #         return -1, -1

    #     contact_index_l, contact_index_r = jax.lax.cond(jnp.all(geom1_id_is_floor),true_fn, false_fn) 
    #     # contact_index_r = -1  # Placeholder for "no contact"
    #     # contact_index_l = -1

    #     # geom_distance = data.contact.dist
    #     # geom1_id = data.contact.geom1
    #     # geom2_id = data.contact.geom2

    #     # floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    #     # foot_box_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_box_r')
    #     # foot_box_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_box_l')

    #     # geom1_id_is_floor = geom1_id == floor_id
    #     # pred = jnp.all(geom1_id_is_floor)  # Scalar boolean (shape ())

    #     # def true_branch(_):
    #     #     penetration = geom_distance <= 0
    #     #     penetration_indices_geom = jnp.where(penetration)[0]

    #     #     def body_fn(i, carry):
    #     #         contact_index_l, contact_index_r = carry
    #     #         geom2_id_i = geom2_id[..., i]

    #     #         contact_index_r = jnp.where(geom2_id_i == foot_box_r_id, step, contact_index_r)
    #     #         contact_index_l = jnp.where(geom2_id_i == foot_box_l_id, step, contact_index_l)

    #     #         return contact_index_l, contact_index_r

    #     #     contact_index_l, contact_index_r = jax.lax.fori_loop(
    #     #         0,
    #     #         penetration_indices_geom.shape[0],
    #     #         body_fn,
    #     #         (contact_index_l, contact_index_r)
    #     #     )

    #     #     return contact_index_l, contact_index_r

    #     # def false_branch(_):
    #     #     return contact_index_l, contact_index_r

    #     # contact_index_l, contact_index_r = jax.lax.cond(
    #     #     pred,
    #     #     true_branch,
    #     #     false_branch,
    #     #     operand=None
    #     # )

    #     return contact_index_l, contact_index_r

    

    # def get_contact_steps(self, mjx_data, foot_side,  step):
    #     all_step_contact = []  # List to store steps where left foot contacts the ground
    #     all_step_contact = []  # List to store steps where right foot contacts the ground
    #     if 'left' in foot_side:
    #         side_suffix = '_l'
    #     elif 'right' in foot_side:
    #         side_suffix = '_r'
    #     else:
    #         raise ValueError("Invalid foot_side. Expected 'left' or 'right'.")
    #     # Collect all geom2 names and their corresponding distances
    #     for n in range(mjx_data.ncon):
    #         geom2_id = mjx_data.contact.geom2[n]
    #         geom_name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
    #         geom_distance = mjx_data.contact.dist[n]
    #         # Check if the geom_name2 matches the requested side and the contact is happening
    #         if side_suffix in geom_name2 and (geom_distance <= 0).any():
    #           all_step_contact.append(step)
    #     return contact_index_l, contact_index_r



        

        #         if "_l" in geom_name2:
        #             contact_left = step
        #             all_step_contact_left.append(contact_left)
        #         elif "_r" in geom_name2:
        #             contact_right = step
        #             all_step_contact_right.append(contact_right)

        # all_step_contact_left = list(set(all_step_contact_left))  # remove duplicates
        # all_step_contact_right = list(set(all_step_contact_right))

        # return all_step_contact_left, all_step_contact_right


    def get_start_steps(self, all_step_contact):
        """
        Get the start steps for left and right foot based on contact data.
        This function processes the contact data to determine the start of each step.
        """
        # Initialize lists to store the start steps for left and right foot
        if all_step_contact[0]!= 0:
            all_step_start = [all_step_contact[0]]


        # Iterate through the contact data to find non-consecutive steps
        for m in range(len(all_step_contact) - 1):
            if all_step_contact[m + 1] - all_step_contact[m] > 1:
                all_step_start.append(all_step_contact[m])
        
        print(f"All start steps: {all_step_start}")


        return all_step_start
    

    # def get_start_steps(self, all_step_contact_left, all_step_contact_right):
    #     """
    #     Get the start steps for left and right foot based on contact data.
    #     This function processes the contact data to determine the start of each step.
    #     """
    #     # Initialize lists to store the start steps for left and right foot
    #     all_step_start_left = [all_step_contact_left[0]]
    #     all_step_start_right = [all_step_contact_right[0]]

    #     # Iterate through the contact data to find non-consecutive steps
    #     for m in range(len(all_step_contact_left) - 1):
    #         if all_step_contact_left[m + 1] - all_step_contact_left[m] > 1:
    #             all_step_start_left.append(all_step_contact_left[m])
    #     for m in range(len(all_step_contact_right) - 1):
    #         if all_step_contact_right[m + 1] - all_step_contact_right[m] > 1:
    #             all_step_start_right.append(all_step_contact_right[m])

    #     print(f"All start steps left: {all_step_start_left}")
    #     print(f"All start steps right: {all_step_start_right}")


    #     return all_step_start_left, all_step_start_right



    def get_sensor_data(self, mjx_data):
        """
        Retrieve sensor data from the Mujoco environment.
        This function extracts sensor data from the Mujoco data structure.
        
        """
        # Force sensor: creates a 3-axis force sensor. The sensor outputs three numbers, which are the interaction 
        # force between a child and a parent body, expressed in the site frame defining the sensor. The convention 
        # is that the site is attached to the child body, and the force points from the child towards the parent. 
        # The computation here takes into account all forces acting on the system, including contacts as well as 
        # external perturbations. 

        # Retrieve sensor names from the model
        sensor_names = []
        for sensor_id in range(self.model.nsensor):
            sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
            sensor_names.append(sensor_name)

        # print(f"Number of sensors: {self.model.nsensor}")
        # print(f"Sensor names: {sensor_names}")

        # Extract sensor data
        if self.model.nsensor == 1: 
            # tibia sensor is the only sensor available
            sensor_data = mjx_data.sensordata
            # print(f"Human-prosthesis interface sensor data: {sensor_data}") 
            #return sensor_data
        else: 
            sensor_data = {}
            # get sensor_names and iterate through all sensors
            for sensor_id in range(self.model.nsensor):
                sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
                start_idx = sensor_id * 3
                end_idx = start_idx + 3
                sensor_data[sensor_name] = mjx_data.sensordata[0][start_idx:end_idx]
                # sensor_data[sensor_name] = mjx_data.sensordata[...,sensor_id]
        
        return sensor_data, sensor_names
                # print(f"Sensor {sensor_name} data: {sensor_data}")
                # No specific sensor data is saved!!!! FIX!!!!!!!!!!!!!!!!

    def get_sensor_data_batched(self, mjx_data):
        """
        Retrieve sensor data from the Mujoco environment.
        Returns (sensor_data, sensor_names) where
        - sensor_data is a dict name -> [batch, 3] (PyTree, JAX-friendly)
        - sensor_names is a list of names (static Python)
        """

        # Cache sensor names (static, not traced)
        if not hasattr(self, "_batched_sensor_names"):
            self._batched_sensor_names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                for i in range(self.model.nsensor)
            ]
        sensor_names = self._batched_sensor_names
        nsensor = self.model.nsensor

        # sensordata shape: [batch, nsensor*3]
        sensordata = mjx_data.sensordata

        # Split into [batch, nsensor, 3]
        sensordata_split = sensordata.reshape(-1, nsensor, 3)

        # Make a dict {name: [batch, 3]}
        # sensor_data = {name: sensordata_split[:, i, :] for i, name in enumerate(sensor_names)}
        sensor_data = {name: sensordata_split[0, i] for i, name in enumerate(sensor_names)}

        return sensor_data, sensor_names


    def get_sensor_data_batched_2(self, mjx_data):
        """
        Retrieve sensor data from the Mujoco environment.
        Returns (sensor_data, sensor_names) where
        - sensor_data is a dict name -> [batch, 3] (PyTree, JAX-friendly)
        - sensor_names is a list of names (static Python)
        """

        # Cache sensor names (static, not traced)
        if not hasattr(self, "_batched_sensor_names"):
            self._batched_sensor_names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                for i in range(self.model.nsensor)
            ]
        sensor_names = self._batched_sensor_names
        nsensor = self.model.nsensor

        # sensordata shape: [batch, nsensor*3]
        sensordata = mjx_data.sensordata

        # Split into [batch, nsensor, 3]
        sensordata_split = sensordata.reshape(-1, nsensor, 3)

        # Make a dict {name: [batch, 3]}
        # sensor_data = {name: sensordata_split[:, i, :] for i, name in enumerate(sensor_names)}
        sensor_data = {name: sensordata_split[:, i] for i, name in enumerate(sensor_names)}

        return sensor_data, sensor_names

    # def get_sensor_data_batched(self, mjx_data):
    #     """
    #     Retrieve sensor data from the Mujoco environment.
    #     This function extracts sensor data from the Mujoco data structure.
        
    #     """
    #     # Force sensor: creates a 3-axis force sensor. The sensor outputs three numbers, which are the interaction 
    #     # force between a child and a parent body, expressed in the site frame defining the sensor. The convention 
    #     # is that the site is attached to the child body, and the force points from the child towards the parent. 
    #     # The computation here takes into account all forces acting on the system, including contacts as well as 
    #     # external perturbations. 

    #     # Retrieve sensor names from the model
    #     # JAX-compatible sensor data extraction for parallel runs (batched mjx_data)
    #     # Get sensor names from the model (cache for efficiency)
    #     # if not hasattr(self, '_batched_sensor_names'):
    #     self._batched_sensor_names = [
    #     mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    #     for i in range(self.model.nsensor)
    #     ]
    #     # # Optionally add mimic sensors if needed
    #     # mimic_names = ["hip_mimic", "knee_mimic", "foot_mimic"]
    #     # self._batched_sensor_names = (
    #     # [name + '_l' for name in mimic_names] +
    #     # [name + '_r' for name in mimic_names] +
    #     # self._batched_sensor_names
    #     # )

    #     sensor_names = self._batched_sensor_names
    #     nsensor = self.model.nsensor

    #     # mjx_data.sensordata shape: [batch, nsensor*3] or [batch, N] (N = nsensor*3)
    #     sensordata = mjx_data.sensordata #jnp.atleast_2d(mjx_data.sensordata)

    #     # Always return a dict of sensor_name -> [batch, 3] arrays
    #     sensor_data = {}
    #     for i, name in enumerate(sensor_names):  # Only real sensors, not mimic
    #         start = i * 3
    #         end = start + 3
    #         sensor_data[name] = sensordata[:, start:end]

    #     return sensor_data, sensor_names


    # def get_sensor_data_batched(self, mjx_data):
        # """
        # JAX-JIT compatible version with consistent output structure for jax.lax.cond
        # """
        
        # # Cache sensor information
        # if not hasattr(self, '_sensor_names') or not hasattr(self, '_nsensor'):
        #     self._nsensor = self.model.nsensor
        #     self._sensor_names = []
        #     for sensor_id in range(self._nsensor):
        #         sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
        #         self._sensor_names.append(sensor_name)
        
        # nsensor = self._nsensor
        # sensor_names = self._sensor_names
        
        # def single_sensor_case():
        #     """Handle case with only one sensor - return dict for consistency"""
        #     sensor_data = {sensor_names[0]: mjx_data.sensordata}
        #     return sensor_data
        
        # def multiple_sensor_case():
        #     """Handle case with multiple sensors"""
        #     sensor_data = {}
            
        #     # Ensure consistent 2D shape
        #     sensordata_2d = jnp.atleast_2d(mjx_data.sensordata)
            
        #     for sensor_id in range(nsensor):
        #         sensor_name = sensor_names[sensor_id]
        #         start_idx = sensor_id * 3
        #         end_idx = start_idx + 3
        #         sensor_data[sensor_name] = sensordata_2d[0, start_idx:end_idx]
            
        #     return sensor_data
        
        # # Both branches now return dictionaries - consistent pytree structure
        # sensor_data = jax.lax.cond(
        #     nsensor == 1,
        #     single_sensor_case,
        #     multiple_sensor_case
        # )
        
        # return sensor_data, sensor_names

    # def get_sensor_data_batched(self, mjx_data):
    #     # Get sensor count and names (cached)
    #     if not hasattr(self, '_simple_sensor_cache'):
    #         self._simple_nsensor = self.model.nsensor
    #         self._simple_sensor_names = []
    #         for i in range(self._simple_nsensor):
    #             name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    #             self._simple_sensor_names.append(name)
    #         self._simple_sensor_cache = True

        
    #     # Extract data - always return dict
    #     sensor_data = {}
    #     data = jnp.atleast_1d(mjx_data.sensordata)
        
    #     if self._simple_nsensor == 1:
    #         sensor_data[self._simple_sensor_names[0]] = data
    #     else:
    #         # Multiple sensors: 3 values each
    #         for i in range(self._simple_nsensor):
    #             start = i * 3
    #             end = start + 3
    #             sensor_data[self._simple_sensor_names[i]] = data[start:end]
        
    #     return sensor_data, self._simple_sensor_names
    # # Most robust version that handles edge cases
    # def get_sensor_data_batched(self, mjx_data):
    #     """
    #     Most robust JAX-JIT compatible implementation
    #     """
        
    #     # Initialize cached sensor info
    #     # if not hasattr(self, '_jax_sensor_cache'):
    #     self._jax_nsensor = self.model.nsensor
    #     self._jax_sensor_names = []
    #     for sensor_id in range(self._jax_nsensor):
    #         sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
    #         self._jax_sensor_names.append(sensor_name)
    #     self._jax_sensor_cache = True
        
    #     nsensor = self._jax_nsensor
    #     sensor_names = self._jax_sensor_names
        
    #     def process_sensors():
    #         """Process sensor data based on number of sensors"""
            
    #         def single_sensor():
    #             return mjx_data.sensordata
            
    #         def multiple_sensors():
    #             sensor_data = {}
                
    #             # Ensure sensordata is at least 2D for consistent indexing
    #             sensordata_shaped = jnp.reshape(mjx_data.sensordata, (1, -1)) if len(mjx_data.sensordata.shape) == 1 else mjx_data.sensordata
                
    #             # Extract data for each sensor (3 values per sensor)
    #             for sensor_id in range(nsensor):
    #                 sensor_name = sensor_names[sensor_id]
    #                 start_idx = sensor_id * 3
    #                 end_idx = start_idx + 3
    #                 sensor_data[sensor_name] = sensordata_shaped[0, start_idx:end_idx]
                
    #             return sensor_data
            
    #         # Choose processing method based on sensor count
    #         return jax.lax.cond(
    #             nsensor == 1,
    #             single_sensor,
    #             multiple_sensors
    #         )
        
    #     sensor_data = process_sensors()
    #     return sensor_data, sensor_names



    
    def get_grf(self, mjx_data, foot_name):
        """
        Get the ground reaction forces (GRF) from the Mujoco data.
        This function extracts the GRF data from the Mujoco data structure.
        """
        # if foot_side == 'left_side':
        #     foot_name = f"{foot_name}_l"
        # elif foot_side == 'right_side':
        #     foot_name = f"{foot_name}_r"

        # Get the body IDs for left and right foot
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, foot_name)


        # Extract the GRF for left and right foot
        grf = mjx_data.cfrc_ext[0][body_id]


        # self.all_grf_left.append(grf_l)
        # self.all_grf_right.append(grf_r)

        return grf
    


    def get_grf_batched(self, mjx_data, foot_name):
        """
        Get the ground reaction forces (GRF) from the Mujoco data.
        This function extracts the GRF data from the Mujoco data structure.
        """
        # if foot_side == 'left_side':
        #     foot_name = f"{foot_name}_l"
        # elif foot_side == 'right_side':
        #     foot_name = f"{foot_name}_r"

        # Get the body IDs for left and right foot
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, foot_name)


        # Extract the GRF for left and right foot
        grf = mjx_data.cfrc_ext[:, body_id,:] #...][body_id]


        # self.all_grf_left.append(grf_l)
        # self.all_grf_right.append(grf_r)

        return grf
    

    def filter_start_step_indices(self, all_step_start, min_walk_step_length):
        """
        Filter the start step indices to ensure they are sufficiently spaced apart.
        This function removes start steps that are too close together, based on a minimum step length.
        
        Args:
        all_step_start (list): The list of start step indices.
        min_walk_step_length (int): The minimum length of a walk step to consider it valid.

        Returns:
        list: A filtered list of start step indices.
        """
        # if difference between start_step[i] and start_step[i+1] is smaller than 50 than delete start_step[i]
        filtered_start_steps = []
        for i in range(len(all_step_start) - 1):
            if all_step_start[i + 1] - all_step_start[i] > min_walk_step_length:
                filtered_start_steps.append(all_step_start[i])

        return filtered_start_steps
    


    def get_parameter_per_step(self, parameter_data, start_indices):
        """
        Get the parameter values for each step based on the start indices until the next start index -1 .
        This function extracts the specified parameter values from the saved parameter data.
        
        Args:
        chosen_parameter (str): The parameter to extract. Options: "joint_angles", "joint_velocities", "joint_forces", "joint_torques", "muscle_actions".
        start_indices (list): The start indices of the steps.

        Returns:
        list: A list of parameter values for each step.
        """
        all_parameter_per_step = []
        
        for i in range(len(start_indices) - 1):
            start_index = start_indices[i]
            end_index = start_indices[i + 1] - 1

            parameter_per_step = parameter_data[start_index:end_index]  # Extract the parameter values for the step

            parameter_per_step = jnp.array(parameter_per_step)  # Convert to JAX array for further processing
            all_parameter_per_step.append(parameter_per_step)  # Append the parameter values for the step to the list
        
        return all_parameter_per_step
    




    # def extract_steps(self, data, step, all_data):
    #     """
    #     Extract the steps from the data based on the ground reaction forces.
    #     Return the indices of the steps for left and right. Gets the first indix of grf not equal to zero for each step 
    #     """
    #     left_steps = jnp.array([])  # Initialize as an empty array
    #     right_steps = jnp.array([])  # Initialize as an empty array
        

    #     for n in range(len(all_data)):
    #         data = all_data[n]
           
    #         for i in range(data.ncon):
    #             geom1_id = data.contact.geom1[i]
    #             geom2_id = data.contact.geom2[i]

    #             # print('geom1_id:', geom1_id, 'geom2_id:', geom2_id)
                
    #             # Get names if needed
    #             geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
    #             geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)

    #             # print(f"Contact between {geom1_name} and {geom2_name}")

    #             if geom2_name == "foot_box_l":
    #                 # check if force is not zero 
    #                 result = np.zeros(6, dtype=np.float64)  # 3 force + 3 torque
    #                 result = mujoco.mj_contactForce(self.model, data, i, result)
    #                 # Check when 0:3 in results is not zero, save left step index 
    #                 # print(f"Contact Force Result: {result}")
    #                 if np.any(result[0:3] != 0):
    #                     left_step_index = step #data.time[i]
    #                     # print(f"Left step detected at index {left_step_index}")
    #                     left_steps = jnp.array([left_step_index])
    #             elif geom2_name == "foot_box_r":
    #                 # check if force is not zero 
    #                 result = np.zeros(6, dtype=np.float64)  # 3 force + 3 torque
    #                 result = mujoco.mj_contactForce(self.model, data, i, result)
    #                 # print(f"Contact Force Result: {result}")
    #                 # Check when 0:3 in results is not zero, save right step index
    #                 if np.any(result[0:3] != 0):
    #                     right_step_index = step #data.time[i]
    #                     # print(f"Right step detected at index {right_step_index}")
    #                     right_steps = jnp.array([right_step_index]) 

    #     right_indices = []
    #     left_indices = []

    #     # Only get the start of the step, so not consequitive in left_steps and right_steps
    #     if left_steps.size != 0:
    #         for i in range(1, len(left_steps)):
    #             if left_steps[i] == left_steps[i-1] + 1:
    #                 left_steps = jnp.delete(left_steps, i)
    #         for i in range(len(left_steps) - 1):
    #             start_index = left_steps[i]
    #             end_index = left_steps[i + 1] - 1
    #             left_indices.extend([start_index, end_index + 1])
    #         left_indices = jnp.array(left_indices)
    #     if right_steps.size != 0:
    #         for i in range(1, len(right_steps)):
    #             if right_steps[i] == right_steps[i-1] + 1:
    #                 right_steps = jnp.delete(right_steps, i)
    #         for i in range(len(right_steps) - 1):
    #             start_index = right_steps[i]
    #             end_index = right_steps[i + 1] - 1
    #             right_indices.extend([start_index, end_index + 1])
    #         right_indices = jnp.array(right_indices)


    #     # Get start and end of each step 
    #     # left_indices = []
    #     # for i in range(len(left_steps) - 1):
    #     #     start_index = left_steps[i]
    #     #     end_index = left_steps[i + 1] - 1
    #     #     left_indices.extend([start_index, end_index + 1])
    #     # left_indices = jnp.array(left_indices)

    #     # right_indices = []
    #     # for i in range(len(right_steps) - 1):
    #     #     start_index = right_steps[i]
    #     #     end_index = right_steps[i + 1] - 1
    #     #     right_indices.extend([start_index, end_index + 1])
    #     # right_indices = jnp.array(right_indices)
       
    #     return left_indices, right_indices
    

    # def get knee contact force 

    
class PostProcessMetricsHandler(): 

    def get_start_steps(self, all_step_contact, min_walk_step_length):
        """
        Get the start steps for left and right foot based on contact data.
        This function processes the contact data to determine the start of each step.
        Args:
        all_step_contact (list): A list of contact indices where steps occur.
        min_walk_step_length (int): The minimum length of a walk step to consider it valid.
        """
        all_step_start  = []
        if all_step_contact[0]!= 0:
            all_step_start = [all_step_contact[0]]
        for m in range(len(all_step_contact) - 1):
            if all_step_contact[m + 1] - all_step_contact[m] > 1: #min_walk_step_length:
                all_step_start.append(all_step_contact[m+1])
        
        filtered_start_steps = []
        for i in range(len(all_step_start) - 1):
            if all_step_start[i + 1] - all_step_start[i] > min_walk_step_length:
                filtered_start_steps.append(all_step_start[i])
        # if all_step_start[m] - all_start_step[m+1] > min_walk_step_length:
        #     all_step_start.append(all_step_contact[m+1])
        return filtered_start_steps #all_step_start
    
    def get_start_steps_from_grfZ(self, grfZ, min_walk_step_length=60, threshold=0.5):
        """
        Detect step start indices from vertical GRF data.

        Args:
            grfZ (list or np.array): Vertical ground reaction force data.
            min_walk_step_length (int): Minimum number of frames between steps.
            threshold (float): GRF threshold to detect foot contact.

        Returns:
            list: Indices where steps start.
        """
        # Alternative only next time step still above 0 
        # grfZ = np.array(grfZ)
        # above_threshold = grfZ > threshold
        # step_starts = []

        # for i in range(1, len(above_threshold)):
        #     # Detect rising edge: from below to above threshold
        #     if above_threshold[i] and not above_threshold[i - 1] and above_threshold[i+2]:
        #         if not step_starts or (i - step_starts[-1]) > min_walk_step_length:
        #             step_starts.append(i)

        # return step_starts
        

        # Alternative only next 20 time steps still above 0 
        grfZ = np.array(grfZ)
        above_threshold = grfZ > threshold
        step_starts = []

        for i in range(1, len(above_threshold) - 20):  # Ensure we have 20 values ahead
            # Detect rising edge and check next 20 values
            if above_threshold[i] and not above_threshold[i - 1]:
                if np.all(grfZ[i:i+20] > 0):
                    if not step_starts or (i - step_starts[-1]) > min_walk_step_length:
                        step_starts.append(i)

        return step_starts

        # # Alternative only 20 time steps before has to be 0 
        # grfZ = np.array(grfZ)
        # above_threshold = grfZ > threshold
        # step_starts = []

        # for i in range(30, len(above_threshold)):  # Ensure we have 30 values before and 20 ahead
        #     # Check if GRF was zero for at least 30 frames before
        #     if np.all(grfZ[i-30:i] == 0):
        #         # Detect rising edge and check next 20 values
        #         if above_threshold[i] and not above_threshold[i - 1]:
        #             if not step_starts or (i - step_starts[-1]) > min_walk_step_length:
        #                 step_starts.append(i)

        # return step_starts
        
    
    def get_contact_lengths(self, contact_data, min_walk_step_length):
        """
        Identify contact segments and compute their lengths based on gaps in contact data.
        
        Args:
        contact_data (list): A list of contact indices (integers).
        min_walk_step_length (int): Minimum length of a valid contact segment.
        
        Returns:
        list: Lengths of valid contact segments.
        """
        contact_lengths = []
        start_index = contact_data[0]

        for i in range(1, len(contact_data)):
            if contact_data[i] - contact_data[i - 1] > 1:
                end_index = contact_data[i - 1]
                length = end_index - start_index + 1
                if length >= min_walk_step_length:
                    contact_lengths.append(length)
                start_index = contact_data[i]

        # Handle the final segment
        final_length = contact_data[-1] - start_index + 1
        if final_length >= min_walk_step_length:
            contact_lengths.append(final_length)

        return contact_lengths


    def get_contact_lengths_from_grfZ(self, grfZ, min_walk_step_length=20, threshold=50):
        """
        Detect contact lengths from vertical GRF data.

        Args:
            grfZ (list or np.array): Vertical ground reaction force data.
            min_walk_step_length (int): Minimum number of frames to consider a valid contact.
            threshold (float): GRF threshold to define contact (default is 50 N).

        Returns:
            list: Lengths of valid contact segments.
        """


        grfZ = np.array(grfZ)
        above_threshold = grfZ > threshold

        contact_lengths = []
        in_contact = False
        start_idx = None

        for i, val in enumerate(above_threshold):
            if val and not in_contact:
                # Start of contact
                in_contact = True
                start_idx = i
            elif not val and in_contact:
                # End of contact
                end_idx = i
                length = end_idx - start_idx
                if length >= min_walk_step_length:
                    contact_lengths.append(length)
                in_contact = False

        # Handle case where contact continues till the end
        if in_contact:
            length = len(grfZ) - start_idx
            if length >= min_walk_step_length:
                contact_lengths.append(length)

        return contact_lengths

    def get_contact_from_grfZ(self, grfZ, threshold=50):
        grfZ = np.array(grfZ)
        above_threshold = grfZ > threshold
        in_contact = 100*above_threshold
        return in_contact

    def compute_single_support_time(self ,in_contact_right, in_contact_left, total_steps = 1000):
        """
        Computes single support time for left and right legs.
        
        Parameters:
            in_contact_right (list of bool): Right foot contact per frame
            in_contact_left (list of bool): Left foot contact per frame
            frame_rate (int): Frames per second (default 100 Hz)
        
        Returns:
            dict: Single support time in seconds for left and right
        """
        single_support_right_frames = 0
        single_support_left_frames = 0

        for r, l in zip(in_contact_right, in_contact_left):
            if r and not l:
                single_support_right_frames += 1
            elif l and not r:
                single_support_left_frames += 1

        single_support_right_time = single_support_right_frames/1000
        single_support_left_time = single_support_left_frames/1000


        return single_support_right_time, single_support_left_time


    def compute_single_support_per_step(self, in_contact_right,in_contact_left, step_start_right, step_start_left):
        """
        Computes single support time per step for left and right legs.

        Parameters:
            run_data (dict): Contains 'in_contact_right', 'in_contact_left',
                            'step_start_right', 'step_start_left'
            frame_rate (int): Sampling rate in Hz

        Returns:
            dict: Lists of single support times per step for left and right
        """
        # in_contact_right = run_data["in_contact_right"]
        # in_contact_left = run_data["in_contact_left"]
        # step_start_right = run_data["step_start_right"]
        # step_start_left = run_data["step_start_left"]

        step_count = min(len(step_start_right), len(step_start_left)) - 1
        if step_count < 1:
            return {"right": [], "left": []}

        single_support_right = []
        single_support_left = []

        for i in range(step_count):
            # Right step window
            start_r = step_start_right[i]
            end_r = step_start_right[i + 1]
            ss_r_frames = sum(
                1 for r, l in zip(in_contact_right[start_r:end_r], in_contact_left[start_r:end_r])
                if r and not l
            )
            single_support_right.append(ss_r_frames)

            # Left step window
            start_l = step_start_left[i]
            end_l = step_start_left[i + 1]
            ss_l_frames = sum(
                1 for r, l in zip(in_contact_right[start_l:end_l], in_contact_left[start_l:end_l])
                if l and not r
            )
            single_support_left.append(ss_l_frames)

        return single_support_right, single_support_left
    


    def compute_double_support_per_step(self, in_contact_right,in_contact_left, step_start_right, step_start_left):
        """
        Compute double support time per step for left and right legs.
        """
        step_count = min(len(step_start_right), len(step_start_left)) - 1
        if step_count < 1:
            return []

        double_support = []

        for i in range(step_count):
            # Right step window
            start_r = step_start_right[i]
            end_r = step_start_right[i + 1]
            ds_r_frames = sum(
                1 for r, l in zip(in_contact_right[start_r:end_r], in_contact_left[start_r:end_r])
                if r and l
            )
            double_support.append(ds_r_frames)

        return double_support


    def compute_step_cadence(self, episode_length, in_contact_right,in_contact_left, step_start_right, step_start_left):
        """
        Compute step cadence in walking steps per minute.
        Look for how many steps take place on average in 6000 steps
        """
        step_count = min(len(step_start_right), len(step_start_left)) - 1
        if step_count < 1:
            return 0

        # Compute amount of steps in 6000 frames
        steps_in_6000 = (6000/episode_length) *step_count
        # steps_in_6000 = 6000 / (step_count / 60) if step_count > 0 else 0

        # step_count = min(len(step_start_right), len(step_start_left)) - 1
        # if step_count < 1:
        #     return 0

        # # Compute total step duration in seconds
        # total_duration = 0
        # for i in range(step_count):
        #     start_r = step_start_right[i]
        #     end_r = step_start_right[i + 1]
        #     start_l = step_start_left[i]
        #     end_l = step_start_left[i + 1]
        #     duration = max(end_r, end_l) - min(start_r, start_l)
        #     total_duration *= 100 # 1000 steps is 10 seconds
        #     total_duration += duration

        # # Convert to minutes and compute cadence
        # total_duration_minutes = total_duration / 60
        # cadence = step_count / total_duration_minutes if total_duration_minutes > 0 else 0
        # return cadence
        return steps_in_6000
    

    @staticmethod
    def _get_step_slices(data, step_indices):
        steps = []
        for i in range(len(step_indices) - 1):
            start = step_indices[i]
            end = step_indices[i + 1]
            steps.append(data[start:end])
        return steps

    def _plot_steps(
        self,
        steps, 
        ax, 
        interp_mode="none", 
        interp_len=None, 
        deg=False, 
        legend_prefix="", 
        side="left", 
        col=None
    ):
        for i, step in enumerate(steps):
            if len(step) < 2:
                continue
            if col is not None:
                y = [float(np.array(a)[0, col] if np.array(a).ndim > 1 else np.array(a)[col]) for a in step]
            else:
                y = [float(a) for a in step]
            if interp_mode == "interp":
                x_old = np.linspace(0, 1, len(y))
                x_new = np.linspace(0, 1, interp_len or min([len(s) for s in steps if len(s) > 1]))
                y = np.interp(x_new, x_old, y)
            if deg:
                y = np.rad2deg(y)
            ax.plot(y, label=f"{legend_prefix}{side} step {i+1}")

    def plot_parameter_per_step(
        self,
        data,
        step_indices_left,
        step_indices_right,
        param_name="parameter",
        interp_mode="none",
        interp_len=None,
        columns=None,
        ylabel=None,
        title=None,
        legend_prefix="",
        deg=False,
    ):
        if isinstance(data, dict):
            nplots = len(data)
            fig, axes = plt.subplots(nplots, 1, figsize=(10, 3 * nplots), sharex=True)
            if nplots == 1:
                axes = [axes]
            for ax, (key, arr) in zip(axes, data.items()):
                self.plot_parameter_per_step(
                    arr,
                    step_indices_left,
                    step_indices_right,
                    param_name=param_name,
                    interp_mode=interp_mode,
                    interp_len=interp_len,
                    columns=columns,
                    ylabel=ylabel or param_name,
                    title=title or key,
                    legend_prefix=key + " ",
                    deg=deg,
                )
                ax.set_title(key)
            plt.tight_layout()
            plt.show()
            return

        if isinstance(data, list) and len(data) > 0 and hasattr(data[0], "shape"):
            arr0 = np.array(data[0])
            ncols = arr0.shape[-1] if arr0.ndim > 1 else 1
            if columns is None:
                columns = list(range(ncols))
            elif isinstance(columns, int):
                columns = [columns]
            nplots = len(columns)
            fig, axes = plt.subplots(nplots, 1, figsize=(10, 3 * nplots), sharex=True)
            if nplots == 1:
                axes = [axes]
            for idx, col in enumerate(columns):
                ax = axes[idx]
                if step_indices_left and len(step_indices_left) > 1:
                    steps = self._get_step_slices(data, step_indices_left)
                    self._plot_steps(
                        steps, ax, interp_mode, interp_len, deg, legend_prefix, "left", col
                    )
                if step_indices_right and len(step_indices_right) > 1:
                    steps = self._get_step_slices(data, step_indices_right)
                    self._plot_steps(
                        steps, ax, interp_mode, interp_len, deg, legend_prefix, "right", col
                    )
                ax.set_ylabel(ylabel or f"{param_name} col {col}")
                ax.legend()
                ax.set_title(title or f"{param_name} col {col}")
            axes[-1].set_xlabel("Interpolated Step (%)" if interp_mode == "interp" else "Step index")
            plt.tight_layout()
            plt.show()
            return

        fig, ax = plt.subplots(figsize=(10, 3))
        if step_indices_left and len(step_indices_left) > 1:
            steps = self._get_step_slices(data, step_indices_left)
            self._plot_steps(
                steps, ax, interp_mode, interp_len, deg, legend_prefix, "left"
            )
        if step_indices_right and len(step_indices_right) > 1:
            steps = self._get_step_slices(data, step_indices_right)
            self._plot_steps(
                steps, ax, interp_mode, interp_len, deg, legend_prefix, "right"
            )
        ax.set_ylabel(ylabel or param_name)
        ax.set_title(title or param_name)
        ax.legend()
        ax.set_xlabel("Interpolated Step (%)" if interp_mode == "interp" else "Step index")
        plt.tight_layout()
        plt.show()

    def plot_parameter_mean_per_step(
        self,
        data,
        step_indices_left,
        step_indices_right,
        param_name="parameter",
        interp_mode="none",
        interp_len=None,
        columns=None,
        ylabel=None,
        title=None,
        legend_prefix="",
        deg=False,
    ):

        def compute_mean_std(steps, interp_mode, interp_len, deg, col=None):
            y_steps = []
            for step in steps:
                if len(step) < 2:
                    continue
                if col is not None:
                    y = [float(np.array(a)[0, col] if np.array(a).ndim > 1 else np.array(a)[col]) for a in step]
                else:
                    y = [float(a) for a in step]
                if interp_mode == "interp":
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len or min([len(s) for s in steps if len(s) > 1]))
                    y = np.interp(x_new, x_old, y)
                if deg:
                    y = np.rad2deg(y)
                y_steps.append(y)
            if not y_steps:
                return None, None
            y_steps = np.array(y_steps)
            mean = np.mean(y_steps, axis=0)
            std = np.std(y_steps, axis=0)
            return mean, std

        if isinstance(data, dict):
            nplots = len(data)
            fig, axes = plt.subplots(nplots, 1, figsize=(10, 3 * nplots), sharex=True)
            if nplots == 1:
                axes = [axes]
            for ax, (key, arr) in zip(axes, data.items()):
                self.plot_parameter_mean_per_step(
                    arr,
                    step_indices_left,
                    step_indices_right,
                    param_name=param_name,
                    interp_mode=interp_mode,
                    interp_len=interp_len,
                    columns=columns,
                    ylabel=ylabel or param_name,
                    title=title or key,
                    legend_prefix=key + " ",
                    deg=deg,
                )
                ax.set_title(key)
            plt.tight_layout()
            plt.show()
            return

        if isinstance(data, list) and len(data) > 0 and hasattr(data[0], "shape"):
            arr0 = np.array(data[0])
            ncols = arr0.shape[-1] if arr0.ndim > 1 else 1
            if columns is None:
                columns = list(range(ncols))
            elif isinstance(columns, int):
                columns = [columns]
            nplots = len(columns)
            fig, axes = plt.subplots(nplots, 1, figsize=(10, 3 * nplots), sharex=True)
            if nplots == 1:
                axes = [axes]
            for idx, col in enumerate(columns):
                ax = axes[idx]
                if step_indices_left and len(step_indices_left) > 1:
                    steps = self._get_step_slices(data, step_indices_left)
                    mean, std = compute_mean_std(steps, interp_mode, interp_len, deg, col)
                    if mean is not None:
                        x = np.arange(len(mean))
                        ax.plot(x, mean, label=f"{legend_prefix}left mean")
                        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
                if step_indices_right and len(step_indices_right) > 1:
                    steps = self._get_step_slices(data, step_indices_right)
                    mean, std = compute_mean_std(steps, interp_mode, interp_len, deg, col)
                    if mean is not None:
                        x = np.arange(len(mean))
                        ax.plot(x, mean, label=f"{legend_prefix}right mean")
                        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
                ax.set_ylabel(ylabel or f"{param_name} col {col}")
                ax.legend()
                ax.set_title(title or f"{param_name} col {col}")
            axes[-1].set_xlabel("Interpolated Step (%)" if interp_mode == "interp" else "Step index")
            plt.tight_layout()
            plt.show()
            return

        fig, ax = plt.subplots(figsize=(10, 3))
        if step_indices_left and len(step_indices_left) > 1:
            steps = self._get_step_slices(data, step_indices_left)
            mean, std = compute_mean_std(steps, interp_mode, interp_len, deg)
            if mean is not None:
                x = np.arange(len(mean))
                ax.plot(x, mean, label=f"{legend_prefix}left mean")
                ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        if step_indices_right and len(step_indices_right) > 1:
            steps = self._get_step_slices(data, step_indices_right)
            mean, std = compute_mean_std(steps, interp_mode, interp_len, deg)
            if mean is not None:
                x = np.arange(len(mean))
                ax.plot(x, mean, label=f"{legend_prefix}right mean")
                ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        ax.set_ylabel(ylabel or param_name)
        ax.set_title(title or param_name)
        ax.legend()
        ax.set_xlabel("Interpolated Step (%)" if interp_mode == "interp" else "Step index")
        plt.tight_layout()
        plt.show()

    def plot_joint_parameter_per_step(
        self,
        joint_pairs,
        joint_data,
        step_indices_left,
        step_indices_right,
        parameter_names,
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=None,
    ):
        def get_step_indices(joint):
            if joint.endswith("_l"):
                return step_indices_left
            elif joint.endswith("_r"):
                return step_indices_right
            else:
                return [step_indices_left, step_indices_right]

        if interp_mode == "interp" and interp_len is None:
            step_lengths = []
            for pair in joint_pairs:
                for joint in pair:
                    indices = get_step_indices(joint)
                    if isinstance(indices, list) and indices and isinstance(indices[0], list):
                        for idx in indices:
                            if idx and len(idx) > 1:
                                step_lengths += [idx[i+1] - idx[i] for i in range(len(idx)-1)]
                    elif indices and len(indices) > 1:
                        step_lengths += [indices[i+1] - indices[i] for i in range(len(indices)-1)]
            interp_len = min(step_lengths) if step_lengths else 1

        n_rows = len(joint_pairs)
        n_cols = len(parameter_names)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes] if n_rows > 1 else [[axes]]

        for row_idx, pair in enumerate(joint_pairs):
            for col_idx, parameter_name in enumerate(parameter_names):
                ax = axes[row_idx][col_idx]
                for joint in pair:
                    if joint not in joint_data or parameter_name not in joint_data[joint]:
                        continue
                    indices = get_step_indices(joint)
                    if isinstance(indices, list) and indices and isinstance(indices[0], list):
                        for side_indices, side in zip(indices, ["left", "right"]):
                            self._plot_joint_steps(
                                ax, joint, joint_data[joint][parameter_name], side_indices, interp_mode, interp_len, convert_to_deg, parameter_name, side
                            )
                    else:
                        side = "left" if joint.endswith("_l") else "right" if joint.endswith("_r") else ""
                        self._plot_joint_steps(
                            ax, joint, joint_data[joint][parameter_name], indices, interp_mode, interp_len, convert_to_deg, parameter_name, side
                        )
                ax.set_ylabel(f"{parameter_name.capitalize()} ({'deg' if convert_to_deg and parameter_name == 'angle' else 'rad'})")
                ax.set_title(" / ".join(pair) + f" - {parameter_name}")
                ax.legend()
        for col_idx in range(n_cols):
            axes[-1][col_idx].set_xlabel("Interpolated Step (%)" if interp_mode == "interp" else "Step index")
        plt.tight_layout()
        plt.show()

    def _plot_joint_steps(self, ax, joint, data, step_indices, interp_mode, interp_len, convert_to_deg, parameter_name, side):
        if step_indices is None or len(step_indices) < 2:
            return
        for i in range(len(step_indices) - 1):
            start = step_indices[i]
            end = step_indices[i + 1]
            y = [float(np.array(a).squeeze()) for a in data[start:end]]
            if len(y) < 2:
                continue
            if interp_mode == "interp":
                x_old = np.linspace(0, 1, len(y))
                x_new = np.linspace(0, 1, interp_len)
                y = np.interp(x_new, x_old, y)
            if convert_to_deg and parameter_name in ("angle", "velocity"):
                y = np.rad2deg(y)
            label = f"{joint} {side} step {i+1}" if side else f"{joint} step {i+1}"
            ax.plot(y, label=label)

    def plot_joint_parameter_mean_per_step(
        self,
        joint_pairs,
        joint_data,
        step_indices_left,
        step_indices_right,
        parameter_names,
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=None,
    ):
        def get_step_indices(joint):
            if joint.endswith("_l"):
                return step_indices_left
            elif joint.endswith("_r"):
                return step_indices_right
            else:
                return [step_indices_left, step_indices_right]

        if interp_mode == "interp" and interp_len is None:
            step_lengths = []
            for pair in joint_pairs:
                for joint in pair:
                    indices = get_step_indices(joint)
                    if isinstance(indices, list) and indices and isinstance(indices[0], list):
                        for idx in indices:
                            if idx and len(idx) > 1:
                                step_lengths += [idx[i+1] - idx[i] for i in range(len(idx)-1)]
                    elif indices and len(indices) > 1:
                        step_lengths += [indices[i+1] - indices[i] for i in range(len(indices)-1)]
            interp_len = min(step_lengths) if step_lengths else 1

        n_rows = len(joint_pairs)
        n_cols = len(parameter_names)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes] if n_rows > 1 else [[axes]]

        for row_idx, pair in enumerate(joint_pairs):
            for col_idx, parameter_name in enumerate(parameter_names):
                ax = axes[row_idx][col_idx]
                for joint in pair:
                    if joint not in joint_data or parameter_name not in joint_data[joint]:
                        continue
                    indices = get_step_indices(joint)
                    if isinstance(indices, list) and indices and isinstance(indices[0], list):
                        for side_indices, side in zip(indices, ["left", "right"]):
                            self._plot_joint_mean_std(
                                ax, joint, joint_data[joint][parameter_name], side_indices, interp_mode, interp_len, convert_to_deg, parameter_name, side
                            )
                    else:
                        side = "left" if joint.endswith("_l") else "right" if joint.endswith("_r") else ""
                        self._plot_joint_mean_std(
                            ax, joint, joint_data[joint][parameter_name], indices, interp_mode, interp_len, convert_to_deg, parameter_name, side
                        )
                ax.set_ylabel(f"{parameter_name.capitalize()} ({'deg' if convert_to_deg and parameter_name == 'angle' else 'rad'})")
                ax.set_title(" / ".join(pair) + f" - {parameter_name} (mean ± std)")
                ax.legend()
        for col_idx in range(n_cols):
            axes[-1][col_idx].set_xlabel("Interpolated Step (%)" if interp_mode == "interp" else "Step index")
        plt.tight_layout()
        plt.show()

    def _plot_joint_mean_std(self, ax, joint, data, step_indices, interp_mode, interp_len, convert_to_deg, parameter_name, side):
        if step_indices is None or len(step_indices) < 2:
            return
        y_steps = []
        for i in range(len(step_indices) - 1):
            start = step_indices[i]
            end = step_indices[i + 1]
            y = [float(np.array(a).squeeze()) for a in data[start:end]]
            if len(y) < 2:
                continue
            if interp_mode == "interp":
                x_old = np.linspace(0, 1, len(y))
                x_new = np.linspace(0, 1, interp_len)
                y = np.interp(x_new, x_old, y)
            if convert_to_deg and parameter_name in ("angle", "velocity"):
                y = np.rad2deg(y)
            y_steps.append(y)
        if not y_steps:
            return
        y_steps = np.array(y_steps)
        mean = np.mean(y_steps, axis=0)
        std = np.std(y_steps, axis=0)
        label = f"{joint} {side} mean" if side else f"{joint} mean"
        x = np.arange(len(mean))
        ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    def plot_muscle_activations_per_step(
        self,
        muscle_actions,
        muscle_names,
        step_indices_left,
        step_indices_right,
        interp_mode="interp",
        interp_len=None,
    ):
        if interp_mode == "interp" and interp_len is None:
            step_lengths = []
            for name in muscle_names:
                steps = step_indices_left if name.endswith("_l") else step_indices_right if name.endswith("_r") else None
                if steps and len(steps) > 1:
                    step_lengths += [steps[i+1] - steps[i] for i in range(len(steps)-1)]
            interp_len = min(step_lengths) if step_lengths else 1

        n = len(muscle_names)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for i, name in enumerate(muscle_names):
            ax = axes[i]
            actions = muscle_actions.get(name, None)
            if actions is None:
                ax.set_title(f"{name} (no data)")
                continue
            if name.endswith("_l"):
                steps = step_indices_left
                side = "Left"
            elif name.endswith("_r"):
                steps = step_indices_right
                side = "Right"
            else:
                ax.set_title(f"{name} (unknown side)")
                continue
            if steps and len(steps) > 1:
                for j in range(len(steps) - 1):
                    start = steps[j]
                    end = steps[j + 1]
                    y = [float(np.array(a).squeeze()) for a in actions[start:end]]
                    if len(y) < 2:
                        continue
                    if interp_mode == "interp":
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y = np.interp(x_new, x_old, y)
                    ax.plot(y, label=f"{side} step {j+1}")
            ax.set_title(f"{name} Activation per Step")
            ax.set_ylabel("Activation Level")
            ax.legend()
        axes[-1].set_xlabel("Interpolated Step (%)" if interp_mode == "interp" else "Step index")
        plt.tight_layout()
        plt.show()


    def plot_muscle_activations_mean_per_step(
        self,
        muscle_actions,
        muscle_names,
        step_indices_left,
        step_indices_right,
        interp_mode="interp",
        interp_len=None,
    ):
        if interp_mode == "interp" and interp_len is None:
            step_lengths = []
            for name in muscle_names:
                steps = step_indices_left if name.endswith("_l") else step_indices_right if name.endswith("_r") else None
                if steps and len(steps) > 1:
                    step_lengths += [steps[i+1] - steps[i] for i in range(len(steps)-1)]
            interp_len = min(step_lengths) if step_lengths else 1

        n = len(muscle_names)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for i, name in enumerate(muscle_names):
            ax = axes[i]
            actions = muscle_actions.get(name, None)
            if actions is None:
                ax.set_title(f"{name} (no data)")
                continue
            if name.endswith("_l"):
                steps = step_indices_left
                side = "Left"
            elif name.endswith("_r"):
                steps = step_indices_right
                side = "Right"
            else:
                ax.set_title(f"{name} (unknown side)")
                continue
            y_steps = []
            if steps and len(steps) > 1:
                for j in range(len(steps) - 1):
                    start = steps[j]
                    end = steps[j + 1]
                    y = [float(np.array(a).squeeze()) for a in actions[start:end]]
                    if len(y) < 2:
                        continue
                    if interp_mode == "interp":
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y = np.interp(x_new, x_old, y)
                    y_steps.append(y)
            if y_steps:
                y_steps = np.array(y_steps)
                mean = np.mean(y_steps, axis=0)
                std = np.std(y_steps, axis=0)
                x = np.arange(len(mean))
                ax.plot(x, mean, label=f"{side} mean")
                ax.fill_between(x, mean - std, mean + std, alpha=0.2)
            ax.set_title(f"{name} Activation Mean per Step")
            ax.set_ylabel("Activation Level")
            ax.legend()
        axes[-1].set_xlabel("Interpolated Step (%)" if interp_mode == "interp" else "Step index")
        plt.tight_layout()
        plt.show()






    def plot_muscle_activation_symmetry(
        self,
        muscle_names_list,
        all_actuator_names,
        all_actions,
        all_step_start_left,
        all_step_start_right,
        interp_len=100,
    ):
        def find_actuator_index(actuator_names, name):
            try:
                return actuator_names.index(name)
            except ValueError:
                print(f"{name} not found in all_actuator_names.")
                return None

        def collect_activations(actions, idx):
            return [actions[i][0][idx] for i in range(len(actions))]

        def compute_interpolated_steps(acts, step_starts, interp_len):
            steps = []
            if step_starts and len(step_starts) > 1:
                for j in range(len(step_starts) - 1):
                    start, end = step_starts[j], step_starts[j + 1]
                    y = [float(a) for a in acts[start:end]]
                    if len(y) < 2:
                        continue
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    y_interp = np.interp(x_new, x_old, y)
                    steps.append(y_interp)
            return steps

        for muscle_name in muscle_names_list:
            left_name = muscle_name + "_l"
            right_name = muscle_name + "_r"
            idx_left = find_actuator_index(all_actuator_names, left_name)
            idx_right = find_actuator_index(all_actuator_names, right_name)
            if idx_left is None or idx_right is None:
                continue

            left_acts = collect_activations(all_actions, idx_left)
            right_acts = collect_activations(all_actions, idx_right)

            left_steps = compute_interpolated_steps(left_acts, all_step_start_left, interp_len)
            right_steps = compute_interpolated_steps(right_acts, all_step_start_right, interp_len)

            if left_steps and right_steps:
                mean_left = np.mean(left_steps, axis=0)
                std_left = np.std(left_steps, axis=0)
                mean_right = np.mean(right_steps, axis=0)
                std_right = np.std(right_steps, axis=0)
                diff = mean_left - mean_right
                x = np.linspace(0, 100, interp_len)
                plt.figure(figsize=(10, 5))
                plt.plot(x, mean_left, label=f"{muscle_name}_l mean", color='blue')
                plt.fill_between(x, mean_left - std_left, mean_left + std_left, color='blue', alpha=0.2)
                plt.plot(x, mean_right, label=f"{muscle_name}_r mean", color='red')
                plt.fill_between(x, mean_right - std_right, mean_right + std_right, color='red', alpha=0.2)
                plt.plot(x, diff, label="Difference (left - right)", color='black', linestyle='--')
                plt.axhline(0, color='gray', linestyle=':', linewidth=1)
                plt.title(f"Activation symmetry: {muscle_name}_l vs {muscle_name}_r")
                plt.xlabel("Interpolated Step (%)")
                plt.ylabel("Activation / Difference")
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                print(f"Not enough steps for {muscle_name} to compute difference.")





    def plot_grf_component_symmetry(
        self,
        grf_components,
        all_grf_l,
        all_grf_r,
        all_step_start_left,
        all_step_start_right,
        interp_len=100,
    ):
        """
        Plot mean and std of GRF components for left and right steps, and their difference.
        """
        for i, comp in enumerate(grf_components):
            # Compute mean per time step in a walk step for left
            left_steps = []
            if all_grf_l and all_step_start_left and len(all_step_start_left) > 1:
                for j in range(len(all_step_start_left) - 1):
                    start, end = all_step_start_left[j]-1, all_step_start_left[j + 1]-1
                    y = [float(np.array(a)[i]) for a in all_grf_l[start:end]]
                    if len(y) < 2:
                        continue
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    y_interp = np.interp(x_new, x_old, y)
                    left_steps.append(y_interp)
            # Compute mean per time step in a walk step for right
            right_steps = []
            if all_grf_r and all_step_start_right and len(all_step_start_right) > 1:
                for j in range(len(all_step_start_right) - 1):
                    start, end = all_step_start_right[j]-1, all_step_start_right[j + 1]-1
                    y = [float(np.array(a)[i]) for a in all_grf_r[start:end]]
                    if len(y) < 2:
                        continue
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    y_interp = np.interp(x_new, x_old, y)
                    right_steps.append(y_interp)
            # Calculate mean and std
            if left_steps and right_steps:
                mean_left = np.mean(left_steps, axis=0)
                std_left = np.std(left_steps, axis=0)
                mean_right = np.mean(right_steps, axis=0)
                std_right = np.std(right_steps, axis=0)
                diff = mean_left - mean_right
                x = np.linspace(0, 100, interp_len)
                plt.figure(figsize=(10, 5))
                # Plot mean and std for left
                plt.plot(x, mean_left, label=f"Left mean", color='blue')
                plt.fill_between(x, mean_left - std_left, mean_left + std_left, color='blue', alpha=0.2)
                # Plot mean and std for right
                plt.plot(x, mean_right, label=f"Right mean", color='red')
                plt.fill_between(x, mean_right - std_right, mean_right + std_right, color='red', alpha=0.2)
                # Plot difference
                plt.plot(x, diff, label="Difference (left - right)", color='black', linestyle='--')
                plt.axhline(0, color='gray', linestyle=':', linewidth=1)
                plt.title(f"GRF {comp}: mean, std, and difference per time step")
                plt.xlabel("Interpolated Step (%)")
                plt.ylabel(f"{comp} (N or Nm)")
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                print(f"Not enough steps for GRF {comp} to compute difference.")



    def plot_sensor_force_symmetry(
        self,
        sensor_force_names,
        all_sensor_force_data,
        all_step_start_left,
        all_step_start_right,
        interp_len=100,
    ):
        sensor_pairs = []
        for name in sensor_force_names:
            if name.startswith("left_"):
                right_name = name.replace("left_", "right_")
                if right_name in sensor_force_names:
                    sensor_pairs.append((name, right_name))

        for left_sensor, right_sensor in sensor_pairs:
            left_data = all_sensor_force_data[left_sensor]
            right_data = all_sensor_force_data[right_sensor]

            # Collect interpolated steps for left
            left_steps = []
            if all_step_start_left and len(all_step_start_left) > 1:
                for j in range(len(all_step_start_left) - 1):
                    start, end = all_step_start_left[j]-1, all_step_start_left[j + 1]-1
                    y = [float(np.array(a)[1]) for a in left_data[start:end]]
                    if len(y) < 2:
                        continue
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    y_interp = np.interp(x_new, x_old, y)
                    left_steps.append(y_interp)
            # Collect interpolated steps for right
            right_steps = []
            if all_step_start_right and len(all_step_start_right) > 1:
                for j in range(len(all_step_start_right) - 1):
                    start, end = all_step_start_right[j]-1, all_step_start_right[j + 1]-1
                    y = [float(np.array(a)[1]) for a in right_data[start:end]]
                    if len(y) < 2:
                        continue
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    y_interp = np.interp(x_new, x_old, y)
                    right_steps.append(y_interp)
            # Calculate mean and std across steps
            if left_steps and right_steps:
                mean_left = np.mean(left_steps, axis=0)
                std_left = np.std(left_steps, axis=0)
                mean_right = np.mean(right_steps, axis=0)
                std_right = np.std(right_steps, axis=0)
                diff = mean_left - mean_right
                x = np.linspace(0, 100, interp_len)

                plt.figure(figsize=(10, 5))
                plt.plot(x, mean_left, label=f"{left_sensor} mean", color='blue')
                plt.fill_between(x, mean_left - std_left, mean_left + std_left, color='blue', alpha=0.2)
                plt.plot(x, mean_right, label=f"{right_sensor} mean", color='red')
                plt.fill_between(x, mean_right - std_right, mean_right + std_right, color='red', alpha=0.2)
                plt.plot(x, diff, label="Difference (left - right)", color='black', linestyle='--')
                plt.axhline(0, color='gray', linestyle=':', linewidth=1)
                plt.title(f"Force Sensor (y) per time step: {left_sensor} vs {right_sensor}")
                plt.xlabel("Interpolated Step (%)")
                plt.ylabel("Force Value")
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                print(f"Not enough steps for {left_sensor} or {right_sensor} to compute difference.")


            

    def plot_joint_angle_symmetry(
        self,
        joint_names_list,
        joint_data,
        step_indices_left,
        step_indices_right,
        parameter_name="angle",
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=None,
    ):
        """
        Plot mean and difference of joint parameter (e.g., angle or velocity) for symmetry assessment.
        joint_names_list: list of joint base names, e.g. ["knee_angle", "ankle_angle", ...]
        joint_data: dict of {joint_name: {parameter_name: list/array}}
        step_indices_left, step_indices_right: step start indices for left/right
        parameter_name: "angle", "velocity", etc.
        """
        def get_step_indices(joint):
            if joint.endswith("_l"):
                return step_indices_left
            elif joint.endswith("_r"):
                return step_indices_right
            else:
                return None

        # Convert joint_names_list to left/right pairs
        joint_pairs = [(name + "_l", name + "_r") for name in joint_names_list]

        # Determine interpolation length if needed
        if interp_mode == "interp" and interp_len is None:
            step_lengths = []
            for left_joint, right_joint in joint_pairs:
                for joint in (left_joint, right_joint):
                    indices = get_step_indices(joint)
                    if indices and len(indices) > 1:
                        step_lengths += [indices[i+1] - indices[i] for i in range(len(indices)-1)]
            interp_len = min(step_lengths) if step_lengths else 1

        for left_joint, right_joint in joint_pairs:
            left_indices = get_step_indices(left_joint)
            right_indices = get_step_indices(right_joint)
            left_data = joint_data.get(left_joint, {}).get(parameter_name, None)
            right_data = joint_data.get(right_joint, {}).get(parameter_name, None)
            if left_data is None or right_data is None:
                print(f"Missing data for {left_joint} or {right_joint}")
                continue

            def collect_steps(data, indices):
                steps = []
                if indices and len(indices) > 1:
                    for j in range(len(indices) - 1):
                        start, end = indices[j], indices[j + 1]
                        y = [float(np.array(a).squeeze()) for a in data[start:end]]
                        if len(y) < 2:
                            continue
                        if interp_mode == "interp":
                            x_old = np.linspace(0, 1, len(y))
                            x_new = np.linspace(0, 1, interp_len)
                            y = np.interp(x_new, x_old, y)
                        if convert_to_deg and parameter_name in ("angle", "velocity"):
                            y = np.rad2deg(y)
                        steps.append(y)
                return steps

            left_steps = collect_steps(left_data, left_indices)
            right_steps = collect_steps(right_data, right_indices)

            if left_steps and right_steps:
                mean_left = np.mean(left_steps, axis=0)
                std_left = np.std(left_steps, axis=0)
                mean_right = np.mean(right_steps, axis=0)
                std_right = np.std(right_steps, axis=0)
                diff = mean_left - mean_right
                x = np.linspace(0, 100, interp_len)
                plt.figure(figsize=(10, 5))
                plt.plot(x, mean_left, label=f"{left_joint} mean", color='blue')
                plt.fill_between(x, mean_left - std_left, mean_left + std_left, color='blue', alpha=0.2)
                plt.plot(x, mean_right, label=f"{right_joint} mean", color='red')
                plt.fill_between(x, mean_right - std_right, mean_right + std_right, color='red', alpha=0.2)
                plt.plot(x, diff, label="Difference (left - right)", color='black', linestyle='--')
                plt.axhline(0, color='gray', linestyle=':', linewidth=1)
                plt.title(f"Joint symmetry: {left_joint} vs {right_joint} ({parameter_name})")
                plt.xlabel("Interpolated Step (%)")
                plt.ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                print(f"Not enough steps for {left_joint} or {right_joint} to compute symmetry.")



    # @staticmethod
    # def plot_muscle_activation_symmetry_all_runs(
    #     muscle_names_list,
    #     all_loaded_data,
    #     run_step_data,
    #     interp_len=100,
    #     ):
    #     """
    #     Plot muscle activation symmetry (mean left - mean right per time step) for each muscle in muscle_names_list, for all runs.
    #     """
    #     for muscle_name in muscle_names_list:
    #         plt.figure(figsize=(12, 6))
    #         for run_key, run_dict in all_loaded_data.items():
    #             all_actuator_names = run_dict["all_actuator_names"]
    #             all_actions = run_dict["all_actions"]
    #             step_data = run_step_data[run_key]
    #             all_step_start_left = step_data["step_start_left"]
    #             all_step_start_right = step_data["step_start_right"]

    #             def find_actuator_index(actuator_names, name):
    #                 try:
    #                     return actuator_names.index(name)
    #                 except ValueError:
    #                     print(f"{name} not found in all_actuator_names.")
    #                     return None

    #             idx_left = find_actuator_index(all_actuator_names, muscle_name + "_l")
    #             idx_right = find_actuator_index(all_actuator_names, muscle_name + "_r")
    #             if idx_left is None or idx_right is None:
    #                 continue

    #             # Collect left and right activations per step
    #             left_steps = []
    #             right_steps = []
    #             if all_step_start_left and len(all_step_start_left) > 1:
    #                 for j in range(len(all_step_start_left) - 1):
    #                     start, end = all_step_start_left[j], all_step_start_left[j + 1]
    #                     y = [float(np.array(a)[0][idx_left]) for a in all_actions[start:end]]
    #                     if len(y) < 2:
    #                         continue
    #                     x_old = np.linspace(0, 1, len(y))
    #                     x_new = np.linspace(0, 1, interp_len)
    #                     y_interp = np.interp(x_new, x_old, y)
    #                     left_steps.append(y_interp)
    #             if all_step_start_right and len(all_step_start_right) > 1:
    #                 for j in range(len(all_step_start_right) - 1):
    #                     start, end = all_step_start_right[j], all_step_start_right[j + 1]
    #                     y = [float(np.array(a)[0][idx_right]) for a in all_actions[start:end]]
    #                     if len(y) < 2:
    #                         continue
    #                     x_old = np.linspace(0, 1, len(y))
    #                     x_new = np.linspace(0, 1, interp_len)
    #                     y_interp = np.interp(x_new, x_old, y)
    #                     right_steps.append(y_interp)
    #             # Plot if enough steps
    #             if left_steps and right_steps:
    #                 mean_left = np.mean(left_steps, axis=0)
    #                 std_left = np.std(left_steps, axis=0)
    #                 mean_right = np.mean(right_steps, axis=0)
    #                 std_right = np.std(right_steps, axis=0)
    #                 diff = mean_left - mean_right
    #                 x = np.linspace(0, 100, interp_len)
    #                 plt.plot(x, mean_left, label=f"{run_key} {muscle_name}_l mean")
    #                 plt.fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
    #                 plt.plot(x, mean_right, label=f"{run_key} {muscle_name}_r mean", linestyle='--')
    #                 plt.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
    #                 plt.plot(x, diff, label=f"{run_key} L-R", linestyle=':')
    #         plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    #         plt.title(f"Muscle activation symmetry: {muscle_name} (all runs)")
    #         plt.xlabel("Interpolated Step (%)")
    #         plt.ylabel("Activation")
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()


    @staticmethod
    def plot_muscle_activation_symmetry_all_runs(
        muscle_names_list,
        all_loaded_data,
        run_step_data,
        interp_len=100,
    ):
        """
        Plot mean left of all runs in one plot, mean right of all runs in another, and the difference in a third.
        Each plot includes all runs, shown side by side.
        """
        for muscle_name in muscle_names_list:
            mean_left_all = []
            std_left_all = []
            mean_right_all = []
            std_right_all = []
            diff_all = []
            run_keys = list(all_loaded_data.keys())
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                all_actuator_names = run_dict["all_actuator_names"]
                all_actions = run_dict["all_actions"]
                all_actions = np.array(all_actions)  # Ensure actions are numpy array for indexing
                step_data = run_step_data[run_key]
                all_step_start_left = step_data["step_start_left"]
                all_step_start_right = step_data["step_start_right"]

                def find_actuator_index(actuator_names, name):
                    try:
                        return actuator_names.index(name)
                    except ValueError:
                        print(f"{name} not found in all_actuator_names.")
                        return None

                idx_left = find_actuator_index(all_actuator_names, muscle_name + "_l")
                idx_right = find_actuator_index(all_actuator_names, muscle_name + "_r")
                if idx_left is None or idx_right is None:
                    continue

                # Collect left and right activations per step
                left_steps = []
                right_steps = []
                if all_step_start_left and len(all_step_start_left) > 1:
                    for j in range(len(all_step_start_left) - 1):
                        start, end = all_step_start_left[j]-1, all_step_start_left[j + 1]-1
                        # print("all_actions shape:", all_actions.shape)
                        # print("a shape:", np.array(all_actions[start:end]).shape)
                        # print("idx_left:", idx_left)
                        # y = [float(a[idx_left]) for a in all_actions[start:end]]
                        # y = [float(np.array(a)[0][idx_left]) for a in all_actions[start:end]]
                        if hasattr(all_actions[start], 'shape') and len(np.array(all_actions[start]).shape) > 1:
                            y = [float(np.array(a)[0][idx_left]) for a in all_actions[start:end]]
                        else:
                            y = [float(a[idx_left]) for a in all_actions[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        left_steps.append(y_interp)
                if all_step_start_right and len(all_step_start_right) > 1:
                    for j in range(len(all_step_start_right) - 1):
                        start, end = all_step_start_right[j]-1, all_step_start_right[j + 1]-1
                        
                        if hasattr(all_actions[start], 'shape') and len(np.array(all_actions[start]).shape) > 1:
                            y = [float(np.array(a)[0][idx_right]) for a in all_actions[start:end]]
                        else: 
                            y = [float(a[idx_right]) for a in all_actions[start:end]]

                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        right_steps.append(y_interp)
                # Save means/stds if enough steps
                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0)
                    std_left = np.std(left_steps, axis=0)
                    mean_right = np.mean(right_steps, axis=0)
                    std_right = np.std(right_steps, axis=0)
                    # diff = mean_right - mean_left #
                    diff = mean_left - mean_right
                    mean_left_all.append((run_key, mean_left, std_left))
                    mean_right_all.append((run_key, mean_right, std_right))
                    diff_all.append((run_key, diff))

            x = np.linspace(0, 100, interp_len)
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
            # Plot mean left
            for run_key, mean_left, std_left in mean_left_all:
                axes[1].plot(x, mean_left, label=f"{run_key} {muscle_name}_l mean")
                axes[1].fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
            axes[1].set_title(f"{muscle_name}_l mean (all runs)")
            axes[1].set_xlabel("Interpolated Step (%)")
            axes[1].set_ylabel("Activation")
            # axes[0].legend()
            # Plot mean right
            for run_key, mean_right, std_right in mean_right_all:
                axes[0].plot(x, mean_right, label=f"{run_key} {muscle_name}_r mean")
                axes[0].fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
            axes[0].set_title(f"{muscle_name}_r mean (all runs)")
            axes[0].set_xlabel("Interpolated Step (%)")
            axes[0].set_ylabel("Activation")
            # axes[1].legend()
            # Plot difference
            for run_key, diff in diff_all:
                axes[2].plot(x, diff, label=f"{run_key} L-R")
            axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
            axes[2].set_title(f"{muscle_name} L-R difference (all runs)")
            axes[2].set_xlabel("Interpolated Step (%)")
            axes[2].set_ylabel("Activation Difference")
            axes[2].legend()
            plt.tight_layout()
            plt.show()
            plt.close()



    # Across all runs
    @staticmethod
    def plot_muscle_activation_symmetry_all_runs_summed(
        muscle_names_list,
        all_loaded_data,
        run_step_data,
        interp_len=100,
    ):
        """
        Plot mean ± std across runs for left, right, and difference (L-R).
        """
        for muscle_name in muscle_names_list:
            mean_left_all = []
            mean_right_all = []
            diff_all = []

            run_keys = list(all_loaded_data.keys())
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                all_actuator_names = run_dict["all_actuator_names"]
                all_actions = np.array(run_dict["all_actions"])  # Ensure actions are numpy array
                step_data = run_step_data[run_key]
                all_step_start_left = step_data["step_start_left"]
                all_step_start_right = step_data["step_start_right"]

                def find_actuator_index(actuator_names, name):
                    try:
                        return actuator_names.index(name)
                    except ValueError:
                        print(f"{name} not found in all_actuator_names.")
                        return None

                idx_left = find_actuator_index(all_actuator_names, muscle_name + "_l")
                idx_right = find_actuator_index(all_actuator_names, muscle_name + "_r")
                if idx_left is None or idx_right is None:
                    continue

                # Collect left and right activations per step
                left_steps, right_steps = [], []
                if all_step_start_left and len(all_step_start_left) > 1:
                    for j in range(len(all_step_start_left) - 1):
                        start, end = all_step_start_left[j]-1, all_step_start_left[j + 1]-1
                        if hasattr(all_actions[start], 'shape') and len(np.array(all_actions[start]).shape) > 1:
                            y = [float(np.array(a)[0][idx_left]) for a in all_actions[start:end]]
                        else:
                            y = [float(a[idx_left]) for a in all_actions[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        left_steps.append(np.interp(x_new, x_old, y))

                if all_step_start_right and len(all_step_start_right) > 1:
                    for j in range(len(all_step_start_right) - 1):
                        start, end = all_step_start_right[j]-1, all_step_start_right[j + 1]-1
                        if hasattr(all_actions[start], 'shape') and len(np.array(all_actions[start]).shape) > 1:
                            y = [float(np.array(a)[0][idx_right]) for a in all_actions[start:end]]
                        else:
                            y = [float(a[idx_right]) for a in all_actions[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        right_steps.append(np.interp(x_new, x_old, y))

                # Aggregate per run
                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0)
                    mean_right = np.mean(right_steps, axis=0)
                    diff = mean_left - mean_right
                    mean_left_all.append(mean_left)
                    mean_right_all.append(mean_right)
                    diff_all.append(diff)

            # --- Aggregate across runs ---
            if not mean_left_all or not mean_right_all:
                print(f"Skipping {muscle_name}, no data found.")
                continue

            mean_left_all = np.array(mean_left_all)
            mean_right_all = np.array(mean_right_all)
            diff_all = np.array(diff_all)

            mean_left = np.mean(mean_left_all, axis=0)
            std_left = np.std(mean_left_all, axis=0)
            mean_right = np.mean(mean_right_all, axis=0)
            std_right = np.std(mean_right_all, axis=0)
            mean_diff = np.mean(diff_all, axis=0)
            std_diff = np.std(diff_all, axis=0)

            x = np.linspace(0, 100, interp_len)
            plt.figure(figsize=(8, 5))

            # Left
            plt.plot(x, mean_left, label=f"{muscle_name}_l mean", color="blue")
            plt.fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.2, color="blue")
            # axes[0].set_title(f"{muscle_name}_l mean ± std (across runs)")
            # axes[0].set_xlabel("Interpolated Step (%)")
            # axes[0].set_ylabel("Activation")

            # Right
            plt.plot(x, mean_right, label=f"{muscle_name}_r mean", color="red")
            plt.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.2, color="red")
            # axes[1].set_title(f"{muscle_name}_r mean ± std (across runs)")
            # axes[1].set_xlabel("Interpolated Step (%)")
            # axes[1].set_ylabel("Activation")

            # Difference
            plt.plot(x, mean_diff, label=f"{muscle_name} L-R mean", color="green")
            plt.fill_between(x, mean_diff - std_diff, mean_diff + std_diff, alpha=0.2, color="green")
            plt.axhline(0, color='gray', linestyle=':', linewidth=1)
            # # plt.set_title(f"{muscle_name} L-R difference ± std (across runs)")
            # plt.set_xlabel("Interpolated Step (%)")
            # plt.set_ylabel("Activation Difference")
            plt.legend()
            plt.title(f"Muscle activation symmetry: {muscle_name} (across runs)")
            plt.xlabel("Interpolated Step (%)")
            plt.ylabel("Activation")
            plt.tight_layout()
            plt.show()
            plt.close()


    @staticmethod
    def plot_muscle_activation_symmetry_all_dirs(
        muscle_names_list,
        all_loaded_data,
        run_step_data,
        interp_len=100,
    ):
        """
        Plot mean ± std across runs for left, right, and difference (L-R),
        comparing multiple directories side-by-side.
        """
        dir_labels = list(all_loaded_data.keys())
        colors = plt.cm.tab10.colors
        for muscle_name in muscle_names_list:
            plt.figure(figsize=(20, 5))
            x = np.linspace(0, 100, interp_len)

            for dir_idx, dir_label in enumerate(dir_labels):
                mean_left_all = []
                mean_right_all = []
                diff_all = []

                runs = all_loaded_data[dir_label]
                for run_key, run_dict in runs.items():
                    all_actuator_names = run_dict.get("all_actuator_names", [])
                    all_actions = np.array(run_dict.get("all_actions", []))
                    step_data = run_step_data[dir_label][run_key]
                    all_step_start_left = step_data["step_start_left"]
                    all_step_start_right = step_data["step_start_right"]

                    def find_actuator_index(actuator_names, name):
                        try:
                            return actuator_names.index(name)
                        except ValueError:
                            return None

                    idx_left = find_actuator_index(all_actuator_names, muscle_name + "_l")
                    idx_right = find_actuator_index(all_actuator_names, muscle_name + "_r")
                    if idx_left is None or idx_right is None:
                        continue

                    # Collect left/right activations per step
                    left_steps, right_steps = [], []

                    if all_step_start_left and len(all_step_start_left) > 1:
                        for j in range(len(all_step_start_left) - 1):
                            start, end = all_step_start_left[j]-1, all_step_start_left[j + 1]-1
                            y = [float(a[idx_left]) for a in all_actions[start:end]]
                            if len(y) < 2:
                                continue
                            left_steps.append(np.interp(np.linspace(0, 1, interp_len),
                                                        np.linspace(0, 1, len(y)), y))

                    if all_step_start_right and len(all_step_start_right) > 1:
                        for j in range(len(all_step_start_right) - 1):
                            start, end = all_step_start_right[j]-1, all_step_start_right[j + 1]-1
                            y = [float(a[idx_right]) for a in all_actions[start:end]]
                            if len(y) < 2:
                                continue
                            right_steps.append(np.interp(np.linspace(0, 1, interp_len),
                                                        np.linspace(0, 1, len(y)), y))

                    if left_steps and right_steps:
                        mean_left_all.append(np.mean(left_steps, axis=0))
                        mean_right_all.append(np.mean(right_steps, axis=0))
                        diff_all.append(np.mean(left_steps, axis=0) - np.mean(right_steps, axis=0))

                if not mean_left_all or not mean_right_all:
                    print(f"Skipping {muscle_name} in {dir_label}, no data found.")
                    continue

                mean_left = np.mean(mean_left_all, axis=0)
                std_left = np.std(mean_left_all, axis=0)
                mean_right = np.mean(mean_right_all, axis=0)
                std_right = np.std(mean_right_all, axis=0)
                mean_diff = np.mean(diff_all, axis=0)
                std_diff = np.std(diff_all, axis=0)

                # Color mapping per directory
                # Use a larger color palette for more directories
                # colors = [
                #     "blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan",
                #     "magenta", "gold", "teal", "navy", "maroon", "lime", "indigo", "coral", "turquoise", "darkgreen"
                # ]
                
                color = colors[dir_idx % len(colors)]

                plt.plot(x, mean_left, label=f"{dir_label} {muscle_name}_l", color=color, linestyle='-')
                plt.fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.2, color=color)

                plt.plot(x, mean_right, label=f"{dir_label} {muscle_name}_r", color=color, linestyle='--')
                plt.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.2, color=color)

                plt.plot(x, mean_diff, label=f"{dir_label} {muscle_name} L-R", color=color, linestyle=':')
                plt.fill_between(x, mean_diff - std_diff, mean_diff + std_diff, alpha=0.1, color=color)

            plt.axhline(0, color='gray', linestyle=':', linewidth=1)
            plt.xlabel("Interpolated Step (%)")
            plt.ylabel("Activation")
            plt.title(f"Muscle activation symmetry comparison: {muscle_name}")
            # Place legend in 3 columns, outside plot on the right
            plt.legend(ncol=3, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            plt.show()
            plt.close()


    # @staticmethod
    # def plot_joint_angle_symmetry_all_runs(
    #         joint_names_list,
    #         all_loaded_data,
    #         run_step_data,
    #         parameter_name="angle",
    #         convert_to_deg=False,
    #         interp_mode="interp",
    #         interp_len=100,
    # ):
    #     """
    #     Plot mean and difference of joint parameter (e.g., angle or velocity) for symmetry assessment for all runs.
    #     joint_names_list: list of joint base names, e.g. ["knee_angle", "ankle_angle", ...]
    #     all_loaded_data: dict of {run_key: loaded_data}
    #     run_step_data: dict of {run_key: {"step_start_left": [...], "step_start_right": [...]} }
    #     parameter_name: "angle", "velocity", etc.
    #     """

    #     joint_pairs = [(name + "_l", name + "_r") for name in joint_names_list]
    #     color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #     color_iter = itertools.cycle(color_cycle)

    #     run_colors = {}  # Dictionary to store assigned colors for each run
    #     for left_joint, right_joint in joint_pairs:
    #         plt.figure(figsize=(12, 6))
    #         has_data = False  # Track if any run has data for this parameter
    #         for run_key, run_dict in all_loaded_data.items():
    #             joint_data = {}
    #             # Build joint_data dict for this run
    #             evaluation_joint_names = run_dict.get("evaluation_joint_names")
    #             joint_parameters = ["angle", "velocity", "forces_constraint", "forces_smooth", "torques", "energy_exp"]
    #             for joint_name in evaluation_joint_names:
    #                 joint_data[joint_name] = {}
    #                 for key in joint_parameters:
    #                     param = run_dict.get(f"{joint_name}_{key}")
    #                     if param is not None:
    #                         joint_data[joint_name][key] = param
    #                         joint_data[joint_name][key] = param

    #             step_indices_left = run_step_data[run_key]["step_start_left"]
    #             step_indices_right = run_step_data[run_key]["step_start_right"]
    #             left_data = joint_data.get(left_joint, {}).get(parameter_name, None)
    #             right_data = joint_data.get(right_joint, {}).get(parameter_name, None)
    #             if left_data is None or right_data is None:
    #                 print(f"[{run_key}] Missing data for {left_joint} or {right_joint}")
    #                 continue

    #             def collect_steps(data, indices):
    #                 steps = []
    #                 if indices and len(indices) > 1:
    #                     for j in range(len(indices) - 1):
    #                         start, end = indices[j], indices[j + 1]
    #                         y = [float(np.array(a).squeeze()) for a in data[start:end]]
    #                         if len(y) < 2:
    #                             continue
    #                         if interp_mode == "interp":
    #                             x_old = np.linspace(0, 1, len(y))
    #                             x_new = np.linspace(0, 1, interp_len)
    #                             y = np.interp(x_new, x_old, y)
    #                         if convert_to_deg and parameter_name in ("angle", "velocity"):
    #                             y = np.rad2deg(y)
    #                         steps.append(y)
    #                 return steps

    #             left_steps = collect_steps(left_data, step_indices_left)
    #             right_steps = collect_steps(right_data, step_indices_right)

    #             if left_steps and right_steps:
    #                 mean_left = np.mean(left_steps, axis=0)
    #                 std_left = np.std(left_steps, axis=0)
    #                 mean_right = np.mean(right_steps, axis=0)
    #                 std_right = np.std(right_steps, axis=0)
    #                 diff = mean_left - mean_right
    #                 x = np.linspace(0, 100, interp_len)
    #                 # Assign a color for this run
    #                 color = run_colors.get(run_key)
    #                 if color is None:
    #                     color = next(color_iter)
    #                     run_colors[run_key] = color
    #                 plt.plot(x, mean_left, label=f"{run_key} {left_joint} mean", color=color)
    #                 plt.fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15, color=color)
    #                 plt.plot(x, mean_right, label=f"{run_key} {right_joint} mean", linestyle='--', color=color)
    #                 plt.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15, color=color)
    #                 plt.plot(x, diff, label=f"{run_key} Diff (L-R)", linestyle=':', color=color)
    #                 has_data = True
    #             else:
    #                 print(f"[{run_key}] Not enough steps for {left_joint} or {right_joint} to compute symmetry.")

    #         if has_data:
    #             plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    #             plt.title(f"Joint symmetry (all runs): {left_joint} vs {right_joint} ({parameter_name})")
    #             plt.xlabel("Interpolated Step (%)")
    #             plt.ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
    #             plt.legend()
    #             plt.tight_layout()
    #             plt.show()
    #         else:
    #             plt.close()


    

    @staticmethod
    def plot_joint_angle_symmetry_all_runs(
        joint_names_list,
        all_loaded_data,
        run_step_data,
        parameter_name="angle",
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=100,
        body_weight_to_normalize = 1,
        plot_baseline = False, 
        baseline_data=None,  # can be added for angle and velocity from loco-mujoco data 
    ):
        """
        Plot mean of right of all runs, mean of left, and the difference, side by side in one figure.
        Only plot runs where both left and right data are available.
        """
        joint_pairs = [(name + "_l", name + "_r") for name in joint_names_list]
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_iter = itertools.cycle(color_cycle)

        for left_joint, right_joint in joint_pairs:
            run_keys = list(all_loaded_data.keys())
            mean_left_all = []
            std_left_all = []
            mean_right_all = []
            std_right_all = []
            diff_all = []
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                joint_data = {}
                evaluation_joint_names = run_dict.get("evaluation_joint_names")
                joint_parameters = ["angle", "velocity", "forces_constraint", "forces_smooth","forces_applied", "torques", "energy_exp"]
                for joint_name in evaluation_joint_names:
                    joint_data[joint_name] = {}
                    for key in joint_parameters:
                        param = run_dict.get(f"{joint_name}_{key}")
                        if param is not None:
                            joint_data[joint_name][key] = param

                step_indices_left = run_step_data[run_key]["step_start_left"]
                step_indices_right = run_step_data[run_key]["step_start_right"]
                left_data = joint_data.get(left_joint, {}).get(parameter_name, None)
                right_data = joint_data.get(right_joint, {}).get(parameter_name, None)
                # Only include runs where both left and right data are available
                if left_data is None or right_data is None:
                    continue

                def collect_steps(data, indices):
                    steps = []
                    if indices and len(indices) > 1:
                        for j in range(len(indices) - 1):
                            start, end = indices[j], indices[j + 1]
                            y = [float(np.array(a).squeeze()) for a in data[start:end]]
                            if len(y) < 2:
                                continue
                            if interp_mode == "interp":
                                x_old = np.linspace(0, 1, len(y))
                                x_new = np.linspace(0, 1, interp_len)
                                y = np.interp(x_new, x_old, y)
                            if convert_to_deg and parameter_name in ("angle", "velocity"):
                                y = np.rad2deg(y)
                            steps.append(y)
                    return steps

                left_steps = collect_steps(left_data, step_indices_left)
                right_steps = collect_steps(right_data, step_indices_right)

                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0)/body_weight_to_normalize
                    std_left = np.std(left_steps, axis=0)/body_weight_to_normalize
                    mean_right = np.mean(right_steps, axis=0)/body_weight_to_normalize
                    std_right = np.std(right_steps, axis=0)/body_weight_to_normalize
                    if 'knee' in left_joint: # Switch sign (-1) for knee sensors to match flexion/extension convention
                        mean_left = -mean_left
                        mean_right = -mean_right
                        std_left = -std_left
                        std_right = -std_right
                    diff = np.abs(mean_left) - np.abs(mean_right)
                    mean_left_all.append((run_key, mean_left, std_left))
                    mean_right_all.append((run_key, mean_right, std_right))
                    diff_all.append((run_key, diff))

            x = np.linspace(0, 100, interp_len)
            # Plot all three plots side by side
            if mean_left_all or mean_right_all or diff_all:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
                # Plot mean of right of all runs
                if mean_right_all:
                    for run_key, mean_right, std_right in mean_right_all:
                        axes[0].plot(x, mean_right, label=f"{run_key} {right_joint} mean")
                        axes[0].fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
                    if plot_baseline and parameter_name in ["angle", "velocity"] and baseline_data is not None:
                        if right_joint in baseline_data and parameter_name in baseline_data[right_joint]:#if np.any(baseline_data[right_joint][parameter_name]):
                            if 'knee' in right_joint: # Switch sign (-1) for knee sensors to match flexion/extension convention
                                axes[0].plot(x, -baseline_data[right_joint][parameter_name], label=f"Baseline {right_joint} mean", color='black')
                            else:
                                axes[0].plot(x, baseline_data[right_joint][parameter_name], label=f"Baseline {right_joint} mean", color='black')
                    axes[0].set_title(f"{right_joint} mean (all runs)")
                    axes[0].set_xlabel("Interpolated Step (%)")
                    axes[0].set_ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                    # axes[0].legend()
                # Plot mean of left of all runs
                if mean_left_all:
                    for run_key, mean_left, std_left in mean_left_all:
                        axes[1].plot(x, mean_left, label=f"{run_key} {left_joint} mean")
                        axes[1].fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
                    if plot_baseline and parameter_name in ["angle", "velocity"] and baseline_data is not None:
                        if right_joint in baseline_data and parameter_name in baseline_data[right_joint]:
                            if 'knee' in right_joint: # Switch sign (-1) for knee sensors to match flexion/extension convention
                                axes[1].plot(x, -baseline_data[right_joint][parameter_name], label=f"Baseline {left_joint} mean", color='black')
                            else:
                                axes[1].plot(x, baseline_data[right_joint][parameter_name], label=f"Baseline {left_joint} mean", color='black')
                    axes[1].set_title(f"{left_joint} mean (all runs)")
                    axes[1].set_xlabel("Interpolated Step (%)")
                    axes[1].set_ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                    # axes[1].legend()
                # Plot difference (left - right) of all runs
                if diff_all:
                    for run_key, diff in diff_all:
                        axes[2].plot(x, diff, label=f"{run_key} Diff (L-R)")
                    axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
                    axes[2].set_title(f" Abs {left_joint} - Abs {right_joint} difference (all runs)")
                    axes[2].set_xlabel("Interpolated Step (%)")
                    axes[2].set_ylabel(f"{parameter_name} difference ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                    axes[2].legend()
                plt.tight_layout()
                plt.show()
                plt.close()



    @staticmethod
    def plot_joint_angle_steps_and_mean(
        joint_names_list,
        all_loaded_data,
        run_step_data,
        parameter_name="angle",
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=100,
        body_weight_to_normalize=1,
        plot_baseline=False, 
        baseline_data=None, 
    ):
        """
        Plots joint angle/velocity/etc. data. For each run, it creates a separate figure 
        showing all individual steps, the run's mean (with std), and the difference 
        between the left and right side means.
        """

        joint_pairs = [(name + "_l", name + "_r") for name in joint_names_list]
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # We need a function to get run keys, which was missing but implied for sorting/iteration
        def get_sorted_run_keys(data):
            run_keys = list(data.keys())
            # Assuming a helper function like the one in previous examples exists for sorting
            # If not, simply use: return run_keys
            try:
                # Placeholder for numeric run key extraction logic if needed for sorting
                return sorted(run_keys)
            except:
                 return run_keys
        
        # Helper function to collect steps (modified to also return steps)
        def collect_steps(data, indices):
            steps = []
            if indices and len(indices) > 1:
                for j in range(len(indices) - 1):
                    start, end = indices[j], indices[j + 1]
                    # Ensure data is handled correctly (e.g., if it's a 1D array or list of single values)
                    y = [float(np.array(a).squeeze()) for a in data[start:end]]
                    if len(y) < 2:
                        continue
                    if interp_mode == "interp":
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y = np.interp(x_new, x_old, y)
                    if convert_to_deg and parameter_name in ("angle", "velocity"):
                        y = np.rad2deg(y)
                    steps.append(y)
            return steps

        # Determine y-axis unit
        if convert_to_deg and parameter_name in ('angle', 'velocity'):
            unit = 'deg' if parameter_name == 'angle' else 'deg/s'
        else:
            unit = 'rad' if parameter_name == 'angle' else 'rad/s'

        for left_joint, right_joint in joint_pairs:
            
            # --- COLLECT DATA FOR ALL RUNS ---
            # We collect all individual steps, means, and diffs run by run first
            all_run_data = {}
            run_keys = get_sorted_run_keys(all_loaded_data)

            for run_key in run_keys:
                run_dict = all_loaded_data.get(run_key, {})
                
                # Retrieve joint data (simplified as it was verbose in original)
                left_data = run_dict.get(f"{left_joint}_{parameter_name}", None)
                right_data = run_dict.get(f"{right_joint}_{parameter_name}", None)
                
                # Check for alternative data structure (as in original function body)
                if left_data is None or right_data is None:
                    try:
                        # Attempt to find data within the nested joint_data structure (like in original)
                        evaluation_joint_names = run_dict.get("evaluation_joint_names", [])
                        joint_data = {}
                        for jn in evaluation_joint_names:
                             joint_data[jn] = {p: run_dict.get(f"{jn}_{p}") for p in ["angle", "velocity", "forces_constraint", "forces_smooth","forces_applied", "torques", "energy_exp"] if run_dict.get(f"{jn}_{p}") is not None}
                        
                        left_data = joint_data.get(left_joint, {}).get(parameter_name, None)
                        right_data = joint_data.get(right_joint, {}).get(parameter_name, None)
                    except:
                        pass # Continue if data retrieval fails

                if left_data is None or right_data is None:
                    continue
                
                step_indices_left = run_step_data.get(run_key, {}).get("step_start_left", [])
                step_indices_right = run_step_data.get(run_key, {}).get("step_start_right", [])

                left_steps = collect_steps(left_data, step_indices_left)
                right_steps = collect_steps(right_data, step_indices_right)

                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0) / body_weight_to_normalize
                    std_left = np.std(left_steps, axis=0) / body_weight_to_normalize
                    mean_right = np.mean(right_steps, axis=0) / body_weight_to_normalize
                    std_right = np.std(right_steps, axis=0) / body_weight_to_normalize
                    
                    # if 'knee' in left_joint: # Apply sign switch
                    #     mean_left, std_left = -mean_left, -std_left
                    #     mean_right, std_right = -mean_right, -std_right
                        
                    diff = mean_left - mean_right # Changed to (L-R) difference for steps plot
                    
                    all_run_data[run_key] = {
                        'left_steps': left_steps,
                        'right_steps': right_steps,
                        'mean_left': mean_left,
                        'std_left': std_left,
                        'mean_right': mean_right,
                        'std_right': std_right,
                        'diff': diff,
                        'num_steps': len(left_steps) # Assuming left and right have equal steps
                    }

            x = np.linspace(0, 100, interp_len)
            
            # --- PLOT INDIVIDUAL RUNS ---
            for run_key, data in all_run_data.items():
                
                fig, axes = plt.subplots(1, 2, figsize=(18, 4), sharex=True)
                
                num_steps = data['num_steps']
                steps_left = data['left_steps']
                steps_right = data['right_steps']
                mean_left = data['mean_left']
                std_left = data['std_left']
                mean_right = data['mean_right']
                std_right = data['std_right']
                diff = data['diff']
                
                # --- Subplot 1: Left Side Steps and Mean ---
                color_iter = itertools.cycle(color_cycle)
                mean_color = next(color_iter)
                
                # Plot individual steps
                for i, step in enumerate(steps_left):
                    step_color = color_cycle[i % len(color_cycle)]
                    axes[0].plot(x, step, color=step_color, alpha=0.3, linewidth=1, label=f"Step {i+1}" if i < 10 else None) 
                
                # Plot mean and std
                axes[0].plot(x, mean_left, label=f"Mean (N={num_steps})", color='black', linestyle='-', linewidth=2)
                axes[0].fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.9, color='gray')
                
                axes[0].set_title(f"{run_key} - {left_joint} (All Steps)")
                axes[0].set_xlabel("Interpolated Step (%)")
                axes[0].set_ylabel(f"{parameter_name} ({unit})")
                axes[0].legend(ncol=2, title="Steps (L)", fontsize='small')


                # --- Subplot 2: Right Side Steps and Mean ---
                # Reset color cycle for steps in this plot for clear distinction
                color_iter = itertools.cycle(color_cycle) 
                
                # Plot individual steps
                for i, step in enumerate(steps_right):
                    step_color = color_cycle[i % len(color_cycle)]
                    axes[1].plot(x, step, color=step_color, alpha=0.3, linewidth=1, label=f"Step {i+1}" if i < 10 else None)
                
                # Plot mean and std
                axes[1].plot(x, mean_right, label=f"Mean (N={num_steps})", color='black', linestyle='-', linewidth=2)
                axes[1].fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.9, color='gray')

                axes[1].set_title(f"{run_key} - {right_joint} (All Steps)")
                axes[1].set_xlabel("Interpolated Step (%)")
                axes[1].set_ylabel(f"{parameter_name} ({unit})")
                axes[1].legend(ncol=2, title="Steps (R)", fontsize='small')


                # # --- Subplot 3: Difference ---
                # axes[2].plot(x, diff, label=f"{run_key} Diff (L Mean - R Mean)", color='red', linewidth=2)
                # axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
                
                # axes[2].set_title(f"Difference ({left_joint} - {right_joint})")
                # axes[2].set_xlabel("Interpolated Step (%)")
                # axes[2].set_ylabel(f"{parameter_name} Difference ({unit})")
                # axes[2].legend(loc='best', fontsize='small')
                
                # --- Finalize Figure ---
                fig.suptitle(f"Joint Parameter Analysis for Joint: {left_joint[:-2]} | Run: {run_key}", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
                plt.show()
                plt.close()



    @staticmethod
    def plot_joint_angle_symmetry_all_runs_prosthesis_baseline(
        joint_names_list,
        all_loaded_data,
        run_step_data,
        baseline_file,
        parameter_name="angle",
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=100,
        body_weight_to_normalize=1,
        mass_baseline = 75*10,
    ):
        """
        Plot mean of right of all runs, mean of left, and the difference, side by side in one figure.
        Only plot runs where both left and right data are available.
        Also plot baseline data from baseline_file.
        """

        joint_pairs = [(name + "_l", name + "_r") for name in joint_names_list]
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_iter = itertools.cycle(color_cycle)

        # Load baseline data from file
        baseline_data = {}
        current_label = None
        with open(baseline_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if 'Schmalz' in baseline_file:
                    label_match = re.match(r'^([A-Z_]+)\s*=\s*\[?', line)
                else:
                    label_match = re.match(r'^([A-Za-z_]+)\s*=\s*\[?', line)
                if label_match:
                    current_label = label_match.group(1)
                    baseline_data[current_label] = []
                    continue
                if current_label:
                    vals = line.split(';')
                    if len(vals) == 2:
                        try:
                            x, y = map(float, vals)
                            baseline_data[current_label].append((x, y))
                        except ValueError:
                            continue

        for left_joint, right_joint in joint_pairs:
            run_keys = list(all_loaded_data.keys())
            mean_left_all = []
            std_left_all = []
            mean_right_all = []
            std_right_all = []
            diff_all = []
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                joint_data = {}
                evaluation_joint_names = run_dict.get("evaluation_joint_names")
                joint_parameters = ["angle", "velocity", "forces_constraint", "forces_smooth", "forces_applied", "torques", "energy_exp"]
                for joint_name in evaluation_joint_names:
                    joint_data[joint_name] = {}
                    for key in joint_parameters:
                        param = run_dict.get(f"{joint_name}_{key}")
                        if param is not None:
                            joint_data[joint_name][key] = param

                step_indices_left = run_step_data[run_key]["step_start_left"]
                step_indices_right = run_step_data[run_key]["step_start_right"]
                left_data = joint_data.get(left_joint, {}).get(parameter_name, None)
                right_data = joint_data.get(right_joint, {}).get(parameter_name, None)
                if left_data is None or right_data is None:
                    continue

                def collect_steps(data, indices):
                    steps = []
                    if indices and len(indices) > 1:
                        for j in range(len(indices) - 1):
                            start, end = indices[j], indices[j + 1]
                            y = [float(np.array(a).squeeze()) for a in data[start:end]]
                            if len(y) < 2:
                                continue
                            if interp_mode == "interp":
                                x_old = np.linspace(0, 1, len(y))
                                x_new = np.linspace(0, 1, interp_len)
                                y = np.interp(x_new, x_old, y)
                            if convert_to_deg and parameter_name in ("angle", "velocity"):
                                y = np.rad2deg(y)
                            steps.append(y)
                    return steps

                left_steps = collect_steps(left_data, step_indices_left)
                right_steps = collect_steps(right_data, step_indices_right)

                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0) / body_weight_to_normalize
                    std_left = np.std(left_steps, axis=0) / body_weight_to_normalize
                    mean_right = np.mean(right_steps, axis=0) / body_weight_to_normalize
                    std_right = np.std(right_steps, axis=0) / body_weight_to_normalize
                    if 'knee' in left_joint:
                        mean_left = -mean_left
                        mean_right = -mean_right
                        std_left = -std_left
                        std_right = -std_right
                    diff = np.abs(mean_left) - np.abs(mean_right)
                    mean_left_all.append((run_key, mean_left, std_left))
                    mean_right_all.append((run_key, mean_right, std_right))
                    diff_all.append((run_key, diff))

            x = np.linspace(0, 100, interp_len)
            if mean_left_all or mean_right_all or diff_all:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
                # Plot mean of right of all runs
                if mean_right_all:
                    for run_key, mean_right, std_right in mean_right_all:
                        axes[0].plot(x, mean_right, label=f"{run_key} {right_joint} mean")
                        axes[0].fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
                    # Plot baseline data for right_joint if available
                    for label, points in baseline_data.items():
                        if label.lower() in right_joint.lower() and points:
                            xs, ys = zip(*points)
                            if 'torques' in parameter_name:
                                ys = np.array(ys) * -1
                            axes[0].plot(xs, ys, marker='o', linestyle='-', color='black', label=f"Baseline {label}")
                    axes[0].set_title(f"{right_joint} mean (all runs)")
                    axes[0].set_xlabel("Interpolated Step (%)")
                    axes[0].set_ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                # Plot mean of left of all runs
                if mean_left_all:
                    for run_key, mean_left, std_left in mean_left_all:
                        axes[1].plot(x, mean_left, label=f"{run_key} {left_joint} mean")
                        axes[1].fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
                    # Plot baseline data for left_joint if available
                    for label, points in baseline_data.items():
                        if label.lower() in left_joint.lower() and points:
                            xs, ys = zip(*points)
                            if 'torques' in parameter_name:
                                ys = np.array(ys) * -1
                            axes[1].plot(xs, ys, marker='o', linestyle='-', color='black', label=f"Baseline {label}")
                    axes[1].set_title(f"{left_joint} mean (all runs)")
                    axes[1].set_xlabel("Interpolated Step (%)")
                    axes[1].set_ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                # Plot difference (left - right) of all runs
                if diff_all:
                    for run_key, diff in diff_all:
                        axes[2].plot(x, diff, label=f"{run_key} Diff (L-R)")
                    axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
                    axes[2].set_title(f"Abs {left_joint} - Abs {right_joint} difference (all runs)")
                    axes[2].set_xlabel("Interpolated Step (%)")
                    axes[2].set_ylabel(f"{parameter_name} difference ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                    axes[2].legend()
                plt.tight_layout()
                plt.show()
                plt.close()




    @staticmethod
    def plot_joint_angle_symmetry_all_runs_prosthesis_intact_baseline(
        joint_names_list,
        all_loaded_data,
        run_step_data,
        baseline_file,
        parameter_name="angle",
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=100,
        body_weight_to_normalize=1,
        mass_baseline = 75*10,
    ):
        """
        Plot mean of right of all runs, mean of left, and the difference, side by side in one figure.
        Only plot runs where both left and right data are available.
        Also plot baseline data from baseline_file.
        """

        joint_pairs = [(name + "_l", name + "_r") for name in joint_names_list]
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_iter = itertools.cycle(color_cycle)

        # Load baseline data from file
        baseline_data = {}
        current_label = None
        with open(baseline_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if 'Schmalz' in baseline_file:
                    label_match = re.match(r'^([A-Z_]+)\s*=\s*\[?', line)
                else:
                    label_match = re.match(r'^([A-Za-z_]+)\s*=\s*\[?', line)
                if label_match:
                    current_label = label_match.group(1)
                    baseline_data[current_label] = []
                    continue
                if current_label:
                    vals = line.split(';')
                    if len(vals) == 2:
                        try:
                            x, y = map(float, vals)
                            baseline_data[current_label].append((x, y))
                        except ValueError:
                            continue

        for left_joint, right_joint in joint_pairs:
            run_keys = list(all_loaded_data.keys())
            mean_left_all = []
            std_left_all = []
            mean_right_all = []
            std_right_all = []
            diff_all = []
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                joint_data = {}
                evaluation_joint_names = run_dict.get("evaluation_joint_names")
                joint_parameters = ["angle", "velocity", "forces_constraint", "forces_smooth", "forces_applied", "torques", "energy_exp"]
                for joint_name in evaluation_joint_names:
                    joint_data[joint_name] = {}
                    for key in joint_parameters:
                        param = run_dict.get(f"{joint_name}_{key}")
                        if param is not None:
                            joint_data[joint_name][key] = param

                step_indices_left = run_step_data[run_key]["step_start_left"]
                step_indices_right = run_step_data[run_key]["step_start_right"]
                left_data = joint_data.get(left_joint, {}).get(parameter_name, None)
                right_data = joint_data.get(right_joint, {}).get(parameter_name, None)
                if left_data is None or right_data is None:
                    continue

                def collect_steps(data, indices):
                    steps = []
                    if indices and len(indices) > 1:
                        for j in range(len(indices) - 1):
                            start, end = indices[j], indices[j + 1]
                            y = [float(np.array(a).squeeze()) for a in data[start:end]]
                            if len(y) < 2:
                                continue
                            if interp_mode == "interp":
                                x_old = np.linspace(0, 1, len(y))
                                x_new = np.linspace(0, 1, interp_len)
                                y = np.interp(x_new, x_old, y)
                            if convert_to_deg and parameter_name in ("angle", "velocity"):
                                y = np.rad2deg(y)
                            steps.append(y)
                    return steps

                left_steps = collect_steps(left_data, step_indices_left)
                right_steps = collect_steps(right_data, step_indices_right)

                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0) / body_weight_to_normalize
                    std_left = np.std(left_steps, axis=0) / body_weight_to_normalize
                    mean_right = np.mean(right_steps, axis=0) / body_weight_to_normalize
                    std_right = np.std(right_steps, axis=0) / body_weight_to_normalize
                    if 'knee' in left_joint:
                        mean_left = -mean_left
                        mean_right = -mean_right
                        std_left = -std_left
                        std_right = -std_right
                    diff = np.abs(mean_left) - np.abs(mean_right)
                    mean_left_all.append((run_key, mean_left, std_left))
                    mean_right_all.append((run_key, mean_right, std_right))
                    diff_all.append((run_key, diff))

            x = np.linspace(0, 100, interp_len)
            if mean_left_all or mean_right_all or diff_all:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
                # Plot mean of right of all runs
                if mean_right_all:
                    for run_key, mean_right, std_right in mean_right_all:
                        axes[0].plot(x, mean_right, label=f"{run_key} {right_joint} mean")
                        axes[0].fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
                    # Plot baseline data for right_joint if available
                    for label, points in baseline_data.items():
                        if '_' in label:
                            label_joint = label.split('_')[0]  # e.g., 'KNEE_INT' -> 'KNEE'
                            if label_joint.lower() in right_joint.lower() and 'intact' in label.lower() and points:
                                xs, ys = zip(*points)
                                if 'torques' in parameter_name:
                                    ys = np.array(ys) * -1
                                    if 'Banks' in baseline_file: 
                                        ys = np.array(ys)/ mass_baseline
                                        if not 'knee' in right_joint:
                                            ys = np.array(ys)* -1
                                if 'angle' in parameter_name: 
                                    if 'Banks' in baseline_file and 'knee' in right_joint:
                                        ys = np.array(ys)* -1
                                axes[0].plot(xs, ys, marker='o', linestyle='-', color='black', label=f"Baseline {label}")
                    axes[0].set_title(f"{right_joint} mean (all runs)")
                    axes[0].set_xlabel("Interpolated Step (%)")
                    axes[0].set_ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                # Plot mean of left of all runs
                if mean_left_all:
                    for run_key, mean_left, std_left in mean_left_all:
                        axes[1].plot(x, mean_left, label=f"{run_key} {left_joint} mean")
                        axes[1].fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
                    # Plot baseline data for left_joint if available
                    for label, points in baseline_data.items():
                        if label.lower() in left_joint.lower() and points:
                            xs, ys = zip(*points)
                            if 'torques' in parameter_name:
                                ys = np.array(ys) * -1
                                if 'Banks' in baseline_file: 
                                    ys = np.array(ys)/ mass_baseline
                                    if not 'knee' in right_joint:
                                        ys = np.array(ys)* -1
                            if 'angle' in parameter_name: 
                                if 'Banks' in baseline_file and 'knee' in left_joint:
                                    ys = np.array(ys)* -1
                            axes[1].plot(xs, ys, marker='o', linestyle='-', color='black', label=f"Baseline {label}")
                    axes[1].set_title(f"{left_joint} mean (all runs)")
                    axes[1].set_xlabel("Interpolated Step (%)")
                    axes[1].set_ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                # Plot difference (left - right) of all runs
                if diff_all:
                    for run_key, diff in diff_all:
                        axes[2].plot(x, diff, label=f"{run_key} Diff (L-R)")
                    axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
                    axes[2].set_title(f"Abs {left_joint} - Abs {right_joint} difference (all runs)")
                    axes[2].set_xlabel("Interpolated Step (%)")
                    axes[2].set_ylabel(f"{parameter_name} difference ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                    axes[2].legend()
                plt.tight_layout()
                plt.show()
                plt.close()





    @staticmethod 
    def plot_joint_angle_symmetry_all_runs_prosthesis_intact_baseline_comp(
        joint_names_list,
        all_loaded_data,
        run_step_data,
        baseline_file,
        baseline_file_2,
        baseline_data_3 = 'None',
        parameter_name="angle",
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=100,
        body_weight_to_normalize=1,
        mass_baseline = 75*10,
        mass_baseline_2 = 75*10,
    ):
        """
        Plot mean of right of all runs, mean of left, and the difference, side by side in one figure.
        Only plot runs where both left and right data are available.
        Also plot baseline data from baseline_file.
        """

        joint_pairs = [(name + "_l", name + "_r") for name in joint_names_list]
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_iter = itertools.cycle(color_cycle)

        # Load baseline data from file
        baseline_data = {}
        current_label = None
        baseline_data_2 = {}
        with open(baseline_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if 'Schmalz' in baseline_file:
                    label_match = re.match(r'^([A-Z_]+)\s*=\s*\[?', line)
                else:
                    label_match = re.match(r'^([A-Za-z_]+)\s*=\s*\[?', line)
                if label_match:
                    current_label = label_match.group(1)
                    baseline_data[current_label] = []
                    continue
                if current_label:
                    vals = line.split(';')
                    if len(vals) == 2:
                        try:
                            x, y = map(float, vals)
                            baseline_data[current_label].append((x, y))
                        except ValueError:
                            continue

        with open(baseline_file_2, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if 'Schmalz' in baseline_file_2:
                    label_match = re.match(r'^([A-Z_]+)\s*=\s*\[?', line)
                else:
                    label_match = re.match(r'^([A-Za-z_]+)\s*=\s*\[?', line)
                if label_match:
                    current_label = label_match.group(1)
                    baseline_data_2[current_label] = []
                    continue
                if current_label:
                    vals = line.split(';')
                    if len(vals) == 2:
                        try:
                            x, y = map(float, vals)
                            baseline_data_2[current_label].append((x, y))
                        except ValueError:
                            continue

        for left_joint, right_joint in joint_pairs:
            run_keys = list(all_loaded_data.keys())
            mean_left_all = []
            std_left_all = []
            mean_right_all = []
            std_right_all = []
            diff_all = []
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                joint_data = {}
                evaluation_joint_names = run_dict.get("evaluation_joint_names")
                joint_parameters = ["angle", "velocity", "forces_constraint", "forces_smooth", "forces_applied", "torques", "energy_exp"]
                for joint_name in evaluation_joint_names:
                    joint_data[joint_name] = {}
                    for key in joint_parameters:
                        param = run_dict.get(f"{joint_name}_{key}")
                        if param is not None:
                            joint_data[joint_name][key] = param

                step_indices_left = run_step_data[run_key]["step_start_left"]
                step_indices_right = run_step_data[run_key]["step_start_right"]
                left_data = joint_data.get(left_joint, {}).get(parameter_name, None)
                right_data = joint_data.get(right_joint, {}).get(parameter_name, None)
                if left_data is None or right_data is None:
                    continue

                def collect_steps(data, indices):
                    steps = []
                    if indices and len(indices) > 1:
                        for j in range(len(indices) - 1):
                            start, end = indices[j], indices[j + 1]
                            y = [float(np.array(a).squeeze()) for a in data[start:end]]
                            if len(y) < 2:
                                continue
                            if interp_mode == "interp":
                                x_old = np.linspace(0, 1, len(y))
                                x_new = np.linspace(0, 1, interp_len)
                                y = np.interp(x_new, x_old, y)
                            if convert_to_deg and parameter_name in ("angle", "velocity"):
                                y = np.rad2deg(y)
                            steps.append(y)
                    return steps

                left_steps = collect_steps(left_data, step_indices_left)
                right_steps = collect_steps(right_data, step_indices_right)

                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0) / body_weight_to_normalize
                    std_left = np.std(left_steps, axis=0) / body_weight_to_normalize
                    mean_right = np.mean(right_steps, axis=0) / body_weight_to_normalize
                    std_right = np.std(right_steps, axis=0) / body_weight_to_normalize
                    if 'knee' in left_joint:
                        mean_left = -mean_left
                        mean_right = -mean_right
                        std_left = -std_left
                        std_right = -std_right
                    diff = np.abs(mean_left) - np.abs(mean_right)
                    mean_left_all.append((run_key, mean_left, std_left))
                    mean_right_all.append((run_key, mean_right, std_right))
                    diff_all.append((run_key, diff))

            x = np.linspace(0, 100, interp_len)
            if mean_left_all or mean_right_all or diff_all:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
                # Plot mean of right of all runs
                if mean_right_all:
                    for run_key, mean_right, std_right in mean_right_all:
                        axes[0].plot(x, mean_right, label=f"{run_key}")
                        axes[0].fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
                    # Plot baseline data for right_joint if available
                    for label, points in baseline_data.items():
                        if '_' in label:
                            label_joint = label.split('_')[0]  # e.g., 'KNEE_INT' -> 'KNEE'
                            if label_joint.lower() in right_joint.lower() and 'intact' in label.lower() and points:
                                xs, ys = zip(*points)
                                if 'torques' in parameter_name:
                                    ys = np.array(ys) * -1
                                    if 'Banks' in baseline_file: 
                                        ys = np.array(ys)/ mass_baseline
                                        if not 'knee' in right_joint:
                                            ys = np.array(ys)* -1
                                if 'angle' in parameter_name: 
                                    if 'Banks' in baseline_file and 'knee' in right_joint:
                                        ys = np.array(ys)* -1
                                if 'Turcot' in baseline_file: 
                                    baseline_label = 'Turcot et al.'
                                elif 'Banks' in baseline_file:
                                    baseline_label = 'Banks et al.'
                                axes[0].plot(xs, ys, marker='o', linestyle='-', color='black', label=f"Baseline {baseline_label}")
                                print('Minimum Healthy: ', baseline_label, ' ,',  right_joint, ' ,',np.min(ys))
                                print('Maximum Healthy: ', baseline_label, ' ,', right_joint, ' ,', np.max(ys))
                    if parameter_name in ["angle", "velocity"] and baseline_data_3 is not None:
                        if right_joint in baseline_data_3 and parameter_name in baseline_data_3[right_joint]:#if np.any(baseline_data[right_joint][parameter_name]):
                            if 'knee' in right_joint: # Switch sign (-1) for knee sensors to match flexion/extension convention
                                axes[0].plot(x, -baseline_data_3[right_joint][parameter_name], label=f"Baseline Healthy", color='black')
                                print('Minimum Healthy Knee Right:', np.min(-baseline_data_3[right_joint][parameter_name]))
                                print('Maximum Healthy Knee Right:', np.max(-baseline_data_3[right_joint][parameter_name]))
                            else:
                                axes[0].plot(x, baseline_data_3[right_joint][parameter_name], label=f"Baseline Healthy", color='black')
                                print('Minimum Healthy:', np.min(baseline_data_3[right_joint][parameter_name]))
                                print('Maximum Healthy:', np.max(baseline_data_3[right_joint][parameter_name]))

                    axes[0].set_title(f"{right_joint} mean (all runs)")
                    axes[0].set_xlabel("Interpolated Step (%)")
                    axes[0].set_ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                    axes[0].legend()
                # Plot mean of left of all runs
                if mean_left_all:
                    for run_key, mean_left, std_left in mean_left_all:
                        axes[1].plot(x, mean_left, label=f"{run_key}")
                        axes[1].fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
                    # Plot baseline data for left_joint if available
                    for label, points in baseline_data.items():
                        if label.lower() in left_joint.lower() and points:
                            xs, ys = zip(*points)
                            if 'torques' in parameter_name:
                                ys = np.array(ys) * -1
                                if 'Banks' in baseline_file: 
                                    ys = np.array(ys)/ mass_baseline
                                    if not 'knee' in right_joint:
                                        ys = np.array(ys)* -1
                            if 'angle' in parameter_name: 
                                if 'Banks' in baseline_file and 'knee' in left_joint:
                                    ys = np.array(ys)* -1
                            if 'Turcot' in baseline_file: 
                                baseline_label = 'Turcot et al.'
                            elif 'Banks' in baseline_file:
                                baseline_label = 'Banks et al.'
                            axes[1].plot(xs, ys, marker='o', linestyle='-', color='black', label=f"Baseline {baseline_label}")
                            print('Minimum: ', baseline_label, ' ,',  right_joint, ' ,',np.min(ys))
                            print('Maximum: ', baseline_label, ' ,', right_joint, ' ,', np.max(ys))

                    for label, points in baseline_data_2.items():
                        if label.lower() in left_joint.lower() and points:
                            xs, ys = zip(*points)
                            if 'torques' in parameter_name:
                                ys = np.array(ys) * -1
                                if 'Banks' in baseline_file_2: 
                                    ys = np.array(ys)/ mass_baseline_2
                                    if not 'knee' in right_joint:
                                        ys = np.array(ys)* -1
                            if 'angle' in parameter_name: 
                                if 'Banks' in baseline_file_2 and 'knee' in left_joint:
                                    ys = np.array(ys)* -1
                            if 'Turcot' in baseline_file_2: 
                                baseline_label = 'Turcot et al.'
                            elif 'Banks' in baseline_file_2:
                                baseline_label = 'Banks et al.'
                            axes[1].plot(xs, ys, marker='o', linestyle='-', color='grey', label=f"Baseline {baseline_label}")
                            print('Minimum: ', baseline_label, ' ,',  right_joint, ' ,',np.min(ys))
                            print('Maximum: ', baseline_label, ' ,', right_joint, ' ,', np.max(ys))


                    if parameter_name in ["angle", "velocity"] and baseline_data_3 is not None:
                        if right_joint in baseline_data_3 and parameter_name in baseline_data_3[right_joint]:#if np.any(baseline_data[right_joint][parameter_name]):
                            if 'knee' in right_joint: # Switch sign (-1) for knee sensors to match flexion/extension convention
                                axes[1].plot(x, -baseline_data_3[right_joint][parameter_name], label=f"Baseline Healthy", color='black')
                            else:
                                axes[1].plot(x, baseline_data_3[right_joint][parameter_name], label=f"Baseline Healthy", color='black')

                    axes[1].set_title(f"{left_joint} mean (all runs)")
                    axes[1].set_xlabel("Interpolated Step (%)")
                    axes[1].set_ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                    axes[1].legend()
                # Plot difference (left - right) of all runs
                if diff_all:
                    for run_key, diff in diff_all:
                        axes[2].plot(x, diff, label=f"{run_key} Diff (L-R)")
                    axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
                    axes[2].set_title(f"Abs {left_joint} - Abs {right_joint} difference (all runs)")
                    axes[2].set_xlabel("Interpolated Step (%)")
                    axes[2].set_ylabel(f"{parameter_name} difference ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                    axes[2].legend()
                plt.tight_layout()
                plt.show()
                plt.close()



    


    @staticmethod 
    def plot_joint_angle_symmetry_all_runs_prosthesis_intact_baseline_comp_pros_vs_health(
        joint_names_list,
        all_loaded_data,
        run_step_data,
        baseline_file,
        baseline_file_2,
        baseline_data_3 = 'None',
        parameter_name="angle",
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=100,
        body_weight_to_normalize=1,
        mass_baseline = 75*10,
        mass_baseline_2 = 75*10,
        healthy = ['right'],
    ):
        """
        Plot mean of right of all runs, mean of left, and the difference, side by side in one figure.
        Only plot runs where both left and right data are available.
        Also plot baseline data from baseline_file.
        """

        joint_pairs = [(name + "_l", name + "_r") for name in joint_names_list]
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_iter = itertools.cycle(color_cycle)

        # define axis_healthy and axis_prosthesis
        run_keys = list(all_loaded_data.keys())
        if len(healthy)==1:
            healthy = healthy*len(run_keys)
        
        # fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

        # for run_idx, run_key in enumerate(run_keys):
        #     if healthy[run_idx]=='right':
        #         axis_healthy = axes[0]
        #         axis_prosthesis = axes[1]
        #     else:
        #         axis_healthy = axes[1]
        #         axis_prosthesis = axes[0]
        # Load baseline data from file
        baseline_data = {}
        current_label = None
        baseline_data_2 = {}
        with open(baseline_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if 'Schmalz' in baseline_file:
                    label_match = re.match(r'^([A-Z_]+)\s*=\s*\[?', line)
                else:
                    label_match = re.match(r'^([A-Za-z_]+)\s*=\s*\[?', line)
                if label_match:
                    current_label = label_match.group(1)
                    baseline_data[current_label] = []
                    continue
                if current_label:
                    vals = line.split(';')
                    if len(vals) == 2:
                        try:
                            x, y = map(float, vals)
                            baseline_data[current_label].append((x, y))
                        except ValueError:
                            continue

        with open(baseline_file_2, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if 'Schmalz' in baseline_file_2:
                    label_match = re.match(r'^([A-Z_]+)\s*=\s*\[?', line)
                else:
                    label_match = re.match(r'^([A-Za-z_]+)\s*=\s*\[?', line)
                if label_match:
                    current_label = label_match.group(1)
                    baseline_data_2[current_label] = []
                    continue
                if current_label:
                    vals = line.split(';')
                    if len(vals) == 2:
                        try:
                            x, y = map(float, vals)
                            baseline_data_2[current_label].append((x, y))
                        except ValueError:
                            continue

        for left_joint, right_joint in joint_pairs:
            # run_keys = list(all_loaded_data.keys())
            mean_left_all = []
            std_left_all = []
            mean_right_all = []
            std_right_all = []
            diff_all = []
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                joint_data = {}
                evaluation_joint_names = run_dict.get("evaluation_joint_names")
                joint_parameters = ["angle", "velocity", "forces_constraint", "forces_smooth", "forces_applied", "torques", "energy_exp"]
                for joint_name in evaluation_joint_names:
                    joint_data[joint_name] = {}
                    for key in joint_parameters:
                        param = run_dict.get(f"{joint_name}_{key}")
                        if param is not None:
                            joint_data[joint_name][key] = param

                step_indices_left = run_step_data[run_key]["step_start_left"]
                step_indices_right = run_step_data[run_key]["step_start_right"]
                left_data = joint_data.get(left_joint, {}).get(parameter_name, None)
                right_data = joint_data.get(right_joint, {}).get(parameter_name, None)
                if left_data is None or right_data is None:
                    continue

                def collect_steps(data, indices):
                    steps = []
                    if indices and len(indices) > 1:
                        for j in range(len(indices) - 1):
                            start, end = indices[j], indices[j + 1]
                            y = [float(np.array(a).squeeze()) for a in data[start:end]]
                            if len(y) < 2:
                                continue
                            if interp_mode == "interp":
                                x_old = np.linspace(0, 1, len(y))
                                x_new = np.linspace(0, 1, interp_len)
                                y = np.interp(x_new, x_old, y)
                            if convert_to_deg and parameter_name in ("angle", "velocity"):
                                y = np.rad2deg(y)
                            steps.append(y)
                    return steps

                left_steps = collect_steps(left_data, step_indices_left)
                right_steps = collect_steps(right_data, step_indices_right)

                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0) / body_weight_to_normalize
                    std_left = np.std(left_steps, axis=0) / body_weight_to_normalize
                    mean_right = np.mean(right_steps, axis=0) / body_weight_to_normalize
                    std_right = np.std(right_steps, axis=0) / body_weight_to_normalize
                    if 'knee' in left_joint:
                        mean_left = -mean_left
                        mean_right = -mean_right
                        std_left = -std_left
                        std_right = -std_right
                    diff = np.abs(mean_left) - np.abs(mean_right)
                    mean_left_all.append((run_key, mean_left, std_left))
                    mean_right_all.append((run_key, mean_right, std_right))
                    diff_all.append((run_key, diff))

            x = np.linspace(0, 100, interp_len)
            if mean_left_all or mean_right_all or diff_all:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
                axis_healthy = axes[0]
                axis_prosthesis = axes[1]
                # for run_idx, run_key in enumerate(run_keys):
                #     if healthy[run_idx]=='right':
                #         axis_healthy = axes[0]
                #         axis_prosthesis = axes[1]
                #     else:
                #         axis_healthy = axes[1]
                #         axis_prosthesis = axes[0]
                # Plot mean of right of all runs
                if mean_right_all:
                    for run_idx, run_key in enumerate(run_keys):
                        for idx, (rk, mean_right, std_right) in enumerate(mean_right_all):
                            if rk == run_key:
                                if healthy[run_idx] == 'right':
                                    axis_healthy.plot(x, mean_right, label=f"{run_key}")
                                    axis_healthy.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
                                else:
                                    axis_prosthesis.plot(x, mean_right, label=f"{run_key}")
                                    axis_prosthesis.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
                                break
                            # axis_healthy.plot(x, mean_right, label=f"{run_key}")
                            # axis_healthy.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
                        else:
                            axis_prosthesis.plot(x, mean_right, label=f"{run_key}")
                            axis_prosthesis.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
                    # Plot baseline data for right_joint if available
                    for label, points in baseline_data.items():
                        if '_' in label:
                            label_joint = label.split('_')[0]  # e.g., 'KNEE_INT' -> 'KNEE'
                            if label_joint.lower() in right_joint.lower() and 'intact' in label.lower() and points:
                                xs, ys = zip(*points)
                                if 'torques' in parameter_name:
                                    ys = np.array(ys) * -1
                                    if 'Banks' in baseline_file: 
                                        ys = np.array(ys)/ mass_baseline
                                        if not 'knee' in right_joint:
                                            ys = np.array(ys)* -1
                                if 'angle' in parameter_name: 
                                    if 'Banks' in baseline_file and 'knee' in right_joint:
                                        ys = np.array(ys)* -1
                                if 'Turcot' in baseline_file: 
                                    baseline_label = 'Turcot et al.'
                                elif 'Banks' in baseline_file:
                                    baseline_label = 'Banks et al.'
                                axis_healthy.plot(xs, ys, marker='o', linestyle='-', color='black', label=f"Baseline {baseline_label}")
                                print('Minimum Healthy: ', baseline_label, ' ,',  right_joint, ' ,',np.min(ys))
                                print('Maximum Healthy: ', baseline_label, ' ,', right_joint, ' ,', np.max(ys))
                    if parameter_name in ["angle", "velocity"] and baseline_data_3 is not None:
                        if right_joint in baseline_data_3 and parameter_name in baseline_data_3[right_joint]:#if np.any(baseline_data[right_joint][parameter_name]):
                            if 'knee' in right_joint: # Switch sign (-1) for knee sensors to match flexion/extension convention
                                axis_healthy.plot(x, -baseline_data_3[right_joint][parameter_name], label=f"Baseline Healthy", color='black')
                                print('Minimum Healthy Knee Right:', np.min(-baseline_data_3[right_joint][parameter_name]))
                                print('Maximum Healthy Knee Right:', np.max(-baseline_data_3[right_joint][parameter_name]))
                            else:
                                axis_healthy.plot(x, baseline_data_3[right_joint][parameter_name], label=f"Baseline Healthy", color='black')
                                print('Minimum Healthy:', np.min(baseline_data_3[right_joint][parameter_name]))
                                print('Maximum Healthy:', np.max(baseline_data_3[right_joint][parameter_name]))

                    axis_healthy.set_title(f"{right_joint} mean (all runs) HEALTHY")
                    axis_healthy.set_xlabel("Interpolated Step (%)")
                    axis_healthy.set_ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                    # axis_healthy.legend()
                # Plot mean of left of all runs
                if mean_left_all:
                    for run_idx, run_key in enumerate(run_keys):
                        for idx, (rk, mean_left, std_left) in enumerate(mean_left_all):
                            if rk == run_key:
                                if healthy[run_idx] == 'left':
                                    axis_healthy.plot(x, mean_left, label=f"{run_key}")
                                    axis_healthy.fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
                                else:
                                    axis_prosthesis.plot(x, mean_left, label=f"{run_key}")
                                    axis_prosthesis.fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
                                break
                            # axis_healthy.plot(x, mean_right, label=f"{run_key}")
                            # axis_healthy.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
                        else:
                            axis_prosthesis.plot(x, mean_right, label=f"{run_key}")
                            axis_prosthesis.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
                        # axis_prosthesis.plot(x, mean_left, label=f"{run_key}")
                        # axis_prosthesis.fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
                    # Plot baseline data for left_joint if available
                    for label, points in baseline_data.items():
                        if label.lower() in left_joint.lower() and points:
                            xs, ys = zip(*points)
                            if 'torques' in parameter_name:
                                ys = np.array(ys) * -1
                                if 'Banks' in baseline_file: 
                                    ys = np.array(ys)/ mass_baseline
                                    if not 'knee' in right_joint:
                                        ys = np.array(ys)* -1
                            if 'angle' in parameter_name: 
                                if 'Banks' in baseline_file and 'knee' in left_joint:
                                    ys = np.array(ys)* -1
                            if 'Turcot' in baseline_file: 
                                baseline_label = 'Turcot et al.'
                            elif 'Banks' in baseline_file:
                                baseline_label = 'Banks et al.'
                            axis_prosthesis.plot(xs, ys, marker='o', linestyle='-', color='black', label=f"Baseline {baseline_label}")
                            print('Minimum: ', baseline_label, ' ,',  right_joint, ' ,',np.min(ys))
                            print('Maximum: ', baseline_label, ' ,', right_joint, ' ,', np.max(ys))

                    for label, points in baseline_data_2.items():
                        if label.lower() in left_joint.lower() and points:
                            xs, ys = zip(*points)
                            if 'torques' in parameter_name:
                                ys = np.array(ys) * -1
                                if 'Banks' in baseline_file_2: 
                                    ys = np.array(ys)/ mass_baseline_2
                                    if not 'knee' in right_joint:
                                        ys = np.array(ys)* -1
                            if 'angle' in parameter_name: 
                                if 'Banks' in baseline_file_2 and 'knee' in left_joint:
                                    ys = np.array(ys)* -1
                            if 'Turcot' in baseline_file_2: 
                                baseline_label = 'Turcot et al.'
                            elif 'Banks' in baseline_file_2:
                                baseline_label = 'Banks et al.'
                            axis_prosthesis.plot(xs, ys, marker='o', linestyle='-', color='grey', label=f"Baseline {baseline_label}")
                            print('Minimum: ', baseline_label, ' ,',  right_joint, ' ,',np.min(ys))
                            print('Maximum: ', baseline_label, ' ,', right_joint, ' ,', np.max(ys))


                    if parameter_name in ["angle", "velocity"] and baseline_data_3 is not None:
                        if right_joint in baseline_data_3 and parameter_name in baseline_data_3[right_joint]:#if np.any(baseline_data[right_joint][parameter_name]):
                            if 'knee' in right_joint: # Switch sign (-1) for knee sensors to match flexion/extension convention
                                axis_prosthesis.plot(x, -baseline_data_3[right_joint][parameter_name], label=f"Baseline Healthy", color='black')
                            else:
                                axis_prosthesis.plot(x, baseline_data_3[right_joint][parameter_name], label=f"Baseline Healthy", color='black')

                    axis_prosthesis.set_title(f"{left_joint} mean (all runs) IMPAIRED")
                    axis_prosthesis.set_xlabel("Interpolated Step (%)")
                    axis_prosthesis.set_ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                    axis_prosthesis.legend()
                    axis_healthy.legend()
                # Plot difference (left - right) of all runs
                if diff_all:
                    for run_key, diff in diff_all:
                        axes[2].plot(x, diff, label=f"{run_key} Diff (L-R)")
                    axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
                    axes[2].set_title(f"Abs {left_joint} - Abs {right_joint} difference (all runs)")
                    axes[2].set_xlabel("Interpolated Step (%)")
                    axes[2].set_ylabel(f"{parameter_name} difference ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                    axes[2].legend()
                plt.tight_layout()
                plt.show()
                plt.close()




    @staticmethod
    def _load_baseline_file(baseline_file):
        """Loads data points from text files."""
        baseline_data = {}
        current_label = None
        with open(baseline_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                # Use regex to handle various label formats
                label_match = re.match(r'^([A-Z_]+)\s*=\s*\[?', line) or re.match(r'^([A-Za-z_]+)\s*=\s*\[?', line)
                if label_match:
                    current_label = label_match.group(1)
                    baseline_data[current_label] = []
                    continue
                if current_label and '[' not in line and ']' not in line: # Avoid processing bracket lines as data
                    vals = line.split(';')
                    if len(vals) == 2:
                        try:
                            x, y = map(float, vals)
                            baseline_data[current_label].append((x, y))
                        except ValueError: continue
        return baseline_data


    @staticmethod
    def plot_joint_angle_symmetry_all_runs_prosthesis_intact_baseline_comp_helper(
        joint_names_list,
        all_loaded_data,
        run_step_data,
        baseline_file,
        baseline_file_2,
        baseline_data_3='None',
        parameter_name="angle",
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=100,
        body_weight_to_normalize=1,
        mass_baseline=75*10,
        mass_baseline_2=75*10,
        adjust_talus_offset=False,
    ):
        """
        Processes and aggregates mean curves for all runs and all baselines.
        """
        
        # --- 1. Load Baseline Files (Baseline 1 and 2) ---
        baseline_data_1 = PostProcessMetricsHandler._load_baseline_file(baseline_file)
        baseline_data_2 = PostProcessMetricsHandler._load_baseline_file(baseline_file_2)
        
        joint_pairs = [(name + "_l", name + "_r") for name in joint_names_list]
        aggregated_mean_data = {} 

        # Define collect_steps helper function (for run data)
        def collect_steps(data, indices):
            steps = []
            if indices and len(indices) > 1:
                for j in range(len(indices) - 1):
                    start, end = indices[j], indices[j + 1]
                    y = [float(np.array(a).squeeze()) for a in data[start:end]]
                    if len(y) < 2: continue
                    if interp_mode == "interp":
                        x_old = np.linspace(0, 1, len(y)); x_new = np.linspace(0, 1, interp_len)
                        y = np.interp(x_new, x_old, y)
                    if convert_to_deg and parameter_name in ("angle", "velocity"):
                        y = np.rad2deg(y)
                    steps.append(y)
            return steps

        for left_joint, right_joint in joint_pairs:
            # joint_base_name is e.g. 'hip_flexion'
            joint_base_name = left_joint.rsplit('_', 1)[0]
            if joint_base_name not in aggregated_mean_data:
                aggregated_mean_data[joint_base_name] = {}
            
            # --- 2. Process Runs (Existing Logic) ---
            run_keys = list(all_loaded_data.keys())
            for run_key in run_keys:
                # ... (Run data extraction logic as provided in the prompt) ...
                run_dict = all_loaded_data[run_key]
                joint_data = {}
                evaluation_joint_names = run_dict.get("evaluation_joint_names", [])
                joint_parameters = ["angle", "velocity", "forces_constraint", "forces_smooth", "forces_applied", "torques", "energy_exp"]
                for joint_name in evaluation_joint_names:
                    joint_data[joint_name] = {key: run_dict.get(f"{joint_name}_{key}") 
                                              for key in joint_parameters if run_dict.get(f"{joint_name}_{key}") is not None}

                step_indices_left = run_step_data.get(run_key, {}).get("step_start_left", [])
                step_indices_right = run_step_data.get(run_key, {}).get("step_start_right", [])
                left_data = joint_data.get(left_joint, {}).get(parameter_name, None)
                right_data = joint_data.get(right_joint, {}).get(parameter_name, None)
                if left_data is None or right_data is None: continue
                
                left_steps = collect_steps(left_data, step_indices_left)
                right_steps = collect_steps(right_data, step_indices_right)

                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0) / body_weight_to_normalize
                    mean_right = np.mean(right_steps, axis=0) / body_weight_to_normalize
                    
                    if 'knee' in joint_base_name:
                        mean_left, mean_right = -mean_left, -mean_right
                    if 'ankle' in joint_base_name and adjust_talus_offset: 
                        if 'deg' in run_key: 
                            # Look for offset in the run_key, e.g., '-5deg' or '5deg'
                            match = re.search(r'(-?\d+)\s*deg', run_key)
                            if match:
                                offset = float(match.group(1))
                                mean_left -= offset
                                # mean_right += offset
                    # Store mean curves for features 
                    aggregated_mean_data[joint_base_name][run_key] = {
                        'mean_l': mean_left, 
                        'mean_r': mean_right,
                        # ... (std, steps, diff data can remain here if needed elsewhere)
                    }

            for i, (b_data, b_label, b_mass, b_file) in enumerate(zip(
                [baseline_data_1, baseline_data_2], 
                ['Baseline_1', 'Baseline_2'], 
                [mass_baseline, mass_baseline_2],
                [baseline_file, baseline_file_2])):
                
                if 'Banks' in b_file: 
                    baseline_key = 'Banks' #b_label
                elif 'Turcot' in b_file:
                    baseline_key = 'Turcot' #b_label
                
                # Default to zero array if data is missing, which is safe for plotting/extraction
                zero_curve = np.zeros(interp_len)
                mean_curve_l = zero_curve
                mean_curve_r = zero_curve
                data_found = False

                # ⚠️ CORRECTED LABEL SEARCH: Use only the simplified base joint name
                simple_joint_name = joint_base_name.split('_')[0].upper() # e.g., 'HIP' from 'hip_flexion'

                # --- Search for Intact and Non-Intact labels ---
                # Search for label containing the simple joint name AND '_INTACT'
                label_intact = next((label for label in b_data 
                                     if simple_joint_name in label.upper() and '_INTACT' in label.upper() and b_data[label]), None)
                
                # Search for label containing the simple joint name BUT NOT '_INTACT' (the generic curve)
                label_non_intact = next((label for label in b_data 
                                         if simple_joint_name in label.upper() and '_INTACT' not in label.upper() and b_data[label]), None)
                
                # --- Baseline 1: Intact (Right) and Non-Intact (Left) ---
                if baseline_key == 'Banks':
                    
                    # 1. Process Intact (Maps to Right side)
                    if label_intact:
                        xs, ys = zip(*b_data[label_intact])
                        x_new = np.linspace(0, 100, interp_len)
                        mean_curve_r = np.interp(x_new, xs, ys)
                        data_found = True
                    
                    # 2. Process Non-Intact (Maps to Left side)
                    if label_non_intact:
                        xs, ys = zip(*b_data[label_non_intact])
                        x_new = np.linspace(0, 100, interp_len)
                        mean_curve_l = np.interp(x_new, xs, ys)
                        data_found = True

                    # Fallback: If only the Intact curve is found, assume symmetry for BL1
                    if data_found and not label_non_intact and label_intact:
                        mean_curve_l = mean_curve_r
                        
                # --- Baseline 2: Only Left Side Data Available ---
                elif baseline_key == 'Turcot':
                    # For BL2, use the most relevant curve found and map it to Left
                    target_label = None
                    if label_non_intact:
                        target_label = label_non_intact
                    elif label_intact:
                        target_label = label_intact
                        
                    if target_label:
                        xs, ys = zip(*b_data[target_label])
                        x_new = np.linspace(0, 100, interp_len)
                        mean_curve_l = np.interp(x_new, xs, ys)
                        # mean_curve_r remains zero_curve as requested
                        data_found = True


                # --- Apply Normalization and Sign Corrections (Applied to the resulting curves) ---
                if data_found:
                    
                    # --- Normalization and Sign Switch for Torques/Angles (must be applied to L and R) ---
                    for side_curve in [mean_curve_l, mean_curve_r]:
                        if side_curve is not zero_curve: # Only apply to curves that actually have data
                            if 'torques' in parameter_name.lower(): 
                                side_curve *= -1
                                if 'Banks' in b_file: side_curve /= b_mass
                                if 'Banks' in b_file and 'knee' not in joint_base_name: side_curve *= -1

                            if 'angle' in parameter_name.lower() and 'Banks' in b_file and 'knee' in joint_base_name:
                                side_curve *= -1
                            
                    # --- Store the results ---
                    aggregated_mean_data[joint_base_name][baseline_key] = {
                        'mean_l': mean_curve_l, 
                        'mean_r': mean_curve_r,
                    }

            # --- 4. Process Baseline 3 (Curve Array Data - CORRECTED) ---
            # Correct SyntaxWarning: use != 'None' for string comparison
            if baseline_data_3 is not None and baseline_data_3 != 'None':
                baseline_key = 'Baseline_Healthy'
                
                # Keys are the full joint name with side suffix (e.g., 'hip_flexion_l')
                l_key = left_joint
                r_key = right_joint
                
                # Check for existence and then retrieve the array
                mean_curve_l = baseline_data_3.get(l_key, {}).get(parameter_name, None)
                mean_curve_r = baseline_data_3.get(r_key, {}).get(parameter_name, None)
                
                if mean_curve_r is not None or mean_curve_l is not None:
                    
                    # Fallback: Use R curve for L if L is missing (common for healthy baselines)
                    if mean_curve_r is not None and mean_curve_l is None:
                        mean_curve_l = mean_curve_r
                    
                    # Ensure mean curve length is interpolated to interp_len if needed
                    # Note: Need to handle the case where the curve might not be a NumPy array yet
                    mean_curves = []
                    for side_curve in [mean_curve_l, mean_curve_r]:
                        if side_curve is not None:
                             side_curve = np.array(side_curve).squeeze()
                             if len(side_curve) != interp_len:
                                x_new = np.linspace(0, 1, interp_len)
                                x_old = np.linspace(0, 1, len(side_curve))
                                side_curve = np.interp(x_new, x_old, side_curve)
                        mean_curves.append(side_curve)
                    mean_curve_l, mean_curve_r = mean_curves[0], mean_curves[1]
                    
                    # Apply knee sign correction (as shown in the plotting snippet)
                    if 'knee' in joint_base_name:
                         if mean_curve_l is not None: mean_curve_l = -mean_curve_l
                         if mean_curve_r is not None: mean_curve_r = -mean_curve_r
                
                    # Store the data (only if at least one side was found/derived)
                    if mean_curve_l is not None or mean_curve_r is not None:
                        aggregated_mean_data[joint_base_name][baseline_key] = {
                            'mean_l': mean_curve_l, 
                            'mean_r': mean_curve_r,
                        }

        # The function logic for the original curve plotting goes here (omitted for brevity)
        
        return aggregated_mean_data
    
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------

    # @staticmethod
    # def extract_gait_parameters(processed_data, interp_len=100):
    #     """
    #     Extracts specific kinematic features (max/min/timing) for Hip, Knee, and Ankle 
    #     from the processed mean joint angle data.
    #     """
    #     gait_features = {}
    #     time_x = np.linspace(0, 100, interp_len)

    #     for joint_name, run_data in processed_data.items():
    #         gait_features[joint_name] = {}

    #         for run_key, data in run_data.items():
    #             gait_features[joint_name][run_key] = {}

    #             for side in ['l', 'r']:
    #                 mean_curve = data[f'mean_{side}']
    #                 features = {}

    #                 if 'hip' in joint_name.lower():
    #                     # Hip: Timing and value of maximum
    #                     max_val = np.max(mean_curve)
    #                     max_idx = np.argmax(mean_curve)
    #                     features['max_val'] = max_val
    #                     features['t_max'] = time_x[max_idx]

    #                 elif 'knee' in joint_name.lower():
    #                     # Knee: Maximum in first 40% and its time, Minimum, Maximum value in second part and timing
    #                     split_idx = int(0.40 * interp_len) # 40% of step cycle
                        
    #                     # Max in first 40%
    #                     max1_val = np.max(mean_curve[:split_idx])
    #                     max1_idx = np.argmax(mean_curve[:split_idx])
    #                     features['max1_val'] = max1_val
    #                     features['t_max1'] = time_x[max1_idx]

    #                     # Global Minimum
    #                     min_val = np.min(mean_curve)
    #                     min_idx = np.argmin(mean_curve)
    #                     features['min_val'] = min_val
    #                     features['t_min'] = time_x[min_idx]

    #                     # Max in second part (40% to 100%)
    #                     max2_val = np.max(mean_curve[split_idx:])
    #                     max2_idx = np.argmax(mean_curve[split_idx:]) + split_idx
    #                     features['max2_val'] = max2_val
    #                     features['t_max2'] = time_x[max2_idx]
                        
    #                 elif 'ankle' in joint_name.lower():
    #                     # Ankle: Time and value of maximum, time and value of minimum
    #                     max_val = np.max(mean_curve)
    #                     max_idx = np.argmax(mean_curve)
    #                     min_val = np.min(mean_curve)
    #                     min_idx = np.argmin(mean_curve)
                        
    #                     features['max_val'] = max_val
    #                     features['t_max'] = time_x[max_idx]
    #                     features['min_val'] = min_val
    #                     features['t_min'] = time_x[min_idx]

    #                 gait_features[joint_name][run_key][side] = features
    #     return gait_features

    # # ----------------------------------------------------------------------
    # # ----------------------------------------------------------------------

    # @staticmethod
    # def plot_gait_parameter_bars(gait_features, parameter_name):
    #     """
    #     Generates bar plots for each extracted kinematic feature, comparing runs.

    #     ADAPTED: Correctly maps simplified joint names ('hip', 'knee', 'ankle') 
    #     to the full keys in gait_features ('hip_flexion', 'knee_angle', etc.).
    #     """
    #     feature_names = {
    #         'hip': [('max_val', 'Max. Value'), ('t_max', 'Time to Max. (%)')],
    #         'knee': [('max1_val', 'Max. Value (0-40%)'), ('t_max1', 'Time to Max. 1 (%)'),
    #                  ('min_val', 'Min. Value'), ('t_min', 'Time to Min. (%)'),
    #                  ('max2_val', 'Max. Value (40-100%)'), ('t_max2', 'Time to Max. 2 (%)')],
    #         'ankle': [('max_val', 'Max. Value'), ('t_max', 'Time to Max. (%)'),
    #                   ('min_val', 'Min. Value'), ('t_min', 'Time to Min. (%)')]
    #     }

    #     # Iterate over the defined simplified joint names (hip, knee, ankle)
    #     for joint_base_name, feature_list in feature_names.items():
            
    #         # --- FIX: Find the actual full key in gait_features ---
    #         # Search gait_features keys for the one containing the simplified base name
    #         # e.g., Find 'hip_flexion' from 'hip'
    #         full_joint_key = None
    #         for key in gait_features.keys():
    #             if joint_base_name in key:
    #                 full_joint_key = key
    #                 break
            
    #         # If no matching joint data is found (e.g., 'hip' data wasn't extracted), skip.
    #         if full_joint_key is None:
    #             continue

    #         # Plot each specific feature (e.g., Hip Max Value, Hip T_Max)
    #         for feature_key, feature_label in feature_list:
                
    #             # --- Prepare data for a single bar plot (e.g., Hip Max Value) ---
    #             # Use the now-found 'full_joint_key'
    #             run_keys = list(gait_features[full_joint_key].keys())
    #             num_runs = len(run_keys)
                
    #             # Collect data for left and right bars
    #             left_values = [gait_features[full_joint_key][rk]['l'].get(feature_key, 0) for rk in run_keys]
    #             right_values = [gait_features[full_joint_key][rk]['r'].get(feature_key, 0) for rk in run_keys]

    #             # --- Plotting ---
    #             x = np.arange(num_runs)
    #             width = 0.35  # Width of the bars

    #             fig, ax = plt.subplots(figsize=(10, 6))

    #             # Plot Left Side
    #             rects1 = ax.bar(x - width/2, left_values, width, label='Left Side', color='tab:blue')
    #             # Plot Right Side
    #             rects2 = ax.bar(x + width/2, right_values, width, label='Right Side', color='tab:red')

    #             # --- Formatting ---
    #             # Determine Y-label based on feature type
    #             y_label = feature_label.split('(')[0].strip() 
    #             if 'Time' in feature_label:
    #                  y_unit = '(%)'
    #             elif 'Value' in feature_label:
    #                  # Check if original parameter was angle/velocity for unit clarity
    #                  if 'angle' in parameter_name.lower() or 'velocity' in parameter_name.lower():
    #                      y_unit = f"({parameter_name.split('_')[0]} {'(deg)' if 'angle' in parameter_name.lower() else 'unit'})"
    #                  else:
    #                       y_unit = f"({parameter_name} unit)"
    #             else:
    #                 y_unit = ''

    #             ax.set_ylabel(f'{y_label} {y_unit}')
    #             ax.set_title(f'{full_joint_key.replace("_", " ").title()} Kinematic Feature: {feature_label}')
    #             ax.set_xticks(x)
    #             ax.set_xticklabels(run_keys, rotation=45, ha="right")
    #             ax.legend()
    #             ax.grid(axis='y', linestyle='--', alpha=0.7)
    #             plt.tight_layout()
    #             plt.show()
    #             plt.close()

    

    @staticmethod
    def _calculate_joint_features(joint_base_name, mean_curve, time_x, interp_len):
        """Internal helper to calculate features for a single mean curve (retained)."""
        features = {}

        if 'hip' in joint_base_name.lower():
            # Hip: Timing and value of maximum
            min_val = np.min(mean_curve); min_idx = np.argmin(mean_curve)
            features['min_val'] = min_val; features['t_min'] = time_x[min_idx]

        elif 'knee' in joint_base_name.lower():
            # Knee features
            split_idx = int(0.40 * interp_len)
            max1_val = np.max(mean_curve[:split_idx]); max1_idx = np.argmax(mean_curve[:split_idx])
            features['max1_val'] = max1_val; features['t_max1'] = time_x[max1_idx]
            min_val = np.min(mean_curve); min_idx = np.argmin(mean_curve)
            features['min_val'] = min_val; features['t_min'] = time_x[min_idx]
            max2_val = np.max(mean_curve[split_idx:]); max2_idx = np.argmax(mean_curve[split_idx:]) + split_idx
            features['max2_val'] = max2_val; features['t_max2'] = time_x[max2_idx]
            
        elif 'ankle' in joint_base_name.lower():
            # Ankle features
            max_val = np.max(mean_curve); max_idx = np.argmax(mean_curve)
            min_val = np.min(mean_curve); min_idx = np.argmin(mean_curve)
            features['max_val'] = max_val; features['t_max'] = time_x[max_idx]
            features['min_val'] = min_val; features['t_min'] = time_x[min_idx]

        return features

    @staticmethod
    def extract_gait_parameters(processed_data, parameter_name, interp_len=100):
        """
        Extracts kinematic features from ALL aggregated mean data (runs and baselines).
        """
        gait_features = {}
        time_x = np.linspace(0, 100, interp_len)
        
        # Process all entries (Runs, Baseline_1, Baseline_2, Baseline_Healthy)
        for joint_full_name, joint_data in processed_data.items():
            gait_features[joint_full_name] = {}
            joint_base_name = joint_full_name.split('_')[0] # e.g., 'hip'
            
            for run_key, data in joint_data.items():
                gait_features[joint_full_name][run_key] = {}

                for side in ['l', 'r']:
                    mean_curve = data.get(f'mean_{side}')
                    
                    # Handle cases where a specific baseline side might be None (e.g., if data was missing)
                    if mean_curve is None or len(mean_curve) == 0:
                        # Use NaN for numerical results to indicate missing data
                        features = {'max_val': np.nan, 't_max': np.nan, 'min_val': np.nan, 't_min': np.nan, 
                                    'max1_val': np.nan, 't_max1': np.nan, 'max2_val': np.nan, 't_max2': np.nan}
                    else:
                        features = PostProcessMetricsHandler._calculate_joint_features(
                            joint_base_name, mean_curve, time_x, interp_len
                        )
                    gait_features[joint_full_name][run_key][side] = features
                    
        return gait_features



    @staticmethod
    def plot_gait_parameter_bars(gait_features, parameter_name):
        """
        Generates bar plots for each extracted kinematic feature, comparing all runs 
        and all three baselines (Banks, Turcot, Baseline_Healthy).
        """
        feature_names = {
            'hip': [('min_val', 'Min. Value'), ('t_min', 'Time to Min. (%)')],
            'knee': [('max1_val', 'Max. Value (0-40%)'), ('t_max1', 'Time to Max. 1 (%)'),
                    ('min_val', 'Min. Value'), ('t_min', 'Time to Min. (%)'),
                    ('max2_val', 'Max. Value (40-100%)'), ('t_max2', 'Time to Max. 2 (%)')],
            'ankle': [('max_val', 'Max. Value'), ('t_max', 'Time to Max. (%)'),
                    ('min_val', 'Min. Value'), ('t_min', 'Time to Min. (%)')]
        }
        
        # Define unique colors and hatching for runs vs. baselines
        RUN_COLOR_L, RUN_COLOR_R = 'tab:blue', 'tab:red'
        # ⚠️ BL_COLOR_MAP is correctly defined with the new keys
        BL_COLOR_MAP = {
            'Banks': ('darkgreen', 'green'),
            'Turcot': ('darkmagenta', 'purple'),
            'Baseline_Healthy': ('saddlebrown', 'gold'), 
        }
        BL_HATCH = '///' # Hatch pattern for all baselines
        
        # Define the list of expected baseline keys for easy filtering/sorting
        BASELINE_NAMES = list(BL_COLOR_MAP.keys())

        for joint_base_name, feature_list in feature_names.items():
            
            full_joint_key = next((key for key in gait_features.keys() if joint_base_name in key), None)
            if full_joint_key is None: continue

            for feature_key, feature_label in feature_list:
                
                all_keys = list(gait_features[full_joint_key].keys())
                
                # ⚠️ CORRECTED BASELINE KEY SEPARATION
                # Identify runs: keys not in the BASELINE_NAMES list
                run_keys = [k for k in all_keys if k not in BASELINE_NAMES]
                
                # Identify baselines: keys that ARE in the BASELINE_NAMES list
                baseline_keys = [k for k in all_keys if k in BASELINE_NAMES]
                
                # Sort baselines by a predefined order for consistency (Banks, Turcot, Healthy)
                baseline_keys_sorted = sorted(baseline_keys, key=lambda k: BASELINE_NAMES.index(k))
                
                # Final plotting order
                plotting_keys = run_keys + baseline_keys_sorted
                num_groups = len(plotting_keys)
                
                # --- Plotting Setup ---
                x = np.arange(num_groups)
                width = 0.35

                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Dictionary to store unique handles/labels for the final legend
                legend_handles_map = {}

                # Iterate through all keys to plot
                for i, key in enumerate(plotting_keys):
                    data = gait_features[full_joint_key][key]
                    x_pos = x[i]
                    
                    # Get the value, using 0 if NaN (which happens if data was missing)
                    l_val = data['l'].get(feature_key, 0)
                    r_val = data['r'].get(feature_key, 0)
                    
                    is_baseline = key in BASELINE_NAMES
                    
                    # --- Assign Colors and Labels ---
                    if is_baseline:
                        # Key matches the name in BL_COLOR_MAP directly (e.g., 'Banks')
                        l_color, r_color = BL_COLOR_MAP.get(key, ('gray', 'darkgray'))
                        hatch = BL_HATCH
                        label_l = f"{key.replace('_', ' ')} L" # e.g., 'Banks L'
                        label_r = f"{key.replace('_', ' ')} R" # e.g., 'Banks R'
                    else:
                        l_color, r_color = RUN_COLOR_L, RUN_COLOR_R
                        hatch = None
                        label_l = 'Run L'
                        label_r = 'Run R'
                    
                    # Plot Left Side
                    h_l = ax.bar(x_pos - width/2, l_val, width, color=l_color, hatch=hatch, label=label_l)
                    # Plot Right Side
                    h_r = ax.bar(x_pos + width/2, r_val, width, color=r_color, hatch=hatch, label=label_r)
                    
                    # Store the handles for the final legend
                    # This correctly stores unique entries for Banks L, Banks R, Turcot L, etc.
                    legend_handles_map[label_l] = h_l
                    legend_handles_map[label_r] = h_r


                # --- Formatting ---
                y_label = feature_label.split('(')[0].strip() 
                if 'Time' in feature_label: y_unit = '(%)'
                elif 'Value' in feature_label:
                    y_unit = f"({parameter_name.split('_')[0]} {'(deg)' if 'angle' in parameter_name.lower() else 'unit'})"
                else: y_unit = ''

                ax.set_ylabel(f'{y_label} {y_unit}')
                ax.set_title(f'{full_joint_key.replace("_", " ").title()} Kinematic Feature: {feature_label}')
                
                # Set x-ticks to include all run and baseline keys
                ax.set_xticks(x)
                ax.set_xticklabels(plotting_keys, rotation=45, ha="right")
                
                # Create consolidated legend using the stored unique handles
                labels = list(legend_handles_map.keys())
                handles = [legend_handles_map[l] for l in labels]
                
                # Reorder labels to put runs first, then baselines (using the existing sorting key)
                sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: 0 if x[1].startswith('Run') else 1)

                ax.legend(*zip(*sorted_handles_labels), loc='best', ncol=4, fontsize='small')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.show()
                plt.close()



    @staticmethod
    def plot_joint_angle_symmetry_baseline_vs_prosthesis(
        joint_names_list,
        all_loaded_data,
        run_step_data,
        baseline_file,
        prosthesis_side="right",   # can be "right" or "left"
        parameter_name="angle",
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=100,
        body_weight_to_normalize=1,
        mass_baseline = 1,
    ):
        """
        Plot mean of prosthesis side across all runs against baseline data.
        Only two curves: Baseline and SACH (prosthesis).
        """

        import re, itertools
        import numpy as np
        import matplotlib.pyplot as plt

        joint_pairs = [(name + "_l", name + "_r") for name in joint_names_list]

        # Load baseline data
        baseline_data = {}
        current_label = None
        with open(baseline_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                label_match = re.match(r'^([A-Za-z_]+)\s*=\s*\[?', line)
                if label_match:
                    current_label = label_match.group(1)
                    baseline_data[current_label] = []
                    continue
                if current_label:
                    vals = line.split(';')
                    if len(vals) == 2:
                        try:
                            x, y = map(float, vals)
                            baseline_data[current_label].append((x, y))
                        except ValueError:
                            continue

        for left_joint, right_joint in joint_pairs:
            prosthesis_joint = right_joint if prosthesis_side.lower() == "right" else left_joint

            run_keys = list(all_loaded_data.keys())
            mean_prosth_all = []
            std_prosth_all = []

            # Collect prosthesis side data
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                evaluation_joint_names = run_dict.get("evaluation_joint_names", [])
                joint_data = {}
                for joint_name in evaluation_joint_names:
                    joint_data[joint_name] = {}
                    for key in ["angle", "velocity", "forces_constraint",
                                "forces_smooth", "forces_applied", "torques", "energy_exp"]:
                        param = run_dict.get(f"{joint_name}_{key}")
                        if param is not None:
                            joint_data[joint_name][key] = param

                step_indices = run_step_data[run_key].get(
                    "step_start_" + ("right" if prosthesis_side.lower() == "right" else "left"), []
                )
                side_data = joint_data.get(prosthesis_joint, {}).get(parameter_name, None)
                if side_data is None:
                    continue

                def collect_steps(data, indices):
                    steps = []
                    if indices and len(indices) > 1:
                        for j in range(len(indices) - 1):
                            start, end = indices[j], indices[j + 1]
                            y = [float(np.array(a).squeeze()) for a in data[start:end]]
                            if len(y) < 2:
                                continue
                            if interp_mode == "interp":
                                x_old = np.linspace(0, 1, len(y))
                                x_new = np.linspace(0, 1, interp_len)
                                y = np.interp(x_new, x_old, y)
                            if convert_to_deg and parameter_name in ("angle", "velocity"):
                                y = np.rad2deg(y)
                            steps.append(y)
                    return steps

                side_steps = collect_steps(side_data, step_indices)

                if side_steps:
                    mean_prosth = np.mean(side_steps, axis=0) / body_weight_to_normalize
                    std_prosth = np.std(side_steps, axis=0) / body_weight_to_normalize
                    # Flip sign for knees if needed
                    if 'knee' in prosthesis_joint:
                        mean_prosth = -mean_prosth
                        std_prosth = -std_prosth
                    mean_prosth_all.append(mean_prosth)
                    std_prosth_all.append(std_prosth)

            # Plot baseline vs prosthesis
            if mean_prosth_all:
                x = np.linspace(0, 100, interp_len)
                mean_prosth = np.mean(mean_prosth_all, axis=0)
                std_prosth = np.mean(std_prosth_all, axis=0)

                plt.figure(figsize=(8, 5))
                # Prosthesis side (SACH)
                plt.plot(x, mean_prosth, label="SACH Sim", color="tab:red")
                plt.fill_between(x, mean_prosth - std_prosth, mean_prosth + std_prosth,
                                alpha=0.15, color="tab:red")

                # Baseline
                for label, points in baseline_data.items():
                    if label.lower() in prosthesis_joint.lower() and points:
                        xs, ys = zip(*points)
                        if 'torques' in parameter_name:
                            if 'Banks' in baseline_file:
                                ys = np.array(ys)/mass_baseline
                                if 'knee' in prosthesis_joint:
                                    ys = np.array(ys) * -1
                            # elif not 'knee' in prosthesis_joint and 'Turcot' in baseline_file: # CHANGED TO NOT HAVE KNEE
                            #     pass
                            else:
                                ys = np.array(ys) * -1
                        if 'angle' in parameter_name:
                            if 'knee' in prosthesis_joint and 'Banks' in baseline_file: # CHANGED TO NOT HAVE KNEE
                                ys = np.array(ys) * -1
                        plt.plot(xs, ys, color="black", linestyle="-", marker="o", label="SACH Exp")

                plt.title(f"{prosthesis_joint}", fontsize=18)
                plt.xlabel("Gait Cycle (%)", fontsize=18)
                if 'angle' in parameter_name: 
                    if convert_to_deg:
                        plt.ylabel(f"{parameter_name} in °", fontsize=20)
                    else:
                        plt.ylabel(f"{parameter_name} in rad", fontsize=18)
                elif 'torques' in parameter_name:
                    plt.ylabel(f"{parameter_name} in Nm/kg", fontsize=18)
            
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.legend(fontsize=16)
                plt.tight_layout()
                plt.show()
                plt.close()




    



    @staticmethod 
    def plot_joint_angle_single_side_all_runs(
        joint_names_list,
        all_loaded_data,
        run_step_data,
        parameter_name="angle",
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=100,
        plot_baseline=False, 
        baseline_data=None,
        side="left"  # could be "left" or "right"
        ):
        """
        Plot mean joint data for the specified side over all runs.
        Only includes runs where data for the joint is available.
        """
        side_suffix = "_l" if side == "left" else "_r"
        joint_names = [name + side_suffix for name in joint_names_list]
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_iter = itertools.cycle(color_cycle)

        for joint in joint_names:
            run_keys = list(all_loaded_data.keys())
            mean_all = []
            std_all = []

            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                joint_data = {}
                evaluation_joint_names = run_dict.get("evaluation_joint_names", [])
                joint_parameters = ["angle", "velocity", "forces_constraint", "forces_smooth","forces_applied", "torques", "energy_exp"]

                for joint_name in evaluation_joint_names:
                    joint_data[joint_name] = {}
                    for key in joint_parameters:
                        param = run_dict.get(f"{joint_name}_{key}")
                        if param is not None:
                            joint_data[joint_name][key] = param

                step_indices = run_step_data[run_key].get(f"step_start_{side}", [])
                data = joint_data.get(joint, {}).get(parameter_name, None)
                if data is None:
                    continue

                def collect_steps(data, indices):
                    steps = []
                    if indices and len(indices) > 1:
                        for j in range(len(indices) - 1):
                            start, end = indices[j], indices[j + 1]
                            y = [float(np.array(a).squeeze()) for a in data[start:end]]
                            if len(y) < 2:
                                continue
                            if interp_mode == "interp":
                                x_old = np.linspace(0, 1, len(y))
                                x_new = np.linspace(0, 1, interp_len)
                                y = np.interp(x_new, x_old, y)
                            if convert_to_deg and parameter_name in ("angle", "velocity"):
                                features = ['angle', 'flexion', 'adduction', 'rotation']
                                if any(feature in joint for feature in features):
                                    y = np.rad2deg(y)
                            steps.append(y)
                    return steps

                steps = collect_steps(data, step_indices)

                if steps:
                    mean_val = np.mean(steps, axis=0)
                    std_val = np.std(steps, axis=0)
                    mean_all.append((run_key, mean_val, std_val))

            x = np.linspace(0, 100, interp_len)
            if mean_all:
                plt.figure(figsize=(8, 5))
                for run_key, mean_val, std_val in mean_all:
                    plt.plot(x, mean_val, label=f"{run_key}")
                    plt.fill_between(x, mean_val - std_val, mean_val + std_val, alpha=0.15)
                if plot_baseline and parameter_name in ["angle", "velocity"] and baseline_data is not None:
                    if joint in baseline_data and parameter_name in baseline_data[joint]:
                        plt.plot(x, baseline_data[joint][parameter_name], label=f"Baseline {joint}", color='black')
                plt.title(f"{joint} mean (all runs)")
                plt.xlabel("Interpolated Step (%)")
                plt.ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
                plt.legend()
                plt.tight_layout()
                plt.show()
                plt.close()



    @staticmethod
    def plot_joint_angle_summed_sides_all_runs(
        joint_names_list,
        all_loaded_data,
        run_step_data,
        parameter_name="angle",
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=100,
        baseline_data=None
    ):
        """
        For each joint in joint_names_list, plot mean ± std curves for LEFT, RIGHT, and SYMMETRY (R-L),
        aggregated across runs. One plot per joint. Optionally plot baseline data.
        """

        def get_side_data(base_name, side_suffix, run_key, run_dict):
            evaluation_joint_names = run_dict.get("evaluation_joint_names", [])
            joint_parameters = ["angle", "velocity", "forces_constraint", "forces_smooth", "torques", "energy_exp"]

            joint_data = {}
            for joint_name in evaluation_joint_names:
                joint_data[joint_name] = {}
                for key in joint_parameters:
                    param = run_dict.get(f"{joint_name}_{key}")
                    if param is not None:
                        joint_data[joint_name][key] = param

            step_indices = run_step_data[run_key].get(f"step_start_{'left' if side_suffix == '_l' else 'right'}", [])
            joint = base_name + side_suffix
            data = joint_data.get(joint, {}).get(parameter_name, None)
            if data is None:
                return None

            steps = []
            if step_indices and len(step_indices) > 1:
                for j in range(len(step_indices) - 1):
                    start, end = step_indices[j], step_indices[j + 1]
                    y = [float(np.array(a).squeeze()) for a in data[start:end]]
                    if len(y) < 2:
                        continue
                    if interp_mode == "interp":
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y = np.interp(x_new, x_old, y)
                    if convert_to_deg and parameter_name in ("angle", "velocity"):
                        features = ['angle', 'flexion', 'adduction', 'rotation']
                        if any(feature in joint for feature in features):
                            y = np.rad2deg(y)
                    steps.append(y)
            if steps:
                return np.mean(steps, axis=0)  # average over steps
            return None

        run_keys = list(all_loaded_data.keys())

        for base_name in joint_names_list:
            left_runs, right_runs, sym_runs = [], [], []

            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                left_curve = get_side_data(base_name, "_l", run_key, run_dict)
                right_curve = get_side_data(base_name, "_r", run_key, run_dict)
                if left_curve is not None and right_curve is not None:
                    left_runs.append(left_curve)
                    right_runs.append(right_curve)
                    sym_runs.append( left_curve- right_curve)

            def aggregate(curves):
                curves = np.array(curves)
                mean = np.mean(curves, axis=0)
                std = np.std(curves, axis=0)
                return mean, std

            x = np.linspace(0, 100, interp_len)
            plt.figure(figsize=(8, 5))

            if left_runs:
                mean_left, std_left = aggregate(left_runs)
                plt.plot(x, mean_left, label="Left (mean)", color="blue")
                plt.fill_between(x, mean_left - std_left, mean_left + std_left, color="blue", alpha=0.2)

            if right_runs:
                mean_right, std_right = aggregate(right_runs)
                plt.plot(x, mean_right, label="Right (mean)", color="red")
                plt.fill_between(x, mean_right - std_right, mean_right + std_right, color="red", alpha=0.2)

            if sym_runs:
                mean_sym, std_sym = aggregate(sym_runs)
                plt.plot(x, mean_sym, label="Symmetry (R-L)", color="green")
                plt.fill_between(x, mean_sym - std_sym, mean_sym + std_sym, color="green", alpha=0.2)

            # baseline
            if baseline_data is not None:
                for side_suffix, color, label_suffix in [("_l", "blue", "Left"), ("_r", "red", "Right")]:
                    joint_baseline = base_name + side_suffix
                    if joint_baseline in baseline_data and parameter_name in baseline_data[joint_baseline]:
                        y_baseline = baseline_data[joint_baseline][parameter_name]
                        # if convert_to_deg and parameter_name in ("angle", "velocity"):
                        #     y_baseline = np.rad2deg(y_baseline)
                        plt.plot(x, y_baseline, label=f"Baseline {label_suffix}", color="black", linestyle="--")

            plt.title(f"{base_name}: {parameter_name} (summed over runs)")
            plt.xlabel("Interpolated Step (%)")
            plt.ylabel(f"{parameter_name} ({'deg' if convert_to_deg and parameter_name in ('angle', 'velocity') else 'rad'})")
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.close()




    @staticmethod
    def plot_joint_angle_summed_sides_all_dirs(
        joint_names_list,
        all_loaded_data,
        run_step_data,
        parameter_name="angle",
        convert_to_deg=False,
        interp_mode="interp",
        interp_len=100,
        baseline_data=None
    ):
        """
        For each joint in joint_names_list, plot mean ± std curves for LEFT, RIGHT, and SYMMETRY (L-R),
        aggregated across runs, comparing multiple directories in one plot.
        """

        def get_side_data(base_name, side_suffix, run_dict, step_data):
            evaluation_joint_names = run_dict.get("evaluation_joint_names", [])
            joint_parameters = ["angle", "velocity", "forces_constraint", "forces_smooth", "torques", "energy_exp"]

            joint_data = {}
            for joint_name in evaluation_joint_names:
                joint_data[joint_name] = {}
                for key in joint_parameters:
                    param = run_dict.get(f"{joint_name}_{key}")
                    if param is not None:
                        joint_data[joint_name][key] = param

            step_indices = step_data.get(f"step_start_{'left' if side_suffix == '_l' else 'right'}", [])
            joint = base_name + side_suffix
            data = joint_data.get(joint, {}).get(parameter_name, None)
            if data is None:
                return None

            steps = []
            if step_indices and len(step_indices) > 1:
                for j in range(len(step_indices) - 1):
                    start, end = step_indices[j], step_indices[j + 1]
                    y = [float(np.array(a).squeeze()) for a in data[start:end]]
                    if len(y) < 2:
                        continue
                    if interp_mode == "interp":
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y = np.interp(x_new, x_old, y)
                    if convert_to_deg and parameter_name in ("angle", "velocity"):
                        features = ['angle', 'flexion', 'adduction', 'rotation']
                        if any(feature in joint for feature in features):
                            y = np.rad2deg(y)
                    steps.append(y)
            if steps:
                return np.mean(steps, axis=0)
            return None
        colors = plt.cm.tab10.colors
        dir_labels = list(all_loaded_data.keys())
        # colors = [
        #             "blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan",
        #             "magenta", "gold", "teal", "navy", "maroon", "lime", "indigo", "coral", "turquoise", "darkgreen"
        #         ]

        for base_name in joint_names_list:
            plt.figure(figsize=(20, 5))
            x = np.linspace(0, 100, interp_len)

            for dir_idx, dir_label in enumerate(dir_labels):
                left_runs, right_runs, sym_runs = [], [], []
                runs = all_loaded_data[dir_label]

                for run_key, run_dict in runs.items():
                    step_data = run_step_data[dir_label][run_key]
                    left_curve = get_side_data(base_name, "_l", run_dict, step_data)
                    right_curve = get_side_data(base_name, "_r", run_dict, step_data)

                    if left_curve is not None and right_curve is not None:
                        left_runs.append(left_curve)
                        right_runs.append(right_curve)
                        sym_runs.append(left_curve - right_curve)

                if not left_runs or not right_runs:
                    print(f"Skipping {base_name} in {dir_label}, no data found.")
                    continue

                # Aggregate per directory
                def aggregate(curves):
                    curves = np.array(curves)
                    return np.mean(curves, axis=0), np.std(curves, axis=0)

                mean_left, std_left = aggregate(left_runs)
                mean_right, std_right = aggregate(right_runs)
                mean_sym, std_sym = aggregate(sym_runs)

                color = colors[dir_idx % len(colors)]
                plt.plot(x, mean_left, label=f"{dir_label} Left", color=color, linestyle='-')
                plt.fill_between(x, mean_left - std_left, mean_left + std_left, color=color, alpha=0.2)
                plt.plot(x, mean_right, label=f"{dir_label} Right", color=color, linestyle='--')
                plt.fill_between(x, mean_right - std_right, mean_right + std_right, color=color, alpha=0.2)
                plt.plot(x, mean_sym, label=f"{dir_label} Sym (L-R)", color=color, linestyle=':')
                plt.fill_between(x, mean_sym - std_sym, mean_sym + std_sym, color=color, alpha=0.1)

            # Baseline if provided
            if baseline_data:
                for side_suffix, label_suffix, linestyle in [("_l", "Left", "--"), ("_r", "Right", "--")]:
                    joint_baseline = base_name + side_suffix
                    if joint_baseline in baseline_data and parameter_name in baseline_data[joint_baseline]:
                        y_baseline = baseline_data[joint_baseline][parameter_name]
                        plt.plot(x, y_baseline, label=f"Baseline {label_suffix}", color="black", linestyle=linestyle)

            plt.title(f"{base_name}: {parameter_name} (summed over runs, all directories)")
            plt.xlabel("Interpolated Step (%)")
            plt.ylabel(f"{parameter_name} ({'deg' if convert_to_deg else 'rad'})")
            plt.legend(ncol=3, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            plt.show()
            plt.close()


    
    @staticmethod
    def plot_joint_angle_summed_sides_all_dirs_with_baseline(
        joint_names_list,
        all_loaded_data,
        run_step_data,
        prosthesis_side="right",
        parameter_name="angle",
        convert_to_deg=True,
        interp_mode="interp",
        interp_len=100,
        baseline_data_1=None,
        baseline_data_2=None,
        baseline_data_healthy=None,  # Added as requested
    ):
        """
        Plots two subplots: Healthy Side and Impaired Side.
        Distributes reference data from Banks, Turcot, and baseline_data_healthy correctly.
        """

        def get_side_data(base_name, side_suffix, run_dict, step_data):
            evaluation_joint_names = run_dict.get("evaluation_joint_names", [])
            joint_parameters = ["angle", "velocity", "forces_constraint", "forces_smooth", "torques", "energy_exp"]
            
            joint_data = {}
            for joint_name in evaluation_joint_names:
                joint_data[joint_name] = {}
                for key in joint_parameters:
                    param = run_dict.get(f"{joint_name}_{key}")
                    if param is not None:
                        joint_data[joint_name][key] = param

            side_key = 'left' if side_suffix == '_l' else 'right'
            step_indices = step_data.get(f"step_start_{side_key}", [])
            joint = base_name + side_suffix
            data = joint_data.get(joint, {}).get(parameter_name, None)
            
            if data is None:
                return None

            steps = []
            if step_indices and len(step_indices) > 1:
                for j in range(len(step_indices) - 1):
                    start, end = step_indices[j], step_indices[j + 1]
                    y = [float(np.array(a).squeeze()) for a in data[start:end]]
                    if len(y) < 2: continue
                    if interp_mode == "interp":
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y = np.interp(x_new, x_old, y)
                    if convert_to_deg and parameter_name in ("angle", "velocity"):
                        if any(feat in joint for feat in ['angle', 'flexion', 'adduction', 'rotation']):
                            y = np.rad2deg(y)
                    steps.append(y)
            return np.mean(steps, axis=0) if steps else None

        def parse_baseline(file_path):
            if not file_path: return {}
            data_dict = {}
            current_label = None
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        regex = r'^([A-Za-z_]+)\s*=\s*\[?'
                        if 'Schmalz' in str(file_path): regex = r'^([A-Z_]+)\s*=\s*\[?'
                        label_match = re.match(regex, line)
                        if label_match:
                            current_label = label_match.group(1)
                            data_dict[current_label] = []
                            continue
                        if current_label:
                            vals = line.split(';')
                            if len(vals) == 2:
                                try:
                                    data_dict[current_label].append((float(vals[0]), float(vals[1])))
                                except ValueError: continue
            except FileNotFoundError:
                print(f"Warning: Baseline file {file_path} not found.")
            return data_dict

        # 1. Parse all baseline sources
        b1_dict = parse_baseline(baseline_data_1)
        b2_dict = parse_baseline(baseline_data_2)
        # bh_dict = parse_baseline(baseline_data_healthy)
        # x = np.linspace(0, 100, interp_len)
        
        colors = plt.cm.tab10.colors
        dir_labels = list(all_loaded_data.keys())

        for base_name in joint_names_list:
            imp_suffix = "_l" if prosthesis_side.lower() == "left" else "_r"
            hea_suffix = "_r" if prosthesis_side.lower() == "left" else "_l"

            fig, (ax_h, ax_i) = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
            x_axis = np.linspace(0, 100, interp_len)

            # 2. Plot Simulation Data
            for dir_idx, dir_label in enumerate(dir_labels):
                healthy_runs, impaired_runs = [], []
                runs = all_loaded_data[dir_label]
                color = colors[dir_idx % len(colors)]

                for run_key, run_dict in runs.items():
                    step_data = run_step_data[dir_label][run_key]
                    h_curve = get_side_data(base_name, hea_suffix, run_dict, step_data)
                    i_curve = get_side_data(base_name, imp_suffix, run_dict, step_data)
                    if h_curve is not None: healthy_runs.append(h_curve)
                    if i_curve is not None: impaired_runs.append(i_curve)

                def plot_agg(ax, data_list, label_side):
                    if not data_list: return
                    arr = np.array(data_list)
                    m, s = np.mean(arr, axis=0), np.std(arr, axis=0)
                    if 'knee' in base_name:
                        m = -m
                    ax.plot(x_axis, m, label=f"{dir_label} {label_side}", color=color, linewidth=2)
                    ax.fill_between(x_axis, m - s, m + s, color=color, alpha=0.15)

                plot_agg(ax_h, healthy_runs, "Sim Healthy")
                plot_agg(ax_i, impaired_runs, "Sim Impaired")

            # 3. Baseline Plotting Logic
            def plot_ref(ax, b_dict, file_path, target_side):
                """
                target_side: "healthy" or "impaired"
                """
                if not b_dict: return
                for label, points in b_dict.items():
                    if not points: continue
                    # Basic joint matching
                    label_joint = label.split('_')[0].lower()
                    if label_joint not in base_name.lower(): continue
                    
                    # Determine if this specific data point is 'intact'
                    is_intact = 'intact' in label.lower() or 'healthy' in str(file_path).lower()
                    
                    # Logic: plot healthy-labeled data on healthy ax, everything else on impaired ax
                    if target_side == "healthy" and not is_intact: continue
                    if target_side == "impaired" and is_intact: continue

                    xs, ys = zip(*points)
                    ys = np.array(ys)
                    
                    # Apply data transformations from your original code
                    if 'torques' in parameter_name: ys *= -1
                    # if file_path and 'Banks' in str(file_path):
                    #     ys *= -1
                    #     if 'knee' not in base_name: ys *= -1
                    if 'angle' in parameter_name and file_path and 'Banks' in str(file_path) and 'knee' in base_name:
                        ys *= -1

                    # Styling
                    src_name = 'Turcot' if 'Turcot' in str(file_path) else 'Banks' if 'Banks' in str(file_path) else 'Baseline'
                    if file_path == baseline_data_healthy: src_name = "Healthy Ref"
                    
                    fmt = 'k--' if 'Turcot' in src_name else 'k-.' if 'Banks' in src_name else 'k:'
                    ax.plot(xs, ys, fmt, alpha=0.8, label=f"Ref: {src_name} ({label})")

            if parameter_name in ["angle", "velocity"] and baseline_data_healthy is not None:
                if base_name+'_r' in baseline_data_healthy and parameter_name in baseline_data_healthy[base_name+'_r']:#if np.any(baseline_data[right_joint][parameter_name]):
                    if 'knee' in base_name+'_r': # Switch sign (-1) for knee sensors to match flexion/extension convention
                        ax_h.plot(x_axis, -baseline_data_healthy[base_name+'_r'][parameter_name],  color='black', label=f"Baseline Healthy")
                        ax_i.plot(x_axis, -baseline_data_healthy[base_name+'_r'][parameter_name],  color='black', label=f"Baseline Healthy")
                        # print('Minimum Healthy Knee Right:', np.min(-baseline_data_healthy[base_name+hea_suffix][parameter_name]))
                        # print('Maximum Healthy Knee Right:', np.max(-baseline_data_healthy[base_name+hea_suffix][parameter_name]))
                    else:
                        ax_h.plot(x_axis, baseline_data_healthy[base_name+'_r'][parameter_name], color='black', label=f"Baseline Healthy")
                        ax_i.plot(x_axis, baseline_data_healthy[base_name+'_r'][parameter_name], color='black', label=f"Baseline Healthy")

            # Apply baselines to subplots
            for ax, side in [(ax_h, "healthy"), (ax_i, "impaired")]:
                # plot_ref(ax, bh_dict, baseline_data_healthy, side)
                # plot_ref(ax, bh_dict, -baseline_data_healthy[base_name + hea_suffix][parameter_name], side)
                plot_ref(ax, b1_dict, baseline_data_1, side)
                plot_ref(ax, b2_dict, baseline_data_2, side)
                ax.set_xlabel("Gait Cycle (%)")
                ax.grid(True, linestyle=':', alpha=0.6)
                # Add legend to BOTH axes so all reference lines are visible
                ax.legend(fontsize='small', loc='best')



            ax_h.set_title(f"HEALTHY SIDE ({hea_suffix.upper()})")
            ax_i.set_title(f"PROSTHETIC/IMPAIRED SIDE ({imp_suffix.upper()})")
            ax_h.set_ylabel(f"{parameter_name} ({'deg' if convert_to_deg else 'rad'})")
            
            plt.suptitle(f"Comparison: {base_name} {parameter_name}", fontsize=14, y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()



    #@staticmethod 
    # def plot_sensor_force_symmetry_all_runs(all_loaded_data, run_step_data, interp_len=100, sensor_force_names=None):
    #     """
    #     Plot mean, std, and left-right difference for each left/right sensor force pair across all runs.
    #     """
    #     # Use sensor_force_names from the first run if not provided
    #     if sensor_force_names is None:
    #         first_run = next(iter(all_loaded_data))
    #         sensor_force_names = all_loaded_data[first_run]["sensor_force_names"]

    #     # Find sensor pairs (left/right)
    #     sensor_pairs = []
    #     for name in sensor_force_names:
    #         if name.startswith("left_"):
    #             right_name = name.replace("left_", "right_")
    #             if right_name in sensor_force_names:
    #                 sensor_pairs.append((name, right_name))

    #     for left_sensor, right_sensor in sensor_pairs:
    #         plt.figure(figsize=(12, 6))
    #         for run_key in all_loaded_data:
    #             run_dict = all_loaded_data[run_key]
    #             step_data = run_step_data[run_key]
    #             left_data = run_dict["all_sensor_force"][left_sensor]
    #             right_data = run_dict["all_sensor_force"][right_sensor]
    #             left_steps = []
    #             right_steps = []
    #             all_step_start_left = step_data["step_start_left"]
    #             all_step_start_right = step_data["step_start_right"]

    #             # Interpolate left steps
    #             if all_step_start_left and len(all_step_start_left) > 1:
    #                 for j in range(len(all_step_start_left) - 1):
    #                     start, end = all_step_start_left[j], all_step_start_left[j + 1]
    #                     y = [float(np.array(a)[1]) for a in left_data[start:end]]
    #                     if len(y) < 2:
    #                         continue
    #                     x_old = np.linspace(0, 1, len(y))
    #                     x_new = np.linspace(0, 1, interp_len)
    #                     y_interp = np.interp(x_new, x_old, y)
    #                     left_steps.append(y_interp)
    #             # Interpolate right steps
    #             if all_step_start_right and len(all_step_start_right) > 1:
    #                 for j in range(len(all_step_start_right) - 1):
    #                     start, end = all_step_start_right[j], all_step_start_right[j + 1]
    #                     y = [float(np.array(a)[1]) for a in right_data[start:end]]
    #                     if len(y) < 2:
    #                         continue
    #                     x_old = np.linspace(0, 1, len(y))
    #                     x_new = np.linspace(0, 1, interp_len)
    #                     y_interp = np.interp(x_new, x_old, y)
    #                     right_steps.append(y_interp)
    #             # Plot if enough steps
    #             if left_steps and right_steps:
    #                 mean_left = np.mean(left_steps, axis=0)
    #                 std_left = np.std(left_steps, axis=0)
    #                 mean_right = np.mean(right_steps, axis=0)
    #                 std_right = np.std(right_steps, axis=0)
    #                 diff = mean_left - mean_right
    #                 x = np.linspace(0, 100, interp_len)
    #                 plt.plot(x, mean_left, label=f"{run_key} {left_sensor} mean")
    #                 plt.fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
    #                 plt.plot(x, mean_right, label=f"{run_key} {right_sensor} mean", linestyle='--')
    #                 plt.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
    #                 plt.plot(x, diff, label=f"{run_key} Left-Right", linestyle=':')
    #         plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    #         plt.title(f"Force Sensor (y) per time step: {left_sensor} vs {right_sensor} (all runs)")
    #         plt.xlabel("Interpolated Step (%)")
    #         plt.ylabel("Force Value")
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()


    @staticmethod
    def plot_sensor_force_symmetry_all_runs(
        all_loaded_data, run_step_data, direction, interp_len=100, body_weight_to_normalize=1, sensor_force_names=None
    ):
        """
        Plot mean of right of all runs, mean of left, and the difference, side by side in one figure.
        Each plot includes all runs.
        direction: "x", "y", or "z" to specify the force direction. [0, 1, 2] for x, y, z respectively.
        """
        if direction == "x":
            force_direction = 0
        elif direction == "y":
            force_direction = 1
        elif direction == "z":
            force_direction = 2

        # direction_map = {"x": 0, "y": 1, "z": 2}
        # force_direction = direction_map.get(direction, 0)
        # Use sensor_force_names from the first run if not provided
        if sensor_force_names is None:
            first_run = next(iter(all_loaded_data))
            sensor_force_names = all_loaded_data[first_run]["sensor_force_names"]

        # Find sensor pairs (left/right)
        sensor_pairs = []
        for name in sensor_force_names:
            if name.startswith("left_"):
                right_name = name.replace("left_", "right_")
                if right_name in sensor_force_names:
                    sensor_pairs.append((name, right_name))

        for left_sensor, right_sensor in sensor_pairs:
            # Prepare data for all runs
            run_keys = list(all_loaded_data.keys())
            mean_left_all = []
            std_left_all = []
            mean_right_all = []
            std_right_all = []
            diff_all = []
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                step_data = run_step_data[run_key]
                left_data = run_dict["all_sensor_force"][left_sensor]
                right_data = run_dict["all_sensor_force"][right_sensor]
                all_step_start_left = step_data["step_start_left"]
                all_step_start_right = step_data["step_start_right"]

                # Interpolate left steps
                left_steps = []
                if all_step_start_left and len(all_step_start_left) > 1:
                    for j in range(len(all_step_start_left) - 1):
                        start, end = all_step_start_left[j]-1, all_step_start_left[j + 1]-1
                        y = [float(np.array(a)[force_direction]) for a in left_data[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        left_steps.append(y_interp)
                # Interpolate right steps
                right_steps = []
                if all_step_start_right and len(all_step_start_right) > 1:
                    for j in range(len(all_step_start_right) - 1):
                        start, end = all_step_start_right[j]-1, all_step_start_right[j + 1]-1
                        # Plot y axis in site coordinate system [1]
                        y = [float(np.array(a)[force_direction]) for a in right_data[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        right_steps.append(y_interp)
                # Compute means and stds
                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0)/body_weight_to_normalize
                    std_left = np.std(left_steps, axis=0)/body_weight_to_normalize
                    mean_right = np.mean(right_steps, axis=0)/body_weight_to_normalize
                    std_right = np.std(right_steps, axis=0)/body_weight_to_normalize
                    if 'knee' in left_sensor: # Switch sign (-1) for knee sensors to match flexion/extension convention
                        mean_left = -mean_left
                        mean_right = -mean_right
                        std_left = -std_left
                        std_right = -std_right
                    diff = mean_left - mean_right
                    mean_left_all.append((run_key, mean_left, std_left))
                    mean_right_all.append((run_key, mean_right, std_right))
                    diff_all.append((run_key, diff))
                else:
                    mean_left_all.append((run_key, None, None))
                    mean_right_all.append((run_key, None, None))
                    diff_all.append((run_key, None))

            x = np.linspace(0, 100, interp_len)
            # Plot all three plots side by side
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
            # Plot mean of right of all runs
            for run_key, mean_right, std_right in mean_right_all:
                if mean_right is not None:
                    axes[0].plot(x, mean_right, label=f"{run_key} {right_sensor} mean")
                    axes[0].fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
            axes[0].set_title(f"{right_sensor} mean (all runs)")
            axes[0].set_xlabel("Interpolated Step (%)")
            axes[0].set_ylabel("Force Value")
            # axes[0].legend()

            # Plot mean of left of all runs
            for run_key, mean_left, std_left in mean_left_all:
                if mean_left is not None:
                    axes[1].plot(x, mean_left, label=f"{run_key} {left_sensor} mean")
                    axes[1].fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
            axes[1].set_title(f"{left_sensor} mean (all runs)")
            axes[1].set_xlabel("Interpolated Step (%)")
            axes[1].set_ylabel("Force Value")
            # axes[1].legend()

            # Plot difference (left - right) of all runs
            for run_key, diff in diff_all:
                if diff is not None:
                    axes[2].plot(x, diff, label=f"{run_key} Left-Right")
            axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
            axes[2].set_title(f"{left_sensor} - {right_sensor} difference (all runs)")
            axes[2].set_xlabel("Interpolated Step (%)")
            axes[2].set_ylabel("Force Value Difference")
            axes[2].legend()

            plt.tight_layout()
            plt.show()
            plt.close()


    

    @staticmethod
    def plot_sensor_force_pylon_all_runs(
        all_loaded_data, run_step_data, direction, interp_len=100, body_weight_to_normalize=1, sensor_force_names=None
    ):
        """
        Plot mean of all runs for pylon sensor data (prosthesis side only).
        Only sensors with 'pylon' in their name are plotted.
        direction: "x", "y", or "z" to specify the force direction. [0, 1, 2] for x, y, z respectively.
        """
        force_direction = {"x": 0, "y": 1, "z": 2}.get(direction, 0)

        # Use sensor_force_names from the first run if not provided
        if sensor_force_names is None:
            # first_run = next(iter(all_loaded_data))
            run_keys = list(all_loaded_data.keys())
            if len(run_keys) > 1:
                run = run_keys[1]
            else:
                run = run_keys[0]
            sensor_force_names = all_loaded_data[run]["sensor_force_names"]
            # sensor_force_names = all_loaded_data[run]["sensor_force_names"]

        # Only keep sensors with 'pylon' in their name
        pylon_sensors = [name for name in sensor_force_names if "pylon" in name or "socket" in name]

        for sensor_name in pylon_sensors:
            run_keys = list(all_loaded_data.keys())
            mean_all = []
            std_all = []
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                step_data = run_step_data[run_key]
                if sensor_name in run_dict.get("all_sensor_force", {}):
                    sensor_data = run_dict["all_sensor_force"][sensor_name]
                else:
                    continue  # Skip this run if sensor data is missing
                all_step_start = step_data["step_start_right"] if "_r" in sensor_name else step_data["step_start_left"]

                steps = []
                if all_step_start and len(all_step_start) > 1:
                    for j in range(len(all_step_start) - 1):
                        start, end = all_step_start[j]-1, all_step_start[j + 1]-1
                        y = [float(np.array(a)[force_direction]) for a in sensor_data[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        steps.append(y_interp)
                if steps:
                    mean_val = np.mean(steps, axis=0) / body_weight_to_normalize
                    std_val = np.std(steps, axis=0) / body_weight_to_normalize
                    if force_direction == 2:
                        mean_val*= -1
                        std_val *= -1
                    mean_all.append((run_key, mean_val, std_val))

            x = np.linspace(0, 100, interp_len)
            if mean_all:
                plt.figure(figsize=(8, 5))
                for run_key, mean_val, std_val in mean_all:
                    plt.plot(x, mean_val, label=f"{run_key}")
                    plt.fill_between(x, mean_val - std_val, mean_val + std_val, alpha=0.15)
                plt.title(f"{sensor_name} mean (all runs)")
                plt.xlabel("Interpolated Step (%)")
                plt.ylabel("Force Value")
                plt.legend()
                plt.tight_layout()
                plt.show()
                plt.close()


    @staticmethod
    def plot_sensor_force_pylon_all_runs_prosthesis_side(
        all_loaded_data, run_step_data, direction, interp_len=100, body_weight_to_normalize=1, sensor_force_names=[], prosthesis_side=["left"]
    ):
        """
        Plot mean of all runs for pylon sensor data (prosthesis side only).
        Only sensors with 'pylon' in their name are plotted.
        direction: "x", "y", or "z" to specify the force direction. [0, 1, 2] for x, y, z respectively.
        """
        force_direction = {"x": 0, "y": 1, "z": 2}.get(direction, 0)

        run_keys = list(all_loaded_data.keys())

        # Use sensor_force_names from the first run if not provided
        if not sensor_force_names:
            for run_key in run_keys:
                sensor_force_names.extend(all_loaded_data[run_key]["sensor_force_names"])

        # Only keep sensors with 'pylon' in their name
        pylon_sensors = [name for name in sensor_force_names if "torque" in name and ("pylon" in name or "socket" in name)]

        if len(prosthesis_side) == 1:
            prosthesis_side = prosthesis_side * len(run_keys)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for sensor_name in pylon_sensors:
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                step_data = run_step_data[run_key]
                if sensor_name in run_dict.get("all_sensor_force", {}):
                    sensor_data = run_dict["all_sensor_force"][sensor_name]
                else:
                    continue  # Skip this run if sensor data is missing
                all_step_start = step_data["step_start_right"] if "_r" in sensor_name else step_data["step_start_left"]

                steps = []
                if all_step_start and len(all_step_start) > 1:
                    for j in range(len(all_step_start) - 1):
                        start, end = all_step_start[j] - 1, all_step_start[j + 1] - 1
                        y = [float(np.array(a)[force_direction]) for a in sensor_data[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        steps.append(y_interp)
                
                if steps:
                    mean_val = np.mean(steps, axis=0) / body_weight_to_normalize
                    std_val = np.std(steps, axis=0) / body_weight_to_normalize
                    if force_direction == 2:
                        mean_val *= -1
                        std_val *= -1
                    
                    x = np.linspace(0, 100, interp_len)
                    ax.plot(x, mean_val, label=f"{run_key} - {sensor_name}", linewidth=2)
                    ax.fill_between(x, mean_val - std_val, mean_val + std_val, alpha=0.15)
        
        ax.set_title("Pylon/Socket Sensors - All Runs and Sensors")
        ax.set_xlabel("Interpolated Step (%)")
        ax.set_ylabel("Force Value")
        ax.legend(loc='best', fontsize='small', ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.close(fig)


    @staticmethod
    def plot_sensor_force_pylon_all_runs_stance_sorted(
        all_loaded_data, run_step_data, direction, left_switches, right_switches, interp_len=100, body_weight_to_normalize=1, sensor_force_names=None
    ):
        """
        Plot mean of all runs for pylon sensor data (prosthesis side only).
        Only sensors with 'pylon' in their name are plotted.
        Interpolate each step to 100%, then take the part until left_switches or right_switches (depending on '_l' or '_r'),
        then interpolate this part again to 100%.
        direction: "x", "y", or "z" to specify the force direction. [0, 1, 2] for x, y, z respectively.
        """
        force_direction = {"x": 0, "y": 1, "z": 2}.get(direction, 0)

        # Use sensor_force_names from the first run if not provided
        if sensor_force_names is None:
            run_keys = list(all_loaded_data.keys())
            run = run_keys[1] if len(run_keys) > 1 else run_keys[0]
            sensor_force_names = all_loaded_data[run]["sensor_force_names"]

        # Only keep sensors with 'pylon' or 'socket' in their name
        pylon_sensors = [name for name in sensor_force_names if "pylon" in name or "socket" in name]

        for sensor_name in pylon_sensors:
            if 'torque' in sensor_name:
                run_keys = list(all_loaded_data.keys())
                mean_all = []
                std_all = []
                for run_key in run_keys:
                    run_dict = all_loaded_data[run_key]
                    step_data = run_step_data[run_key]
                    if sensor_name in run_dict.get("all_sensor_force", {}):
                        sensor_data = run_dict["all_sensor_force"][sensor_name]
                    else:
                        continue  # Skip this run if sensor data is missing

                    # Determine side and switches
                    if sensor_name.endswith("_r"):
                        all_step_start = step_data["step_start_right"]
                        switches = right_switches.get(run_key, [])
                    else:
                        all_step_start = step_data["step_start_left"]
                        switches = left_switches.get(run_key, [])

                    steps = []
                    if all_step_start and len(all_step_start) > 1 and switches:
                        for j in range(len(all_step_start) - 1):
                            start, end = all_step_start[j]-1, all_step_start[j + 1]-1
                            y = [float(np.array(a)[force_direction]) for a in sensor_data[start:end]]
                            if len(y) < 2:
                                continue
                            # First interpolate to 100%
                            x_old = np.linspace(0, 1, len(y))
                            x_new = np.linspace(0, 1, interp_len)
                            y_interp = np.interp(x_new, x_old, y)
                            # Take part until switch index (stance phase)
                            switch_idx = switches[0] if switches and switches[0] < interp_len else interp_len
                            stance_phase = y_interp[:switch_idx]
                            # Interpolate stance phase again to 100%
                            if len(stance_phase) > 1:
                                x_stance_old = np.linspace(0, 1, len(stance_phase))
                                x_stance_new = np.linspace(0, 1, interp_len)
                                stance_phase_interp = np.interp(x_stance_new, x_stance_old, stance_phase)
                                steps.append(stance_phase_interp)
                    if steps:
                        mean_val = np.mean(steps, axis=0) / body_weight_to_normalize
                        std_val = np.std(steps, axis=0) / body_weight_to_normalize
                        if force_direction == 2:
                            mean_val *= -1
                            std_val *= -1
                        mean_all.append((run_key, mean_val, std_val))

                x = np.linspace(0, 100, interp_len)
                if mean_all:
                    plt.figure(figsize=(8, 5))
                    for run_key, mean_val, std_val in mean_all:
                        plt.plot(x, mean_val, label=f"{run_key}")
                        plt.fill_between(x, mean_val - std_val, mean_val + std_val, alpha=0.15)
                    plt.title(f"{sensor_name} mean (stance phase, all runs)")
                    plt.xlabel("Interpolated Step (%)")
                    plt.ylabel("Force Value")
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                    plt.close()



    @staticmethod
    def plot_sensor_force_pylon_all_runs_stance_sorted_flock(
        all_loaded_data, run_step_data, direction, left_switches, right_switches,folder_name, interp_len=100, body_weight_to_normalize=1, sensor_force_names=None, smooth_data=False, 
        body_in_sensor_names = ['pylon', 'socket'], knee_alignment_baseline_file= None, knee_alignment_data_mass= 85*9.81, prosthesis_side = 'left', pylon_optimal_alignment_baseline_file = None
    ):
        """
        Plot mean of all runs for pylon sensor data (prosthesis side only).
        Only sensors with 'pylon' in their name are plotted.
        Interpolate each step to 100%, then take the part until left_switches or right_switches (depending on '_l' or '_r'),
        then interpolate this part again to 100%.
        direction: "x", "y", or "z" to specify the force direction. [0, 1, 2] for x, y, z respectively.
        """
        force_direction = {"x": 0, "y": 1, "z": 2}.get(direction, 0)
        index_45_percent = int(np.round(0.45 * (interp_len - 1)))
        index_30_percent = int(np.round(0.30 * (interp_len - 1))) 
        index_75_percent = int(np.round(0.75 * (interp_len - 1))) 
        # Use sensor_force_names from the first run if not provided
        if sensor_force_names is None:
            run_keys = list(all_loaded_data.keys())
            run = run_keys[1] if len(run_keys) > 1 else run_keys[0]
            sensor_force_names = all_loaded_data[run]["sensor_force_names"]

        # Only keep sensors with 'pylon' or 'socket' in their name
        pylon_sensors = [name for name in sensor_force_names if any(body_part in name for body_part in body_in_sensor_names)]
        # pylon_sensors = [name for name in sensor_force_names if "pylon" in name or "socket" in name]

        for sensor_name in pylon_sensors:
            if 'torque' in sensor_name:
                run_keys = list(all_loaded_data.keys())
                mean_all = []
                std_all = []
                bar_plot_data = {} 
                for run_key in run_keys:
                    run_dict = all_loaded_data[run_key]
                    step_data = run_step_data[run_key]
                    if sensor_name in run_dict.get("all_sensor_force", {}):
                        sensor_data = run_dict["all_sensor_force"][sensor_name]
                    else:
                        continue  # Skip this run if sensor data is missing

                    # Determine side and switches
                    if '_r_' in sensor_name: #.endswith("_r"):
                        all_step_start = step_data["step_start_right"]
                        switches = right_switches.get(run_key, [])
                    else:
                        all_step_start = step_data["step_start_left"]
                        switches = left_switches.get(run_key, [])

                    steps = []
                    if all_step_start and len(all_step_start) > 1 and switches:
                        for j in range(len(all_step_start) - 1):
                            start, end = all_step_start[j]-1, all_step_start[j + 1]-1
                            y = [float(np.array(a)[force_direction]) for a in sensor_data[start:end]]
                            if len(y) < 2:
                                continue
                            # First interpolate to 100%
                            x_old = np.linspace(0, 1, len(y))
                            x_new = np.linspace(0, 1, interp_len)
                            y_interp = np.interp(x_new, x_old, y)
                            # Take part until switch index (stance phase)
                            switch_idx = switches[0] if switches and switches[0] < interp_len else interp_len
                            stance_phase = y_interp[:switch_idx]
                            # Interpolate stance phase again to 100%
                            if len(stance_phase) > 1:
                                x_stance_old = np.linspace(0, 1, len(stance_phase))
                                x_stance_new = np.linspace(0, 1, interp_len)
                                stance_phase_interp = np.interp(x_stance_new, x_stance_old, stance_phase)
                                steps.append(stance_phase_interp)
                    if steps:
                        mean_val = np.mean(steps, axis=0) / body_weight_to_normalize
                        std_val = np.std(steps, axis=0) / body_weight_to_normalize
                        if force_direction == 2:
                            mean_val *= -1
                            std_val *= -1
                        # Smooth mean and std if requested
                        if smooth_data:
                            mean_val = gaussian_filter1d(mean_val, sigma=3)
                            std_val = gaussian_filter1d(std_val, sigma=3)
                        mean_all.append((run_key, mean_val, std_val))

                        # --- Collect Bar Plot Data if direction is 'z' (Sagittal Moment) ---
                        if direction == 'z':
                            min_moment = np.min(mean_val)
                            max_moment = np.max(mean_val)
                            moment_at_45 = mean_val[index_45_percent]
                            
                            bar_plot_data[run_key] = {
                                'min': min_moment,
                                'max': max_moment,
                                'at_45': moment_at_45
                            }
                        # --- New Bar Plot Data Collection for direction == 'x' ---
                        elif direction == 'x':
                            moment_at_30 = mean_val[index_30_percent]
                            moment_at_75 = mean_val[index_75_percent]
                            
                            bar_plot_data[run_key] = {
                                'at_30': moment_at_30,
                                'at_75': moment_at_75
                            }

                x = np.linspace(0, 100, interp_len)

                # Helper: try to get a numeric value from run_key
                def extract_numeric(run_key):
                    try:
                        return float(run_key)
                    except Exception:
                        # fallback: extract first number (with optional sign/decimal) via regex
                        m = re.search(r'[-+]?\d*\.?\d+', str(run_key))
                        if m:
                            try:
                                return float(m.group())
                            except Exception:
                                return None
                        return None

                # Sort by numeric value when possible, otherwise by string
                def parse_run_key_for_sort(run_key):
                    num = extract_numeric(run_key)
                    return num if num is not None else str(run_key)

                sorted_mean_all = sorted(mean_all, key=lambda tup: parse_run_key_for_sort(tup[0]))

                # Filter sorted list to only include keys with data (important for color/linestyle mapping)
                sorted_mean_all = [tup for tup in sorted_mean_all if tup[0] in bar_plot_data]

                # ADDED CODE: direction is 'z', sort bar data to match line plot order
                # if direction == 'z':
                #     sorted_run_keys = [tup[0] for tup in sorted_mean_all]
                    
                #     min_moments = [bar_plot_data[rk]['min'] for rk in sorted_run_keys if rk in bar_plot_data]
                #     max_moments = [bar_plot_data[rk]['max'] for rk in sorted_run_keys if rk in bar_plot_data]
                #     moments_at_45 = [bar_plot_data[rk]['at_45'] for rk in sorted_run_keys if rk in bar_plot_data]
                    
                #     # Ensure only keys with data are used for bar labels/colors
                #     bar_labels = [rk for rk in sorted_run_keys if rk in bar_plot_data]
                    
                #     # Update sorted_mean_all to only include entries with data (for consistent color mapping)
                #     sorted_mean_all = [tup for tup in sorted_mean_all if tup[0] in bar_plot_data]
                if bar_plot_data:
                    sorted_run_keys = [tup[0] for tup in sorted_mean_all]
                    bar_labels = [rk for rk in sorted_run_keys if rk in bar_plot_data]

                    if direction == 'z':
                        min_moments = [bar_plot_data[rk]['min'] for rk in bar_labels]
                        max_moments = [bar_plot_data[rk]['max'] for rk in bar_labels]
                        moments_at_45 = [bar_plot_data[rk]['at_45'] for rk in bar_labels]
                    elif direction == 'x':
                        moments_at_30 = [bar_plot_data[rk]['at_30'] for rk in bar_labels]
                        moments_at_75 = [bar_plot_data[rk]['at_75'] for rk in bar_labels]


                # Linestyles to cycle for the Blue/varied-linestyles group
                linestyles = [ (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (3, 5, 1, 5)),"--", "-.", ":", ]

                # Helper to decide whether this run_key should have the reversed mapping
                def _reverse_mapping_for_key(rk):
                    rk_l = str(rk).lower()
                    return (('z' in rk_l and 'mm' in rk_l) or ('x' in rk_l and 'deg' in rk_l)) or ('x' in rk_l and 'mm' in rk_l)

                # Build visual groups (visual_negative == Blue group with varied linestyles)
                visual_neg_keys = []  # keys that will be shown with Blues colormap + varied linestyles
                visual_pos_keys = []  # keys that will be shown with Reds colormap + solid linestyle

                # Gather numeric values for normalization according to the visual grouping
                vis_neg_vals = []
                vis_pos_vals = []

                for run_key, _, _ in sorted_mean_all:
                    rk_str = str(run_key)
                    num = extract_numeric(run_key)
                    # determine original sign presence
                    original_neg = ('-' in rk_str)
                    # compute per-key reverse flag
                    reverse_flag = _reverse_mapping_for_key(rk_str)
                    # visual group: XOR -> if True, key is treated as "negative" visual group
                    visual_neg = (original_neg != reverse_flag)

                    if (num is not None) and (num == 0):
                        # skip zeros in normalization
                        continue

                    if visual_neg:
                        visual_neg_keys.append(rk_str)
                        if num is not None:
                            vis_neg_vals.append(abs(num))
                    else:
                        visual_pos_keys.append(rk_str)
                        # store absolute magnitude for consistent shading
                        if num is not None:
                            vis_pos_vals.append(abs(num))

                # Map visual-negative (Blue) group keys to linestyles
                # Sort visual_neg_keys by absolute numeric magnitude (ascending), then assign
                # linestyles from the reversed list so the smallest magnitude gets the 'largest' linestyle
                linestyle_map = {}
                try:
                    key_vals = []
                    for k in visual_neg_keys:
                        num = extract_numeric(k)
                        key_vals.append((k, num))
                    # sort by abs(num) ascending; put non-numeric at end
                    key_vals_sorted = sorted(key_vals, key=lambda t: (float('inf') if t[1] is None else abs(t[1])))
                    rev_styles = linestyles[::-1]
                    for i, (k, _) in enumerate(key_vals_sorted):
                        linestyle_map[k] = rev_styles[i % len(rev_styles)]
                except Exception:
                    # fallback: assign in original order
                    linestyle_map = {k: linestyles[i % len(linestyles)] for i, k in enumerate(visual_neg_keys)}

                # Fallback ranges if empty
                min_neg, max_neg = (min(vis_neg_vals), max(vis_neg_vals)) if vis_neg_vals else (0.0, 1.0)
                min_pos, max_pos = (min(vis_pos_vals), max(vis_pos_vals)) if vis_pos_vals else (0.0, 1.0)

                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                color_idx = 0
                run_colors = {} # Store colors for bar plots

                plt.figure(figsize=(8, 5))

                for run_key, mean_val, std_val in sorted_mean_all:
                    run_key_str = str(run_key)
                    num = extract_numeric(run_key)

                    # default linewidth
                    linewidth = 1.5

                    # Zero: exact numeric zero or string "0" -> black solid
                    if (num is not None and num == 0) or run_key_str in ("0", "0.0"):
                        color = "black"
                        linestyle = "-"
                        alpha = 1.0

                    else:
                        # determine per-key visual group
                        original_neg = ('-' in run_key_str)
                        reverse_flag = _reverse_mapping_for_key(run_key_str)
                        visual_neg = (original_neg != reverse_flag)

                        # If not numeric, fallback to color cycle and a default linestyle
                        if num is None:
                            color = color_cycle[color_idx % len(color_cycle)]
                            color_idx += 1
                            # pick linestyle from map if in visual-neg group, else solid
                            linestyle = linestyle_map.get(run_key_str, "-") if visual_neg else "-"
                            alpha = 1.0

                        else:
                            # Numeric: pick colormap and normalize according to visual group
                            if visual_neg:
                                # Blue group (varied linestyles)
                                abs_val = abs(num)
                                if max_neg != min_neg:
                                    norm = (abs_val - min_neg) / (max_neg - min_neg)
                                else:
                                    norm = 0.0
                                # original inversion & clipping so larger abs -> lighter
                                norm = 1.0 - np.clip(norm, 0.0, 0.6)
                                cmap = matplotlib.cm.get_cmap('Blues')
                                color = cmap(norm)
                                linestyle = linestyle_map.get(run_key_str, "-")
                                # keep constant linewidth
                                alpha = 1.0
                            else:
                                # Red group (solid lines)
                                # use absolute magnitude for red-group normalization so larger abs -> lighter shade
                                abs_val = abs(num)
                                if max_pos != min_pos:
                                    norm = (abs_val - min_pos) / (max_pos - min_pos)
                                else:
                                    norm = 0.0
                                # original inversion & clipping so larger abs -> lighter
                                norm = 1.0 - np.clip(norm, 0, 0.6)
                                cmap = matplotlib.cm.get_cmap('Reds')
                                color = cmap(norm)
                                linestyle = "-"
                                # keep constant linewidth
                                alpha = 1.0

                    # Store color for bar plots
                    run_colors[run_key_str] = color

                    # Ensure arrays are numpy arrays
                    mean_arr = np.asarray(mean_val)
                    std_arr = np.asarray(std_val)

                    # Plot mean and shaded std dev (use linewidth scaling)
                    plt.plot(x, mean_arr, label=f"{run_key_str}", color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
                    plt.fill_between(x, mean_arr - std_arr, mean_arr + std_arr, alpha=0.15, color=color)

                if direction=='z':
                    plane_name = 'Sagittal'
                elif direction=='x':
                    plane_name = 'Coronal'


                # plot the alignment baseline data if left_knee in sensor_name and direction == 'z'
                if 'knee' in sensor_name and prosthesis_side in sensor_name and direction == 'z':
                    baseline_data = {}
                    # read baseline_data from csv file
                    if knee_alignment_baseline_file and os.path.isfile(knee_alignment_baseline_file):
                        with open(knee_alignment_baseline_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                if 'Schmalz' in knee_alignment_baseline_file:
                                    label_match = re.match(r'^([A-Za-z+\-_]+)\s*=\s*\[?', line)
                                if label_match:
                                    current_label = label_match.group(1)
                                    baseline_data[current_label] = []
                                    continue
                                if current_label:
                                    vals = line.split(';')
                                    if len(vals) == 2:
                                        try:
                                            x, y = map(float, vals)
                                            baseline_data[current_label].append((x, y))
                                        except ValueError:
                                            continue   
                    # if run_keys have z and deg then plot the FOOT_PLA, FOOT_DOR, OPT from baseline
                    if 'z' in run_key and 'deg' in run_key:
                        if baseline_data.get('FOOT_PLA', []):
                            xs, ys = zip(*baseline_data.get('FOOT_PLA', []))
                            ys = np.array(ys) / knee_alignment_data_mass
                            plt.plot(xs, ys, label="Baseline PLA_-10°", color='lightgray', linestyle='--', linewidth=2)
                        if baseline_data.get('FOOT_DOR', []):
                            xs, ys = zip(*baseline_data.get('FOOT_DOR', []))
                            ys = np.array(ys) / knee_alignment_data_mass
                            plt.plot(xs, ys, label="Baseline DOR_10°", color='darkgrey', linestyle='--', linewidth=2)
                        if baseline_data.get('OPT', []):
                            xs, ys = zip(*baseline_data.get('OPT', []))
                            ys = np.array(ys) / knee_alignment_data_mass
                            plt.plot(xs, ys, label="Baseline OPT", color='dimgray', linestyle='--', linewidth=2)
                    elif 'x' in run_key and 'mm' in run_key:
                        if baseline_data.get('FOOT_ANT', []):
                            xs, ys = zip(*baseline_data.get('FOOT_ANT', []))
                            ys = np.array(ys) / knee_alignment_data_mass
                            plt.plot(xs, ys, label="Baseline ANT_20mm", color='lightgray', linestyle='--', linewidth=2)
                        if baseline_data.get('FOOT_POS', []):
                            xs, ys = zip(*baseline_data.get('FOOT_POS', []))
                            ys = np.array(ys) / knee_alignment_data_mass
                            plt.plot(xs, ys, label="Baseline POS_-20mm", color='darkgrey', linestyle='--', linewidth=2)
                        if baseline_data.get('OPT', []):
                            xs, ys = zip(*baseline_data.get('OPT', []))
                            ys = np.array(ys) / knee_alignment_data_mass
                            plt.plot(xs, ys, label="Baseline OPT", color='dimgray', linestyle='--', linewidth=2)
                    else: 
                        if baseline_data.get('OPT', []):
                            xs, ys = zip(*baseline_data.get('OPT', []))
                            ys = np.array(ys) / knee_alignment_data_mass
                            plt.plot(xs, ys, label="Baseline OPT", color='dimgray', linestyle='--', linewidth=2)


                if 'pylon' in sensor_name and pylon_optimal_alignment_baseline_file:
                    baseline_data = {}
                    # read baseline_data from csv file
                    if os.path.isfile(pylon_optimal_alignment_baseline_file):
                        with open(pylon_optimal_alignment_baseline_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                label_match = re.match(r'^([A-Za-z+\-_]+)\s*=\s*\[?', line)
                                if label_match: 
                                    current_label = label_match.group(1)
                                    baseline_data[current_label] = []
                                    continue
                                if current_label:
                                    vals = line.split(';')
                                    if len(vals) == 2:
                                        try:
                                            x, y = map(float, vals)
                                            baseline_data[current_label].append((x, y))
                                        except ValueError:
                                            continue   

                    # Labels in pylon optimal alignment file are: Sagittal, Sagittal+std, Sagittal-std, or Coronal, Coronal+std, Coronal-std
                    if direction == 'z' and baseline_data.get('Sagittal', []):
                        xs, ys = zip(*baseline_data.get('Sagittal', []))
                        # Sort data by x before plotting
                        xs = np.array(xs) * 100  # Convert from 0-1 to 0-100
                        ys = np.array(ys)
                        sort_idx = np.argsort(xs)
                        xs = xs[sort_idx]
                        ys = ys[sort_idx]
                        plt.plot(xs, ys, label="OPT", color='grey', linestyle='-', linewidth=2)
                        if baseline_data.get('Sagittal+std', []) and baseline_data.get('Sagittal-std', []):
                            xs_std_p, ys_std_p = zip(*baseline_data.get('Sagittal+std', []))
                            xs_std_m, ys_std_m = zip(*baseline_data.get('Sagittal-std', []))
                            xs_std_p = np.array(xs_std_p) * 100
                            xs_std_m = np.array(xs_std_m) * 100
                            ys_std_p = np.array(ys_std_p)
                            ys_std_m = np.array(ys_std_m)
                            # Sort std data by x before plotting
                            sort_idx_p = np.argsort(xs_std_p)
                            sort_idx_m = np.argsort(xs_std_m)
                            xs_std_p = xs_std_p[sort_idx_p]
                            ys_std_p = ys_std_p[sort_idx_p]
                            xs_std_m = xs_std_m[sort_idx_m]
                            ys_std_m = ys_std_m[sort_idx_m]
                            # Interpolate ys_std_m and ys_std_p to xs_std_p if lengths mismatch
                            if len(xs_std_p) != len(ys_std_m):
                                ys_std_m_interp = np.interp(xs_std_p, xs_std_m, ys_std_m)
                            else:
                                ys_std_m_interp = ys_std_m
                            if len(xs_std_p) != len(ys_std_p):
                                ys_std_p_interp = np.interp(xs_std_p, xs_std_p, ys_std_p)
                            else:
                                ys_std_p_interp = ys_std_p
                            plt.fill_between(xs_std_p, ys_std_m_interp, ys_std_p_interp, color='lightgrey', alpha=0.6)
                    elif direction == 'x' and baseline_data.get('Coronal', []):
                        xs, ys = zip(*baseline_data.get('Coronal', []))
                        # Sort data by x before plotting
                        xs = np.array(xs) * 100  # Convert from 0-1 to 0-100
                        ys = np.array(ys)
                        sort_idx = np.argsort(xs)
                        xs = xs[sort_idx]
                        ys = ys[sort_idx]
                        plt.plot(xs, ys, label="OPT", color='grey', linestyle='-', linewidth=2)
                        if baseline_data.get('Coronal+std', []) and baseline_data.get('Coronal-std', []):
                            xs_std_p, ys_std_p = zip(*baseline_data.get('Coronal+std', []))
                            xs_std_m, ys_std_m = zip(*baseline_data.get('Coronal-std', []))
                            xs_std_p = np.array(xs_std_p) * 100
                            xs_std_m = np.array(xs_std_m) * 100
                            ys_std_p = np.array(ys_std_p)
                            ys_std_m = np.array(ys_std_m)
                            # Sort std data by x before plotting
                            sort_idx_p = np.argsort(xs_std_p)
                            sort_idx_m = np.argsort(xs_std_m)
                            xs_std_p = xs_std_p[sort_idx_p]
                            ys_std_p = ys_std_p[sort_idx_p]
                            xs_std_m = xs_std_m[sort_idx_m]
                            ys_std_m = ys_std_m[sort_idx_m]
                            plt.fill_between(xs_std_p, ys_std_m, ys_std_p, color='lightgrey', alpha=0.6)


                plt.title(f"{plane_name} Socket Moment")
                plt.xlabel("Stance Phase (%)")
                plt.ylabel("Normalized Socket Reaction Moment (Nm/kg)")
                plt.legend()
                plt.tight_layout()
                date_str = datetime.now().strftime("%Y%m%d")
                time_str = datetime.now().strftime("%H%M%S")
                save_dir = os.path.join(folder_name)
                # os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f"{sensor_name}_moment_{plane_name.lower()}_{date_str}_{time_str}.png"), dpi=300)
                plt.show()
                plt.close()


                # Add reference data to bar plots for pylon/socket sensors
                # Reference values for different conditions (example values, replace with actual reference if needed)
                # The reference arrays must match the length/order of bar_labels

                # Helper: check for keywords in bar_labels
                bar_labels_str = " ".join(str(lbl) for lbl in bar_labels).lower()
                ref_plotted = False

                if direction == 'z':
                    if 'z' in bar_labels_str and 'deg' in bar_labels_str:
                        # Example reference data for z direction, degrees
                        moment_45_data = [0.160, 0.176, 0.217, 0.363, 0.396]
                        moment_max_data = [0.793, 0.755, 0.719, 0.672, 0.609]
                        moment_min_data = [-0.077, -0.104, -0.147, -0.144, -0.158]
                        x_data = [-6,-3,0,3,6]
                        ref_plotted = True
                    elif 'z' in bar_labels_str and 'mm' in bar_labels_str:
                        moment_min_data = [-0.137, -0.122, -0.147, -0.119, -0.132]
                        moment_45_data = [0.217, 0.250, 0.217, 0.224, 0.304]
                        moment_max_data = [0.735, 0.722, 0.719, 0.730, 0.718]
                        x_data = [-10,-5,0,5,10]
                        ref_plotted = True
                    elif 'x' in bar_labels_str and 'mm' in bar_labels_str:
                        moment_min_data = [-0.059, -0.095, -0.147, -0.163, -0.187]
                        moment_45_data = [0.287, 0.252, 0.217, 0.227, 0.203]
                        moment_max_data = [0.821, 0.776, 0.719, 0.693, 0.613]
                        x_data = [-10,-5,0,5,10]
                        ref_plotted = True
                    elif 'x' in bar_labels_str and 'deg' in bar_labels_str:
                        moment_min_data = [-0.142, -0.137, -0.147, -0.128, -0.142]
                        moment_max_data = [0.729, 0.708, 0.719, 0.726, 0.722]
                        moment_45_data = [0.249, 0.255, 0.217, 0.248, 0.269]
                        x_data = [-6,-3,0,3,6]
                        ref_plotted = True
                elif direction == 'x':
                    if 'x' in bar_labels_str and 'deg' in bar_labels_str:
                        moment_30_data = [-0.174, -0.130, -0.077, -0.006, -0.074][::-1]
                        moment_75_data = [-0.089, -0.046, 0.013, 0.049, 0.111][::-1]
                        x_data = [-6,-3,0,3,6]
                        ref_plotted = True
                    elif 'z' in bar_labels_str and 'mm' in bar_labels_str:
                        moment_30_data = [-0.163, -0.109, -0.077, -0.033, 0.025]
                        moment_75_data = [-0.062, -0.023, 0.013, 0.054, 0.096]
                        x_data = [-10,-5,0,5,10]
                        ref_plotted = True
                    elif 'z' in bar_labels_str and 'deg' in bar_labels_str:
                        moment_30_data = [-0.029, -0.033,-0.077, -0.073, -0.072][::-1]
                        moment_75_data = [-0.019, -0.006, 0.013,0.001, 0.003][::-1]
                        x_data = [-6,-3,0,3,6]
                        ref_plotted = True
                    elif 'x' in bar_labels_str and 'mm' in bar_labels_str:
                        moment_30_data = [-0.055, -0.067, -0.077,-0.060, -0.072][::-1]
                        moment_75_data = [0.002, 0.010, 0.013,0.016, 0.021][::-1]
                        x_data = [-10,-5,0,5,10]
                        ref_plotted = True

                # --- New Bar Plots for Z-Direction Moment ---
                if direction == 'z' and bar_labels:
                    bar_colors = [run_colors[label] for label in bar_labels]
                    x_pos = np.arange(len(bar_labels))

                    fig_bars, axes_bars = plt.subplots(1, 3, figsize=(15, 5))

                    # 1. Minimum Moment
                    axes_bars[0].bar(x_pos, min_moments, color=bar_colors, label="Simulation")
                    axes_bars[0].axhline(0, color='grey', linewidth=0.8)
                    axes_bars[0].set_xticks(x_pos)
                    axes_bars[0].set_xticklabels(bar_labels, rotation=45, ha='right')
                    axes_bars[0].set_title("Minimum Sagittal Moment")
                    axes_bars[0].set_ylabel("Normalized Moment (Nm/kg)")
                    axes_bars[0].tick_params(axis='x', which='major', labelsize=8)
                    # Reference as dots
                    if ref_plotted and 'moment_min_data' in locals() and len(moment_min_data) == len(x_pos):
                        axes_bars[0].scatter(x_pos, moment_min_data, color='grey', marker='o', s=60, label="Reference")
                        axes_bars[0].legend()
                    # mismatched lengths
                    elif ref_plotted and 'moment_min_data' in locals() and len(moment_min_data) != len(x_pos):
                        # Only plot reference dots for x_pos values that match
                        for idx, label in enumerate(bar_labels):
                            try:
                                label_num = float(label)
                            except Exception:
                                # fallback: extract first number via regex  
                                m = re.search(r'[-+]?\d*\.?\d+', str(label))
                                label_num = float(m.group()) if m else None
                            if label_num is not None and label_num in x_data:
                                ref_idx = x_data.index(label_num)
                                axes_bars[0].scatter(x_pos[idx], moment_min_data[ref_idx], color='grey', marker='o', s=60, label="Reference" if idx == 0 else None)
                        axes_bars[0].legend()

                    # 2. Moment at 45% Stance
                    axes_bars[1].bar(x_pos, moments_at_45, color=bar_colors, label="Simulation")
                    axes_bars[1].axhline(0, color='grey', linewidth=0.8)
                    axes_bars[1].set_xticks(x_pos)
                    axes_bars[1].set_xticklabels(bar_labels, rotation=45, ha='right')
                    axes_bars[1].set_title("Sagittal Moment at 45% Stance")
                    axes_bars[1].set_ylabel("Normalized Moment (Nm/kg)")
                    axes_bars[1].tick_params(axis='x', which='major', labelsize=8)
                    # Reference as dots
                    # if ref_plotted and 'moment_45_data' in locals() and len(moment_45_data) == len(x_pos):
                    #     axes_bars[1].scatter(x_pos, moment_45_data, color='black', marker='o', s=60, label="Reference")
                    #     axes_bars[1].legend()
                    # if not the same length match the x_pos with x_data
                    if ref_plotted and 'moment_45_data' in locals() and len(moment_45_data) == len(x_pos):
                        axes_bars[1].scatter(x_pos, moment_45_data, color='grey', marker='o', s=60, label="Reference")
                        axes_bars[1].legend()
                    # mismatched lengths
                    elif ref_plotted and 'moment_45_data' in locals() and len(moment_45_data) != len(x_pos):
                        # Only plot reference dots for x_pos values that match
                        for idx, label in enumerate(bar_labels):
                            try:
                                label_num = float(label)
                            except Exception:
                                # fallback: extract first number via regex  
                                m = re.search(r'[-+]?\d*\.?\d+', str(label))
                                label_num = float(m.group()) if m else None
                            if label_num is not None and label_num in x_data:
                                ref_idx = x_data.index(label_num)
                                axes_bars[1].scatter(x_pos[idx], moment_45_data[ref_idx], color='grey', marker='o', s=60, label="Reference" if idx == 0 else None)
                        axes_bars[1].legend()


                    # 3. Maximum Moment
                    axes_bars[2].bar(x_pos, max_moments, color=bar_colors, label="Simulation")
                    axes_bars[2].axhline(0, color='grey', linewidth=0.8)
                    axes_bars[2].set_xticks(x_pos)
                    axes_bars[2].set_xticklabels(bar_labels, rotation=45, ha='right')
                    axes_bars[2].set_title("Maximum Sagittal Moment")
                    axes_bars[2].set_ylabel("Normalized Moment (Nm/kg)")
                    axes_bars[2].tick_params(axis='x', which='major', labelsize=8)
                    # Reference as dots
                    if ref_plotted and 'moment_max_data' in locals() and len(moment_max_data) == len(x_pos):
                        axes_bars[2].scatter(x_pos, moment_max_data, color='grey', marker='o', s=60, label="Reference")
                        axes_bars[2].legend()
                    # mismatched lengths
                    elif ref_plotted and 'moment_max_data' in locals() and len(moment_max_data) != len(x_pos):
                        # Only plot reference dots for x_pos values that match
                        for idx, label in enumerate(bar_labels):
                            try:
                                label_num = float(label)
                            except Exception:
                                # fallback: extract first number via regex
                                m = re.search(r'[-+]?\d*\.?\d+', str(label))
                                label_num = float(m.group()) if m else None
                            if label_num is not None and label_num in x_data:
                                ref_idx = x_data.index(label_num)
                                axes_bars[2].scatter(x_pos[idx], moment_max_data[ref_idx], color='grey', marker='o', s=60, label="Reference" if idx == 0 else None)
                        axes_bars[2].legend()


                    fig_bars.suptitle(f"{sensor_name} Key Moment Values", fontsize=14)
                    fig_bars.tight_layout(rect=[0, 0.03, 1, 0.95])
                    fig_bars.savefig(os.path.join(save_dir, f"{sensor_name}_z_bars_{date_str}_{time_str}.png"), dpi=300)
                    plt.show()
                    plt.close(fig_bars)

                # --- New Bar Plots for X-Direction Moment (Coronal) ---
                if direction == 'x' and bar_labels:
                    bar_colors = [run_colors[label] for label in bar_labels]
                    x_pos = np.arange(len(bar_labels))

                    fig_bars_x, axes_bars_x = plt.subplots(1, 2, figsize=(10, 5))

                    # 1. Moment at 30% Stance
                    axes_bars_x[0].bar(x_pos, moments_at_30, color=bar_colors, label="Simulation")
                    axes_bars_x[0].axhline(0, color='grey', linewidth=0.8)
                    axes_bars_x[0].set_xticks(x_pos)
                    axes_bars_x[0].set_xticklabels(bar_labels, rotation=45, ha='right')
                    axes_bars_x[0].set_title("Coronal Moment at 30% Stance")
                    axes_bars_x[0].set_ylabel("Normalized Moment (Nm/kg)")
                    axes_bars_x[0].tick_params(axis='x', which='major', labelsize=8)
                    # Reference as dots
                    if ref_plotted and 'moment_30_data' in locals() and len(moment_30_data) == len(x_pos):
                        axes_bars_x[0].scatter(x_pos, moment_30_data, color='grey', marker='o', s=60, label="Reference")
                        axes_bars_x[0].legend()
                    # mismatched lengths
                    elif ref_plotted and 'moment_30_data' in locals() and len(moment_30_data) != len(x_pos):
                        # Only plot reference dots for x_pos values that match
                        for idx, label in enumerate(bar_labels):
                            try:
                                label_num = float(label)
                            except Exception:
                                # fallback: extract first number via regex
                                m = re.search(r'[-+]?\d*\.?\d+', str(label))
                                label_num = float(m.group()) if m else None
                            if label_num is not None and label_num in x_data:
                                ref_idx = x_data.index(label_num)
                                axes_bars_x[0].scatter(x_pos[idx], moment_30_data[ref_idx], color='grey', marker='o', s=60, label="Reference" if idx == 0 else None)
                        axes_bars_x[0].legend()

                    # 2. Moment at 75% Stance
                    axes_bars_x[1].bar(x_pos, moments_at_75, color=bar_colors, label="Simulation")
                    axes_bars_x[1].axhline(0, color='grey', linewidth=0.8)
                    axes_bars_x[1].set_xticks(x_pos)
                    axes_bars_x[1].set_xticklabels(bar_labels, rotation=45, ha='right')
                    axes_bars_x[1].set_title("Coronal Moment at 75% Stance")
                    axes_bars_x[1].set_ylabel("Normalized Moment (Nm/kg)")
                    axes_bars_x[1].tick_params(axis='x', which='major', labelsize=8)
                    # Reference as dots
                    if ref_plotted and 'moment_75_data' in locals() and len(moment_75_data) == len(x_pos):
                        axes_bars_x[1].scatter(x_pos, moment_75_data, color='grey', marker='o', s=60, label="Reference")
                        axes_bars_x[1].legend()
                    # mismatched lengths
                    elif ref_plotted and 'moment_75_data' in locals() and len(moment_75_data) != len(x_pos):
                        # Only plot reference dots for x_pos values that match
                        for idx, label in enumerate(bar_labels):
                            try:
                                label_num = float(label)
                            except Exception:
                                # fallback: extract first number via regex
                                m = re.search(r'[-+]?\d*\.?\d+', str(label))
                                label_num = float(m.group()) if m else None
                            if label_num is not None and label_num in x_data:
                                ref_idx = x_data.index(label_num)
                                axes_bars_x[1].scatter(x_pos[idx], moment_75_data[ref_idx], color='grey', marker='o', s=60, label="Reference" if idx == 0 else None)
                        axes_bars_x[1].legend()

                    fig_bars_x.suptitle(f"{sensor_name} Key Coronal Moment Values", fontsize=14)
                    fig_bars_x.tight_layout(rect=[0, 0.03, 1, 0.95])
                    fig_bars_x.savefig(os.path.join(save_dir, f"socket_moment_x_bars_{date_str}_{time_str}.png"), dpi=300)
                    plt.show()
                    plt.close(fig_bars_x)



                # # --- New Bar Plots for Z-Direction Moment ---
                # if direction == 'z' and bar_labels:
                #     # Get the colors in the correct order for the bars
                #     bar_colors = [run_colors[label] for label in bar_labels]
                #     x_pos = np.arange(len(bar_labels))
                    
                #     fig_bars, axes_bars = plt.subplots(1, 3, figsize=(15, 5))
                    
                #     # 1. Minimum Moment
                #     axes_bars[0].bar(x_pos, min_moments, color=bar_colors)
                #     axes_bars[0].axhline(0, color='grey', linewidth=0.8)
                #     axes_bars[0].set_xticks(x_pos)
                #     axes_bars[0].set_xticklabels(bar_labels, rotation=45, ha='right')
                #     axes_bars[0].set_title("Minimum Sagittal Moment")
                #     axes_bars[0].set_ylabel("Normalized Moment (Nm/kg)")
                #     axes_bars[0].tick_params(axis='x', which='major', labelsize=8)

                #     # 2. Moment at 45% Stance
                #     axes_bars[1].bar(x_pos, moments_at_45, color=bar_colors)
                #     axes_bars[1].axhline(0, color='grey', linewidth=0.8)
                #     axes_bars[1].set_xticks(x_pos)
                #     axes_bars[1].set_xticklabels(bar_labels, rotation=45, ha='right')
                #     axes_bars[1].set_title("Sagittal Moment at 45% Stance")
                #     axes_bars[1].set_ylabel("Normalized Moment (Nm/kg)")
                #     axes_bars[1].tick_params(axis='x', which='major', labelsize=8)

                #     # 3. Maximum Moment
                #     axes_bars[2].bar(x_pos, max_moments, color=bar_colors)
                #     axes_bars[2].axhline(0, color='grey', linewidth=0.8)
                #     axes_bars[2].set_xticks(x_pos)
                #     axes_bars[2].set_xticklabels(bar_labels, rotation=45, ha='right')
                #     axes_bars[2].set_title("Maximum Sagittal Moment")
                #     axes_bars[2].set_ylabel("Normalized Moment (Nm/kg)")
                #     axes_bars[2].tick_params(axis='x', which='major', labelsize=8)
                    
                #     fig_bars.suptitle(f"{sensor_name} Key Moment Values", fontsize=14)
                #     fig_bars.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
                    
                #     # Saving/Showing Bar Plot
                #     fig_bars.savefig(os.path.join(save_dir, f"{sensor_name}_z_bars_{date_str}_{time_str}.png"), dpi=300)
                #     plt.show()
                #     plt.close(fig_bars)


                # # --- New Bar Plots for X-Direction Moment (Coronal) ---
                # if direction == 'x' and bar_labels:
                #     bar_colors = [run_colors[label] for label in bar_labels]
                #     x_pos = np.arange(len(bar_labels))
                    
                #     fig_bars_x, axes_bars_x = plt.subplots(1, 2, figsize=(10, 5))
                    
                #     # 1. Moment at 30% Stance
                #     axes_bars_x[0].bar(x_pos, moments_at_30, color=bar_colors)
                #     axes_bars_x[0].axhline(0, color='grey', linewidth=0.8)
                #     axes_bars_x[0].set_xticks(x_pos)
                #     axes_bars_x[0].set_xticklabels(bar_labels, rotation=45, ha='right')
                #     axes_bars_x[0].set_title("Coronal Moment at 30% Stance")
                #     axes_bars_x[0].set_ylabel("Normalized Moment (Nm/kg)")
                #     axes_bars_x[0].tick_params(axis='x', which='major', labelsize=8)

                #     # 2. Moment at 75% Stance
                #     axes_bars_x[1].bar(x_pos, moments_at_75, color=bar_colors)
                #     axes_bars_x[1].axhline(0, color='grey', linewidth=0.8)
                #     axes_bars_x[1].set_xticks(x_pos)
                #     axes_bars_x[1].set_xticklabels(bar_labels, rotation=45, ha='right')
                #     axes_bars_x[1].set_title("Coronal Moment at 75% Stance")
                #     axes_bars_x[1].set_ylabel("Normalized Moment (Nm/kg)")
                #     axes_bars_x[1].tick_params(axis='x', which='major', labelsize=8)
                    
                #     fig_bars_x.suptitle(f"{sensor_name} Key Coronal Moment Values", fontsize=14)
                #     fig_bars_x.tight_layout(rect=[0, 0.03, 1, 0.95])
                    
                #     fig_bars_x.savefig(os.path.join(save_dir, f"socket_moment_x_bars_{date_str}_{time_str}.png"), dpi=300)
                #     plt.show()
                #     plt.close(fig_bars_x)


    # @staticmethod
    # def plot_sensor_force_pylon_all_runs_stance_sorted_flock(
    #     all_loaded_data, run_step_data, direction, left_switches, right_switches,folder_name, interp_len=100, body_weight_to_normalize=1, sensor_force_names=None, smooth_data=False, 
    #     body_in_sensor_names = ['pylon', 'socket'], knee_alignment_baseline_file= None, knee_alignment_data_mass= 85*9.81, prosthesis_side = 'left'
    # ):
    #     """
    #     Plot mean of all runs for pylon sensor data (prosthesis side only) with the revised 
    #     Blue/Red, unique linestyle, and absolute-magnitude color progression scheme.
    #     """
    #     force_direction = {"x": 0, "y": 1, "z": 2}.get(direction, 0)
    #     index_45_percent = int(np.round(0.45 * (interp_len - 1)))
    #     index_30_percent = int(np.round(0.30 * (interp_len - 1))) 
    #     index_75_percent = int(np.round(0.75 * (interp_len - 1))) 
        
    #     # Use sensor_force_names from the first run if not provided
    #     if sensor_force_names is None:
    #         run_keys = list(all_loaded_data.keys())
    #         run = run_keys[1] if len(run_keys) > 1 else run_keys[0]
    #         if run in all_loaded_data and "sensor_force_names" in all_loaded_data[run]:
    #             sensor_force_names = all_loaded_data[run]["sensor_force_names"]
    #         else:
    #             # Fallback if initial run key structure is missing
    #             print("Warning: Could not determine sensor_force_names. Skipping plot.")
    #             return

    #     # Only keep sensors with 'pylon' or 'socket' in their name
    #     pylon_sensors = [name for name in sensor_force_names if any(body_part in name for body_part in body_in_sensor_names)]

    #     for sensor_name in pylon_sensors:
    #         if 'torque' in sensor_name:
    #             run_keys = list(all_loaded_data.keys())
    #             mean_all = []
    #             std_all = []
    #             bar_plot_data = {} 
    #             for run_key in run_keys:
    #                 run_dict = all_loaded_data[run_key]
    #                 step_data = run_step_data[run_key]
    #                 if sensor_name in run_dict.get("all_sensor_force", {}):
    #                     sensor_data = run_dict["all_sensor_force"][sensor_name]
    #                 else:
    #                     continue  # Skip this run if sensor data is missing

    #                 # Determine side and switches
    #                 if sensor_name.endswith("_r"):
    #                     all_step_start = step_data["step_start_right"]
    #                     switches = right_switches.get(run_key, [])
    #                 else:
    #                     all_step_start = step_data["step_start_left"]
    #                     switches = left_switches.get(run_key, [])

    #                 steps = []
    #                 if all_step_start and len(all_step_start) > 1 and switches:
    #                     for j in range(len(all_step_start) - 1):
    #                         start, end = all_step_start[j], all_step_start[j + 1]
    #                         y = [float(np.array(a)[force_direction]) for a in sensor_data[start:end]]
    #                         if len(y) < 2:
    #                             continue
    #                         # First interpolate to 100%
    #                         x_old = np.linspace(0, 1, len(y))
    #                         x_new = np.linspace(0, 1, interp_len)
    #                         y_interp = np.interp(x_new, x_old, y)
    #                         # Take part until switch index (stance phase)
    #                         switch_idx = switches[0] if switches and switches[0] < interp_len else interp_len
    #                         stance_phase = y_interp[:switch_idx]
    #                         # Interpolate stance phase again to 100%
    #                         if len(stance_phase) > 1:
    #                             x_stance_old = np.linspace(0, 1, len(stance_phase))
    #                             x_stance_new = np.linspace(0, 1, interp_len)
    #                             stance_phase_interp = np.interp(x_stance_new, x_stance_old, stance_phase)
    #                             steps.append(stance_phase_interp)
    #                 if steps:
    #                     # Normalize by body weight (assuming normalization factor is already applied)
    #                     mean_val = np.mean(steps, axis=0) / body_weight_to_normalize
    #                     std_val = np.std(steps, axis=0) / body_weight_to_normalize
    #                     if force_direction == 2:
    #                         mean_val *= -1
    #                         std_val *= -1
    #                     # Smooth mean and std if requested
    #                     if smooth_data:
    #                         mean_val = gaussian_filter1d(mean_val, sigma=3)
    #                         std_val = gaussian_filter1d(std_val, sigma=3)
    #                     mean_all.append((run_key, mean_val, std_val))

    #                     # --- Collect Bar Plot Data if direction is 'z' or 'x' ---
    #                     if direction == 'z':
    #                         min_moment = np.min(mean_val)
    #                         max_moment = np.max(mean_val)
    #                         moment_at_45 = mean_val[index_45_percent]
                            
    #                         bar_plot_data[run_key] = {
    #                             'min': min_moment,
    #                             'max': max_moment,
    #                             'at_45': moment_at_45
    #                         }
    #                     elif direction == 'x':
    #                         moment_at_30 = mean_val[index_30_percent]
    #                         moment_at_75 = mean_val[index_75_percent]
                            
    #                         bar_plot_data[run_key] = {
    #                             'at_30': moment_at_30,
    #                             'at_75': moment_at_75
    #                         }

    #             x = np.linspace(0, 100, interp_len)

    #             # Helper: try to get a numeric value from run_key
    #             def extract_numeric(run_key):
    #                 try:
    #                     return float(run_key)
    #                 except Exception:
    #                     # fallback: extract first number (with optional sign/decimal) via regex
    #                     m = re.search(r'[-+]?\d*\.?\d+', str(run_key))
    #                     if m:
    #                         try:
    #                             return float(m.group())
    #                         except Exception:
    #                             return None
    #                     return None

    #             # Sort by numeric value when possible, otherwise by string
    #             def parse_run_key_for_sort(run_key):
    #                 num = extract_numeric(run_key)
    #                 return num if num is not None else str(run_key)

    #             sorted_mean_all = sorted(mean_all, key=lambda tup: parse_run_key_for_sort(tup[0]))

    #             # Filter sorted list to only include keys with data
    #             sorted_mean_all = [tup for tup in sorted_mean_all if tup[0] in bar_plot_data]

    #             if bar_plot_data:
    #                 sorted_run_keys = [tup[0] for tup in sorted_mean_all]
    #                 bar_labels = [rk for rk in sorted_run_keys if rk in bar_plot_data]

    #                 if direction == 'z':
    #                     min_moments = [bar_plot_data[rk]['min'] for rk in bar_labels]
    #                     max_moments = [bar_plot_data[rk]['max'] for rk in bar_labels]
    #                     moments_at_45 = [bar_plot_data[rk]['at_45'] for rk in bar_labels]
    #                 elif direction == 'x':
    #                     moments_at_30 = [bar_plot_data[rk]['at_30'] for rk in bar_labels]
    #                     moments_at_75 = [bar_plot_data[rk]['at_75'] for rk in bar_labels]

    #             # --------------------------------------------------------
    #             # LOGIC FOR COLOR/LINESTYLE MAPPING SCHEMES
    #             # --------------------------------------------------------

    #             # Linestyles to cycle for the "Blue Group"
    #             linestyles = [ (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (3, 5, 1, 5)),"--", "-.", ":", ]

    #             # Determine the Blue vs. Red assignment based on keywords
    #             first_run_key = next((str(tup[0]) for tup in sorted_mean_all if str(tup[0]) not in ("0", "0.0")), None)

    #             # positive_is_blue_group = True means POSITIVE values go to the Blue Group (unique linestyle)
    #             positive_is_blue_group = False
                
    #             if first_run_key:
    #                 has_z = 'z' in first_run_key.lower()
    #                 has_x = 'x' in first_run_key.lower()
    #                 has_mm = 'mm' in first_run_key.lower()
    #                 has_deg = 'deg' in first_run_key.lower()

    #                 # Blue Group Trigger: (z and mm) or (x and deg) -> POSITIVE is Blue Group
    #                 if (has_z and has_mm) or (has_x and has_deg):
    #                     positive_is_blue_group = True


    #             # Gather absolute numeric values for normalization:
    #             neg_vals = []   # absolute numeric values < 0
    #             pos_vals = []   # numeric values > 0

    #             for run_key, _, _ in sorted_mean_all:
    #                 num = extract_numeric(run_key)
    #                 if num is not None and num != 0:
    #                     if num < 0:
    #                         neg_vals.append(abs(num))
    #                     else:
    #                         pos_vals.append(num)

    #             # Determine the min/max of the non-zero absolute values for normalization
    #             min_neg, max_neg = (min(neg_vals), max(neg_vals)) if neg_vals else (0.0, 1.0)
    #             min_pos, max_pos = (min(pos_vals), max(pos_vals)) if pos_vals else (0.0, 1.0)

                
    #             # Map run keys to linestyles for the Blue Group
    #             blue_group_run_keys = []
                
    #             if positive_is_blue_group:
    #                 # Keys with POSITIVE numeric value go to Blue Group
    #                 blue_group_run_keys = [str(tup[0]) for tup in sorted_mean_all if extract_numeric(tup[0]) is not None and extract_numeric(tup[0]) > 0]
    #             else:
    #                 # Keys with NEGATIVE numeric value go to Blue Group
    #                 blue_group_run_keys = [str(tup[0]) for tup in sorted_mean_all if extract_numeric(tup[0]) is not None and extract_numeric(tup[0]) < 0]

    #             linestyle_map = {k: linestyles[i % len(linestyles)] for i, k in enumerate(blue_group_run_keys)}
                
    #             # --------------------------------------------------------
    #             # PLOTTING LOOP WITH CORRECTED ABSOLUTE-MAGNITUDE SHADING
    #             # --------------------------------------------------------

    #             color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #             run_colors = {} # Store colors for bar plots

    #             plt.figure(figsize=(8, 5))

    #             for run_key, mean_val, std_val in sorted_mean_all:
    #                 run_key_str = str(run_key)
    #                 num = extract_numeric(run_key)
                    
    #                 # Default to safe values
    #                 color = "gray"
    #                 linestyle = "-"
    #                 alpha = 1.0

    #                 # 1. Zero Run: Black and Solid
    #                 if (num is not None and num == 0) or run_key_str in ("0", "0.0"):
    #                     color = "black"
    #                     linestyle = "-"
    #                     alpha = 1.0
                    
    #                 # Check if the current run_key is in the unique-linestyle (Blue) group
    #                 is_blue_group = run_key_str in blue_group_run_keys

    #                 # 2. Numeric Positive Runs (num > 0)
    #                 if num is not None and num > 0:
                        
    #                     if is_blue_group:
    #                         cmap = matplotlib.cm.get_cmap('Blues')
    #                         linestyle = linestyle_map.get(run_key_str, "-")
    #                     else:
    #                         cmap = matplotlib.cm.get_cmap('Reds')
    #                         linestyle = "-" # Solid line

    #                     # --- Color Calculation: Lighter for larger absolute value ---
    #                     max_val, min_val = max_pos, min_pos
    #                     if max_val > min_val:
    #                         # Normalize to 0 (smallest absolute) to 1 (largest absolute)
    #                         norm = (num - min_val) / (max_val - min_val) 
    #                     else:
    #                         norm = 0.0
                        
    #                     # Inverted Norm: Map 0.0 (smallest abs) to 1.0 (dark) 
    #                     # and 1.0 (largest abs) to 0.0 (light). The color range is [0.6, 0.9]
    #                     # This ensures runs close to zero are dark, and far from zero are light.
    #                     inverted_norm = 1.0 - norm
    #                     # Map [0,1] to [0.6, 0.9] - 0.9 is dark, 0.6 is light
    #                     norm_clip = 0.6 + np.clip(inverted_norm, 0, 1) * 0.3 
                        
    #                     color = cmap(norm_clip)
                    
    #                 # 3. Numeric Negative Runs (num < 0)
    #                 elif num is not None and num < 0:
    #                     abs_val = abs(num)
                        
    #                     if is_blue_group:
    #                         cmap = matplotlib.cm.get_cmap('Blues')
    #                         linestyle = linestyle_map.get(run_key_str, "-")
    #                     else:
    #                         cmap = matplotlib.cm.get_cmap('Reds')
    #                         linestyle = "-" # Solid line

    #                     # --- Color Calculation: Lighter for larger absolute value ---
    #                     max_val, min_val = max_neg, min_neg
    #                     if max_val > min_val:
    #                         # Normalize to 0 (smallest absolute) to 1 (largest absolute)
    #                         norm = (abs_val - min_val) / (max_val - min_val)
    #                     else:
    #                         norm = 0.0
                        
    #                     # Inverted Norm: Map 0.0 (smallest abs) to 1.0 (dark) 
    #                     # and 1.0 (largest abs) to 0.0 (light). The color range is [0.6, 0.9]
    #                     inverted_norm = 1.0 - norm
    #                     # Map [0,1] to [0.6, 0.9] - 0.9 is dark, 0.6 is light
    #                     norm_clip = 0.6 + np.clip(inverted_norm, 0, 1) * 0.3 
                        
    #                     color = cmap(norm_clip)
                    
    #                 # 4. Non-numeric runs (fallback) - Fix for 'tab:blue' error
    #                 else:
    #                     color = "tab:blue" 
    #                     linestyle = "-"


    #                 # Store color for bar plots
    #                 run_colors[run_key_str] = color

    #                 # Ensure arrays are numpy arrays
    #                 mean_arr = np.asarray(mean_val)
    #                 std_arr = np.asarray(std_val)

    #                 # Plot mean and shaded std dev
    #                 plt.plot(x, mean_arr, label=f"{run_key_str}", color=color, linestyle=linestyle, alpha=alpha)
    #                 plt.fill_between(x, mean_arr - std_arr, mean_arr + std_arr, alpha=0.15, color=color)

    #             # --------------------------------------------------------
    #             # PLOT DECORATION AND SAVING (Unchanged from previous versions)
    #             # --------------------------------------------------------
                
    #             if direction=='z':
    #                 plane_name = 'Sagittal'
    #             elif direction=='x':
    #                 plane_name = 'Coronal'

    #             # plot the alignment baseline data if left_knee in sensor_name and direction == 'z'
    #             if 'knee' in sensor_name and prosthesis_side in sensor_name and direction == 'z':
    #                 baseline_data = {}
    #                 # read baseline_data from csv file
    #                 if knee_alignment_baseline_file and os.path.isfile(knee_alignment_baseline_file):
    #                     current_label = None # Initialize current_label outside the loop
    #                     with open(knee_alignment_baseline_file, 'r') as f:
    #                         for line in f:
    #                             line = line.strip()
    #                             if not line:
    #                                 continue
    #                             if 'Schmalz' in knee_alignment_baseline_file:
    #                                 label_match = re.match(r'^([A-Z_]+)\s*=\s*\[?', line)
    #                             else:
    #                                 # Generic matching for potential keys
    #                                 label_match = re.match(r'^([A-Z_]+)\s*=\s*\[?', line)

    #                             if label_match:
    #                                 current_label = label_match.group(1)
    #                                 baseline_data[current_label] = []
    #                                 continue
    #                             if current_label and not label_match: # Process data lines only if a label was found
    #                                 vals = line.replace('[','').replace(']','').split(';')
    #                                 if len(vals) == 2:
    #                                     try:
    #                                         x_val, y_val = map(float, vals)
    #                                         # Normalize y_val by mass
    #                                         y_val_normalized = y_val / (knee_alignment_data_mass * 9.81) if knee_alignment_data_mass != 0 else y_val
    #                                         baseline_data[current_label].append((x_val, y_val_normalized))
    #                                     except ValueError:
    #                                         continue  
                    
    #                 # Apply baseline logic based on run key keywords
    #                 if first_run_key:
    #                     if 'z' in first_run_key and 'deg' in first_run_key:
    #                         baseline_keys = {'FOOT_PLA': "Baseline PLA_-10°", 'FOOT_DOR': "Baseline DOR_10°", 'OPT': "Baseline OPT"}
    #                     elif 'x' in first_run_key and 'mm' in first_run_key:
    #                         baseline_keys = {'FOOT_ANT': "Baseline ANT_20mm", 'FOOT_POS': "Baseline POS_-20mm", 'OPT': "Baseline OPT"}
    #                     else:
    #                         baseline_keys = {}

    #                     baseline_styles = {
    #                         'FOOT_PLA': ('lightgray', '--'), 
    #                         'FOOT_DOR': ('darkgrey', '--'), 
    #                         'OPT': ('dimgray', '--'),
    #                         'FOOT_ANT': ('lightgray', '--'), 
    #                         'FOOT_POS': ('darkgrey', '--'),
    #                     }

    #                     for key, label in baseline_keys.items():
    #                         if key in baseline_data and baseline_data[key]:
    #                             xs, ys = zip(*baseline_data[key])
    #                             color, linestyle = baseline_styles.get(key, ('gray', '--'))
    #                             plt.plot(xs, ys, label=label, color=color, linestyle=linestyle, linewidth=2)


    #             plt.title(f"{plane_name} Socket Moment")
    #             plt.xlabel("Stance Phase (%)")
    #             plt.ylabel("Normalized Socket Reaction Moment (Nm/kg)")
    #             plt.legend()
    #             plt.tight_layout()
    #             date_str = datetime.now().strftime("%Y%m%d")
    #             time_str = datetime.now().strftime("%H%M%S")
    #             save_dir = os.path.join(folder_name)
                
    #             # Ensure folder exists
    #             os.makedirs(save_dir, exist_ok=True) 

    #             plt.savefig(os.path.join(save_dir, f"{sensor_name}_moment_{plane_name.lower()}_{date_str}_{time_str}.png"), dpi=300)
    #             plt.show()
    #             plt.close()


    #             # --- Bar Plots for Z-Direction Moment ---
    #             if direction == 'z' and bar_labels:
    #                 # Get the colors in the correct order for the bars
    #                 bar_colors = [run_colors[label] for label in bar_labels]
    #                 x_pos = np.arange(len(bar_labels))
                    
    #                 fig_bars, axes_bars = plt.subplots(1, 3, figsize=(15, 5))
                    
    #                 # 1. Minimum Moment
    #                 axes_bars[0].bar(x_pos, min_moments, color=bar_colors)
    #                 axes_bars[0].axhline(0, color='grey', linewidth=0.8)
    #                 axes_bars[0].set_xticks(x_pos)
    #                 axes_bars[0].set_xticklabels(bar_labels, rotation=45, ha='right')
    #                 axes_bars[0].set_title("Minimum Sagittal Moment")
    #                 axes_bars[0].set_ylabel("Normalized Moment (Nm/kg)")
    #                 axes_bars[0].tick_params(axis='x', which='major', labelsize=8)

    #                 # 2. Moment at 45% Stance
    #                 axes_bars[1].bar(x_pos, moments_at_45, color=bar_colors)
    #                 axes_bars[1].axhline(0, color='grey', linewidth=0.8)
    #                 axes_bars[1].set_xticks(x_pos)
    #                 axes_bars[1].set_xticklabels(bar_labels, rotation=45, ha='right')
    #                 axes_bars[1].set_title("Sagittal Moment at 45% Stance")
    #                 axes_bars[1].set_ylabel("Normalized Moment (Nm/kg)")
    #                 axes_bars[1].tick_params(axis='x', which='major', labelsize=8)

    #                 # 3. Maximum Moment
    #                 axes_bars[2].bar(x_pos, max_moments, color=bar_colors)
    #                 axes_bars[2].axhline(0, color='grey', linewidth=0.8)
    #                 axes_bars[2].set_xticks(x_pos)
    #                 axes_bars[2].set_xticklabels(bar_labels, rotation=45, ha='right')
    #                 axes_bars[2].set_title("Maximum Sagittal Moment")
    #                 axes_bars[2].set_ylabel("Normalized Moment (Nm/kg)")
    #                 axes_bars[2].tick_params(axis='x', which='major', labelsize=8)
                    
    #                 fig_bars.suptitle(f"{sensor_name} Key Moment Values", fontsize=14)
    #                 fig_bars.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
                    
    #                 # Saving/Showing Bar Plot
    #                 fig_bars.savefig(os.path.join(save_dir, f"{sensor_name}_z_bars_{date_str}_{time_str}.png"), dpi=300)
    #                 plt.show()
    #                 plt.close(fig_bars)


    #             # --- Bar Plots for X-Direction Moment (Coronal) ---
    #             if direction == 'x' and bar_labels:
    #                 bar_colors = [run_colors[label] for label in bar_labels]
    #                 x_pos = np.arange(len(bar_labels))
                    
    #                 fig_bars_x, axes_bars_x = plt.subplots(1, 2, figsize=(10, 5))
                    
    #                 # 1. Moment at 30% Stance
    #                 axes_bars_x[0].bar(x_pos, moments_at_30, color=bar_colors)
    #                 axes_bars_x[0].axhline(0, color='grey', linewidth=0.8)
    #                 axes_bars_x[0].set_xticks(x_pos)
    #                 axes_bars_x[0].set_xticklabels(bar_labels, rotation=45, ha='right')
    #                 axes_bars_x[0].set_title("Coronal Moment at 30% Stance")
    #                 axes_bars_x[0].set_ylabel("Normalized Moment (Nm/kg)")
    #                 axes_bars_x[0].tick_params(axis='x', which='major', labelsize=8)

    #                 # 2. Moment at 75% Stance
    #                 axes_bars_x[1].bar(x_pos, moments_at_75, color=bar_colors)
    #                 axes_bars_x[1].axhline(0, color='grey', linewidth=0.8)
    #                 axes_bars_x[1].set_xticks(x_pos)
    #                 axes_bars_x[1].set_xticklabels(bar_labels, rotation=45, ha='right')
    #                 axes_bars_x[1].set_title("Coronal Moment at 75% Stance")
    #                 axes_bars_x[1].set_ylabel("Normalized Moment (Nm/kg)")
    #                 axes_bars_x[1].tick_params(axis='x', which='major', labelsize=8)
                    
    #                 fig_bars_x.suptitle(f"{sensor_name} Key Coronal Moment Values", fontsize=14)
    #                 fig_bars_x.tight_layout(rect=[0, 0.03, 1, 0.95])
                    
    #                 fig_bars_x.savefig(os.path.join(save_dir, f"socket_moment_x_bars_{date_str}_{time_str}.png"), dpi=300)
    #                 plt.show()
    #                 plt.close(fig_bars_x)



    @staticmethod
    def plot_sensor_force_pylon_all_dirs_stance_sorted_flock(
        all_loaded_data, run_step_data, direction, left_switches, right_switches, folder_name, 
        interp_len=100, body_weight_to_normalize=1, sensor_force_names=None, smooth_data=False, 
        body_in_sensor_names=['pylon', 'socket'], knee_alignment_baseline_file=None, 
        knee_alignment_data_mass=85*9.81, prosthesis_side='left', pylon_optimal_alignment_baseline_file=None
    ):
        """
        Groups seeds by orientation (category) and plots the average across seeds.
        """
        force_direction = {"x": 0, "y": 1, "z": 2}.get(direction, 0)
        index_45_percent = int(np.round(0.45 * (interp_len - 1)))
        index_30_percent = int(np.round(0.30 * (interp_len - 1))) 
        index_75_percent = int(np.round(0.75 * (interp_len - 1))) 

        if sensor_force_names is None:
            # Get names from the first available seed
            first_cat = list(all_loaded_data.keys())[0]
            first_seed = list(all_loaded_data[first_cat].keys())[0]
            sensor_force_names = all_loaded_data[first_cat][first_seed].get("sensor_force_names", [])

        pylon_sensors = [name for name in sensor_force_names if any(b in name for b in body_in_sensor_names)]

        for sensor_name in pylon_sensors:
            if 'torque' not in sensor_name: continue
            
            category_means = [] # To store (cat_key, mean_across_seeds, std_across_seeds)
            bar_plot_data = {}

            # --- 1. Iterate Categories (Orientations) ---
            for cat_key, seeds_dict in all_loaded_data.items():
                all_seeds_processed_curves = []

                # --- 2. Process each seed within this Category ---
                for seed_key, run_dict in seeds_dict.items():
                    if sensor_name not in run_dict.get("all_sensor_force", {}): continue

                    sensor_data = run_dict["all_sensor_force"][sensor_name]
                    side = 'right' if '_r_' in sensor_name else 'left'
                    
                    # Extract Switches for this specific seed
                    sw_source = right_switches if side == 'right' else left_switches
                    sw_container = sw_source.get(cat_key, [])
                    
                    if isinstance(sw_container, dict):
                        current_switches = sw_container.get(seed_key, [])
                    else:
                        current_switches = sw_container

                    if not current_switches: continue

                    # Get step timings
                    step_data = run_step_data.get(cat_key, {}).get(seed_key, {})
                    all_step_start = step_data.get(f"step_start_{side}", [])

                    seed_steps = []
                    if len(all_step_start) > 1:
                        for j in range(len(all_step_start) - 1):
                            start, end = all_step_start[j]-1, all_step_start[j+1]-1
                            if start < 0 or end > len(sensor_data): continue
                            
                            y = [float(np.array(a)[force_direction]) for a in sensor_data[start:end]]
                            if len(y) < 2: continue
                            
                            y_interp = np.interp(np.linspace(0, 1, interp_len), np.linspace(0, 1, len(y)), y)
                            
                            # Define stance end using the mean of available switches
                            sw_val = np.mean(current_switches)
                            sw_idx = int(sw_val) if sw_val < interp_len else interp_len
                            
                            stance = y_interp[:sw_idx]
                            if len(stance) > 1:
                                seed_steps.append(np.interp(np.linspace(0, 1, interp_len), np.linspace(0, 1, len(stance)), stance))

                    if seed_steps:
                        # Store the mean curve for this specific seed
                        all_seeds_processed_curves.append(np.mean(seed_steps, axis=0))

                # --- 3. Aggregate Seeds for this Category ---
                if all_seeds_processed_curves:
                    # Calculate mean across seeds
                    cat_mean = np.mean(all_seeds_processed_curves, axis=0) / body_weight_to_normalize
                    cat_std = np.std(all_seeds_processed_curves, axis=0) / body_weight_to_normalize
                    
                    if force_direction == 2: # Tz inversion
                        cat_mean *= -1
                        cat_std *= -1

                    if smooth_data:
                        cat_mean = gaussian_filter1d(cat_mean, sigma=3)
                        cat_std = gaussian_filter1d(cat_std, sigma=3)

                    category_means.append((cat_key, cat_mean, cat_std))
                    
                    # Prepare bar metrics
                    if direction == 'z':
                        bar_plot_data[cat_key] = {'min': np.min(cat_mean), 'max': np.max(cat_mean), 'at_45': cat_mean[index_45_percent]}
                    elif direction == 'x':
                        bar_plot_data[cat_key] = {'at_30': cat_mean[index_30_percent], 'at_75': cat_mean[index_75_percent]}

            if not category_means: continue

            # --- 4. Sorting & Styling ---
            def extract_numeric(rk):
                m = re.search(r'[-+]?\d*\.?\d+', str(rk))
                return float(m.group()) if m else 0.0

            def _is_reversed(rk):
                rk_l = str(rk).lower()
                return (('z' in rk_l and 'mm' in rk_l) or ('x' in rk_l and 'deg' in rk_l)) or ('x' in rk_l and 'mm' in rk_l)

            sorted_cats = sorted(category_means, key=lambda x: extract_numeric(x[0]))
            
            # Color groups
            vis_neg_keys = [t[0] for t in sorted_cats if (('-' in str(t[0])) != _is_reversed(t[0])) and extract_numeric(t[0]) != 0]

            plt.figure(figsize=(10, 6))
            x_axis = np.linspace(0, 100, interp_len)
            cat_colors = {}

            for cat_key, mean_v, std_v in sorted_cats:
                num = extract_numeric(cat_key)
                if num == 0:
                    color = 'black'
                elif cat_key in vis_neg_keys:
                    norm = 1.0 - np.clip(abs(num) / 15, 0, 0.6)
                    color = matplotlib.colormaps['Blues'](norm)
                else:
                    norm = 1.0 - np.clip(abs(num) / 15, 0, 0.6)
                    color = matplotlib.colormaps['Reds'](norm)
                
                cat_colors[cat_key] = color
                plt.plot(x_axis, mean_v, label=f"{cat_key} (Mean of Seeds)", color=color, linewidth=2.5)
                plt.fill_between(x_axis, mean_v - std_v, mean_v + std_v, color=color, alpha=0.15)



            if direction=='z':
                plane_name = 'Sagittal'
            elif direction=='x':
                plane_name = 'Coronal'


            # plot the alignment baseline data if left_knee in sensor_name and direction == 'z'
            if 'knee' in sensor_name and prosthesis_side in sensor_name and direction == 'z':
                baseline_data = {}
                # read baseline_data from csv file
                if knee_alignment_baseline_file and os.path.isfile(knee_alignment_baseline_file):
                    with open(knee_alignment_baseline_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            if 'Schmalz' in knee_alignment_baseline_file:
                                label_match = re.match(r'^([A-Za-z+\-_]+)\s*=\s*\[?', line)
                            if label_match:
                                current_label = label_match.group(1)
                                baseline_data[current_label] = []
                                continue
                            if current_label:
                                vals = line.split(';')
                                if len(vals) == 2:
                                    try:
                                        x, y = map(float, vals)
                                        baseline_data[current_label].append((x, y))
                                    except ValueError:
                                        continue   
                # if run_keys have z and deg then plot the FOOT_PLA, FOOT_DOR, OPT from baseline
                if 'z' in cat_key and 'deg' in cat_key:
                    if baseline_data.get('FOOT_PLA', []):
                        xs, ys = zip(*baseline_data.get('FOOT_PLA', []))
                        ys = np.array(ys) / knee_alignment_data_mass
                        plt.plot(xs, ys, label="Baseline PLA_-10°", color='lightgray', linestyle='--', linewidth=2)
                    if baseline_data.get('FOOT_DOR', []):
                        xs, ys = zip(*baseline_data.get('FOOT_DOR', []))
                        ys = np.array(ys) / knee_alignment_data_mass
                        plt.plot(xs, ys, label="Baseline DOR_10°", color='darkgrey', linestyle='--', linewidth=2)
                    if baseline_data.get('OPT', []):
                        xs, ys = zip(*baseline_data.get('OPT', []))
                        ys = np.array(ys) / knee_alignment_data_mass
                        plt.plot(xs, ys, label="Baseline OPT", color='dimgray', linestyle='--', linewidth=2)
                elif 'x' in cat_key and 'mm' in cat_key:
                    if baseline_data.get('FOOT_ANT', []):
                        xs, ys = zip(*baseline_data.get('FOOT_ANT', []))
                        ys = np.array(ys) / knee_alignment_data_mass
                        plt.plot(xs, ys, label="Baseline ANT_20mm", color='lightgray', linestyle='--', linewidth=2)
                    if baseline_data.get('FOOT_POS', []):
                        xs, ys = zip(*baseline_data.get('FOOT_POS', []))
                        ys = np.array(ys) / knee_alignment_data_mass
                        plt.plot(xs, ys, label="Baseline POS_-20mm", color='darkgrey', linestyle='--', linewidth=2)
                    if baseline_data.get('OPT', []):
                        xs, ys = zip(*baseline_data.get('OPT', []))
                        ys = np.array(ys) / knee_alignment_data_mass
                        plt.plot(xs, ys, label="Baseline OPT", color='dimgray', linestyle='--', linewidth=2)
                else: 
                    if baseline_data.get('OPT', []):
                        xs, ys = zip(*baseline_data.get('OPT', []))
                        ys = np.array(ys) / knee_alignment_data_mass
                        plt.plot(xs, ys, label="Baseline OPT", color='dimgray', linestyle='--', linewidth=2)



            if 'pylon' in sensor_name and pylon_optimal_alignment_baseline_file:
                baseline_data = {}
                # read baseline_data from csv file
                if os.path.isfile(pylon_optimal_alignment_baseline_file):
                    with open(pylon_optimal_alignment_baseline_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            label_match = re.match(r'^([A-Za-z+\-_]+)\s*=\s*\[?', line)
                            if label_match: 
                                current_label = label_match.group(1)
                                baseline_data[current_label] = []
                                continue
                            if current_label:
                                vals = line.split(';')
                                if len(vals) == 2:
                                    try:
                                        x, y = map(float, vals)
                                        baseline_data[current_label].append((x, y))
                                    except ValueError:
                                        continue  

                if direction == 'z' and baseline_data.get('Sagittal', []):
                    xs, ys = zip(*baseline_data.get('Sagittal', []))
                    # Sort data by x before plotting
                    xs = np.array(xs) * 100  # Convert from 0-1 to 0-100
                    ys = np.array(ys)
                    sort_idx = np.argsort(xs)
                    xs = xs[sort_idx]
                    ys = ys[sort_idx]
                    plt.plot(xs, ys, label="OPT", color='grey', linestyle='-', linewidth=2)
                    if baseline_data.get('Sagittal+std', []) and baseline_data.get('Sagittal-std', []):
                        xs_std_p, ys_std_p = zip(*baseline_data.get('Sagittal+std', []))
                        xs_std_m, ys_std_m = zip(*baseline_data.get('Sagittal-std', []))
                        xs_std_p = np.array(xs_std_p) * 100
                        xs_std_m = np.array(xs_std_m) * 100
                        ys_std_p = np.array(ys_std_p)
                        ys_std_m = np.array(ys_std_m)
                        # Sort std data by x before plotting
                        sort_idx_p = np.argsort(xs_std_p)
                        sort_idx_m = np.argsort(xs_std_m)
                        xs_std_p = xs_std_p[sort_idx_p]
                        ys_std_p = ys_std_p[sort_idx_p]
                        xs_std_m = xs_std_m[sort_idx_m]
                        ys_std_m = ys_std_m[sort_idx_m]
                        # Interpolate ys_std_m and ys_std_p to xs_std_p if lengths mismatch
                        if len(xs_std_p) != len(ys_std_m):
                            ys_std_m_interp = np.interp(xs_std_p, xs_std_m, ys_std_m)
                        else:
                            ys_std_m_interp = ys_std_m
                        if len(xs_std_p) != len(ys_std_p):
                            ys_std_p_interp = np.interp(xs_std_p, xs_std_p, ys_std_p)
                        else:
                            ys_std_p_interp = ys_std_p
                        plt.fill_between(xs_std_p, ys_std_m_interp, ys_std_p_interp, color='lightgrey', alpha=0.6)
                elif direction == 'x' and baseline_data.get('Coronal', []):
                    xs, ys = zip(*baseline_data.get('Coronal', []))
                    # Sort data by x before plotting
                    xs = np.array(xs) * 100  # Convert from 0-1 to 0-100
                    ys = np.array(ys)
                    sort_idx = np.argsort(xs)
                    xs = xs[sort_idx]
                    ys = ys[sort_idx]
                    plt.plot(xs, ys, label="OPT", color='grey', linestyle='-', linewidth=2)
                    if baseline_data.get('Coronal+std', []) and baseline_data.get('Coronal-std', []):
                        xs_std_p, ys_std_p = zip(*baseline_data.get('Coronal+std', []))
                        xs_std_m, ys_std_m = zip(*baseline_data.get('Coronal-std', []))
                        xs_std_p = np.array(xs_std_p) * 100
                        xs_std_m = np.array(xs_std_m) * 100
                        ys_std_p = np.array(ys_std_p)
                        ys_std_m = np.array(ys_std_m)
                        # Sort std data by x before plotting
                        sort_idx_p = np.argsort(xs_std_p)
                        sort_idx_m = np.argsort(xs_std_m)
                        xs_std_p = xs_std_p[sort_idx_p]
                        ys_std_p = ys_std_p[sort_idx_p]
                        xs_std_m = xs_std_m[sort_idx_m]
                        ys_std_m = ys_std_m[sort_idx_m]
                        plt.fill_between(xs_std_p, ys_std_m, ys_std_p, color='lightgrey', alpha=0.6)



            plane = 'Sagittal' if direction == 'z' else 'Coronal'
            plt.title(f"{plane} Moment (Seed Averages): {sensor_name}")
            plt.xlabel("Stance Phase (%)")
            plt.ylabel("Nm/kg")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            os.makedirs(folder_name, exist_ok=True)
            plt.savefig(os.path.join(folder_name, f"{sensor_name}_{direction}_CategoryMean.png"), dpi=300)
            plt.show()

            # --- 5. Summary Bar Plots (Category Level) ---
            if bar_plot_data:
                labels = [t[0] for t in sorted_cats]
                x_pos = np.arange(len(labels))
                colors = [cat_colors[l] for l in labels]
                
                if direction == 'z':
                    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
                    for i, m in enumerate(['min', 'max', 'at_45']):
                        ax[i].bar(x_pos, [bar_plot_data[l][m] for l in labels], color=colors)
                        ax[i].set_title(f"{m.capitalize()} {plane}")
                        ax[i].set_xticks(x_pos)
                        ax[i].set_xticklabels(labels, rotation=45)
                elif direction == 'x':
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                    for i, m in enumerate(['at_30', 'at_75']):
                        ax[i].bar(x_pos, [bar_plot_data[l][m] for l in labels], color=colors)
                        ax[i].set_title(f"Coronal {m}")
                        ax[i].set_xticks(x_pos)
                        ax[i].set_xticklabels(labels, rotation=45)
                plt.tight_layout()
                plt.show()



    @staticmethod
    def plot_sensor_force_pylon_all_dirs_stance_sorted_flock_each_seed(
        all_loaded_data, run_step_data, direction, left_switches, right_switches, folder_name, 
        interp_len=100, body_weight_to_normalize=1, sensor_force_names=None, smooth_data=False, 
        body_in_sensor_names=['pylon', 'socket'], knee_alignment_baseline_file=None, 
        knee_alignment_data_mass=85*9.81, prosthesis_side='left', pylon_optimal_alignment_baseline_file=None
    ):
        force_direction = {"x": 0, "y": 1, "z": 2}.get(direction, 0)
        
        # --- 1. Flatten into an explicit list of every seed ---
        all_seeds_to_plot = []
        for cat_key, seeds_dict in all_loaded_data.items():
            if not isinstance(seeds_dict, dict): continue
            for seed_key, run_dict in seeds_dict.items():
                all_seeds_to_plot.append({
                    'cat_key': cat_key,
                    'seed_key': seed_key,
                    'run_label': f"{cat_key}_{seed_key}",
                    'data': run_dict
                })

        if not all_seeds_to_plot:
            print("No data found to plot.")
            return

        if sensor_force_names is None:
            sensor_force_names = all_seeds_to_plot[0]['data'].get("sensor_force_names", [])

        pylon_sensors = [name for name in sensor_force_names if any(b in name for b in body_in_sensor_names)]

        for sensor_name in pylon_sensors:
            if 'torque' not in sensor_name: continue
            
            plot_results = []

            # --- 2. Process Every Seed Independently ---
            for entry in all_seeds_to_plot:
                cat_key = entry['cat_key']
                seed_key = entry['seed_key']
                run_dict = entry['data']

                if sensor_name not in run_dict.get("all_sensor_force", {}): continue

                sensor_data = run_dict["all_sensor_force"][sensor_name]
                side = 'right' if '_r_' in sensor_name else 'left'
                
                # Switch handling for gait phase
                sw_source = right_switches if side == 'right' else left_switches
                sw_container = sw_source.get(cat_key, [])
                current_switches = sw_container.get(seed_key, []) if isinstance(sw_container, dict) else sw_container

                if not current_switches: continue

                step_timing = run_step_data.get(cat_key, {}).get(seed_key, {})
                starts = step_timing.get(f"step_start_{side}", [])

                steps = []
                if len(starts) > 1:
                    for j in range(len(starts) - 1):
                        s, e = starts[j]-1, starts[j+1]-1
                        if s < 0 or e > len(sensor_data): continue
                        
                        y = [float(np.array(a)[force_direction]) for a in sensor_data[s:e]]
                        if len(y) < 2: continue
                        
                        y_interp = np.interp(np.linspace(0, 1, interp_len), np.linspace(0, 1, len(y)), y)
                        sw_val = np.mean(current_switches)
                        sw_idx = int(sw_val) if sw_val < interp_len else interp_len
                        
                        stance = y_interp[:sw_idx]
                        if len(stance) > 1:
                            steps.append(np.interp(np.linspace(0, 1, interp_len), np.linspace(0, 1, len(stance)), stance))

                if steps:
                    mean_v = np.mean(steps, axis=0) / body_weight_to_normalize
                    std_v = np.std(steps, axis=0) / body_weight_to_normalize
                    
                    if force_direction == 2: mean_v *= -1 # Tz convention
                    if smooth_data: 
                        mean_v = gaussian_filter1d(mean_v, sigma=3)
                        std_v = gaussian_filter1d(std_v, sigma=3)

                    plot_results.append({
                        'label': entry['run_label'],
                        'cat': cat_key,
                        'seed': seed_key,
                        'mean': mean_v,
                        'std': std_v
                    })

            # --- 3. Sorting & Multi-Line Plotting ---
            def extract_numeric(s):
                m = re.search(r'[-+]?\d*\.?\d+', str(s))
                return float(m.group()) if m else 0.0

            def _is_reversed(rk):
                rk_l = str(rk).lower()
                return (('z' in rk_l and 'mm' in rk_l) or ('x' in rk_l and 'deg' in rk_l)) or ('x' in rk_l and 'mm' in rk_l)

            # Sort by category numeric value first, then seed name
            plot_results.sort(key=lambda x: (extract_numeric(x['cat']), x['seed']))

            plt.figure(figsize=(12, 7))
            linestyles = ["-", "--", "-.", ":", (0, (3, 5, 1, 5))]
            x_axis = np.linspace(0, 100, interp_len)
            
            # Track how many seeds we've seen for each category to vary the linestyle
            category_seed_tracker = {}

            for res in plot_results:
                cat = res['cat']
                num = extract_numeric(cat)
                
                # Determine Color based on Category
                is_neg = (('-' in cat) != _is_reversed(cat)) and num != 0
                if num == 0:
                    color = 'black'
                elif is_neg:
                    norm = 1.0 - np.clip(abs(num) / 15, 0, 0.6)
                    color = matplotlib.colormaps['Blues'](norm)
                else:
                    norm = 1.0 - np.clip(abs(num) / 15, 0, 0.6)
                    color = matplotlib.colormaps['Reds'](norm)
                
                # Assign unique linestyle to each seed within the category
                if cat not in category_seed_tracker:
                    category_seed_tracker[cat] = 0
                ls = linestyles[category_seed_tracker[cat] % len(linestyles)]
                category_seed_tracker[cat] += 1

                plt.plot(x_axis, res['mean'], label=res['label'], color=color, linestyle=ls, linewidth=1.5)
                plt.fill_between(x_axis, res['mean']-res['std'], res['mean']+res['std'], color=color, alpha=0.05)


            if direction=='z':
                plane_name = 'Sagittal'
            elif direction=='x':
                plane_name = 'Coronal'


            # plot the alignment baseline data if left_knee in sensor_name and direction == 'z'
            if 'knee' in sensor_name and prosthesis_side in sensor_name and direction == 'z':
                baseline_data = {}
                # read baseline_data from csv file
                if knee_alignment_baseline_file and os.path.isfile(knee_alignment_baseline_file):
                    with open(knee_alignment_baseline_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            if 'Schmalz' in knee_alignment_baseline_file:
                                label_match = re.match(r'^([A-Za-z+\-_]+)\s*=\s*\[?', line)
                            if label_match:
                                current_label = label_match.group(1)
                                baseline_data[current_label] = []
                                continue
                            if current_label:
                                vals = line.split(';')
                                if len(vals) == 2:
                                    try:
                                        x, y = map(float, vals)
                                        baseline_data[current_label].append((x, y))
                                    except ValueError:
                                        continue   
                # if run_keys have z and deg then plot the FOOT_PLA, FOOT_DOR, OPT from baseline
                if 'z' in cat_key and 'deg' in cat_key:
                    if baseline_data.get('FOOT_PLA', []):
                        xs, ys = zip(*baseline_data.get('FOOT_PLA', []))
                        ys = np.array(ys) / knee_alignment_data_mass
                        plt.plot(xs, ys, label="Baseline PLA_-10°", color='lightgray', linestyle='--', linewidth=2)
                    if baseline_data.get('FOOT_DOR', []):
                        xs, ys = zip(*baseline_data.get('FOOT_DOR', []))
                        ys = np.array(ys) / knee_alignment_data_mass
                        plt.plot(xs, ys, label="Baseline DOR_10°", color='darkgrey', linestyle='--', linewidth=2)
                    if baseline_data.get('OPT', []):
                        xs, ys = zip(*baseline_data.get('OPT', []))
                        ys = np.array(ys) / knee_alignment_data_mass
                        plt.plot(xs, ys, label="Baseline OPT", color='dimgray', linestyle='--', linewidth=2)
                elif 'x' in cat_key and 'mm' in cat_key:
                    if baseline_data.get('FOOT_ANT', []):
                        xs, ys = zip(*baseline_data.get('FOOT_ANT', []))
                        ys = np.array(ys) / knee_alignment_data_mass
                        plt.plot(xs, ys, label="Baseline ANT_20mm", color='lightgray', linestyle='--', linewidth=2)
                    if baseline_data.get('FOOT_POS', []):
                        xs, ys = zip(*baseline_data.get('FOOT_POS', []))
                        ys = np.array(ys) / knee_alignment_data_mass
                        plt.plot(xs, ys, label="Baseline POS_-20mm", color='darkgrey', linestyle='--', linewidth=2)
                    if baseline_data.get('OPT', []):
                        xs, ys = zip(*baseline_data.get('OPT', []))
                        ys = np.array(ys) / knee_alignment_data_mass
                        plt.plot(xs, ys, label="Baseline OPT", color='dimgray', linestyle='--', linewidth=2)
                else: 
                    if baseline_data.get('OPT', []):
                        xs, ys = zip(*baseline_data.get('OPT', []))
                        ys = np.array(ys) / knee_alignment_data_mass
                        plt.plot(xs, ys, label="Baseline OPT", color='dimgray', linestyle='--', linewidth=2)



            if 'pylon' in sensor_name and pylon_optimal_alignment_baseline_file:
                baseline_data = {}
                # read baseline_data from csv file
                if os.path.isfile(pylon_optimal_alignment_baseline_file):
                    with open(pylon_optimal_alignment_baseline_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            label_match = re.match(r'^([A-Za-z+\-_]+)\s*=\s*\[?', line)
                            if label_match: 
                                current_label = label_match.group(1)
                                baseline_data[current_label] = []
                                continue
                            if current_label:
                                vals = line.split(';')
                                if len(vals) == 2:
                                    try:
                                        x, y = map(float, vals)
                                        baseline_data[current_label].append((x, y))
                                    except ValueError:
                                        continue  

                if direction == 'z' and baseline_data.get('Sagittal', []):
                    xs, ys = zip(*baseline_data.get('Sagittal', []))
                    # Sort data by x before plotting
                    xs = np.array(xs) * 100  # Convert from 0-1 to 0-100
                    ys = np.array(ys)
                    sort_idx = np.argsort(xs)
                    xs = xs[sort_idx]
                    ys = ys[sort_idx]
                    plt.plot(xs, ys, label="OPT", color='grey', linestyle='-', linewidth=2)
                    if baseline_data.get('Sagittal+std', []) and baseline_data.get('Sagittal-std', []):
                        xs_std_p, ys_std_p = zip(*baseline_data.get('Sagittal+std', []))
                        xs_std_m, ys_std_m = zip(*baseline_data.get('Sagittal-std', []))
                        xs_std_p = np.array(xs_std_p) * 100
                        xs_std_m = np.array(xs_std_m) * 100
                        ys_std_p = np.array(ys_std_p)
                        ys_std_m = np.array(ys_std_m)
                        # Sort std data by x before plotting
                        sort_idx_p = np.argsort(xs_std_p)
                        sort_idx_m = np.argsort(xs_std_m)
                        xs_std_p = xs_std_p[sort_idx_p]
                        ys_std_p = ys_std_p[sort_idx_p]
                        xs_std_m = xs_std_m[sort_idx_m]
                        ys_std_m = ys_std_m[sort_idx_m]
                        # Interpolate ys_std_m and ys_std_p to xs_std_p if lengths mismatch
                        if len(xs_std_p) != len(ys_std_m):
                            ys_std_m_interp = np.interp(xs_std_p, xs_std_m, ys_std_m)
                        else:
                            ys_std_m_interp = ys_std_m
                        if len(xs_std_p) != len(ys_std_p):
                            ys_std_p_interp = np.interp(xs_std_p, xs_std_p, ys_std_p)
                        else:
                            ys_std_p_interp = ys_std_p
                        plt.fill_between(xs_std_p, ys_std_m_interp, ys_std_p_interp, color='lightgrey', alpha=0.6)
                elif direction == 'x' and baseline_data.get('Coronal', []):
                    xs, ys = zip(*baseline_data.get('Coronal', []))
                    # Sort data by x before plotting
                    xs = np.array(xs) * 100  # Convert from 0-1 to 0-100
                    ys = np.array(ys)
                    sort_idx = np.argsort(xs)
                    xs = xs[sort_idx]
                    ys = ys[sort_idx]
                    plt.plot(xs, ys, label="OPT", color='grey', linestyle='-', linewidth=2)
                    if baseline_data.get('Coronal+std', []) and baseline_data.get('Coronal-std', []):
                        xs_std_p, ys_std_p = zip(*baseline_data.get('Coronal+std', []))
                        xs_std_m, ys_std_m = zip(*baseline_data.get('Coronal-std', []))
                        xs_std_p = np.array(xs_std_p) * 100
                        xs_std_m = np.array(xs_std_m) * 100
                        ys_std_p = np.array(ys_std_p)
                        ys_std_m = np.array(ys_std_m)
                        # Sort std data by x before plotting
                        sort_idx_p = np.argsort(xs_std_p)
                        sort_idx_m = np.argsort(xs_std_m)
                        xs_std_p = xs_std_p[sort_idx_p]
                        ys_std_p = ys_std_p[sort_idx_p]
                        xs_std_m = xs_std_m[sort_idx_m]
                        ys_std_m = ys_std_m[sort_idx_m]
                        plt.fill_between(xs_std_p, ys_std_m, ys_std_p, color='lightgrey', alpha=0.6)



            plane = 'Sagittal' if direction == 'z' else 'Coronal'
            plt.title(f"{plane} Moment (All Seeds): {sensor_name}")
            plt.xlabel("Stance Phase (%)")
            plt.ylabel("Nm/kg")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small', ncol=2)
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            os.makedirs(folder_name, exist_ok=True)
            plt.savefig(os.path.join(folder_name, f"{sensor_name}_{direction}_AllSeeds.png"), dpi=300)
            plt.show()




    @staticmethod
    def plot_sensor_force_pylon_steps_per_run_stance(
        all_loaded_data, run_step_data, direction, left_switches, right_switches, folder_name, interp_len=100, body_weight_to_normalize=1, sensor_force_names=None, smooth_data=False
    ):
        """
        Plot moment data over the stance phase for each run in a separate plot.
        Individual steps are shown in unique colors and are fully labeled
        as "Step 1", "Step 2", etc., in the legend. (CAUTION: Legend may be large).

        direction: "x", "y", or "z" to specify the force direction. [0, 1, 2] for x, y, z respectively.
        """
        
        force_direction = {"x": 0, "y": 1, "z": 2}.get(direction, 0)
        
        # Helper: try to get a numeric value from run_key (kept from original)
        def extract_numeric(run_key):
            try:
                return float(run_key)
            except Exception:
                m = re.search(r'[-+]?\d*\.?\d+', str(run_key))
                if m:
                    try:
                        return float(m.group())
                    except Exception:
                        return None
                return None

        # Helper to parse run key for sorting (kept from original)
        def parse_run_key_for_sort(run_key):
            num = extract_numeric(run_key)
            return num if num is not None else str(run_key)

        # Use sensor_force_names from the first run if not provided
        if sensor_force_names is None:
            run_keys = list(all_loaded_data.keys())
            if not run_keys:
                 return # No data
            run = run_keys[0]
            sensor_force_names = all_loaded_data[run]["sensor_force_names"]

        # Only keep sensors with 'pylon' or 'socket' AND 'torque' in their name
        moment_sensors = [name for name in sensor_force_names if ("pylon" in name or "socket" in name) and "torque" in name]
        
        run_keys = list(all_loaded_data.keys())
        sorted_run_keys = sorted(run_keys, key=parse_run_key_for_sort)

        # Pre-define a large set of distinct colors for individual steps
        tab20_colors = matplotlib.cm.get_cmap('tab20').colors
        all_step_colors = list(tab20_colors) * 3 + list(mcolors.TABLEAU_COLORS.values())
        
        for sensor_name in moment_sensors:
            
            if direction == 'z':
                plane_name = 'Sagittal'
            elif direction == 'x':
                plane_name = 'Coronal'
            elif direction == 'y':
                plane_name = 'Transverse'
            else:
                plane_name = 'Moment'

            # --- Iterate over each run to create a separate plot ---
            for run_key in sorted_run_keys:
                run_dict = all_loaded_data.get(run_key, {})
                step_data = run_step_data.get(run_key, {})

                if sensor_name not in run_dict.get("all_sensor_force", {}):
                    continue  # Skip this run if sensor data is missing

                sensor_data = run_dict["all_sensor_force"][sensor_name]

                # Determine side and switches
                if sensor_name.endswith("_r"):
                    all_step_start = step_data.get("step_start_right", [])
                    switches = right_switches.get(run_key, [])
                else:
                    all_step_start = step_data.get("step_start_left", [])
                    switches = left_switches.get(run_key, [])

                steps = []
                # List to hold the unique color for each step
                step_colors = [] 
                
                if all_step_start and len(all_step_start) > 1 and switches:
                    
                    # 1. Process and interpolate steps to get stance phase data
                    step_count = 0
                    for j in range(len(all_step_start) - 1):
                        start, end = all_step_start[j]-1, all_step_start[j + 1]-1
                        y = [float(np.array(a)[force_direction]) for a in sensor_data[start:end]]
                        
                        if len(y) < 2:
                            continue

                        # Interpolate to stance phase
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        switch_idx = switches[0] if switches and switches[0] < interp_len else interp_len
                        stance_phase = y_interp[:switch_idx]
                        
                        if len(stance_phase) > 1:
                            x_stance_old = np.linspace(0, 1, len(stance_phase))
                            x_stance_new = np.linspace(0, 1, interp_len)
                            stance_phase_interp = np.interp(x_stance_new, x_stance_old, stance_phase)
                            
                            normalized_step = stance_phase_interp / body_weight_to_normalize
                            if force_direction == 2:
                                normalized_step *= -1 

                            steps.append(normalized_step)
                            
                            # Assign unique color for this step
                            color_index = step_count % len(all_step_colors)
                            step_colors.append(all_step_colors[color_index])
                            step_count += 1


                # 2. Plotting (if steps exist for this run)
                if steps:
                    
                    steps_array = np.array(steps)
                    mean_val = np.mean(steps_array, axis=0)
                    std_val = np.std(steps_array, axis=0)

                    # Smooth mean and std if requested
                    if smooth_data:
                        mean_val = gaussian_filter1d(mean_val, sigma=4)

                    x = np.linspace(0, 100, interp_len)
                    
                    plt.figure(figsize=(10, 6))
                    ax = plt.gca()
                    
                    # Plot all individual steps with unique colors AND the desired numerical label
                    for i, step in enumerate(steps):
                        # The label is set here as requested ("Step 1", "Step 2", etc.)
                        ax.plot(x, step, color=step_colors[i], alpha=0.5, linewidth=1, 
                                label=f"Step {i+1}") 
                        
                    # Plot mean and shaded std dev
                    mean_color = 'black'
                    ax.plot(x, mean_val, label=f"Mean (N={len(steps)})", color=mean_color, linestyle='-', linewidth=2)
                    ax.fill_between(x, mean_val - std_val, mean_val + std_val, alpha=0.15, color=mean_color, label='± 1 SD')

                    print(f"Run: {run_key}, Sensor: {sensor_name}, Std Median: {np.median(std_val):.3f}, Std Mean: {np.mean(std_val):.3f}, Std Max: {np.max(std_val):.3f}, Std Min: {np.min(std_val):.3f}")

                    # Set titles and labels
                    sensor_type = "Pylon" if "pylon" in sensor_name else "Socket"
                    ax.set_title(f"Run: {run_key} - {plane_name} {sensor_type} Moment (N={len(steps)} Steps)")
                    ax.set_xlabel("Stance Phase (%)")
                    ax.set_ylabel(f"Normalized {plane_name} Moment (Nm/kg)")
                    
                    # ----------------------------------------------------------------------
                    # NO LEGEND FILTERING HERE: Matplotlib will now display all individual 
                    # step labels ("Step 1", "Step 2", etc.), Mean, and ± 1 SD.
                    # ----------------------------------------------------------------------
                    ax.legend(loc='best', title=f"Run: {run_key} Legend", ncol=2) # Use ncol to save space

                    ax.grid(True, linestyle='--', alpha=0.6)
                    plt.tight_layout()

                    # Save figure
                    date_str = datetime.now().strftime("%Y%m%d")
                    time_str = datetime.now().strftime("%H%M%S")
                    
                    run_safe_name = run_key.replace('.', '_').replace('-', 'neg')
                    save_dir = os.path.join(folder_name)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    filename = f"{sensor_type.lower()}_moment_{plane_name.lower()}_{run_safe_name}_{date_str}_{time_str}.png"
                    plt.savefig(os.path.join(save_dir, filename), dpi=300)
                    plt.show()
                    plt.close()

    


    @staticmethod
    def plot_sensor_force_symmetry_all_runs_stance(
        all_loaded_data, run_step_data, direction, right_switches, left_switches, interp_len=100, body_weight_to_normalize=1, sensor_force_names=None
    ):
        """
        Plot mean of right of all runs, mean of left, and the difference, side by side in one figure.
        Each plot includes all runs.
        Only plot the stance phase, i.e., interpolate each step from 0 to the first entry of left_switches/right_switches for each run.
        direction: "x", "y", or "z" to specify the force direction. [0, 1, 2] for x, y, z respectively.
        """
        force_direction = {"x": 0, "y": 1, "z": 2}.get(direction, 0)

        if sensor_force_names is None:
            first_run = next(iter(all_loaded_data))
            sensor_force_names = all_loaded_data[first_run]["sensor_force_names"]

        # Find sensor pairs (left/right)
        sensor_pairs = []
        for name in sensor_force_names:
            if name.startswith("left_"):
                right_name = name.replace("left_", "right_")
                if right_name in sensor_force_names:
                    sensor_pairs.append((name, right_name))

        for left_sensor, right_sensor in sensor_pairs:
            run_keys = list(all_loaded_data.keys())
            mean_left_all = []
            std_left_all = []
            mean_right_all = []
            std_right_all = []
            diff_all = []
            max_len = 0
            left_steps_all = []
            right_steps_all = []
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                step_data = run_step_data[run_key]
                left_data = run_dict["all_sensor_force"][left_sensor]
                right_data = run_dict["all_sensor_force"][right_sensor]
                all_step_start_left = step_data["step_start_left"]
                all_step_start_right = step_data["step_start_right"]

                # Use first entry of left_switches/right_switches for stance phase
                switches_left = left_switches.get(run_key, [])
                switches_right = right_switches.get(run_key, [])

                left_steps = []
                if all_step_start_left and len(all_step_start_left) > 1 and switches_left:
                    for j in range(len(all_step_start_left) - 1):
                        start = all_step_start_left[j]
                        end = all_step_start_left[j + 1]
                        y = [float(np.array(a)[force_direction]) for a in left_data[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        # Only plot up to the first entry of switches_left (stance phase)
                        switch_idx = switches_left[0] if switches_left and switches_left[0] < interp_len else interp_len
                        stance_phase = y_interp[:switch_idx]
                        # Re-interpolate stance phase to interp_len
                        if len(stance_phase) > 1:
                            x_stance_old = np.linspace(0, 1, len(stance_phase))
                            x_stance_new = np.linspace(0, 1, interp_len)
                            stance_phase_interp = np.interp(x_stance_new, x_stance_old, stance_phase)
                            left_steps.append(stance_phase_interp)
                        max_len = interp_len
                right_steps = []
                if all_step_start_right and len(all_step_start_right) > 1 and switches_right:
                    for j in range(len(all_step_start_right) - 1):
                        start = all_step_start_right[j]
                        end = all_step_start_right[j + 1]
                        y = [float(np.array(a)[force_direction]) for a in right_data[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        switch_idx = switches_right[0] if switches_right and switches_right[0] < interp_len else interp_len
                        stance_phase = y_interp[:switch_idx]
                        if len(stance_phase) > 1:
                            x_stance_old = np.linspace(0, 1, len(stance_phase))
                            x_stance_new = np.linspace(0, 1, interp_len)
                            stance_phase_interp = np.interp(x_stance_new, x_stance_old, stance_phase)
                            right_steps.append(stance_phase_interp)
                        max_len = interp_len
                # All stance phases are now re-interpolated to interp_len, so no need to pad
                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0) / body_weight_to_normalize
                    std_left = np.std(left_steps, axis=0) / body_weight_to_normalize
                    mean_right = np.mean(right_steps, axis=0) / body_weight_to_normalize
                    std_right = np.std(right_steps, axis=0) / body_weight_to_normalize
                    if 'knee' in left_sensor:
                        mean_left = -mean_left
                        mean_right = -mean_right
                        std_left = -std_left
                        std_right = -std_right
                    diff = mean_left - mean_right
                    mean_left_all.append((run_key, mean_left, std_left))
                    mean_right_all.append((run_key, mean_right, std_right))
                    diff_all.append((run_key, diff))
                else:
                    mean_left_all.append((run_key, None, None))
                    mean_right_all.append((run_key, None, None))
                    diff_all.append((run_key, None))
            # Plot with fixed interp_len for all runs
            x = np.linspace(0, 100, interp_len)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

            # Plot mean of right of all runs
            for run_key, mean_right, std_right in mean_right_all:
                if mean_right is not None:
                    axes[0].plot(x, mean_right, label=f"{run_key} {right_sensor} mean")
                    axes[0].fill_between(x, mean_right - std_right, mean_right + std_right, alpha=1)
            axes[0].set_title(f"{right_sensor} mean (stance phase, all runs)")
            axes[0].set_xlabel("Interpolated Step (%)")
            axes[0].set_ylabel("Force Value")

            # Plot mean of left of all runs
            for run_key, mean_left, std_left in mean_left_all:
                if mean_left is not None:
                    axes[1].plot(x, mean_left, label=f"{run_key} {left_sensor} mean")
                    axes[1].fill_between(x, mean_left - std_left, mean_left + std_left, alpha=1)
            axes[1].set_title(f"{left_sensor} mean (stance phase, all runs)")
            axes[1].set_xlabel("Interpolated Step (%)")
            axes[1].set_ylabel("Force Value")

            # Plot difference (left - right) of all runs
            for run_key, diff in diff_all:
                if diff is not None:
                    axes[2].plot(x, diff, label=f"{run_key} Left-Right")
            axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
            axes[2].set_title(f"{left_sensor} - {right_sensor} difference (stance phase, all runs)")
            axes[2].set_xlabel("Interpolated Step (%)")
            axes[2].set_ylabel("Force Value Difference")
            axes[2].legend()

            plt.tight_layout()
            plt.show()
            plt.close()



    @staticmethod
    def plot_sensor_force_symmetry_all_runs_stance_compare_baseline(
        all_loaded_data, run_step_data, direction, right_switches, left_switches, interp_len=100, body_weight_to_normalize=1, sensor_force_names=None, baseline_data=None
    ):
        """
        Plot mean of right of all runs, mean of left, and the difference, side by side in one figure.
        Only plot the stance phase for the knee_mimic_torque sensor pair.
        Optionally plot baseline_data for knee_z.
        """
        mass_norm = 85*10 # std 15

        force_direction = {"x": 0, "y": 1, "z": 2}.get(direction, 0)

        # Only plot knee_mimic_torque pair
        left_sensor = "left_knee_mimic_torque_sensor"
        right_sensor = "right_knee_mimic_torque_sensor"

        run_keys = list(all_loaded_data.keys())
        mean_left_all = []
        std_left_all = []
        mean_right_all = []
        std_right_all = []
        diff_all = []
        max_len = 0
        left_steps_all = []
        right_steps_all = []
        for run_key in run_keys:
            run_dict = all_loaded_data[run_key]
            step_data = run_step_data[run_key]
            left_data = run_dict["all_sensor_force"][left_sensor]
            right_data = run_dict["all_sensor_force"][right_sensor]
            all_step_start_left = step_data["step_start_left"]
            all_step_start_right = step_data["step_start_right"]

            # Use first entry of left_switches/right_switches for stance phase
            switches_left = left_switches.get(run_key, [])
            switches_right = right_switches.get(run_key, [])

            left_steps = []
            if all_step_start_left and len(all_step_start_left) > 1 and switches_left:
                for j in range(len(all_step_start_left) - 1):
                    start = all_step_start_left[j]
                    end = all_step_start_left[j + 1]
                    y = [float(np.array(a)[force_direction]) for a in left_data[start:end]]
                    if len(y) < 2:
                        continue
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    y_interp = np.interp(x_new, x_old, y)
                    switch_idx = switches_left[0] if switches_left and switches_left[0] < interp_len else interp_len
                    stance_phase = y_interp[:switch_idx]
                    if len(stance_phase) > 1:
                        x_stance_old = np.linspace(0, 1, len(stance_phase))
                        x_stance_new = np.linspace(0, 1, interp_len)
                        stance_phase_interp = np.interp(x_stance_new, x_stance_old, stance_phase)
                        left_steps.append(stance_phase_interp)
                    max_len = interp_len
            right_steps = []
            if all_step_start_right and len(all_step_start_right) > 1 and switches_right:
                for j in range(len(all_step_start_right) - 1):
                    start = all_step_start_right[j]
                    end = all_step_start_right[j + 1]
                    y = [float(np.array(a)[force_direction]) for a in right_data[start:end]]
                    if len(y) < 2:
                        continue
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    y_interp = np.interp(x_new, x_old, y)
                    switch_idx = switches_right[0] if switches_right and switches_right[0] < interp_len else interp_len
                    stance_phase = y_interp[:switch_idx]
                    if len(stance_phase) > 1:
                        x_stance_old = np.linspace(0, 1, len(stance_phase))
                        x_stance_new = np.linspace(0, 1, interp_len)
                        stance_phase_interp = np.interp(x_stance_new, x_stance_old, stance_phase)
                        right_steps.append(stance_phase_interp)
                    max_len = interp_len
            if left_steps and right_steps:
                mean_left = np.mean(left_steps, axis=0) / body_weight_to_normalize
                std_left = np.std(left_steps, axis=0) / body_weight_to_normalize
                mean_right = np.mean(right_steps, axis=0) / body_weight_to_normalize
                std_right = np.std(right_steps, axis=0) / body_weight_to_normalize
                mean_left = -mean_left
                mean_right = -mean_right
                std_left = -std_left
                std_right = -std_right
                diff = mean_left - mean_right
                mean_left_all.append((run_key, mean_left, std_left))
                mean_right_all.append((run_key, mean_right, std_right))
                diff_all.append((run_key, diff))
            else:
                mean_left_all.append((run_key, None, None))
                mean_right_all.append((run_key, None, None))
                diff_all.append((run_key, None))
        x = np.linspace(0, 100, interp_len)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

        # Plot mean of right of all runs
        for run_key, mean_right, std_right in mean_right_all:
            if mean_right is not None:
                axes[0].plot(x, mean_right, label=f"{run_key} {right_sensor} mean")
                axes[0].fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
        axes[0].set_title(f"{right_sensor} mean (stance phase, all runs)")
        axes[0].set_xlabel("Interpolated Step (%)")
        axes[0].set_ylabel("Force Value")

        # Plot mean of left of all runs
        for run_key, mean_left, std_left in mean_left_all:
            if mean_left is not None:
                axes[1].plot(x, mean_left, label=f"{run_key} {left_sensor} mean")
                axes[1].fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
        axes[1].set_title(f"{left_sensor} mean (stance phase, all runs)")
        axes[1].set_xlabel("Interpolated Step (%)")
        axes[1].set_ylabel("Force Value")

        # Plot difference (left - right) of all runs
        for run_key, diff in diff_all:
            if diff is not None:
                axes[2].plot(x, diff, label=f"{run_key} Left-Right")
        axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
        axes[2].set_title(f"{left_sensor} - {right_sensor} difference (stance phase, all runs)")
        axes[2].set_xlabel("Interpolated Step (%)")
        axes[2].set_ylabel("Force Value Difference")
        axes[2].legend()

        # Plot baseline_data for knee_z if provided
        if baseline_data is not None and direction == "z":

            data = {}
            current_label = None
            with open(baseline_data, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if 'Schmalz' in baseline_data:
                        label_match = re.match(r'^([A-Z_]+)\s*=\s*\[?', line)
                    else:
                        label_match = re.match(r'^([A-Za-z_]+)\s*=\s*\[?', line)
                    if label_match:
                        current_label = label_match.group(1)
                        data[current_label] = []
                        continue
                    if current_label:
                        vals = line.split(';')
                    if len(vals) == 2:
                        try:
                            x, y = map(float, vals)
                            data[current_label].append((x, y))
                        except ValueError:
                            continue
                for ax in axes:
                    for label, points in data.items():
                        if label == "OPT" and points:
                            xs, ys = zip(*points)
                            ys = np.array(ys)/mass_norm
                            ax.plot(xs, ys, marker='.', linestyle='-', color='black', label=f"Baseline {label}")
                ax.legend()

        plt.tight_layout()
        plt.show()
        plt.close()





    @staticmethod
    def plot_sensor_torque_symmetry_baseline_comp(
        all_loaded_data,
        run_step_data,
        direction,
        baseline_file,
        baseline_file_2,
        baseline_file_3,
        interp_len=100,
        body_weight_to_normalize=1,
        mass_baseline=97.5,
        mass_baseline_2=75 * 9.81,
        mass_baseline_3 = 85 * 9.81, 
        sensor_force_names=None, # Optional, but good practice
        right_switches = None, 
        left_switches = None,
    ):
        """
        Plot mean of right of all runs, mean of left, and the difference for TORQUE sensors,
        side by side in one figure, including two external baselines.
        direction: "x", "y", or "z" to specify the torque axis. [0, 1, 2] for x, y, z respectively.
        """
        # --- Initialization ---
        direction_map = {"x": 0, "y": 1, "z": 2}
        force_direction = direction_map.get(direction, 0)
        
        # Use sensor_force_names from the first run if not provided
        if sensor_force_names is None:
            first_run = next(iter(all_loaded_data))
            sensor_force_names = all_loaded_data[first_run].get("sensor_force_names", [])

        # Find sensor pairs, filtering for 'torque' in the name
        sensor_pairs = []
        for name in sensor_force_names:
            if 'torque' in name.lower() and name.startswith("left_"):
                right_name = name.replace("left_", "right_")
                if right_name in sensor_force_names:
                    sensor_pairs.append((name, right_name))

        # --- Baseline Loading ---
        baseline_data = {}
        baseline_data_2 = {}
        baseline_data_3 = {}
        
        # Helper for loading baseline (fixed for value stripping and error handling)
        def load_baseline(file_path, data_dict):
            current_label = None
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check for label start
                    if 'Schmalz' in file_path:
                        label_match = re.match(r'^([A-Z_]+)\s*=\s*\[?', line)
                    else:
                        label_match = re.match(r'^([A-Za-z_]+)\s*=\s*\[?', line)
                    
                    if label_match:
                        current_label = label_match.group(1)
                        data_dict[current_label] = []
                        continue
                        
                    # Check for end of list (or just skip if no current_label)
                    if line.endswith(']'):
                        current_label = None
                        continue

                    if current_label:
                        vals = line.split(';')
                        if len(vals) == 2:
                            try:
                                x, y = map(float, [v.strip() for v in vals])
                                data_dict[current_label].append((x, y))
                            except ValueError:
                                continue

        load_baseline(baseline_file, baseline_data)
        load_baseline(baseline_file_2, baseline_data_2)
        load_baseline(baseline_file_3, baseline_data_3)
        
        # --- Main Plotting Loop (Iterates over torque sensor pairs) ---
        for left_sensor, right_sensor in sensor_pairs:
            base_sensor_name_lower = left_sensor.replace("left_", "").replace("_sensor", "").lower()
            
            # Prepare data for all runs
            run_keys = list(all_loaded_data.keys())
            mean_left_all, std_left_all, mean_right_all, std_right_all, diff_all = [], [], [], [], []
            
            # Helper function for data extraction and interpolation
            def collect_steps(data, indices):
                steps = []
                if indices and len(indices) > 1:
                    for j in range(len(indices) - 1):
                        start, end = indices[j], indices[j + 1]
                        # Extract data for the specified direction (torque is a vector)
                        y = [float(np.array(a)[force_direction]) for a in data[start:end]]
                        if len(y) < 2:
                            continue
                        
                        # Interpolation logic
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        steps.append(y_interp)
                return steps
            
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                step_data = run_step_data[run_key]
                
                # Access sensor data
                left_data = run_dict.get("all_sensor_force", {}).get(left_sensor, None)
                right_data = run_dict.get("all_sensor_force", {}).get(right_sensor, None)
                
                if left_data is None or right_data is None:
                    continue
                    
                all_step_start_left = step_data["step_start_left"]
                all_step_start_right = step_data["step_start_right"]

                left_steps = collect_steps(left_data, all_step_start_left)
                right_steps = collect_steps(right_data, all_step_start_right)
                
                # Compute means and stds
                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0) / body_weight_to_normalize
                    std_left = np.std(left_steps, axis=0) / body_weight_to_normalize
                    mean_right = np.mean(right_steps, axis=0) / body_weight_to_normalize
                    std_right = np.std(right_steps, axis=0) / body_weight_to_normalize
                    
                    # Apply sign convention for knee (or other specific sensors)
                    if 'knee' in left_sensor.lower():
                        mean_left = -mean_left
                        mean_right = -mean_right
                        # Note: Do not flip sign for standard deviation (std is always positive)
                        
                    diff = mean_left - mean_right # Symmetry difference
                    
                    mean_left_all.append((run_key, mean_left, std_left))
                    mean_right_all.append((run_key, mean_right, std_right))
                    diff_all.append((run_key, diff))
                else:
                    mean_left_all.append((run_key, None, None))
                    mean_right_all.append((run_key, None, None))
                    diff_all.append((run_key, None))

            # --- Plotting ---
            x = np.linspace(0, 100, interp_len)
            
            if mean_left_all or mean_right_all or diff_all:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
                
                # Plot 1: Right Sensor (axes[0]) - Intact Baseline
                for run_key, mean_right, std_right in mean_right_all:
                    if mean_right is not None:
                        axes[0].plot(x, mean_right, label=f"{run_key}")
                        axes[0].fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
                
                # Helper for plotting baseline data
                def plot_baseline_torque(ax, data_dict, file_path, mass, label_suffix, line_style, line_color, side_label):
                    for label, points in data_dict.items():
                        label_lower = label.lower()
                        if 'ankle' in label_lower:
                            label_lower = label_lower.replace('ankle', 'foot')
                        label_joint = label_lower.split('_')[0]
                        # Match baseline: must contain base sensor name AND 'intact'
                        if 'intact' in label_lower and label_joint in base_sensor_name_lower  and points:
                            xs, ys = zip(*points)
                            ys = np.array(ys) * -1 # Apply common torque sign flip
                            if 'Banks' in file_path:
                                ys = np.array(ys) / mass
                                if 'knee' not in base_sensor_name_lower: # Apply specific normalization flip
                                    ys = np.array(ys) * -1

                            baseline_label = 'Turcot et al.' if 'Turcot' in file_path else 'Banks et al.'
                            ax.plot(xs, ys, marker='o', linestyle=line_style, color=line_color,
                                    label=f"{baseline_label} {label_suffix}")
                

                # Plot Baselines on Right Sensor (Intact side)
                plot_baseline_torque(axes[0], baseline_data, baseline_file, mass_baseline, 
                                     "(Intact 1)", '-', 'black', right_sensor)
                plot_baseline_torque(axes[0], baseline_data_2, baseline_file_2, mass_baseline_2, 
                                     "(Intact 2)", '--', 'grey', right_sensor)


                axes[0].set_title(f"{right_sensor} ({direction}-axis) mean (all runs)")
                axes[0].set_xlabel("Interpolated Step (%)")
                axes[0].set_ylabel(f"Torque ({'Nm/kg' if body_weight_to_normalize != 1 else 'Nm'})")
                
                # Plot 2: Left Sensor (axes[1]) - Prosthetic Baseline
                for run_key, mean_left, std_left in mean_left_all:
                    if mean_left is not None:
                        axes[1].plot(x, mean_left, label=f"{run_key}")
                        axes[1].fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)

                # Helper for plotting baseline data (Prosthesis/impaired Side)
                def plot_baseline_torque_impaired(ax, data_dict, file_path, mass, label_suffix, line_style, line_color, side_label):
                    for label, points in data_dict.items():
                        label_lower = label.lower()
                        if 'ankle' in label_lower:
                            label_lower = label_lower.replace('ankle', 'foot')
                        # Match baseline: must contain base sensor name AND NOT 'intact'
                        if 'intact' not in label_lower and label_lower in base_sensor_name_lower and points:
                            xs, ys = zip(*points)
                            ys = np.array(ys) * -1 # Apply common torque sign flip
                            if 'Banks' in file_path:
                                ys = np.array(ys) / mass
                                if 'knee' not in base_sensor_name_lower:
                                    ys = np.array(ys) * -1
                            
                            baseline_label = 'Turcot et al.' if 'Turcot' in file_path else 'Banks et al.'
                            ax.plot(xs, ys, marker='o', linestyle=line_style, color=line_color, 
                                    label=f"{baseline_label} {label_suffix}")
                
                def plot_baseline_torque_Schmalz(ax, data_dict, file_path, mass, label_suffix, line_style, line_color, side_label):
                    for label, points in data_dict.items():
                        label_lower = label.lower()
                        if 'knee' in label_lower:
                            label_lower = label_lower.replace('knee', 'thigh')
                        # Plot OPT for knee from Schmalz data
                        if 'schmalz' in file_path.lower() and 'opt' in label_lower and 'knee' in base_sensor_name_lower and points:
                            xs, ys = zip(*points)
                            ys = np.array(ys) / mass  # Normalize by mass
                            # Adapt Schmalz x-axis to match stance phase using left_switches
                            # left_switches is a dict: {run_key: [indices]}
                            # Use the first run's left_switches as reference for stance phase length
                            stance_percent = 100  # Default to 100% if not found
                            if left_switches and isinstance(left_switches, dict):
                                # Try to get the first available stance switch index
                                for switches in left_switches.values():
                                    if switches and len(switches) > 0:
                                        stance_percent = (switches[0] / interp_len) * 100
                                        break
                            # Rescale Schmalz xs from [0, 100] to [0, stance_percent]
                            xs = np.array(xs)
                            xs_scaled = xs / 100.0 * stance_percent
                            ax.plot(xs_scaled, ys, marker='o', linestyle=line_style, color=line_color, label=f"Schmalz OPT {label_suffix}")
                            # Plot a vertical line at the stance-to-swing transition (first switch index)
                            if left_switches and isinstance(left_switches, dict):
                                for switches in left_switches.values():
                                    if switches and len(switches) > 0:
                                        stance_percent = (switches[0] / interp_len) * 100
                                        ax.axvline(stance_percent, color='red', linestyle=':', linewidth=2, label="Stance-to-Swing Transition")
                                        break
                        # Match baseline: must contain base sensor name AND 'intact'

                # Plot Baselines on Left Sensor (Prosthetic/Unimpaired side - using non-Intact labels)
                plot_baseline_torque_impaired(axes[1], baseline_data, baseline_file, mass_baseline, 
                                                 "(Impair 1)", '-', 'black', left_sensor)
                plot_baseline_torque_impaired(axes[1], baseline_data_2, baseline_file_2, mass_baseline_2, 
                                                 "(Impair 2)", '--', 'grey', left_sensor)
                plot_baseline_torque_Schmalz(axes[1], baseline_data_3, baseline_file_3, mass_baseline_3,
                                             "(Impair 3)", ':', 'blue', right_sensor)

                axes[1].set_title(f"{left_sensor} ({direction}-axis) mean (all runs)")
                axes[1].set_xlabel("Interpolated Step (%)")
                axes[1].set_ylabel(f"Torque ({'Nm/kg' if body_weight_to_normalize != 1 else 'Nm'})")
                
                # Plot 3: Difference (axes[2])
                for run_key, diff in diff_all:
                    if diff is not None:
                        axes[2].plot(x, diff, label=f"{run_key} Diff (L-R)")
                        
                axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
                axes[2].set_title(f"{left_sensor} - {right_sensor} difference (all runs)")
                axes[2].set_xlabel("Interpolated Step (%)")
                axes[2].set_ylabel(f"Torque Difference ({'Nm/kg' if body_weight_to_normalize != 1 else 'Nm'})")
                
                # Legend consolidation
                for ax in axes:
                    if ax.get_lines() or ax.get_children():
                        lines, labels = ax.get_legend_handles_labels()
                        # Consolidate duplicate labels (e.g., from multiple runs/baselines)
                        unique_labels = dict(zip(labels, lines))
                        ax.legend(unique_labels.values(), unique_labels.keys(), loc='best')

                plt.tight_layout()
                plt.show()
                plt.close()



    
    @staticmethod
    def plot_sensor_torque_symmetry_baseline_comp_pros_vs_health(
        all_loaded_data,
        run_step_data,
        direction,
        baseline_file,
        baseline_file_2,
        baseline_file_3,
        interp_len=100,
        body_weight_to_normalize=1,
        mass_baseline=97.5,
        mass_baseline_2=75 * 9.81,
        mass_baseline_3 = 85, #* 9.81, 
        sensor_force_names=None, # Optional, but good practice
        right_switches = None, 
        left_switches = None,
        healthy = ['right'],
    ):
        """
        Plot mean of right of all runs, mean of left, and the difference for TORQUE sensors,
        side by side in one figure, including two external baselines.
        direction: "x", "y", or "z" to specify the torque axis. [0, 1, 2] for x, y, z respectively.
        """
        # --- Initialization ---
        direction_map = {"x": 0, "y": 1, "z": 2}
        force_direction = direction_map.get(direction, 0)

        run_keys = list(all_loaded_data.keys())

        if len(healthy)==1: 
            healthy = healthy*len(run_keys)
        
        # Use sensor_force_names from the first run if not provided
        if sensor_force_names is None:
            first_run = next(iter(all_loaded_data))
            sensor_force_names = all_loaded_data[first_run].get("sensor_force_names", [])

        # Find sensor pairs, filtering for 'torque' in the name
        sensor_pairs = []
        for name in sensor_force_names:
            if 'torque' in name.lower() and name.startswith("left_"):
                right_name = name.replace("left_", "right_")
                if right_name in sensor_force_names:
                    sensor_pairs.append((name, right_name))

        # --- Baseline Loading ---
        baseline_data = {}
        baseline_data_2 = {}
        baseline_data_3 = {}
        
        # Helper for loading baseline (fixed for value stripping and error handling)
        def load_baseline(file_path, data_dict):
            current_label = None
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check for label start
                    if 'Schmalz' in file_path:
                        label_match = re.match(r'^([A-Z_]+)\s*=\s*\[?', line)
                    else:
                        label_match = re.match(r'^([A-Za-z_]+)\s*=\s*\[?', line)
                    
                    if label_match:
                        current_label = label_match.group(1)
                        data_dict[current_label] = []
                        continue
                        
                    # Check for end of list (or just skip if no current_label)
                    if line.endswith(']'):
                        current_label = None
                        continue

                    if current_label:
                        vals = line.split(';')
                        if len(vals) == 2:
                            try:
                                x, y = map(float, [v.strip() for v in vals])
                                data_dict[current_label].append((x, y))
                            except ValueError:
                                continue

        load_baseline(baseline_file, baseline_data)
        load_baseline(baseline_file_2, baseline_data_2)
        load_baseline(baseline_file_3, baseline_data_3)
        
        # --- Main Plotting Loop (Iterates over torque sensor pairs) ---
        for left_sensor, right_sensor in sensor_pairs:
            base_sensor_name_lower = left_sensor.replace("left_", "").replace("_sensor", "").lower()
            
            # Prepare data for all runs
            # run_keys = list(all_loaded_data.keys())
            mean_left_all, std_left_all, mean_right_all, std_right_all, diff_all = [], [], [], [], []
            
            # Helper function for data extraction and interpolation
            def collect_steps(data, indices):
                steps = []
                if indices and len(indices) > 1:
                    for j in range(len(indices) - 1):
                        start, end = indices[j], indices[j + 1]
                        # Extract data for the specified direction (torque is a vector)
                        y = [float(np.array(a)[force_direction]) for a in data[start:end]]
                        if len(y) < 2:
                            continue
                        
                        # Interpolation logic
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        steps.append(y_interp)
                return steps
            
            for run_key in run_keys:

                run_dict = all_loaded_data[run_key]
                step_data = run_step_data[run_key]
                
                # Access sensor data
                left_data = run_dict.get("all_sensor_force", {}).get(left_sensor, None)
                right_data = run_dict.get("all_sensor_force", {}).get(right_sensor, None)
                
                if left_data is None or right_data is None:
                    continue
                    
                all_step_start_left = step_data["step_start_left"]
                all_step_start_right = step_data["step_start_right"]

                left_steps = collect_steps(left_data, all_step_start_left)
                right_steps = collect_steps(right_data, all_step_start_right)
                
                # Compute means and stds
                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0) / body_weight_to_normalize
                    std_left = np.std(left_steps, axis=0) / body_weight_to_normalize
                    mean_right = np.mean(right_steps, axis=0) / body_weight_to_normalize
                    std_right = np.std(right_steps, axis=0) / body_weight_to_normalize
                    
                    # Apply sign convention for knee (or other specific sensors)
                    if 'knee' in left_sensor.lower():
                        mean_left = -mean_left
                        mean_right = -mean_right
                        # Note: Do not flip sign for standard deviation (std is always positive)
                        
                    diff = mean_left - mean_right # Symmetry difference
                    
                    mean_left_all.append((run_key, mean_left, std_left))
                    mean_right_all.append((run_key, mean_right, std_right))
                    diff_all.append((run_key, diff))
                else:
                    mean_left_all.append((run_key, None, None))
                    mean_right_all.append((run_key, None, None))
                    diff_all.append((run_key, None))

            # --- Plotting ---
            x = np.linspace(0, 100, interp_len)
            
            if mean_left_all or mean_right_all or diff_all:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
                axis_healthy = axes[0]
                axis_impaired = axes[1]

                # if 'right' in healthy[run_keys.index(run_key)].lower():
                #     axis_healthy = axes[0]
                #     axis_impaired = axes[1]
                # else:
                #     axis_healthy = axes[1]
                #     axis_impaired = axes[0]

                
                # Plot 1: Right Sensor (axes[0]) - Intact Baseline
                for run_key, mean_right, std_right in mean_right_all:
                    if mean_right is not None:
                        if healthy[run_keys.index(run_key)].lower() == 'right':
                            axis_healthy.plot(x, mean_right, label=f"{run_key}")
                            axis_healthy.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
                        else:
                            axis_impaired.plot(x, mean_right, label=f"{run_key}")
                            axis_impaired.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
                
                # Helper for plotting baseline data
                def plot_baseline_torque(ax, data_dict, file_path, mass, label_suffix, line_style, line_color, side_label):
                    for label, points in data_dict.items():
                        label_lower = label.lower()
                        if 'ankle' in label_lower:
                            label_lower = label_lower.replace('ankle', 'foot')
                        label_joint = label_lower.split('_')[0]
                        # Match baseline: must contain base sensor name AND 'intact'
                        if 'intact' in label_lower and label_joint in base_sensor_name_lower  and points:
                            xs, ys = zip(*points)
                            ys = np.array(ys) * -1 # Apply common torque sign flip
                            if 'Banks' in file_path:
                                ys = np.array(ys) / mass
                                if 'knee' not in base_sensor_name_lower: # Apply specific normalization flip
                                    ys = np.array(ys) * -1

                            baseline_label = 'Turcot et al.' if 'Turcot' in file_path else 'Banks et al.'
                            ax.plot(xs, ys, marker='o', linestyle=line_style, color=line_color,
                                    label=f"{baseline_label} {label_suffix}")
                
                # Plot Baselines on Right Sensor (Intact side)
                plot_baseline_torque(axis_healthy, baseline_data, baseline_file, mass_baseline, 
                                     "(Intact 1)", '-', 'black', right_sensor)
                plot_baseline_torque(axis_healthy, baseline_data_2, baseline_file_2, mass_baseline_2, 
                                     "(Intact 2)", '--', 'grey', right_sensor)


                axis_healthy.set_title(f"{right_sensor} ({direction}-axis) mean (all runs) HEALTHY")
                axis_healthy.set_xlabel("Interpolated Step (%)")
                axis_healthy.set_ylabel(f"Torque ({'Nm/kg' if body_weight_to_normalize != 1 else 'Nm'})")
                
                # Plot 2: Left Sensor (axes[1]) - Prosthetic Baseline
                for run_key, mean_left, std_left in mean_left_all:
                    if mean_left is not None:
                        if healthy[run_keys.index(run_key)].lower() == 'right':
                            axis_impaired.plot(x, mean_left, label=f"{run_key}")
                            axis_impaired.fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
                        else:
                            axis_healthy.plot(x, mean_left, label=f"{run_key}")
                            axis_healthy.fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
                        # axis_impaired.plot(x, mean_left, label=f"{run_key}")
                        # axis_impaired.fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)

                # Helper for plotting baseline data (Prosthesis/impaired Side)
                def plot_baseline_torque_impaired(ax, data_dict, file_path, mass, label_suffix, line_style, line_color, side_label):
                    for label, points in data_dict.items():
                        label_lower = label.lower()
                        if 'ankle' in label_lower:
                            label_lower = label_lower.replace('ankle', 'foot')
                        # Match baseline: must contain base sensor name AND NOT 'intact'
                        if 'intact' not in label_lower and label_lower in base_sensor_name_lower and points:
                            xs, ys = zip(*points)
                            ys = np.array(ys) * -1 # Apply common torque sign flip
                            if 'Banks' in file_path:
                                ys = np.array(ys) / mass
                                if 'knee' not in base_sensor_name_lower:
                                    ys = np.array(ys) * -1
                            
                            baseline_label = 'Turcot et al.' if 'Turcot' in file_path else 'Banks et al.'
                            ax.plot(xs, ys, marker='o', linestyle=line_style, color=line_color, 
                                    label=f"{baseline_label} {label_suffix}")
                
                def plot_baseline_torque_Schmalz(ax, data_dict, file_path, mass, label_suffix, line_style, line_color, side_label):
                    for label, points in data_dict.items():
                        label_lower = label.lower()
                        if 'knee' in label_lower:
                            label_lower = label_lower.replace('knee', 'thigh')
                        # Plot OPT for knee from Schmalz data
                        if 'schmalz' in file_path.lower() and 'opt' in label_lower and 'knee' in base_sensor_name_lower and points:
                            xs, ys = zip(*points)
                            ys = np.array(ys) / mass  # Normalize by mass
                            # Adapt Schmalz x-axis to match stance phase using left_switches
                            # left_switches is a dict: {run_key: [indices]}
                            # Use the first run's left_switches as reference for stance phase length
                            stance_percent = 100  # Default to 100% if not found
                            if left_switches and isinstance(left_switches, dict):
                                # Try to get the first available stance switch index
                                for switches in left_switches.values():
                                    if switches and len(switches) > 0:
                                        stance_percent = (switches[0] / interp_len) * 100
                                        break
                            # Rescale Schmalz xs from [0, 100] to [0, stance_percent]
                            xs = np.array(xs)
                            xs_scaled = xs / 100.0 * stance_percent
                            ax.plot(xs_scaled, ys, marker='o', linestyle=line_style, color=line_color, label=f"Schmalz OPT {label_suffix}")
                            # Plot a vertical line at the stance-to-swing transition (first switch index)
                            if left_switches and isinstance(left_switches, dict):
                                for switches in left_switches.values():
                                    if switches and len(switches) > 0:
                                        stance_percent = (switches[0] / interp_len) * 100
                                        ax.axvline(stance_percent, color='red', linestyle=':', linewidth=2, label="Stance-to-Swing Transition")
                                        break
                        # Match baseline: must contain base sensor name AND 'intact'

                # Plot Baselines on Left Sensor (Prosthetic/Unimpaired side - using non-Intact labels)
                plot_baseline_torque_impaired(axis_impaired, baseline_data, baseline_file, mass_baseline, 
                                                 "(Impair 1)", '-', 'black', left_sensor)
                plot_baseline_torque_impaired(axis_impaired, baseline_data_2, baseline_file_2, mass_baseline_2, 
                                                 "(Impair 2)", '--', 'grey', left_sensor)
                plot_baseline_torque_Schmalz(axis_impaired, baseline_data_3, baseline_file_3, mass_baseline_3,
                                             "(Impair 3)", ':', 'blue', right_sensor)

                axis_impaired.set_title(f"{left_sensor} ({direction}-axis) mean (all runs) IMPAIRED")
                axis_impaired.set_xlabel("Interpolated Step (%)")
                axis_impaired.set_ylabel(f"Torque ({'Nm/kg' if body_weight_to_normalize != 1 else 'Nm'})")
                
                # Plot 3: Difference (axes[2])
                for run_key, diff in diff_all:
                    if diff is not None:
                        axes[2].plot(x, diff, label=f"{run_key} Diff (L-R)")
                        
                axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
                axes[2].set_title(f"{left_sensor} - {right_sensor} difference (all runs)")
                axes[2].set_xlabel("Interpolated Step (%)")
                axes[2].set_ylabel(f"Torque Difference ({'Nm/kg' if body_weight_to_normalize != 1 else 'Nm'})")
                
                # Legend consolidation
                for ax in axes:
                    if ax.get_lines() or ax.get_children():
                        lines, labels = ax.get_legend_handles_labels()
                        # Consolidate duplicate labels (e.g., from multiple runs/baselines)
                        unique_labels = dict(zip(labels, lines))
                        ax.legend(unique_labels.values(), unique_labels.keys(), loc='best')

                plt.tight_layout()
                plt.show()
                plt.close()




    @staticmethod
    def plot_sensor_force_symmetry_four_panel_with_params(
        all_loaded_data, run_step_data, right_switches, left_switches,
        prosthesis_side='left', interp_len=100, body_weight_to_normalize=1, baseline_data_path=None
    ):
        """
        Plots a 2x2 figure comparing specific runs to baseline data, with updated colors.
        Each subplot shows the mean and standard deviation of a run against a baseline.
        """
        
        # Define the configurations for each of the four subplots
        plot_configs = [
            {'baseline_label': 'FOOT_ANT', 'run_keys': ['P2', 'P2_X20mm'], 'title': 'FOOT_ANT vs. P2_X20mm'},
            {'baseline_label': 'FOOT_POS', 'run_keys': ['P2', 'P2_X-20mm'], 'title': 'FOOT_POS vs. P2_X-20mm'},
            {'baseline_label': 'FOOT_PLA', 'run_keys': ['P2', 'P2_Z-5°'], 'title': 'FOOT_PLA vs. P2_Z-5°'},
            {'baseline_label': 'FOOT_DOR', 'run_keys': ['P2', 'P2_Z5°'], 'title': 'FOOT_DOR vs. P2_Z5°'},
            # {'baseline_label': 'FOOT_PLA', 'run_keys': ['P2', 'P2_Z-10°'], 'title': 'FOOT_PLA vs. P2, P2_Z-10°'},
            # {'baseline_label': 'FOOT_DOR', 'run_keys': ['P2', 'P2_Z10°'], 'title': 'FOOT_DOR vs. P2, P2_Z10°'},
        ]

        mass_norm = 85 * 9.81  # std 15
        direction = 'z' # Assuming Z-axis force for all plots
        force_direction = {"x": 0, "y": 1, "z": 2}.get(direction, 0)

        # Create a figure with a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        #fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration

        # Load baseline data
        baseline_data = {}
        if baseline_data_path is not None:
            with open(baseline_data_path, 'r') as f:
                current_label = None
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if 'Schmalz' in baseline_data_path:
                        label_match = re.match(r'^([A-Z_]+)\s*=\s*\[?', line)
                    else:
                        label_match = re.match(r'^([A-Za-z_]+)\s*=\s*\[?', line)
                    
                    if label_match:
                        current_label = label_match.group(1)
                        baseline_data[current_label] = []
                        continue
                    
                    if current_label:
                        vals = line.split(';')
                        if len(vals) == 2:
                            try:
                                x_val, y_val = map(float, vals)
                                baseline_data[current_label].append((x_val, y_val))
                            except ValueError:
                                continue
        
        # Iterate through each subplot configuration and plot the data
        for i, config in enumerate(plot_configs):
            ax = axs[i]
            
            # Plot baseline data
            if baseline_data is not None:
                baseline_label = config['baseline_label']
                opt_label = "OPT"
                if baseline_label in baseline_data and opt_label in baseline_data:
                    # Plot 'FOOT_...' baseline in blue
                    xs, ys = zip(*baseline_data[baseline_label])
                    ys_norm = np.array(ys) / mass_norm
                    ax.plot(xs, ys_norm, linestyle='-', color='blue', label=f"Baseline {baseline_label}")

                    # Plot 'OPT' baseline in black
                    xs, ys = zip(*baseline_data[opt_label])
                    ys_norm = np.array(ys) / mass_norm
                    ax.plot(xs, ys_norm, linestyle='-', color='black', label=f"Baseline {opt_label}")
            
            # Plot run data
            for run_key in config['run_keys']:
                if run_key in all_loaded_data:
                    run_dict = all_loaded_data[run_key]
                    step_data = run_step_data[run_key]
                    
                    # Use the passed prosthesis_side to select the correct sensor and data
                    if prosthesis_side == "left":
                        sensor = "left_knee_mimic_torque_sensor"
                        all_step_start = step_data.get("step_start_left", [])
                        switches = left_switches.get(run_key, [])
                        data = run_dict["all_sensor_force"].get(sensor, [])
                    else:  # "right"
                        sensor = "right_knee_mimic_torque_sensor"
                        all_step_start = step_data.get("step_start_right", [])
                        switches = right_switches.get(run_key, [])
                        data = run_dict["all_sensor_force"].get(sensor, [])
                    
                    steps = []
                    if all_step_start and len(all_step_start) > 1 and switches:
                        for j in range(len(all_step_start) - 1):
                            start = all_step_start[j]
                            end = all_step_start[j + 1]
                            y = [float(np.array(a)[force_direction]) for a in data[start:end]]
                            if len(y) < 2:
                                continue
                            
                            # Apply the interp_len for interpolation
                            x_old = np.linspace(0, 1, len(y))
                            x_new = np.linspace(0, 1, interp_len)
                            y_interp = np.interp(x_new, x_old, y)
                            switch_idx = switches[0] if switches and switches[0] < interp_len else interp_len
                            stance_phase = y_interp[:switch_idx]
                            if len(stance_phase) > 1:
                                x_stance_old = np.linspace(0, 1, len(stance_phase))
                                x_stance_new = np.linspace(0, 1, interp_len)
                                stance_phase_interp = np.interp(x_stance_new, x_stance_old, stance_phase)
                                steps.append(stance_phase_interp)
                    
                    if steps:
                        mean_val = np.mean(steps, axis=0) / body_weight_to_normalize
                        std_val = np.std(steps, axis=0) / body_weight_to_normalize
                        mean_val = -mean_val
                        std_val = -std_val
                        x = np.linspace(0, 100, interp_len)

                        # Determine color based on run_key
                        if run_key == 'P2':
                            line_color = 'grey'
                        else: # P2_X20mm, P2_X-20mm, etc.
                            line_color = 'skyblue' # baby blue color
                            
                        ax.plot(x, mean_val, label=f"{run_key} {prosthesis_side} mean", color=line_color)
                        ax.fill_between(x, mean_val - std_val, mean_val + std_val, alpha=0.15, color=line_color)
            
            # Set titles and labels for the subplot
            ax.set_title(config['title'], fontsize=16)
            ax.set_xlabel("Stance phase in %", fontsize=16)
            ax.set_ylabel("Coronal Knee Moment in Nm/kg", fontsize=16)
            ax.legend(fontsize=14)

            # increase font size of ticks
            ax.tick_params(axis='both', which='major', labelsize=16)

            # add grid lines at y= 0
            # ax.axhline(0, color='gray', linestyle=':', linewidth=1)
            ax.grid(True, which='both') #, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
        plt.close()





    @staticmethod
    def plot_sensor_force_by_run_and_couple(
        all_loaded_data, run_step_data, direction, right_switches, left_switches,
        prosthesis_side='left', interp_len=100, body_weight_to_normalize=1, baseline_data=None
    ):
        """
        Plot stance-phase mean for the prosthesis side for each run_key.
        Separate plot for each run_key. Each plot shows one of these couples:
        Foot_Ant with 20mm, Foot_Pos with -20mm, Foot_Pla -5°, Foot_Dor 5°.
        Overlay baseline P2 and OPT if provided.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import re

        mass_norm = 85 * 10
        force_direction = {"x": 0, "y": 1, "z": 2}.get(direction, 0)

        left_sensor = "left_knee_mimic_torque_sensor"
        right_sensor = "right_knee_mimic_torque_sensor"

        # Couples to plot
        couples = {
            "Foot_Ant": 20,
            "Foot_Pos": -20,
            "Foot_Pla": -5,
            "Foot_Dor": 5
        }

        # Load baseline if provided
        baseline_dict = {}
        if baseline_data is not None:
            current_label = None
            with open(baseline_data, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if 'Schmalz' in baseline_data:
                        label_match = re.match(r'^([A-Z_]+)\s*=\s*\[?', line)
                    else:
                        label_match = re.match(r'^([A-Za-z_]+)\s*=\s*\[?', line)
                    if label_match:
                        current_label = label_match.group(1)
                        baseline_dict[current_label] = []
                        continue
                    if current_label:
                        vals = line.split(';')
                        if len(vals) == 2:
                            try:
                                x_val, y_val = map(float, vals)
                                baseline_dict[current_label].append((x_val, y_val))
                            except ValueError:
                                continue

        # Iterate over runs
        for run_key in all_loaded_data.keys():
            run_dict = all_loaded_data[run_key]
            step_data = run_step_data[run_key]

            if prosthesis_side == "left":
                sensor = left_sensor
                all_step_start = step_data["step_start_left"]
                switches = left_switches.get(run_key, [])
                data = run_dict[left_sensor] if left_sensor in run_dict else run_dict["all_sensor_force"][left_sensor]
            else:
                sensor = right_sensor
                all_step_start = step_data["step_start_right"]
                switches = right_switches.get(run_key, [])
                data = run_dict[right_sensor] if right_sensor in run_dict else run_dict["all_sensor_force"][right_sensor]

            steps = []
            if all_step_start and len(all_step_start) > 1 and switches:
                for j in range(len(all_step_start) - 1):
                    start = all_step_start[j]
                    end = all_step_start[j + 1]
                    y = [float(np.array(a)[force_direction]) for a in data[start:end]]
                    if len(y) < 2:
                        continue
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    y_interp = np.interp(x_new, x_old, y)
                    switch_idx = switches[0] if switches and switches[0] < interp_len else interp_len
                    stance_phase = y_interp[:switch_idx]
                    if len(stance_phase) > 1:
                        x_stance_old = np.linspace(0, 1, len(stance_phase))
                        x_stance_new = np.linspace(0, 1, interp_len)
                        stance_phase_interp = np.interp(x_stance_new, x_stance_old, stance_phase)
                        steps.append(stance_phase_interp)

            if not steps:
                continue

            mean_val = -np.mean(steps, axis=0) / body_weight_to_normalize
            std_val = -np.std(steps, axis=0) / body_weight_to_normalize
            x = np.linspace(0, 100, interp_len)

            # Plot for each couple
            for couple_name, couple_value in couples.items():
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(x, mean_val, label=f"{run_key} mean ({couple_name} {couple_value})")
                ax.fill_between(x, mean_val - std_val, mean_val + std_val, alpha=0.15)

                # Overlay baseline P2 and OPT if available
                for label in ["P2", "OPT"]:
                    if label in baseline_dict:
                        points = baseline_dict[label]
                        xs, ys = zip(*points)
                        ys = np.array(ys) / mass_norm
                        ax.plot(xs, ys, marker='.', linestyle='-', label=f"Baseline {label}")

                ax.set_title(f"{run_key} - {couple_name} ({couple_value})")
                ax.set_xlabel("Interpolated Step (%)")
                ax.set_ylabel("Force Value")
                ax.legend()
                plt.tight_layout()
                plt.show()
                plt.close()




    @staticmethod
    def plot_sensor_force_per_run_4subplots(
        all_loaded_data, run_step_data, direction, right_switches, left_switches,
        prosthesis_side='left', interp_len=100, body_weight_to_normalize=1, baseline_data=None
    ):
        """
        For each run_key in all_loaded_data, create one figure with 4 subplots:
        1) baseline: FOOT_ANT & OPT   + run: P2 and P2_X20mm
        2) baseline: FOOT_POS & OPT   + run: P2 and P2_X-20mm
        3) baseline: FOOT_PLA & OPT   + run: P2 and P2_Z5°
        4) baseline: FOOT_DOR & OPT   + run: P2 and P2_Z-5°
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import re
        import warnings

        mass_norm = 85 * 10
        force_direction = {"x": 0, "y": 1, "z": 2}.get(direction, 0)

        # mapping: (p2_label_in_run_dict, foot_label_in_baseline, subplot_title)
        subplot_configs = [
            ("P2_X20mm", "FOOT_ANT", "ANT (X +20 mm)"),
            ("P2_X-20mm", "FOOT_POS", "POS (X -20 mm)"),
            ("P2_Z5°", "FOOT_PLA", "PLA (Z +5°)"),
            ("P2_Z-5°", "FOOT_DOR", "DOR (Z -5°)"),
        ]

        # helper: read baseline file into dict(label -> list of (x,y))
        baseline_dict = {}
        if baseline_data is not None:
            current_label = None
            try:
                with open(baseline_data, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        if 'Schmalz' in baseline_data:
                            label_match = re.match(r'^([A-Z_]+)\s*=\s*\[?', line)
                        else:
                            label_match = re.match(r'^([A-Za-z_]+)\s*=\s*\[?', line)
                        if label_match:
                            current_label = label_match.group(1)
                            baseline_dict[current_label] = []
                            continue
                        if current_label:
                            vals = line.split(';')
                            if len(vals) == 2:
                                try:
                                    x_val, y_val = map(float, vals)
                                    baseline_dict[current_label].append((x_val, y_val))
                                except ValueError:
                                    continue
            except FileNotFoundError:
                warnings.warn(f"Baseline file not found: {baseline_data}")
            except Exception as e:
                warnings.warn(f"Error reading baseline file {baseline_data}: {e}")

        # helper: get a time-series signal from run_dict for a given label
        def get_run_signal(run_dict, label):
            # 1) direct top-level key
            if label in run_dict:
                return run_dict[label]
            # 2) if all_sensor_force exists and is a dict
            af = run_dict.get("all_sensor_force", None)
            if isinstance(af, dict) and label in af:
                return af[label]
            # 3) sometimes P2 signals might be nested under a sensor key (e.g., sensor_name -> label)
            # try to search dict-of-dict
            if isinstance(af, dict):
                for k, v in af.items():
                    if isinstance(v, dict) and label in v:
                        return v[label]
            # not found
            return None

        # helper: compute mean & std stance-phase curve for a signal (list-like of vectors)
        def compute_mean_std_from_signal(signal, all_step_start, switches):
            if signal is None:
                return None, None
            steps_list = []
            if not all_step_start or len(all_step_start) <= 1 or not switches:
                return None, None
            for j in range(len(all_step_start) - 1):
                start = all_step_start[j]
                end = all_step_start[j + 1]
                # guard indices
                if start < 0 or end <= start:
                    continue
                seg = signal[start:end]
                # seg expected to be iterable of vectors (e.g., (fx,fy,fz))
                try:
                    y = [float(np.array(a)[force_direction]) for a in seg]
                except Exception:
                    # if elements are scalars already
                    try:
                        y = [float(a) for a in seg]
                    except Exception:
                        continue
                if len(y) < 2:
                    continue
                x_old = np.linspace(0, 1, len(y))
                x_new = np.linspace(0, 1, interp_len)
                y_interp = np.interp(x_new, x_old, y)
                switch_idx = switches[0] if switches and switches[0] < interp_len else interp_len
                stance_phase = y_interp[:switch_idx]
                if len(stance_phase) > 1:
                    x_stance_old = np.linspace(0, 1, len(stance_phase))
                    x_stance_new = np.linspace(0, 1, interp_len)
                    stance_phase_interp = np.interp(x_stance_new, x_stance_old, stance_phase)
                    steps_list.append(stance_phase_interp)
            if not steps_list:
                return None, None
            mean_val = -np.mean(steps_list, axis=0) / body_weight_to_normalize
            std_val = -np.std(steps_list, axis=0) / body_weight_to_normalize
            return mean_val, std_val

        # iterate runs: create one figure per run_key
        for run_key in all_loaded_data.keys():
            run_dict = all_loaded_data[run_key]
            step_data = run_step_data.get(run_key, {})

            # choose step starts & switches according to prosthesis side
            if prosthesis_side == "left":
                all_step_start = step_data.get("step_start_left", [])
                switches = left_switches.get(run_key, [])
            else:
                all_step_start = step_data.get("step_start_right", [])
                switches = right_switches.get(run_key, [])

            fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
            axes = axes.flatten()

            # common x for run curves
            x_run = np.linspace(0, 100, interp_len)

            # plot each subplot according to mapping
            for i, (p2_variant, foot_label, subtitle) in enumerate(subplot_configs):
                ax = axes[i]

                # Baseline OPT (if present)
                if "OPT" in baseline_dict and baseline_dict["OPT"]:
                    xs_opt, ys_opt = zip(*baseline_dict["OPT"])
                    ys_opt = np.array(ys_opt) / mass_norm
                    ax.plot(xs_opt, ys_opt, marker='.', linestyle='-', label="OPT (baseline)", zorder=1)

                # Baseline FOOT_x (if present)
                if foot_label in baseline_dict and baseline_dict[foot_label]:
                    xs_f, ys_f = zip(*baseline_dict[foot_label])
                    ys_f = np.array(ys_f) / mass_norm
                    ax.plot(xs_f, ys_f, linestyle='--', label=f"{foot_label} (baseline)", zorder=1)

                # P2 (from run_dict)
                p2_signal = get_run_signal(run_dict, "P2")
                mean_p2, std_p2 = compute_mean_std_from_signal(p2_signal, all_step_start, switches)
                if mean_p2 is not None:
                    ax.plot(x_run, mean_p2, label="P2 (run)", linewidth=1.5, zorder=2)
                    ax.fill_between(x_run, mean_p2 - std_p2, mean_p2 + std_p2, alpha=0.18, zorder=1)

                # P2 variant (from run_dict) e.g., P2_X20mm, P2_Z5°, etc.
                p2v_signal = get_run_signal(run_dict, p2_variant)
                mean_p2v, std_p2v = compute_mean_std_from_signal(p2v_signal, all_step_start, switches)
                if mean_p2v is not None:
                    ax.plot(x_run, mean_p2v, label=f"{p2_variant} (run)", linewidth=1.5, zorder=3)
                    ax.fill_between(x_run, mean_p2v - std_p2v, mean_p2v + std_p2v, alpha=0.18, zorder=2)

                ax.set_title(f"{run_key} — {subtitle}")
                ax.set_xlabel("Interpolated Step (%)")
                if i % 2 == 0:
                    ax.set_ylabel("Force value (normalized)")

                ax.legend(fontsize='small', loc='best')

            plt.suptitle(f"{run_key} — Prosthesis side: {prosthesis_side}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
            plt.close(fig)







    @staticmethod
    def plot_sensor_force_prosthesis_vs_baseline(self,
        all_loaded_data, run_step_data, direction, right_switches, interp_len=100,
        body_weight_to_normalize=1, baseline_data=None
    ):
        """
        Plot mean of prosthesis side (right) across all runs,
        and optionally compare to baseline_data for knee_z.
        Only stance phase is shown.
        """
        mass_norm = 85 * 10  # std 15
        force_direction = {"x": 0, "y": 1, "z": 2}.get(direction, 0)

        # Prosthesis sensor only
        right_sensor = "right_knee_mimic_torque_sensor"

        run_keys = list(all_loaded_data.keys())
        mean_right_all = []
        std_right_all = []
        max_len = 0

        for run_key in run_keys:
            run_dict = all_loaded_data[run_key]
            step_data = run_step_data[run_key]
            right_data = run_dict["all_sensor_force"][right_sensor]
            all_step_start_right = step_data["step_start_right"]

            switches_right = right_switches.get(run_key, [])
            right_steps = []

            if all_step_start_right and len(all_step_start_right) > 1 and switches_right:
                for j in range(len(all_step_start_right) - 1):
                    start = all_step_start_right[j]
                    end = all_step_start_right[j + 1]
                    y = [float(np.array(a)[force_direction]) for a in right_data[start:end]]
                    if len(y) < 2:
                        continue

                    # Interpolate full stride
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    y_interp = np.interp(x_new, x_old, y)

                    # Stance phase cut
                    switch_idx = switches_right[0] if switches_right and switches_right[0] < interp_len else interp_len
                    stance_phase = y_interp[:switch_idx]

                    if len(stance_phase) > 1:
                        x_stance_old = np.linspace(0, 1, len(stance_phase))
                        x_stance_new = np.linspace(0, 1, interp_len)
                        stance_phase_interp = np.interp(x_stance_new, x_stance_old, stance_phase)
                        right_steps.append(stance_phase_interp)
                    max_len = interp_len

            if right_steps:
                mean_right = -np.mean(right_steps, axis=0) / body_weight_to_normalize
                std_right = -np.std(right_steps, axis=0) / body_weight_to_normalize
                mean_right_all.append((run_key, mean_right, std_right))
            else:
                mean_right_all.append((run_key, None, None))

        # Plot
        x = np.linspace(0, 100, interp_len)
        plt.figure(figsize=(8, 6))

        # Prosthesis side
        for run_key, mean_right, std_right in mean_right_all:
            if mean_right is not None:
                plt.plot(x, mean_right, label=f"{run_key} {right_sensor} mean")
                plt.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)

        # Baseline
        if baseline_data is not None and direction == "z":
            import re
            data = {}
            current_label = None
            with open(baseline_data, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if 'Schmalz' in baseline_data:
                        label_match = re.match(r'^([A-Z_]+)\s*=\s*\[?', line)
                    else:
                        label_match = re.match(r'^([A-Za-z_]+)\s*=\s*\[?', line)
                    if label_match:
                        current_label = label_match.group(1)
                        data[current_label] = []
                        continue
                    if current_label:
                        vals = line.split(';')
                        if len(vals) == 2:
                            try:
                                xx, yy = map(float, vals)
                                data[current_label].append((xx, yy))
                            except ValueError:
                                continue

            for label, points in data.items():
                if label == "OPT" and points:
                    xs, ys = zip(*points)
                    ys = np.array(ys) / mass_norm
                    plt.plot(xs, ys, marker='.', linestyle='-', color='black', label=f"Baseline {label}")

        plt.title(f"{right_sensor} mean (stance phase, prosthesis side only)")
        plt.xlabel("Interpolated Step (%)")
        plt.ylabel("Force Value (normalized)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()


    @staticmethod
    def plot_sensor_force_symmetry_summed_all_runs(
        all_loaded_data, run_step_data, direction, interp_len=100, body_weight_to_normalize=1, sensor_force_names=None
    ):
        """
        Plot combined mean of right, left, and their difference (symmetry) across all runs in one figure.
        """
        force_direction = {"x": 0, "y": 1, "z": 2}.get(direction, 0)

        if sensor_force_names is None:
            first_run = next(iter(all_loaded_data))
            sensor_force_names = all_loaded_data[first_run]["sensor_force_names"]

        # Find left-right sensor pairs
        sensor_pairs = [
            (name, name.replace("left_", "right_"))
            for name in sensor_force_names
            if name.startswith("left_") and name.replace("left_", "right_") in sensor_force_names
        ]

        for left_sensor, right_sensor in sensor_pairs:
            all_left_steps, all_right_steps = [], []

            for run_key in all_loaded_data.keys():
                run_dict = all_loaded_data[run_key]
                step_data = run_step_data[run_key]
                left_data = run_dict["all_sensor_force"][left_sensor]
                right_data = run_dict["all_sensor_force"][right_sensor]

                # Interpolate left steps
                all_step_start_left = step_data.get("step_start_left", [])
                for j in range(len(all_step_start_left) - 1):
                    start, end = all_step_start_left[j]-1, all_step_start_left[j + 1]-1
                    y = [float(np.array(a)[force_direction]) for a in left_data[start:end]]
                    if len(y) < 2:
                        continue
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    all_left_steps.append(np.interp(x_new, x_old, y))

                # Interpolate right steps
                all_step_start_right = step_data.get("step_start_right", [])
                for j in range(len(all_step_start_right) - 1):
                    start, end = all_step_start_right[j]-1, all_step_start_right[j + 1]-1
                    y = [float(np.array(a)[force_direction]) for a in right_data[start:end]]
                    if len(y) < 2:
                        continue
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    all_right_steps.append(np.interp(x_new, x_old, y))

            # Compute overall mean and std across all runs
            mean_left = np.mean(all_left_steps, axis=0) / body_weight_to_normalize
            std_left = np.std(all_left_steps, axis=0) / body_weight_to_normalize
            mean_right = np.mean(all_right_steps, axis=0) / body_weight_to_normalize
            std_right = np.std(all_right_steps, axis=0) / body_weight_to_normalize
            symmetry = mean_left - mean_right

            x = np.linspace(0, 100, interp_len)

            # Plot all three curves in one figure
            plt.figure(figsize=(10, 6))
            plt.plot(x, mean_right, label=f"{right_sensor} mean (all runs)", color='r')
            plt.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15, color='r')

            plt.plot(x, mean_left, label=f"{left_sensor} mean (all runs)", color='b')
            plt.fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15, color='b')

            plt.plot(x, symmetry, label="Left - Right", color='g', linestyle='--')
            plt.axhline(0, color='gray', linestyle=':', linewidth=1)

            plt.title(f"Sensor Force Symmetry: {left_sensor} & {right_sensor}")
            plt.xlabel("Interpolated Step (%)")
            plt.ylabel("Force Value")
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.close()



    def plot_sensor_force_symmetry_summed_all_dirs(self,
        all_loaded_data, run_step_data, direction, interp_len=100, body_weight_to_normalize=1, sensor_force_names=None
    ):
        """
        Plot mean ± std of right, left, and symmetry across runs, comparing directories side by side.
        """
        force_direction = {"x": 0, "y": 1, "z": 2}.get(direction, 0)

        # Use sensor names from first run if not provided
        if sensor_force_names is None:
            first_dir = next(iter(all_loaded_data))
            first_run = next(iter(all_loaded_data[first_dir]))
            sensor_force_names = all_loaded_data[first_dir][first_run]["sensor_force_names"]

        # Find left-right pairs
        sensor_pairs = [
            (name, name.replace("left_", "right_"))
            for name in sensor_force_names
            if name.startswith("left_") and name.replace("left_", "right_") in sensor_force_names
        ]

        colors = plt.cm.tab10.colors

        for left_sensor, right_sensor in sensor_pairs:
            plt.figure(figsize=(20, 6))
            x = np.linspace(0, 100, interp_len)

            for i, dir_label in enumerate(all_loaded_data.keys()):
                # Collect all interpolated steps per directory
                left_steps_dir, right_steps_dir = [], []

                for run_key, run_dict in all_loaded_data[dir_label].items():
                    step_data = run_step_data[dir_label][run_key]
                    left_data = run_dict["all_sensor_force"][left_sensor]
                    right_data = run_dict["all_sensor_force"][right_sensor]

                    # LEFT steps
                    step_starts = step_data.get("step_start_left", [])
                    for j in range(len(step_starts) - 1):
                        start, end = step_starts[j]-1, step_starts[j + 1]-1
                        y = [float(np.array(a)[force_direction]) for a in left_data[start:end]]
                        if len(y) < 2: 
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        left_steps_dir.append(np.interp(np.linspace(0, 1, interp_len), x_old, y))

                    # RIGHT steps
                    step_starts = step_data.get("step_start_right", [])
                    for j in range(len(step_starts) - 1):
                        start, end = step_starts[j]-1, step_starts[j + 1]-1
                        y = [float(np.array(a)[force_direction]) for a in right_data[start:end]]
                        if len(y) < 2: 
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        right_steps_dir.append(np.interp(np.linspace(0, 1, interp_len), x_old, y))

                # Compute mean ± std per directory
                if left_steps_dir and right_steps_dir:
                    left_steps_dir = np.array(left_steps_dir)
                    right_steps_dir = np.array(right_steps_dir)

                    mean_left = np.mean(left_steps_dir, axis=0) / body_weight_to_normalize
                    std_left = np.std(left_steps_dir, axis=0) / body_weight_to_normalize
                    mean_right = np.mean(right_steps_dir, axis=0) / body_weight_to_normalize
                    std_right = np.std(right_steps_dir, axis=0) / body_weight_to_normalize
                    symmetry = mean_left - mean_right

                    color = colors[i % len(colors)]
                    plt.plot(x, mean_left, label=f"{dir_label} Left", color=color, linestyle='-')
                    plt.fill_between(x, mean_left - std_left, mean_left + std_left, color=color, alpha=0.15)

                    plt.plot(x, mean_right, label=f"{dir_label} Right", color=color, linestyle='--')
                    plt.fill_between(x, mean_right - std_right, mean_right + std_right, color=color, alpha=0.1)

                    plt.plot(x, symmetry, label=f"{dir_label} L-R Diff", color=color, linestyle=':')

            plt.axhline(0, color='gray', linestyle=':', linewidth=1)
            plt.xlabel("Interpolated Step (%)")
            plt.ylabel("Force (N or normalized)")
            plt.title(f"Sensor Force Symmetry: {left_sensor} & {right_sensor}")
            plt.legend(ncol=3, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            plt.show()
            plt.close()



    @staticmethod
    def plot_sensor_force_symmetry_summed_all_dirs_with_baseline(
        all_loaded_data,
        run_step_data,
        direction,
        baseline_file,
        baseline_file_2,
        baseline_file_3,
        interp_len=100,
        body_weight_to_normalize=1,
        mass_baseline=97.5,
        mass_baseline_2=75 * 9.81,
        mass_baseline_3 = 85, #* 9.81, 
        sensor_force_names=None,
        healthy = ['right'],
    ):
        """
        Corrected version that iterates through nested directory structure
        to ensure data is found and plotted.
        """
        direction_map = {"x": 0, "y": 1, "z": 2}
        force_direction = direction_map.get(direction, 0)
        
        # Check if data exists
        if not all_loaded_data:
            print("Error: all_loaded_data is empty.")
            return

        # 1. Correctly identify sensor pairs from the nested structure
        if sensor_force_names is None:
            first_dir = next(iter(all_loaded_data))
            first_run = next(iter(all_loaded_data[first_dir]))
            sensor_force_names = all_loaded_data[first_dir][first_run].get("sensor_force_names", [])

        sensor_pairs = []
        for name in sensor_force_names:
            if 'torque' in name:
                if name.startswith("left_"):
                    right_name = name.replace("left_", "right_")
                    if right_name in sensor_force_names:
                        sensor_pairs.append((name, right_name))
        
        if not sensor_pairs:
            print(f"Warning: No sensor pairs found in: {sensor_force_names}")
            return

        # --- Baseline Loading (Logic remains same) ---
        def load_baseline(file_path, data_dict):
            if not file_path or not os.path.exists(file_path): return
            current_label = None
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    regex = r'^([A-Z_]+)\s*=\s*\[?' if 'Schmalz' in file_path else r'^([A-Za-z_]+)\s*=\s*\[?'
                    label_match = re.match(regex, line)
                    if label_match:
                        current_label = label_match.group(1)
                        data_dict[current_label] = []
                        continue
                    if current_label and ';' in line:
                        vals = line.split(';')
                        try:
                            x, y = map(float, [v.strip() for v in vals])
                            data_dict[current_label].append((x, y))
                        except: continue

        b1, b2, b3 = {}, {}, {}
        load_baseline(baseline_file, b1)
        load_baseline(baseline_file_2, b2)
        load_baseline(baseline_file_3, b3)

        # --- Main Plotting Loop ---
        for left_sensor, right_sensor in sensor_pairs:
            print(f"Plotting pair: {left_sensor} / {right_sensor}")
            base_sensor_name_lower = left_sensor.replace("left_", "").replace("_sensor", "").lower()
            
            fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharex=True)
            ax_h, ax_i, ax_diff = axes[0], axes[1], axes[2]
            x_axis = np.linspace(0, 100, interp_len)
            colors = plt.cm.tab10.colors

            # 2. Iterate through Directories (Simulation Data)
            for d_idx, dir_label in enumerate(all_loaded_data.keys()):
                color = colors[d_idx % len(colors)]
                all_l_steps, all_r_steps = [], []
                
                # Determine if this specific directory/run treats 'right' as healthy
                # If healthy is a single string 'right', we apply it to all
                is_right_healthy = (healthy[d_idx % len(healthy)].lower() == 'right')

                for run_key, run_dict in all_loaded_data[dir_label].items():
                    step_data = run_step_data[dir_label][run_key]
                    
                    def get_steps(s_name, side):
                        raw = run_dict.get("all_sensor_force", {}).get(s_name, [])
                        starts = step_data.get(f"step_start_{side}", [])
                        steps = []
                        for j in range(len(starts) - 1):
                            y = [float(np.array(a)[force_direction]) for a in raw[starts[j]:starts[j+1]]]
                            if len(y) > 5:
                                steps.append(np.interp(x_axis, np.linspace(0, 100, len(y)), y))
                        return steps

                    all_l_steps.extend(get_steps(left_sensor, "left"))
                    all_r_steps.extend(get_steps(right_sensor, "right"))

                # 3. Process and Plot simulation data
                for steps_data, side_label, is_side_healthy in [
                    (np.array(all_l_steps), "L", not is_right_healthy),
                    (np.array(all_r_steps), "R", is_right_healthy)
                ]:
                    if steps_data.size == 0: continue
                    
                    target_ax = ax_h if is_side_healthy else ax_i
                    mean_val = np.mean(steps_data, axis=0) / body_weight_to_normalize
                    std_val = np.std(steps_data, axis=0) / body_weight_to_normalize
                    
                    if 'knee' in base_sensor_name_lower: mean_val = -mean_val
                    
                    ls = '-' if side_label == "L" else '--'
                    target_ax.plot(x_axis, mean_val, color=color, linestyle=ls, label=f"{dir_label} ({side_label})")
                    target_ax.fill_between(x_axis, mean_val-std_val, mean_val+std_val, color=color, alpha=0.1)

                # Symmetry Plot
                if len(all_l_steps) > 0 and len(all_r_steps) > 0:
                    diff = (np.mean(all_l_steps, axis=0) - np.mean(all_r_steps, axis=0)) / body_weight_to_normalize
                    if 'knee' in base_sensor_name_lower: diff = -diff
                    ax_diff.plot(x_axis, diff, color=color, label=f"{dir_label} Diff")

            # --- Baseline Plotting (Logic remains same) ---
            def plot_ref(ax, d_dict, f_path, mass, suffix, style, color, find_intact):
                for label, points in d_dict.items():
                    lbl_low = label.lower()
                    match_joint = 'foot' if 'ankle' in lbl_low else lbl_low.split('_')[0]
                    if match_joint not in base_sensor_name_lower: continue
                    if find_intact and 'intact' not in lbl_low: continue
                    if not find_intact and 'intact' in lbl_low: continue
                    
                    xs, ys = zip(*points)
                    ys = np.array(ys) * -1
                    if 'Banks' in str(f_path):
                        ys /= mass
                        if 'knee' not in base_sensor_name_lower: ys *= -1
                    ax.plot(xs, ys, linestyle=style, color=color, label=f"Ref {suffix}", alpha=0.6, linewidth=2)

            plot_ref(ax_h, b1, baseline_file, mass_baseline, "Banks (Intact)", '-', 'black', True)
            plot_ref(ax_i, b1, baseline_file, mass_baseline, "Banks (Impaired)", '-', 'black', False)
            plot_ref(ax_h, b2, baseline_file_2, mass_baseline, "Turcot (Intact)", '-.', 'black', True)
            plot_ref(ax_i, b2, baseline_file_2, mass_baseline, "Turcot (Impaired)", '-.', 'black', False)
            # Plot OPT baseline for knee sensors from Schmalz data
            if 'knee' in base_sensor_name_lower and b3:
                for label, points in b3.items():
                    if label.upper() == 'OPT' and points:
                        xs, ys = zip(*points)
                        ys = np.array(ys) / mass_baseline_3
                        ax_i.plot(xs, ys, linestyle=':', color='blue', label=f"Schmalz OPT", alpha=0.8, linewidth=2)

            # Formatting
            ax_h.set_title("HEALTHY SIDE")
            ax_i.set_title("IMPAIRED SIDE")
            ax_diff.set_title("L-R DIFFERENCE")
            ax_diff.axhline(0, color='black', lw=0.8, ls='--')
            for ax in axes:
                ax.legend(fontsize='xx-small', ncol=1)
                ax.grid(True, alpha=0.2)
            
            plt.tight_layout()
            plt.show()



    @staticmethod
    def plot_grf_component_symmetry_all_runs(
        grf_components,
        all_loaded_data,
        run_step_data,
        interp_len=100,
        body_weight_to_normalize= 1,
    ):
        """
        Plot mean of right of all runs, mean of left, and the difference, side by side in one figure.
        Each plot includes all runs.
        """
        for comp_idx, comp in enumerate(grf_components):
            run_keys = list(all_loaded_data.keys())
            mean_left_all = []
            std_left_all = []
            mean_right_all = []
            std_right_all = []
            diff_all = []
            for run_key in run_keys:
                run_dict = all_loaded_data[run_key]
                step_data = run_step_data[run_key]
                all_grf_l = np.array(run_dict["all_grf_l"])
                all_grf_r = np.array(run_dict["all_grf_r"])
                all_step_start_left = step_data["step_start_left"]
                all_step_start_right = step_data["step_start_right"]

                # Left steps
                left_steps = []
                if all_grf_l is not None and all_step_start_left and len(all_step_start_left) > 1:
                    for j in range(len(all_step_start_left) - 1):
                        start, end = all_step_start_left[j]-1, all_step_start_left[j + 1]-1
                        y = [float(np.array(a)[comp_idx]) for a in all_grf_l[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        left_steps.append(y_interp)
                # Right steps
                right_steps = []
                if all_grf_r is not None and all_step_start_right and len(all_step_start_right) > 1:
                    for j in range(len(all_step_start_right) - 1):
                        start, end = all_step_start_right[j]-1, all_step_start_right[j + 1]-1
                        y = [float(np.array(a)[comp_idx]) for a in all_grf_r[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        right_steps.append(y_interp)
                # Save means/stds if enough steps
                if left_steps and right_steps:
                    mean_left = np.mean(left_steps, axis=0)/body_weight_to_normalize
                    std_left = np.std(left_steps, axis=0)/body_weight_to_normalize
                    mean_right = np.mean(right_steps, axis=0)/body_weight_to_normalize
                    std_right = np.std(right_steps, axis=0)/body_weight_to_normalize
                    diff = mean_left - mean_right
                    mean_left_all.append((run_key, mean_left, std_left))
                    mean_right_all.append((run_key, mean_right, std_right))
                    diff_all.append((run_key, diff))

            x = np.linspace(0, 100, interp_len)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
            # Plot mean of right of all runs
            for run_key, mean_right, std_right in mean_right_all:
                axes[0].plot(x, mean_right, label=f"{run_key}", linestyle='--')
                axes[0].fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15)
            axes[0].set_title(f"{comp} Right mean (all runs)")
            axes[0].set_xlabel("Interpolated Step (%)")
            axes[0].set_ylabel(f"{comp} (N or Nm / BW)")
            axes[0].legend()
            # Plot mean of left of all runs
            for run_key, mean_left, std_left in mean_left_all:
                axes[1].plot(x, mean_left, label=f"{run_key}", linestyle='-')
                axes[1].fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15)
            axes[1].set_title(f"{comp} Left mean (all runs)")
            axes[1].set_xlabel("Interpolated Step (%)")
            axes[1].set_ylabel(f"{comp} (N or Nm / BW)")
            axes[1].legend()
            # Plot difference (left - right) of all runs
            for run_key, diff in diff_all:
                axes[2].plot(x, diff, label=f"{run_key} Left-Right", linestyle=':')
            axes[2].axhline(0, color='gray', linestyle=':', linewidth=1)
            axes[2].set_title(f"{comp} Left-Right difference (all runs)")
            axes[2].set_xlabel("Interpolated Step (%)")
            axes[2].set_ylabel(f"{comp} difference (N or Nm / BW)")
            axes[2].legend()
            plt.tight_layout()
            plt.show()
            plt.close()

    

    def plot_grf_Fz_switch_all_runs(
        self,
        all_loaded_data,
        run_step_data,
        interp_len=100,
        body_weight_to_normalize=1,
        threshold=1e-2,
    ):
        """
        Plot all right GRF Fz (5th entry) for all runs in one plot, and all left in another.
        Mark swing-to-stance phase switches (where mean Fz crosses threshold).
        Returns:
            left_switches (dict): {run_key: [indices]}
            right_switches (dict): {run_key: [indices]}
        """
        # Prepare color cycle for consistent coloring
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        run_keys = list(all_loaded_data.keys())
        right_curves = []
        left_curves = []
        right_switches = {}
        left_switches = {}

        # Collect curves and switches
        for idx, run_key in enumerate(run_keys):
            run_dict = all_loaded_data[run_key]
            step_data = run_step_data[run_key]
            all_grf_l = np.array(run_dict["all_grf_l"])
            all_grf_r = np.array(run_dict["all_grf_r"])
            all_step_start_left = step_data["step_start_left"]
            all_step_start_right = step_data["step_start_right"]

            # Interpolated steps for right
            right_steps = []
            if all_grf_r is not None and all_step_start_right and len(all_step_start_right) > 1:
                for j in range(len(all_step_start_right) - 1):
                    start, end = all_step_start_right[j]-1, all_step_start_right[j + 1]-1
                    y = [float(np.array(a)[5]) for a in all_grf_r[start:end]]
                    if len(y) < 2:
                        continue
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    y_interp = np.interp(x_new, x_old, y)
                    right_steps.append(y_interp)
            if right_steps:
                mean_right = np.mean(right_steps, axis=0) / body_weight_to_normalize
                above = mean_right > threshold
                switches = [i for i in range(1, len(mean_right)) if above[i-1] and not above[i]]
                right_curves.append((run_key, mean_right, idx))
                right_switches[run_key] = switches

            # Interpolated steps for left
            left_steps = []
            if all_grf_l is not None and all_step_start_left and len(all_step_start_left) > 1:
                for j in range(len(all_step_start_left) - 1):
                    start, end = all_step_start_left[j]-1, all_step_start_left[j + 1]-1
                    y = [float(np.array(a)[5]) for a in all_grf_l[start:end]]
                    if len(y) < 2:
                        continue
                    x_old = np.linspace(0, 1, len(y))
                    x_new = np.linspace(0, 1, interp_len)
                    y_interp = np.interp(x_new, x_old, y)
                    left_steps.append(y_interp)
            if left_steps:
                mean_left = np.mean(left_steps, axis=0) / body_weight_to_normalize
                above = mean_left > threshold
                switches = [i for i in range(1, len(mean_left)) if above[i-1] and not above[i]]
                left_curves.append((run_key, mean_left, idx))
                left_switches[run_key] = switches

        x = np.linspace(0, 100, interp_len)
        # Plot all right curves
        plt.figure(figsize=(12, 6))
        for run_key, curve, idx in right_curves:
            color = color_cycle[idx % len(color_cycle)]
            plt.plot(x, curve, label=f"{run_key} Right Fz", color=color)
            for switch_idx in right_switches[run_key]:
                plt.axvline(x[switch_idx], color=color, linestyle='--', alpha=0.7)
        plt.axhline(threshold, color='gray', linestyle=':', linewidth=1, label="Threshold")
        plt.title("All runs: Right GRF Fz (5th entry) & swing/stance switches")
        plt.xlabel("Interpolated Step (%)")
        plt.ylabel("Fz (N or Nm / BW)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

        # Plot all left curves
        plt.figure(figsize=(12, 6))
        for run_key, curve, idx in left_curves:
            color = color_cycle[idx % len(color_cycle)]
            plt.plot(x, curve, label=f"{run_key} Left Fz", color=color)
            for switch_idx in left_switches[run_key]:
                plt.axvline(x[switch_idx], color=color, linestyle='--', alpha=0.7)
        plt.axhline(threshold, color='gray', linestyle=':', linewidth=1, label="Threshold")
        plt.title("All runs: Left GRF Fz (5th entry) & swing/stance switches")
        plt.xlabel("Interpolated Step (%)")
        plt.ylabel("Fz (N or Nm / BW)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

        return left_switches, right_switches



    def plot_grf_Fz_switch_all_dirs(
        self,
        all_loaded_data,
        run_step_data,
        interp_len=100,
        body_weight_to_normalize=1,
        threshold=1e-2,
    ):
        """
        Plots mean GRF Fz for each seed separately.
        Input structure: all_loaded_data[category][seed_key]
        """
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        # Structure to return: {category: {seed: [switches]}}
        right_switches = {}
        left_switches = {}
        
        # Storage for plotting
        plot_data_right = []
        plot_data_left = []
        color_idx = 0

        # 1. Iterate through Categories (e.g., 'z_-6deg')
        for cat_key, seeds_dict in all_loaded_data.items():
            right_switches[cat_key] = {}
            left_switches[cat_key] = {}
            
            # 2. Iterate through Seeds (e.g., 'seed1')
            for seed_key, run_dict in seeds_dict.items():
                
                # --- SAFE ACCESS TO STEP DATA ---
                # We check if cat_key and seed_key exist in run_step_data
                if cat_key not in run_step_data:
                    print(f"Skipping: Category '{cat_key}' not found in run_step_data")
                    continue
                
                cat_step_entry = run_step_data[cat_key]
                
                # Check if run_step_data is nested [cat][seed] or flat [cat]
                if isinstance(cat_step_entry, dict) and seed_key in cat_step_entry:
                    step_data = cat_step_entry[seed_key]
                else:
                    # Fallback if step_data is just indexed by category
                    step_data = cat_step_entry

                # Extract data arrays
                all_grf_l = run_dict.get("all_grf_l", [])
                all_grf_r = run_dict.get("all_grf_r", [])
                all_step_start_left = step_data.get("step_start_left", [])
                all_step_start_right = step_data.get("step_start_right", [])

                # --- Process Right Side ---
                right_steps = []
                if all_grf_r is not None and len(all_step_start_right) > 1:
                    for j in range(len(all_step_start_right) - 1):
                        start, end = int(all_step_start_right[j]-1), int(all_step_start_right[j + 1]-1)
                        # Get 5th index (Fz) for this segment
                        segment = all_grf_r[start:end]
                        if len(segment) < 2: continue
                        
                        y = [float(np.array(a)[5]) for a in segment]
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        right_steps.append(np.interp(x_new, x_old, y))
                
                if right_steps:
                    mean_right = np.mean(right_steps, axis=0) / body_weight_to_normalize
                    above = mean_right > threshold
                    switches = [i for i in range(1, len(mean_right)) if above[i-1] and not above[i]]
                    
                    right_switches[cat_key][seed_key] = switches
                    plot_data_right.append((f"{cat_key}_{seed_key}", mean_right, color_idx, switches))

                # --- Process Left Side ---
                left_steps = []
                if all_grf_l is not None and len(all_step_start_left) > 1:
                    for j in range(len(all_step_start_left) - 1):
                        start, end = int(all_step_start_left[j]-1), int(all_step_start_left[j + 1]-1)
                        segment = all_grf_l[start:end]
                        if len(segment) < 2: continue
                            
                        y = [float(np.array(a)[5]) for a in segment]
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        left_steps.append(np.interp(x_new, x_old, y))
                
                if left_steps:
                    mean_left = np.mean(left_steps, axis=0) / body_weight_to_normalize
                    above = mean_left > threshold
                    switches = [i for i in range(1, len(mean_left)) if above[i-1] and not above[i]]
                    
                    left_switches[cat_key][seed_key] = switches
                    plot_data_left.append((f"{cat_key}_{seed_key}", mean_left, color_idx, switches))
                
                color_idx += 1

        # --- Plotting ---
        x_axis = np.linspace(0, 100, interp_len)
        for side, dataset in [("Right", plot_data_right), ("Left", plot_data_left)]:
            if not dataset: continue
            plt.figure(figsize=(12, 6))
            for label, curve, idx, switches in dataset:
                color = color_cycle[idx % len(color_cycle)]
                plt.plot(x_axis, curve, label=label, color=color)
                for s_idx in switches:
                    plt.axvline(x_axis[s_idx], color=color, linestyle='--', alpha=0.5)
            
            plt.axhline(threshold, color='gray', linestyle=':', label="Threshold")
            plt.title(f"All Seeds: {side} GRF Fz")
            plt.xlabel("Step Progress (%)")
            plt.ylabel("Fz (Normalized)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

        return left_switches, right_switches



    @staticmethod
    def plot_grf_component_symmetry_summed_all_runs(
        grf_components,
        all_loaded_data,
        run_step_data,
        interp_len=100,
        body_weight_to_normalize=1,
    ):
        """
        Plot summed mean of right, left, and difference (symmetry) across all runs
        in one figure per GRF component.
        """
        for comp_idx, comp in enumerate(grf_components):
            left_steps_all_runs = []
            right_steps_all_runs = []

            for run_key, run_dict in all_loaded_data.items():
                step_data = run_step_data[run_key]
                all_grf_l = np.array(run_dict["all_grf_l"])
                all_grf_r = np.array(run_dict["all_grf_r"])
                all_step_start_left = step_data["step_start_left"]
                all_step_start_right = step_data["step_start_right"]

                # Process left steps
                if all_grf_l is not None and all_step_start_left and len(all_step_start_left) > 1:
                    for j in range(len(all_step_start_left) - 1):
                        start, end = all_step_start_left[j]-1, all_step_start_left[j + 1]-1
                        y = [float(np.array(a)[comp_idx]) for a in all_grf_l[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        left_steps_all_runs.append(y_interp)

                # Process right steps
                if all_grf_r is not None and all_step_start_right and len(all_step_start_right) > 1:
                    for j in range(len(all_step_start_right) - 1):
                        start, end = all_step_start_right[j]-1, all_step_start_right[j + 1]-1
                        y = [float(np.array(a)[comp_idx]) for a in all_grf_r[start:end]]
                        if len(y) < 2:
                            continue
                        x_old = np.linspace(0, 1, len(y))
                        x_new = np.linspace(0, 1, interp_len)
                        y_interp = np.interp(x_new, x_old, y)
                        right_steps_all_runs.append(y_interp)

            # Compute means and stds across all runs
            if left_steps_all_runs and right_steps_all_runs:
                mean_left = np.mean(left_steps_all_runs, axis=0) / body_weight_to_normalize
                std_left = np.std(left_steps_all_runs, axis=0) / body_weight_to_normalize
                mean_right = np.mean(right_steps_all_runs, axis=0) / body_weight_to_normalize
                std_right = np.std(right_steps_all_runs, axis=0) / body_weight_to_normalize
                diff = mean_left - mean_right

                x = np.linspace(0, 100, interp_len)
                plt.figure(figsize=(8, 5))

                # Right
                plt.plot(x, mean_right, color='r', label="Right mean")
                plt.fill_between(x, mean_right - std_right, mean_right + std_right, alpha=0.15, color='r')
                # axes[0].set_title(f"{comp} Right mean (summed)")
                # axes[0].set_xlabel("Interpolated Step (%)")
                # axes[0].set_ylabel(f"{comp} (N or Nm / BW)")
                # axes[0].legend()

                # Left
                plt.plot(x, mean_left, color='b', label="Left mean")
                plt.fill_between(x, mean_left - std_left, mean_left + std_left, alpha=0.15, color='b')
                # axes[1].set_title(f"{comp} Left mean (summed)")
                # axes[1].set_xlabel("Interpolated Step (%)")
                # axes[1].set_ylabel(f"{comp} (N or Nm / BW)")
                # axes[1].legend()

                # Difference
                plt.plot(x, diff, color='g', label="Left-Right")
                plt.axhline(0, color='gray', linestyle=':', linewidth=1)
                # axes[2].set_title(f"{comp} Left-Right difference (summed)")
                # axes[2].set_xlabel("Interpolated Step (%)")
                # axes[2].set_ylabel(f"{comp} difference (N or Nm / BW)")
                plt.legend()
                plt.title(f"{comp} (summed across runs)")
                plt.xlabel("Interpolated Step (%)")
                plt.ylabel(f"{comp} (N or Nm / BW)")

                plt.tight_layout()
                plt.show()
                plt.close()




    def plot_grf_component_symmetry_summed_all_dirs(
        self,  # <- important: instance method
        grf_components,
        all_loaded_data,
        run_step_data,
        interp_len=100,
        body_weight_to_normalize=1,
    ):
        """
        Plot summed mean of right, left, and difference (symmetry) across directories.
        One figure per GRF component, showing mean ± std for each directory.
        """

        if not isinstance(grf_components, (list, tuple)):
            raise ValueError("grf_components must be a list or tuple of GRF component names.")

        for comp_idx, comp in enumerate(grf_components):
            plt.figure(figsize=(20, 6))
            colors = plt.cm.tab10.colors  # For different directories

            for i, dir_label in enumerate(all_loaded_data.keys()):
                left_steps_dir = []
                right_steps_dir = []

                for run_key, run_dict in all_loaded_data[dir_label].items():
                    step_data = run_step_data[dir_label][run_key]
                    all_grf_l = np.array(run_dict.get("all_grf_l"))
                    all_grf_r = np.array(run_dict.get("all_grf_r"))
                    all_step_start_left = step_data.get("step_start_left", [])
                    all_step_start_right = step_data.get("step_start_right", [])

                    # Process left steps
                    if all_grf_l is not None and len(all_step_start_left) > 1:
                        for j in range(len(all_step_start_left) - 1):
                            start, end = all_step_start_left[j]-1, all_step_start_left[j + 1]-1
                            y = [float(np.array(a)[comp_idx]) for a in all_grf_l[start:end]]
                            if len(y) < 2:
                                continue
                            x_old = np.linspace(0, 1, len(y))
                            x_new = np.linspace(0, 1, interp_len)
                            y_interp = np.interp(x_new, x_old, y)
                            left_steps_dir.append(y_interp)

                    # Process right steps
                    if all_grf_r is not None and len(all_step_start_right) > 1:
                        for j in range(len(all_step_start_right) - 1):
                            start, end = all_step_start_right[j]-1, all_step_start_right[j + 1]-1
                            y = [float(np.array(a)[comp_idx]) for a in all_grf_r[start:end]]
                            if len(y) < 2:
                                continue
                            x_old = np.linspace(0, 1, len(y))
                            x_new = np.linspace(0, 1, interp_len)
                            y_interp = np.interp(x_new, x_old, y)
                            right_steps_dir.append(y_interp)

                # Aggregate per directory
                if left_steps_dir and right_steps_dir:
                    mean_left = np.mean(left_steps_dir, axis=0) / body_weight_to_normalize
                    std_left = np.std(left_steps_dir, axis=0) / body_weight_to_normalize
                    mean_right = np.mean(right_steps_dir, axis=0) / body_weight_to_normalize
                    std_right = np.std(right_steps_dir, axis=0) / body_weight_to_normalize
                    mean_diff = mean_left - mean_right
                    std_diff = np.sqrt(std_left**2 + std_right**2)

                    x = np.linspace(0, 100, interp_len)
                    color = colors[i % len(colors)]
                    # Left
                    plt.plot(x, mean_left, label=f"{dir_label} Left", color=color, linestyle='-')
                    plt.fill_between(x, mean_left - std_left, mean_left + std_left, color=color, alpha=0.15)
                    # Right
                    plt.plot(x, mean_right, label=f"{dir_label} Right", color=color, linestyle='--')
                    plt.fill_between(x, mean_right - std_right, mean_right + std_right, color=color, alpha=0.1)
                    # Symmetry
                    plt.plot(x, mean_diff, label=f"{dir_label} L-R Diff", color=color, linestyle=':')

            plt.axhline(0, color='gray', linestyle=':', linewidth=1)
            plt.xlabel("Interpolated Step (%)")
            plt.ylabel(f"{comp} (N or Nm / BW)")
            plt.title(f"{comp} GRF Component Symmetry Across Directories")
            plt.legend(ncol=3, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            plt.show()
            plt.close()





    def plot_full_grf_with_step_markers(
    self,
    all_loaded_data,
    run_step_data,
    component_index=5,
    ):
        """
        For each run, plot full GRF data for left and right in separate subplots.
        Add vertical black lines at step start indices.
        
        Args:
        - all_loaded_data (dict): {run_key: {"all_grf_l": array, "all_grf_r": array}}
        - run_step_data (dict): {run_key: {"step_start_left": [...], "step_start_right": [...]} }
        - component_index (int): GRF component to plot (default is 5)
        """
        for run_key in all_loaded_data.keys():
            run_dict = all_loaded_data[run_key]
            step_data = run_step_data[run_key]
            

            all_grf_l = np.array(run_dict["all_grf_l"])
            all_grf_r = np.array(run_dict["all_grf_r"])
            step_start_left = step_data.get("step_start_left", [])
            step_start_right = step_data.get("step_start_right", [])

            grf_l = all_grf_l[:, component_index]
            grf_r = all_grf_r[:, component_index]
            x_l = np.arange(len(grf_l))
            x_r = np.arange(len(grf_r))

            # From GRF Contact detection 
            filtered_left = step_data.get("all_contact_left", [])
            filtered_right = step_data.get("all_contact_right", [])
            # print('filtered_data_left', filtered_left)
            # print('filtered_data_right', filtered_right)

            # From contact 
            # all_foot_ground_contact_left = run_dict.get("all_foot_ground_contact_left", [])
            # all_foot_ground_contact_right = run_dict.get("all_foot_ground_contact_right", [])
            # filtered_left =  np.zeros(len(all_foot_ground_contact_left))
            # filtered_right = np.zeros(len(all_foot_ground_contact_right))
            # for i in range(len(filtered_left)):
            #     if all_foot_ground_contact_left[i] != []:
            #         filtered_left[i]=100
            #     else: 
            #         filtered_left[i] = 0

            # for i in range(len(filtered_right)):
            #     if all_foot_ground_contact_right[i] != []:
            #         filtered_right[i]=100
            #     else: 
            #         filtered_right[i] = 0
              
            fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
            # Left GRF
            axes[0].plot(filtered_left, label="Contact", color='red')
            axes[0].plot(x_l, grf_l, label="Left GRF", color='blue')
            for step in step_start_left:
                axes[0].axvline(x=step, color='black', linestyle='--', linewidth=1)
            axes[0].set_title(f"{run_key} - Left GRF Component [{component_index}]")
            axes[0].set_ylabel("Force (N)")
            axes[0].legend()
            print('step_start_left', step_start_left)
            # Right GRF
            axes[1].plot(filtered_right, label="Contact", color='red')
            axes[1].plot(x_r, grf_r, label="Right GRF", color='green')
            for step in step_start_right:
                axes[1].axvline(x=step, color='black', linestyle='--', linewidth=1)
            axes[1].set_title(f"{run_key} - Right GRF Component [{component_index}]")
            axes[1].set_xlabel("Frame Index")
            axes[1].set_ylabel("Force (N)")
            axes[1].legend()
            print('step_start_right', step_start_right)

            plt.tight_layout()
            plt.show()
            plt.close()


    
    def plot_full_grf_with_step_markers_all_dirs(
        self,
        all_loaded_data,
        run_step_data,
        component_index=5,
    ):
        """
        For each category and seed, plot full GRF data for left and right in separate subplots.
        Add vertical black lines at step start indices.
        
        Structure: all_loaded_data[cat_key][seed_key]
        """
        # 1. Iterate through Categories (e.g., 'z_-6deg')
        for cat_key, seeds_dict in all_loaded_data.items():
            
            # 2. Iterate through Seeds (e.g., 'seed1')
            for seed_key, run_dict in seeds_dict.items():
                
                # --- SAFE ACCESS TO STEP DATA ---
                if cat_key not in run_step_data:
                    print(f"Skipping plot for {cat_key}: not found in run_step_data")
                    continue
                    
                cat_step_entry = run_step_data[cat_key]
                
                # Check if run_step_data is nested [cat][seed] or flat [cat]
                if isinstance(cat_step_entry, dict) and seed_key in cat_step_entry:
                    step_data = cat_step_entry[seed_key]
                else:
                    step_data = cat_step_entry

                # Extract data arrays safely
                # Note: list of arrays converted to 2D numpy array for slicing
                all_grf_l = np.array(run_dict.get("all_grf_l", []))
                all_grf_r = np.array(run_dict.get("all_grf_r", []))
                
                # Check if we actually have data to plot
                if all_grf_l.size == 0 or all_grf_r.size == 0:
                    print(f"No GRF data found for {cat_key} {seed_key}. Skipping.")
                    continue

                step_start_left = step_data.get("step_start_left", [])
                step_start_right = step_data.get("step_start_right", [])
                filtered_left = step_data.get("all_contact_left", [])
                filtered_right = step_data.get("all_contact_right", [])

                # Slicing the specific component (default 5 for Fz)
                grf_l = all_grf_l[:, component_index]
                grf_r = all_grf_r[:, component_index]
                
                # Generate X-axis (frame indices)
                x_l = np.arange(len(grf_l))
                x_r = np.arange(len(grf_r))

                # Create Plot
                fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
                
                # --- Top Plot: Left GRF ---
                # Plot contact signal if it exists and matches length
                if len(filtered_left) == len(x_l):
                    axes[0].plot(x_l, filtered_left, label="Contact Signal", color='red', alpha=0.3)
                
                axes[0].plot(x_l, grf_l, label=f"Left GRF [idx {component_index}]", color='blue')
                
                for step in step_start_left:
                    axes[0].axvline(x=step, color='black', linestyle='--', linewidth=1, alpha=0.7)
                
                axes[0].set_title(f"Category: {cat_key} | {seed_key} - Left Side")
                axes[0].set_ylabel("Force (N)")
                axes[0].legend(loc='upper right')

                # --- Bottom Plot: Right GRF ---
                if len(filtered_right) == len(x_r):
                    axes[1].plot(x_r, filtered_right, label="Contact Signal", color='red', alpha=0.3)
                
                axes[1].plot(x_r, grf_r, label=f"Right GRF [idx {component_index}]", color='green')
                
                for step in step_start_right:
                    axes[1].axvline(x=step, color='black', linestyle='--', linewidth=1, alpha=0.7)
                
                axes[1].set_title(f"Category: {cat_key} | {seed_key} - Right Side")
                axes[1].set_xlabel("Frame Index")
                axes[1].set_ylabel("Force (N)")
                axes[1].legend(loc='upper right')

                plt.tight_layout()
                plt.show()
                plt.close()



    def get_stride_distance_length(self, all_body_poses, foot_body_name, step_start_indices):
        """
        Calculate step lengths in meters based on foot position at step start frames.

        Args:
            all_body_poses (np.array): Array of body poses per frame (shape: [frames, bodies, 3])
            foot_body_name (str): Name of the foot body (e.g., "toes")
            step_start_indices (list): List of frame indices where steps start

        Returns:
            list: Step lengths in meters
        """
        # Assume foot_body_name maps to a known index
        # foot_index = get_body_index(foot_body_name)  # You must define this mapping
        step_lengths = []
        pose_dict = all_body_poses.item()
        # foot_pos = pose_dict[foot_body_name]
        foot_pos_x = [pose[0] for pose in pose_dict[foot_body_name]] #foot_pos[0][:]

        for i in range(len(step_start_indices) - 1):
            start = step_start_indices[i]
            end = step_start_indices[i + 1]

            pos_start = foot_pos_x[start] #foot_pos[:,start]  # X position
            pos_end = foot_pos_x[end]  #foot_pos[:,end]    # X position

            step_length = abs(pos_end - pos_start)
            step_lengths.append(step_length)

        return step_lengths



    def get_step_distance_length(self, all_body_poses, foot_body_name, step_start_indices_right, step_start_indices_left):
        """
        Calculate step lengths in meters based on foot position at step start frames.

        Args:
            all_body_poses (np.array): Array of body poses per frame (shape: [frames, bodies, 3])
            foot_body_name (str): Name of the foot body (e.g., "toes")
            step_start_indices (list): List of frame indices where steps start

        Returns:
            list: Step lengths in meters
        """
        # Assume foot_body_name maps to a known index
        # foot_index = get_body_index(foot_body_name)  # You must define this mapping
        step_lengths_right = []
        step_lengths_left = []
        pose_dict = all_body_poses.item()
        foot_body_name_r = foot_body_name + '_r'
        foot_body_name_l = foot_body_name + '_l'
        # foot_pos = pose_dict[foot_body_name]
        foot_pos_r_x = [pose[0] for pose in pose_dict[foot_body_name_r]] #foot_pos[0][:]
        foot_pos_l_x = [pose[0] for pose in pose_dict[foot_body_name_l]] #foot_pos[0][:]

        # Step length is distance between right and left foot at each heel strike
        # Ensure indices do not go out of bounds
        min_len = min(len(step_start_indices_right), len(step_start_indices_left))
        # Calculate right-to-left step lengths
        for i in range(min_len - 1):
            right_step_start = step_start_indices_right[i]
            left_step_next = step_start_indices_left[i + 1] if step_start_indices_left[0] < step_start_indices_right[0] else step_start_indices_left[i]
            if right_step_start < len(foot_pos_r_x) and left_step_next < len(foot_pos_l_x):
                right_step_pos_start = foot_pos_r_x[right_step_start]
                left_step_pos_next = foot_pos_l_x[left_step_next]
                step_length_left = abs(left_step_pos_next - right_step_pos_start)
                step_lengths_left.append(step_length_left)
        # Calculate left-to-right step lengths
        for i in range(min_len - 1):
            left_step_start = step_start_indices_left[i]
            right_step_next = step_start_indices_right[i + 1] if step_start_indices_right[0] < step_start_indices_left[0] else step_start_indices_right[i]
            if left_step_start < len(foot_pos_l_x) and right_step_next < len(foot_pos_r_x):
                left_step_pos_start = foot_pos_l_x[left_step_start]
                right_step_pos_next = foot_pos_r_x[right_step_next]
                step_length_right= abs(right_step_pos_next - left_step_pos_start)
                step_lengths_right.append(step_length_right)


        return step_lengths_right, step_lengths_left

    

    def plot_body_pos_all_runs(self, all_loaded_data, body):
        plt.figure(figsize=(12, 6))
        for run_key, run_dict in all_loaded_data.items():
            all_body_poses = np.array(run_dict["all_body_poses"])
            pose_dict = all_body_poses.item()
            # print('pose_dict body: ', [pose for pose in pose_dict[body]])
            body_pos_x = np.array([pose[0] for pose in pose_dict[body]])
            body_pos_z = np.array([pose[1] for pose in pose_dict[body]])
            # body_pos_y = np.array([pose[2] for pose in pose_dict[body]])
            body_pos_x = body_pos_x[:-1]
            body_pos_z = body_pos_z[:-1]
            # body_pos_y = body_pos_y[:-1]
            # body_pos_x_norm = - body_pos_x + body_pos_x[0]
            # body_pos_z_norm = -body_pos_z + body_pos_z[0]
            # print('body_pos_x[0]: ', body_pos_x[-1])
            # print('body_pos_z[0]: ', body_pos_z[-1])


            plt.plot(body_pos_x,body_pos_z, label=run_key)
            # plt.plot(body_pos_x,body_pos_y, label=run_key)
            # # Plotting code for body positions
            # # all_body_poses is a dict: {body_name: [Array([x, y, z]), ...]}
            # pose_dict = all_body_poses.item() if hasattr(all_body_poses, "item") else all_body_poses
            # if body not in pose_dict:
            #     print(f"Body '{body}' not found in all_body_poses for run {run_key}")
            #     continue
            # poses = pose_dict[body]
            # xs = [float(p[0]) for p in poses]
            # xs = np.array([float(p[0]) for p in poses])
            # # xs_norm = xs - xs[0]
            # zs = np.array([float(p[2]) for p in poses])
            # zs_norm = zs - zs[0]
            # # plt.plot(xs_norm, zs_norm, label=run_key)
            # plt.plot(zs, label=run_key)
        plt.title(f"Body Position - {body}")
        plt.xlabel("X Position")
        plt.ylabel("Z Position")
        plt.legend()
        plt.show()


    

    def plot_body_pos_all_dirs(self, all_loaded_data, body):
        """
        Plots the X-Z trajectory of a specific body part across all directories and runs.
        Nested structure: all_loaded_data[dir_label][run_key]
        """
        plt.figure(figsize=(12, 7))
        
        # Get a color map to distinguish different directory groups
        colors = plt.cm.tab10.colors
        
        # 1. Loop through Directories
        for d_idx, (dir_label, runs_dict) in enumerate(all_loaded_data.items()):
            color = colors[d_idx % len(colors)]
            
            # 2. Loop through Runs within each directory
            for r_idx, (run_key, run_dict) in enumerate(runs_dict.items()):
                if "all_body_poses" not in run_dict:
                    continue
                    
                # Handle the data extraction
                raw_poses = run_dict["all_body_poses"]
                
                # Use .item() if it's a numpy object, otherwise assume it's a dict
                pose_dict = raw_poses.item() if hasattr(raw_poses, "item") else raw_poses
                
                if body not in pose_dict:
                    print(f"Warning: Body '{body}' not found in {dir_label}/{run_key}")
                    continue
                
                # Extract X and Z coordinates
                # Convention: pose[0] is X, pose[1] is Z (often height or forward depending on your sim)
                body_poses = pose_dict[body]
                body_pos_x = np.array([pose[0] for pose in body_poses])
                body_pos_z = np.array([pose[1] for pose in body_poses])
                
                # Remove the last point if desired (per your original code)
                if len(body_pos_x) > 1:
                    body_pos_x = body_pos_x[:-1]
                    body_pos_z = body_pos_z[:-1]

                # Use a label only for the first run of a directory to avoid legend clutter
                label = dir_label if r_idx == 0 else None
                
                # Plot the trajectory
                # We use alpha=0.6 so overlapping runs from the same directory are visible
                plt.plot(body_pos_x, body_pos_z, label=label, color=color, alpha=0.6, linewidth=1.5)

        plt.title(f"Body Trajectory: {body.upper()} (X vs Z)")
        plt.xlabel("X Position (m)")
        plt.ylabel("Z Position (m)")
        
        # Place legend outside if there are many directories
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()



    
    @staticmethod
    def plot_sensor_force_pylon_all_dirs(
        all_loaded_data, 
        run_step_data, 
        direction, 
        interp_len=100, 
        body_weight_to_normalize=1, 
        pylon_alignment_baseline_file= None,
        pylon_alignment_data_mass= 85*9.81, 
        sensor_force_names=None
    ):
        """
        Plot mean of all runs for pylon/socket sensor data across all directories.
        Handles the nested dict: all_loaded_data[dir_label][run_key]
        """
        force_direction = {"x": 0, "y": 1, "z": 2}.get(direction, 0)

        # 1. Extract sensor names from the first available run if not provided
        if sensor_force_names is None: 
            try:
                first_dir = next(iter(all_loaded_data))
                first_run = next(iter(all_loaded_data[first_dir]))
                sensor_force_names = all_loaded_data[first_dir][first_run]["sensor_force_names"]
            except (StopIteration, KeyError):
                print("Error: Could not find sensor_force_names in data structure.")
                return

        sensor_force_names = [sensor for sensor in sensor_force_names if 'torque' in sensor.lower()]

        # 2. Filter for pylon or socket sensors
        pylon_sensors = [name for name in sensor_force_names if "pylon" in name.lower() or "socket" in name.lower()]

        if not pylon_sensors:
            print("No pylon or socket sensors found.")
            return

        # 3. Main Plotting Loop per sensor
        for sensor_name in pylon_sensors:
            plt.figure(figsize=(10, 6))
            x_axis = np.linspace(0, 100, interp_len)
            colors = plt.cm.tab10.colors
            
            # Iterate through directories (e.g., different conditions or models)
            for i, dir_label in enumerate(all_loaded_data.keys()):
                color = colors[i % len(colors)]
                all_steps_in_dir = []

                # Iterate through runs within that directory
                for run_key, run_dict in all_loaded_data[dir_label].items():
                    step_data = run_step_data[dir_label][run_key]
                    
                    if sensor_name not in run_dict.get("all_sensor_force", {}):
                        continue
                    
                    sensor_data = run_dict["all_sensor_force"][sensor_name]
                    
                    # Determine side based on naming convention
                    # Assumes '_r' or 'right' in name implies right side, else left
                    side_key = "step_start_right" if ("_r" in sensor_name.lower() or "right" in sensor_name.lower()) else "step_start_left"
                    all_step_start = step_data.get(side_key, [])

                    if len(all_step_start) > 1:
                        for j in range(len(all_step_start) - 1):
                            start, end = all_step_start[j], all_step_start[j + 1]
                            
                            # Extract force component
                            y = [float(np.array(a)[force_direction]) for a in sensor_data[start:end]]
                            
                            if len(y) > 5: # Ensure enough points for a valid step
                                x_old = np.linspace(0, 1, len(y))
                                x_new = np.linspace(0, 1, interp_len)
                                all_steps_in_dir.append(np.interp(x_new, x_old, y))

                # 4. Compute and plot mean/std for the directory
                if all_steps_in_dir:
                    all_steps_in_dir = np.array(all_steps_in_dir) / body_weight_to_normalize
                    
                    # Apply sign convention: Flip Z-axis if it's vertical ground reaction force
                    # Often necessary depending on coordinate system orientation
                    if force_direction == 2:
                        all_steps_in_dir *= -1

                    mean_val = np.mean(all_steps_in_dir, axis=0)
                    std_val = np.std(all_steps_in_dir, axis=0)

                    plt.plot(x_axis, mean_val, label=f"{dir_label}", color=color, linewidth=2)
                    plt.fill_between(x_axis, mean_val - std_val, mean_val + std_val, color=color, alpha=0.15)
            
            if 'pylon' in sensor_name and pylon_alignment_baseline_file:
                    baseline_data = {}
                    # read baseline_data from csv file
                    if os.path.isfile(pylon_alignment_baseline_file):
                        with open(pylon_alignment_baseline_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                label_match = re.match(r'^([A-Za-z+\-_]+)\s*=\s*\[?', line)
                                if label_match: 
                                    current_label = label_match.group(1)
                                    baseline_data[current_label] = []
                                    continue
                                if current_label:
                                    vals = line.split(';')
                                    if len(vals) == 2:
                                        try:
                                            x, y = map(float, vals)
                                            baseline_data[current_label].append((x, y))
                                        except ValueError:
                                            continue   

            if direction == 'z' and baseline_data.get('Sagittal', []):
                xs, ys = zip(*baseline_data.get('Sagittal', []))
                # Sort data by x before plotting
                xs = np.array(xs) * 100  # Convert from 0-1 to 0-100
                ys = np.array(ys)
                sort_idx = np.argsort(xs)
                xs = xs[sort_idx]
                ys = ys[sort_idx]
                plt.plot(xs, ys, label="OPT", color='grey', linestyle='-', linewidth=2)
            elif direction == 'x' and baseline_data.get('Coronal', []):
                    xs, ys = zip(*baseline_data.get('Coronal', []))
                    # Sort data by x before plotting
                    xs = np.array(xs) * 100  # Convert from 0-1 to 0-100
                    ys = np.array(ys)
                    sort_idx = np.argsort(xs)
                    xs = xs[sort_idx]
                    ys = ys[sort_idx]
                    plt.plot(xs, ys, label="OPT", color='grey', linestyle='-', linewidth=2)


            # Final Plot adjustments
            plt.title(f"Pylon Force Component: {sensor_name} ({direction}-axis)")
            plt.xlabel("Gait Cycle (%)")
            plt.ylabel("Force (Normalized)" if body_weight_to_normalize != 1 else "Force (N)")
            plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
            plt.close()

