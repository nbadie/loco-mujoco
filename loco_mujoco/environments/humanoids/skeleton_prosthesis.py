import mujoco 
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.humanoids.skeletons import MjxSkeletonMuscle
import numpy as np
from loco_mujoco.core.observations.goals import GoalRandomRootVelocity
from flax import struct
from loco_mujoco.environments.base import  LocoCarry
import jax.numpy as jnp
from loco_mujoco.core.utils import info_property
from collections.abc import Mapping
# from omegaconf import OmegaConf


class MjxSkeletonMuscleProsthesis(MjxSkeletonMuscle):
    """
    Mjx version of SkeletonMuscle with specs for adding a prosthesis.
    """

    mjx_enabled = True

    def __init__(self, timestep: float = 0.002, n_substeps: int = 5, **kwargs):
        """
        Constructor for MjxSkeletonMuscleProsthesis.
        Args:
            timestep (float): The time step for the simulation.
            n_substeps (int): The number of substeps for the simulation.
            **kwargs: Additional keyword arguments for configuration.
        Raises:
            ValueError: If required arguments are missing.
        """
        if "joint_stiffness" in kwargs:
            self.joint_stiffness = kwargs.pop("joint_stiffness")
        if "joint_damping" in kwargs:
            self.joint_damping = kwargs.pop("joint_damping")
        if "delete_joints" in kwargs:
            self.delete_joints = kwargs.pop("delete_joints")
        if "prosthesis_side" not in kwargs:
            raise ValueError("Missing required argument: 'prosthesis_side'")
        if "prosthesis_type" not in kwargs:
            raise ValueError("Missing required argument: 'prosthesis_type'")
        if "add_sensors" in kwargs:
            self.add_sensors = kwargs.pop("add_sensors")
        if "stiffen_and_dampen_joint_names" in kwargs:
            self.stiffen_and_dampen_joint_names = kwargs.pop("stiffen_and_dampen_joint_names")
        if "remove_joint_names" in kwargs:
            self.remove_joint_names = kwargs.pop("remove_joint_names")
        if "use_2_box_per_foot" in kwargs:
            self.use_2_box_per_foot = kwargs.pop("use_2_box_per_foot")
            if self.use_2_box_per_foot:
                use_box_feet = not self.use_2_box_per_foot
                kwargs["ignore_modify_mjx_contact"] = True
            else: 
                use_box_feet = True
                kwargs["ignore_modify_mjx_contact"] = False
        else: 
            use_box_feet = True
            kwargs["ignore_modify_mjx_contact"] = False
        if "amputated_tibia_length" in kwargs: 
            self.amputated_tibia_length = kwargs.pop("amputated_tibia_length")
        if "tibia_socket_overlap" in kwargs: 
            self.tibia_socket_overlap = kwargs.pop("tibia_socket_overlap")
        if "tibia_socket_offset_x" in kwargs:
            self.tibia_socket_offset_x = kwargs.pop("tibia_socket_offset_x")
        if "tibia_socket_offset_z" in kwargs:
            self.tibia_socket_offset_z = kwargs.pop("tibia_socket_offset_z")
        if "reattach_muscle" in kwargs:
            self.reattach_muscle = kwargs.pop("reattach_muscle")
            self.reattach_muscle_offset = kwargs.pop("reattach_muscles_offset")
            self.reattach_muscle_names = list(self.reattach_muscle_offset)
        if "replace_joint" in kwargs:
            self.replace_world_joint = kwargs.pop("replace_joint")
        if "prosthesis_subtype" in kwargs:
            self.prosthesis_subtype = kwargs.pop("prosthesis_subtype")
        if "reward_type" in kwargs: 
            self.reward_type = kwargs.get("reward_type")
        if "limit_knee_extension" in kwargs:
            self.limit_knee_extension = kwargs.get("limit_knee_extension")
            self.knee_extension_limit = kwargs.get("knee_extension_limit")
        if "socket_ty_slack" in kwargs:
            self.socket_ty_slack = kwargs.get("socket_ty_slack")
            print('self.socket_ty_slack in env: ', self.socket_ty_slack)
        else: 
            self.socket_ty_slack = False
        if "add_pos_ori_to_observation" in kwargs:
            # if kwargs.get("add_pos_ori_to_observation"):
            self.add_pos_ori_to_observation = kwargs.get("add_pos_ori_to_observation")
            if self.add_pos_ori_to_observation:
                # if "domain_randomization_params" in kwargs:
                #     domain_randomization_params = kwargs.get("domain_randomization_params")
                #     if "randomize_prosthesis_body_position" in domain_randomization_params:
                #         if domain_randomization_params["randomize_prosthesis_body_position"]:
                #             self.prosthesis_body_position_range = domain_randomization_params["prosthesis_body_position_range"]
                #     if "randomize_prosthesis_body_orientation" in domain_randomization_params:
                #         if domain_randomization_params["randomize_prosthesis_body_orientation"]:
                #             self.prosthesis_body_orientation_range = domain_randomization_params["prosthesis_body_orientation_range"]
                # else: 
                #     # Hardcoded
                # self.prosthesis_body_position_range = {'pylon_socket': {'x': [-0.02,0.02],'y': [-0.1,0.1],'z': [-0.02,0.02]},'talus': {'x': [-0.01,0.01],'z': [-0.01,0.01]}}
                # self.prosthesis_body_orientation_range = {'pylon_socket': {'x': [-0.2,0.2],'y': [-0.3,0.3],'z': [-0.2,0.2]},'talus': {'x': [-0.1,0.1],'z': [-0.1,0.1]}}

                
                # # # if y in pylon_socket in domain_randomization_params["prosthesis_body_position_range"] then take 
                # domain_randomization_params = kwargs.get("domain_randomization_params")
                # if "pylon_socket" in domain_randomization_params["prosthesis_body_position_range"] and 'y' in domain_randomization_params["prosthesis_body_position_range"]["pylon_socket"]:
                #     print('AUTOMATIC OBS FROM RANDOMIZATION PARAMS')
                #     if "randomize_prosthesis_body_position" in domain_randomization_params:
                #         if domain_randomization_params["randomize_prosthesis_body_position"]:
                #             self.prosthesis_body_position_range = domain_randomization_params["prosthesis_body_position_range"]
                #     if "randomize_prosthesis_body_orientation" in domain_randomization_params:
                #         if domain_randomization_params["randomize_prosthesis_body_orientation"]:
                #             self.prosthesis_body_orientation_range = domain_randomization_params["prosthesis_body_orientation_range"]
                # else: 
                #     self.prosthesis_body_position_range = {'pylon_socket': {'x': [-0.02,0.02],'z': [-0.02,0.02]},'talus': {'x': [-0.01,0.01],'z': [-0.01,0.01]}}
                #     self.prosthesis_body_orientation_range = {'pylon_socket': {'x': [-0.2,0.2],'y': [-0.3,0.3],'z': [-0.2,0.2]},'talus': {'x': [-0.1,0.1],'z': [-0.1,0.1]}}
                #     #self.prosthesis_body_orientation_range = {'pylon_socket': {'x': [-0.1,0.1],'y': [-0.1,0.1],'z': [-0.1,0.1]},'talus': {'x': [-0.1,0.1],'y': [-0.1,0.1],'z': [-0.1,0.1]}}

                self.prosthesis_body_position_range = {'pylon_socket': {'x': [-0.02,0.02],'z': [-0.02,0.02]},'talus': {'x': [-0.01,0.01],'z': [-0.01,0.01]}}
                self.prosthesis_body_orientation_range = {'pylon_socket': {'x': [-0.2,0.2],'y': [-0.3,0.3],'z': [-0.2,0.2]},'talus': {'x': [-0.1,0.1],'z': [-0.1,0.1]}}
                    


        else:
            self.add_pos_ori_to_observation = False
        
        if "socket_ty_joint_stiffness" in kwargs: 
            self.socket_ty_joint_stiffness = kwargs.pop("socket_ty_joint_stiffness")
        if "socket_ty_joint_damping" in kwargs:
            self.socket_ty_joint_damping = kwargs.pop("socket_ty_joint_damping")
        if "socket_ty_joint_range" in kwargs:
            self.socket_ty_joint_range = kwargs.pop("socket_ty_joint_range")
        if "socket_type" in kwargs:
            self.socket_type = kwargs.pop("socket_type")
        if "prosthesis_visualization" in kwargs:
            self.prosthesis_visualization = kwargs.pop("prosthesis_visualization")
            
        if "socket_ty_joint" in kwargs: 
            self.socket_ty_joint = kwargs.pop("socket_ty_joint")
        if "socket_tx_joint" in kwargs: 
            self.socket_tx_joint = kwargs.pop("socket_tx_joint")
        if "socket_tz_joint" in kwargs: 
            self.socket_tz_joint = kwargs.pop("socket_tz_joint")
        if "socket_flexion_joint" in kwargs: 
            self.socket_flexion_joint = kwargs.pop("socket_flexion_joint")
        if "socket_rotation_joint" in kwargs: 
            self.socket_rotation_joint = kwargs.pop("socket_rotation_joint")
        if "socket_adduction_joint" in kwargs: 
            self.socket_adduction_joint = kwargs.pop("socket_adduction_joint")

        if "socket_joint" in kwargs: 
            self.socket_joint = kwargs.pop("socket_joint")

        if "contact_solref" in kwargs: 
            self.contact_solref = kwargs.pop("contact_solref")
        if "contact_geom_type" in kwargs:
            self.contact_geom_type = kwargs.pop("contact_geom_type")

        

        self.actuators_removed = []
        self.amputated_joint_names = []
        self.amputated_body_names = []

        side_arg = kwargs.pop("prosthesis_side")
        self.prosthesis_type = kwargs.pop("prosthesis_type")

        if side_arg == "left_side" or side_arg == "bilateral":
            self.prosthesis_side = "_l"
        elif side_arg == "right_side":
            self.prosthesis_side = "_r"
        else:
            raise ValueError("Invalid prosthesis side. Choose 'left_side' or 'right_side'.")
        
        print(f"Prosthesis side: {self.prosthesis_side}")
        print(f"Prosthesis type: {self.prosthesis_type}")

        # Load model specification and modify it
        spec = mujoco.MjSpec.from_file(self.get_default_xml_file_path())
        spec = self.replace_leg_level(spec) 

        if side_arg == 'bilateral':
            self.prosthesis_side = "_r"
            spec = self.replace_leg_level(spec)

        if hasattr(self, "prosthesis_type") and self.prosthesis_type != "None" and hasattr(self, "prosthesis_subtype") and self.prosthesis_subtype == "SACH" and hasattr(self, 'use_2_box_per_foot') and self.use_2_box_per_foot:
            spec = self._add_2_box_per_foot_to_spec(spec)


        if hasattr(self, "replace_world_joint") and self.replace_world_joint: 
            spec = self._replace_joint(spec, 'pelvis', mujoco.mjtJoint.mjJNT_SLIDE, [0,1,0])

        if hasattr(self, 'add_sensors') and self.add_sensors:
                joint_force_sensor_site_name = "hip_mimic"
                self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
                self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
                self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
                self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")
                joint_force_sensor_site_name = "knee_mimic"
                self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
                self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
                self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
                self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")
                joint_force_sensor_site_name = "foot_mimic"
                self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
                self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
                self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
                self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")
                if hasattr(self, "prosthesis_type") and self.prosthesis_type != "None":
                    joint_force_sensor_site_name = f"pylon_mimic{self.prosthesis_side}"
                    self.add_force_sensor(spec, joint_force_sensor_site_name)
                    self.add_torque_sensor(spec, joint_force_sensor_site_name)
                # joint_force_sensor_site_name = f"pylon_socket_joint{self.prosthesis_side}"
                # self.add_force_sensor(spec, joint_force_sensor_site_name)
                # self.add_torque_sensor(spec, joint_force_sensor_site_name)

        if hasattr(self, 'limit_knee_extension') and self.limit_knee_extension:
            kel = getattr(self, "knee_extension_limit", None)

            if isinstance(kel, Mapping):
                # Expect entries like {'knee_angle_r': [-120, 5], 'knee_angle_l': [-120, 10], ...}
                for joint_name, joint_limit in kel.items():
                    # print(f'Limiting joint {joint_name} to range {joint_limit}')
                    if joint_limit is None:
                        continue
                    if not (hasattr(joint_limit, "__len__") and len(joint_limit) == 2):
                        raise ValueError(f"Expected 2-element range for '{joint_name}', got: {joint_limit!r}")
                    self.limit_joint_range_side(spec, joint_name, joint_limit)
            else:
                # Treat as scalar upper limit applied to both sides (e.g. 5)
                if kel is None:
                    pass
                else:
                    try:
                        upper = float(kel)
                    except Exception:
                        raise ValueError(f"knee_extension_limit must be a number or dict, got: {kel!r}")
                    self.limit_joint_range(spec, "knee_angle", upper)

        if 'limit_hip_joints' in kwargs:
            self.limit_hip_rot_add_angles(spec)


        if "joint_stiffness_both_sides" in kwargs:
            self.joint_stiffness_both_sides = kwargs.pop("joint_stiffness_both_sides")
            self.increase_joint_stiffness_both_sides(spec)

        if "joint_stiffness_each_joint" in kwargs:
            self.joint_stiffness_each_joint = kwargs.pop("joint_stiffness_each_joint")
            self.set_joint_stiffness_each_joint(spec)

        if "joint_damping_each_joint" in kwargs:
            self.joint_damping_each_joint = kwargs.pop("joint_damping_each_joint")
            self.set_joint_damping_each_joint(spec)

        # Model option configuration
        model_option_conf = kwargs.pop("model_option_conf", {
            "iterations": 4,
            "ls_iterations": 8,
            "disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        })

        super().__init__(timestep=timestep, n_substeps=n_substeps,
                         model_option_conf=model_option_conf, spec=spec,use_box_feet=use_box_feet, **kwargs)
        

    def limit_joint_range(self, spec, joint_name, upper_limit):
        
        joint_names = [joint_name + '_l', joint_name + '_r']
        for j in spec.joints:
            if j.name in joint_names:
                j.range = [j.range[0], np.deg2rad(upper_limit)]

        
    def limit_joint_range_side(self, spec, joint_name, joint_limit):
        for j in spec.joints:
            if j.name == joint_name: 
                j.range = [np.deg2rad(joint_limit[0]), np.deg2rad(joint_limit[1])]

    
    def limit_hip_rot_add_angles(self, spec):
        joint_names = ['hip_rotation', 'hip_adduction']
        joint_names = [j + '_l' for j in joint_names] + [j + '_r' for j in joint_names]
        for j in spec.joints:
            if j.name in joint_names:
                j.range = [-0.001, 0.001]
        print("Hip rotation and adduction joints limited to 0 range.")

    def add_force_sensor_prosthesis_side(self, spec, site_name):
        """
        Adds a force sensor to the specified body in the mjcf model specification.
        
        Args:
            spec (mjcf.RootElement): The MJCF root model object.
            body_name (str): The name of the body to attach the force sensor to.
        """
        # body = spec.find('body', body_name)
        # for s in body.sites: 
        #     if 'mimic' in s.name:
        #         site_name = s.name

        # if body is None:
        #     raise ValueError(f"Body '{body_name}' not found in the model specification.")
        
        if self.prosthesis_side == "_l":
            site_name = "left_" + site_name
        elif self.prosthesis_side == "_r":
            site_name = "right_" + site_name

        sensor_site = spec.find_site(site_name)

        # print(f"Adding force sensor to site: {site_name}")

        # print(f"Sensor site: {sensor_site}")
        if sensor_site is None:
            raise ValueError(f"Site '{site_name}' not found in the model specification.")
        
        # Add the force sensor
        force_sensor = spec.add_sensor(
        name=f"{sensor_site.name}_force_sensor",
        type=mujoco.mjtSensor.mjSENS_FORCE,
        objtype=mujoco.mjtObj.mjOBJ_SITE,  # Specify that the sensor is attached to a site object
        objname=sensor_site.name      # Provide the name of the specific site
        )

        return force_sensor
    

    def add_force_sensor(self, spec, site_name):
        """
        Adds a force sensor to the specified body in the mjcf model specification.
        
        Args:
            spec (mjcf.RootElement): The MJCF root model object.
            body_name (str): The name of the body to attach the force sensor to.
        """
        sensor_site = spec.find_site(site_name)
        # print(f"Adding force sensor to site: {site_name}")
        # print(f"Sensor site: {sensor_site}")

        # print(f"Adding force sensor to site: {site_name}")

        # print(f"Sensor site: {sensor_site}")
        if sensor_site is None:
            raise ValueError(f"Site '{site_name}' not found in the model specification.")
        
        # Add the force sensor
        force_sensor = spec.add_sensor(
        name=f"{sensor_site.name}_force_sensor",
        type=mujoco.mjtSensor.mjSENS_FORCE,
        objtype=mujoco.mjtObj.mjOBJ_SITE,  # Specify that the sensor is attached to a site object
        objname=sensor_site.name      # Provide the name of the specific site
        )

        return force_sensor
    

    def add_torque_sensor(self, spec, site_name):
        """
        Adds a torque sensor to the specified body in the mjcf model specification.
        
        Args:
            spec (mjcf.RootElement): The MJCF root model object.
            body_name (str): The name of the body to attach the force sensor to.
        """
        sensor_site = spec.find_site(site_name)
        # print(f"Adding force sensor to site: {site_name}")
        # print(f"Sensor site: {sensor_site}")

        # print(f"Adding force sensor to site: {site_name}")

        # print(f"Sensor site: {sensor_site}")
        if sensor_site is None:
            raise ValueError(f"Site '{site_name}' not found in the model specification.")
        
        # Add the torque sensor
        torque_sensor = spec.add_sensor(
        name=f"{sensor_site.name}_torque_sensor",
        type=mujoco.mjtSensor.mjSENS_TORQUE,
        objtype=mujoco.mjtObj.mjOBJ_SITE,  # Specify that the sensor is attached to a site object
        objname=sensor_site.name      # Provide the name of the specific site
        )

        return torque_sensor



    def increase_joint_stiffness(self, spec, amputated_joint_names):
        """
        Increases the stiffness of specified joints in the model specification.
        Args:
            spec (MjSpec): The model specification object.
        """
        for j in spec.joints:
            if j.name in amputated_joint_names:
                # print(f"Increasing stiffness of joint: {j.name}")
                j.stiffness = self.joint_stiffness
                j.damping = self.joint_damping


    def increase_joint_stiffness_each_joint(self, spec):
        """
        Increases the stiffness of specified joints in the model specification.
        Args:
            spec (MjSpec): The model specification object.
        """
        joint_stiffness = self.joint_stiffness
        joint_names =  list(joint_stiffness.keys())
        joint_names = [j+self.prosthesis_side for j in joint_names]
        
        for j in spec.joints:
            if j.name in joint_names:
                # print(f"Increasing stiffness of joint: {j.name}")
                j.stiffness = joint_stiffness[j.name.replace(self.prosthesis_side,'')]


    def increase_joint_stiffness_both_sides(self, spec):
        """
        Increases the stiffness of specified joints in the model specification.
        Args:
            spec (MjSpec): The model specification object.
        """
        joint_stiffness = self.joint_stiffness_both_sides
        joint_names =  list(joint_stiffness.keys())
        joint_names = [j+'_l' for j in joint_names] + [j+'_r' for j in joint_names]
        
        for j in spec.joints:
            if j.name in joint_names:
                # print(f"Increasing stiffness of joint: {j.name}")
                if j.name.endswith('_l'):
                    j.stiffness = joint_stiffness[j.name[:-2]]
                elif j.name.endswith('_r'):
                    j.stiffness = joint_stiffness[j.name[:-2]]

    
    def set_joint_stiffness_each_joint(self, spec):
        """
        Sets the stiffness of specified degree of freedom in the model specification.
        Args:
            spec (MjSpec): The model specification object.
        """

        joint_stiffness = self.joint_stiffness_each_joint
        joint_names =  list(joint_stiffness.keys())

        
        for j in spec.joints:
            if j.name in joint_names:
                # print(f"Increasing stiffness of joint: {j.name}")
                j.stiffness = joint_stiffness[j.name]


    def set_joint_damping_each_joint(self, spec):
        """
        Sets the damping of specified degree of freedom in the model specification.
        Args:
            spec (MjSpec): The model specification object.
        """

        joint_damping = self.joint_damping_each_joint
        joint_names =  list(joint_damping.keys())

        
        for j in spec.joints:
            if j.name in joint_names:
                # print(f"Increasing stiffness of joint: {j.name}")
                j.damping = joint_damping[j.name]



    def increase_joint_damping_each_joint(self, spec):
        """
        Increases the damping of specified degree of freedom in the model specification.
        Args:
            spec (MjSpec): The model specification object.
        """

        joint_damping = self.joint_damping
        joint_names =  list(joint_damping.keys())
        joint_names = [j+self.prosthesis_side for j in joint_names]
        
        for j in spec.joints:
            if j.name in joint_names:
                # print(f"Increasing stiffness of joint: {j.name}")
                j.damping = joint_damping[j.name.replace(self.prosthesis_side,'')]
        
        

    def replace_leg_level(self, spec):
        """
        Replaces the leg level in the model specification based on the prosthesis type.
        Args:
            spec (MjSpec): The model specification object.
        Returns:
            MjSpec: The modified model specification.
        """
        if self.prosthesis_type == "None":
            if hasattr(self, 'use_2_box_per_foot') and self.use_2_box_per_foot:
                spec = self._add_2_box_per_foot_to_spec(spec)
            # if hasattr(self, 'add_sensors') and self.add_sensors:
            #     joint_force_sensor_site_name = "hip_mimic"
            #     self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
            #     self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
            #     self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
            #     self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")
            #     joint_force_sensor_site_name = "knee_mimic"
            #     self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
            #     self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
            #     self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
            #     self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")
            #     joint_force_sensor_site_name = "foot_mimic"
            #     self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
            #     self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
            #     self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
            #     self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")
            return spec
        elif self.prosthesis_type == "transtibial":
            self.amputated_joint_names = [
                f"ankle_angle{self.prosthesis_side}",
                f"subtalar_angle{self.prosthesis_side}",
                f"mtp_angle{self.prosthesis_side}"]
            self.amputated_body_names = [
                f"calcn{self.prosthesis_side}",
                f"toes{self.prosthesis_side}",
                f"talus{self.prosthesis_side}"]
            if hasattr(self, "prosthesis_subtype") and self.prosthesis_subtype == "SACH":
                spec = self.adapt_spec_with_prosthesis_adapter(spec)
                spec = self.add_SACHFoot_properties(spec)
                if hasattr(self, 'reattach_muscle') and self.reattach_muscle:
                    spec = self.reattach_muscles_above_amputation(spec)
                # if hasattr(self, 'use_2_box_per_foot') and self.use_2_box_per_foot:
                #     spec = self._add_2_box_per_foot_to_spec(spec)
                    
                return spec
            else: 
                # return spec
                # spec = self.modify_contact_pairs(spec)
            # if self.delete_joint defined or not 
                if hasattr(self, 'delete_joints') and self.delete_joints:
                    spec = self.transtibial_prosthesis(spec)
                    # if hasattr(self, 'use_2_box_per_foot') and self.use_2_box_per_foot:
                    #     spec = self._add_2_box_per_foot_to_spec(spec)
                    #return spec
                elif not hasattr(self, 'delete_joints'): # Was initally not defined. To use for older policies 
                    spec = self.transtibial_prosthesis(spec)
                    # if hasattr(self, 'use_2_box_per_foot') and self.use_2_box_per_foot:
                    #     spec = self._add_2_box_per_foot_to_spec(spec)
                    #return spec
                else:
                    spec = self.transtibial_prosthesis_with_joints(spec)
                if hasattr(self, 'use_2_box_per_foot') and self.use_2_box_per_foot:
                    spec = self._add_2_box_per_foot_to_spec(spec)
                return spec
        elif self.prosthesis_type == "transfemoral":
            raise NotImplementedError("Transfemoral prosthesis not implemented yet.")
        else:
            raise ValueError(f"Unknown prosthesis type: {self.prosthesis_type}")
        
    def reattach_muscles_above_amputation(self, spec):
        # This function assumes the muscle is attached to the Femur and the calcn. If a muscle has other attachment points, needs to be adapted 
        reattach_muscle_names = [name + self.prosthesis_side for name in self.reattach_muscle_names]
        
        # Find the tibia body
        femur_body = spec.find_body('femur'+ self.prosthesis_side)
        tibia_body = spec.find_body('tibia' + self.prosthesis_side)
        calcn_body = spec.find_body('calcn'+ self.prosthesis_side)
        talus_body = spec.find_body('talus' + self.prosthesis_side)
        pylon_socket = spec.find_body('pylon_socket' + self.prosthesis_side)
        site_pos_tibia_P2={}
        site_pos_tibia_P3 = {}
        new_site_pos = {}
        muscle_site_names = []
        for s in femur_body.sites:
            if 'P2' in s.name: 
                renamed_site = s.name[:-3]
                if renamed_site in reattach_muscle_names: 
                    site_pos_femur = s.pos
                    site_pos_tibia_P2[renamed_site]=  site_pos_femur - tibia_body.pos
                    muscle_site_names.append(renamed_site)

                    tibia_body.add_site(
                    name=f"new_P2_{s.name}",
                    pos = site_pos_tibia_P2[renamed_site], #new_site_pos[renamed_site],
                    size = [0.02,0.02,0.02],
                    rgba = [1,0,0,1],
                )

                    
        for s in calcn_body.sites: 
            renamed_site = s.name[:-3]
            if renamed_site in reattach_muscle_names: 
                # Get site name and position
                site_pos_calcn = s.pos
                site_pos_tibia_P3[renamed_site] = site_pos_calcn + calcn_body.pos + pylon_socket.pos + talus_body.pos
                # new_site_pos[renamed_site] = site_pos_tibia_P3[renamed_site]
                # new_site_pos[1] = self.amputated_tibia_length
                # new_site_pos[renamed_site] += self.reattach_muscle_offset[renamed_site.replace(self.prosthesis_side, '')]

                tibia_body.add_site(
                    name=f"new_P3_{s.name}",
                    pos = site_pos_tibia_P3[renamed_site], #new_site_pos[renamed_site],
                    size = [0.02,0.02,0.02],
                    rgba = [1,0,0,1],
                )

        for site_name in muscle_site_names:
            P2 = site_pos_tibia_P2[site_name]
            P3 = site_pos_tibia_P3[site_name]

            m_x_y = (P2[0]-P3[0])/(P2[1]- P3[1])
            m_z_y = (P2[2]-P3[2])/(P2[1]- P3[1])

            d_x_y = P2[0] - m_x_y*P2[1]
            d_z_y = P2[2] - m_z_y*P2[1]


            new_y = -self.amputated_tibia_length + self.reattach_muscle_offset[site_name.replace(self.prosthesis_side, '')][1]

            new_x = m_x_y*new_y + d_x_y + self.reattach_muscle_offset[site_name.replace(self.prosthesis_side, '')][0]
            new_z = m_z_y*new_y + d_z_y + self.reattach_muscle_offset[site_name.replace(self.prosthesis_side, '')][2]

            [s for s in calcn_body.sites if s.name == f"{site_name}-P3"][0].delete()
            tibia_body.add_site(
                    name=f"{site_name}-P3",
                    pos = [new_x, new_y, new_z],
                    size = [0.02,0.02,0.02],
                    rgba = [0,1,0,1],
                )
            
            new_diff = np.linalg.norm([new_x,new_y,new_z] - P2)
            old_diff = np.linalg.norm(P3-P2)

            length_ratio =  new_diff/old_diff
            for a in spec.actuators:
                if a.name == site_name:
                    a.lengthrange*= length_ratio 

        return spec

    def remove_sites(self, body):
        """
        Removes sites from the specified body that are associated with muscles.
        Args:
            body (MjBody): The body from which to remove sites.
        Returns:
            set: A set of muscle names derived from the removed sites.
        """
        muscle_names = []
        if hasattr(self, 'reattach_muscle') and self.reattach_muscle:
            reattach_muscle_names = [name + self.prosthesis_side for name in self.reattach_muscle_names]
        else: 
            reattach_muscle_names = []
        # self.reattach_muscle_names = [name + self.prosthesis_side for name in self.reattach_muscle_names]
        for s in body.sites:  
            # print(f"Site: {s.name}")
            if '-P' in s.name:
                site_renamed = s.name[:-3] # Take out -P part of site name
                if site_renamed not in reattach_muscle_names:
                    muscle_names.append(site_renamed)
                    s.delete()     
                # print(f"Muscle name: {muscle_names}")
                # print(f"Removing site: {s.name}")
                # s.delete()
        return muscle_names #set(muscle_names)
    

    def remove_tendons(self, spec, muscle_names):
        """
        Removes tendons associated with the specified muscle names.
        Args:
            spec (MjSpec): The model specification object.
            muscle_names (set): A set of muscle names to match against tendon names.
        """
        for t in spec.tendons:
            if any(m in t.name for m in muscle_names):
                t.delete()


    def remove_actuators(self, spec, muscle_names):
        """
        Removes actuators associated with the specified muscle names.
        Args:
            spec (MjSpec): The model specification object.
            muscle_names (set): A set of muscle names to match against actuator names."""
        for a in spec.actuators:
            # print(f"Actuator: {a.name}")
            # print(f"Muscle names: {muscle_names}")
            if any(m in a.name for m in muscle_names):
                # print(f"Removing actuator: {a.name}")
                self.actuators_removed.append(a.name)
                a.delete()


    def remove_joint(self, spec, amputated_joint_names):
        """
        Removes joints specified in self.amputated_joint_names.

        Args:
            spec (MjSpec): The model specification object.
        """
        for j in spec.joints:
            if j.name in amputated_joint_names:
                j.delete()



    def remove_equality(self, spec, amputated_joint_names):
        """
        Removes equality constraints associated with the specified joint names.

        Args:
            spec: The model specification object.
            amputated_joint_names: A list of joint names whose equality constraints should be removed.
        """
        for e in spec.equalities:  # Use list to avoid iteration issues during deletion
            if any(joint_name in e.name for joint_name in amputated_joint_names):
                # print(f"Removing equality constraint: {e.name}")
                e.delete()

    def remove_site_actuator_tendon(self, spec):
        for b in self.amputated_body_names:
            body = spec.find_body(b)
            muscle_names = self.remove_sites(body)
            self.remove_tendons(spec, muscle_names)
            self.remove_actuators(spec, muscle_names)
            for g in body.geoms:
                g.rgba = [0.0, 0.0, 1.0, 1.0]

    def remove_site_actuator_tendon_old(self, spec, body):
        """
        Removes sites, actuators, and tendons from the specified body.
        Args:
            spec (MjSpec): The model specification object.
            body (MjBody): The body from which to remove sites, actuators, and tendons.
        """
        muscle_names = self.remove_sites(body)
        self.remove_tendons(spec, muscle_names)
        self.remove_actuators(spec, muscle_names)

    
    def transtibial_prosthesis_with_joints(self, spec):
        """Handles transtibial prosthesis while keeping joints but increasing stiffness.
        Args:
            spec (MjSpec): The model specification object.
        Returns:
            MjSpec: The modified model specification with increased joint stiffness.
        """
        self.increase_joint_stiffness(spec, self.amputated_joint_names)

        self.remove_site_actuator_tendon(spec)

       
        # # joint force sensor site name only for evaluation
        # if hasattr(self, 'add_sensors') and self.add_sensors:
        #     joint_force_sensor_site_name = "hip_mimic"
        #     self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
        #     self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
        #     self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
        #     self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")
        #     joint_force_sensor_site_name = "knee_mimic"
        #     self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
        #     self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
        #     self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
        #     self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")
        #     joint_force_sensor_site_name = "foot_mimic"
        #     self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
        #     self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
        #     self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
        #     self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")


        return spec


    def transtibial_prosthesis(self, spec):
        """Handles transtibial prosthesis by removing joints, sites, actuators, and tendons.
        Args:
            spec (MjSpec): The model specification object.
        Returns:
            MjSpec: The modified model specification with amputated joints and bodies.
        """
        
        self.remove_joint(spec, self.amputated_joint_names)
        self.remove_equality(spec, self.amputated_joint_names)

        self.remove_site_actuator_tendon(spec)

        # # joint force sensor site name only for evaluation
        # if hasattr(self, 'add_sensors') and self.add_sensors:
        #     joint_force_sensor_site_name = "knee_mimic"
        #     self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
        #     self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
        #     self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
        #     self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")
        #     joint_force_sensor_site_name = "foot_mimic"
        #     self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
        #     self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
        #     self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
        #     self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")

        # for b in self.amputated_body_names:
        #     body = spec.find_body(b)
        #     self.remove_site_actuator_tendon_old(spec, body)
        #     for g in body.geoms:
        #         g.rgba = [0.0, 0.0, 1.0, 1.0]





        # calcn = spec.find_body(f"calcn{self.prosthesis_side}")
        # # print(f"Calcn Body: {calcn}")
        # self.remove_site_actuator_tendon(spec, calcn)

        # toe = spec.find_body(f"toes{self.prosthesis_side}")
        # # print(f"Toe Body: {toe}")
        # self.remove_site_actuator_tendon(spec, toe)

        # talus = spec.find_body(f"talus{self.prosthesis_side}")

        # # Set color of calcn and toe bodies to blue
        # calcn.rgba = [0.0, 0.0, 1.0, 1.0]
        # toe.rgba = [0.0, 0.0, 1.0, 1.0]

        # # get geometries of calcn and toe
        # for g in calcn.geoms + toe.geoms + talus.geoms:
        #     g.rgba = [0.0, 0.0, 1.0, 1.0]

        return spec
    

    def set_socket_parameters(self, spec: mujoco.MjSpec):

        socket_mass = 0.3
        socket_inertia = [0.0136, 0.0021,0.0136, 0, 0,0]
        socket_relative_center_of_mass = np.array([0,0.0491,0])

        # socket_center_of_mass = [0,-tibia_socket_overlap,0] - socket_relative_center_of_mass
        # socket_center_of_mass = - socket_relative_center_of_mass

        return socket_mass, socket_inertia, socket_relative_center_of_mass 



    def calculate_tibia_socket_parameters(self, spec: mujoco.MjSpec, original_talus_pos): #, amputated_tibia_length, tibia_socket_overlap):
        """
        """
        tibia_name = f"tibia{self.prosthesis_side}"
        tibia_body = spec.find_body(tibia_name)
        original_tibia_mass = tibia_body.mass
        original_tibia_fullinertia = tibia_body.fullinertia.copy()
        original_tibia_length = abs(original_talus_pos[1])
        original_tibia_center_of_mass= tibia_body.ipos.copy()

        # SITE POSITON TESTING 
        tibia_body.add_site(
            name=f"tibia_COM_org{self.prosthesis_side}",
            pos=tibia_body.ipos, # This site is at the "bottom" of the shank, relative to prosthetic_shank_body's origin
            size=[0.01,0.01,0.01],
            rgba=[1, 0, 0, 1]
        )

        socket_length = original_tibia_length - self.amputated_tibia_length + self.tibia_socket_overlap

        socket_top_offset = original_tibia_length - socket_length

        # socket_length = socket_top_offset - original_talus_pos 

        # top_socket_pos_relative_to_tibia = np.array([self.tibia_socket_offset_x,-socket_top_offset,self.tibia_socket_offset_z]) # Old implementation 
        
        socket_pos_relative_to_tibia = np.array([self.tibia_socket_offset_x,-socket_top_offset-self.tibia_socket_overlap,self.tibia_socket_offset_z])


        # Get amputation ratio to adapt tibia parameters
        amputation_ratio = self.amputated_tibia_length / original_tibia_length

        # Adapt tibia 
        tibia_body.mass = original_tibia_mass * amputation_ratio

        tibia_radius = self._calculate_cylinder_radius(tibia_body.mass,original_tibia_fullinertia[1])
        # tibia_height_using_x = self._calculate_org_cylinder_height(tibia_body.mass,tibia_radius,original_tibia_fullinertia[0])
        # tibia_height_using_z = self._calculate_org_cylinder_height(tibia_body.mass,tibia_radius,original_tibia_fullinertia[2])
        tibia_body.fullinertia[1] = original_tibia_fullinertia[1]*amputation_ratio

        tibia_body.fullinertia[0] = self._calculate_cylinder_inertia_xorz(tibia_body.mass, tibia_radius, self.amputated_tibia_length) #tibia_height_using_x)
        tibia_body.fullinertia[2] = tibia_body.fullinertia[0]

        # tibia_body.fullinertia= [fullinertia* amputation_ratio for fullinertia in original_tibia_fullinertia] #[1]*amputation_ratio#**3
        # tibia_body.fullinertia = original_tibia_fullinertia* amputation_ratio #[val *amputation_ratio for val in original_tibia_fullinertia]
        tibia_body.ipos[1] = original_tibia_center_of_mass[1]*amputation_ratio
        # tibia_body.ipos = [p*amputation_ratio for p in original_tibia_center_of_mass]


        # SITE POSITON TESTING 
        tibia_body.add_site(
            name=f"tibia_COM{self.prosthesis_side}",
            pos=tibia_body.ipos, # This site is at the "bottom" of the shank, relative to prosthetic_shank_body's origin
            size=[0.01,0.01,0.01],
            rgba=[0, 1, 0, 1]
        )

        # Get original full socket mass, inertia and center of mass 
        original_socket_mass, original_socket_inertia, original_socket_relative_center_of_mass = self.set_socket_parameters(spec)

        socket_ratio = socket_length/original_tibia_length


        socket_radius = self._calculate_cylinder_radius(original_socket_mass,original_socket_inertia[1])
        # socket_height_using_x = self._calculate_org_cylinder_height(original_socket_mass,socket_radius,original_socket_inertia[0])
        # socket_height_using_z = self._calculate_org_cylinder_height(original_socket_mass,socket_radius,original_socket_inertia[2])
        socket_inertia = original_socket_inertia
        socket_inertia[1] = original_socket_inertia[1]*amputation_ratio


        # Scale socket properties depending on length of socket 
        socket_mass = original_socket_mass * socket_ratio

        socket_inertia[0] = self._calculate_cylinder_inertia_xorz(socket_mass, socket_radius, socket_length)
        socket_inertia[2] = socket_inertia[0]

        # socket_inertia = np.array(original_socket_inertia)*socket_ratio#**3 
        socket_relative_center_of_mass = original_socket_relative_center_of_mass
        socket_relative_center_of_mass[1] = socket_ratio*original_socket_relative_center_of_mass[1]
        # socket_center_of_mass = np.array([0,-self.tibia_socket_overlap,0]) - np.array(socket_relative_center_of_mass)
        socket_center_of_mass = np.array(socket_relative_center_of_mass)

        prosthetic_shank_body = self.create_socket(tibia_body, socket_mass, socket_inertia, socket_center_of_mass,socket_pos_relative_to_tibia,socket_radius, socket_length) #,tibia_socket_overlap)


        return  prosthetic_shank_body



    def create_socket(self, tibia_body, socket_mass, socket_inertia, socket_center_of_mass,socket_pos_relative_to_tibia, socket_radius,socket_length): #,tibia_socket_overlap):

        # socket_radius = 0.04
        

        prosthetic_shank_body = tibia_body.add_body(
            name=f"pylon_socket{self.prosthesis_side}",
            pos= socket_pos_relative_to_tibia, # Use the calculated offset here # Start of non-overlaapping with tibia and socket 
            # mass = socket_mass,
            # fullinertia = socket_inertia,
            # ipos = socket_center_of_mass,
        )

        prosthetic_shank_body.add_site(
            name=f"pylon_mimic{self.prosthesis_side}",
            pos=np.array([0,0,0]),
            size = [0.01, 0.01,0.01],
            rgba=[0, 1, 0, 1]
        )

        prosthetic_shank_body.add_site(
            name=f"pylon_0{self.prosthesis_side}",
            pos=np.array([0,0,0])+np.array([0,self.tibia_socket_overlap,0]),
            size = [0.01, 0.01,0.01],
            rgba=[0, 1, 0, 1]
        )

       
        prosthetic_shank_body.add_site(
            name=f"pylon_COM{self.prosthesis_side}",
            pos=socket_center_of_mass, # This site is at the "bottom" of the shank, relative to prosthetic_shank_body's origin
            size=[0.01,0.01,0.01],
            rgba=[0, 1, 0, 1]
        )

        prosthetic_shank_body.add_site(
            name=f"talus_attachment_site_in_pylon{self.prosthesis_side}",
            pos=[0, -socket_length+self.tibia_socket_overlap,0], # This site is at the "bottom" of the shank, relative to prosthetic_shank_body's origin
            size=[0.01,0.01,0.01],
            rgba=[1, 0, 0, 1]
        )

        print(f"Created new prosthetic shank body: '{prosthetic_shank_body.name}' (ID: {id(prosthetic_shank_body)}). Its parent is: '{tibia_body.name}'")

        prosthetic_shank_body.add_geom(
            name=f"pylon_socket_geom{self.prosthesis_side}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[socket_radius,socket_length/2,socket_radius],#, shank_length],
            pos=[0, -socket_length/2+self.tibia_socket_overlap ,0], # Center of the cylinder, relative to prosthetic_shank_body's origin
            euler=[1.571, 0, 0], # Rotate to be vertical if it's currently horizontal
            rgba=[0.5, 0.5, 0.5, 1],
            mass = socket_mass
        )

        if hasattr(self, 'prosthesis_visualization') and self.prosthesis_visualization:
            prosthetic_shank_body.add_geom(
                name=f"socket_visual_geom{self.prosthesis_side}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[socket_radius*4,self.tibia_socket_overlap/2,socket_radius*4],#, shank_length],
                # pos=[0, self.tibia_socket_overlap ,0], # Center of the cylinder, relative to prosthetic_shank_body's origin
                # pos=[0, -socket_length/2 ,0], # Center of the cylinder, relative to prosthetic_shank_body's origin
                pos=[0, self.tibia_socket_overlap/2,0], # Center of the cylinder, relative to prosthetic_shank_body's origin
                # pos=[0, -socket_length/2+self.tibia_socket_overlap/2 ,0], # Center of the cylinder, relative to prosthetic_shank_body's origin
                euler=[1.571, 0, 0], # Rotate to be vertical if it's currently horizontal
                rgba= [0.0, 0.0, 1.0, 1.0], #[0.5, 0.5, 0.5, 1], #[1, 0, 0, 1], #[0.5, 0.5, 0.5, 1],
                # mass = socket_mass
            )

            for g in tibia_body.geoms:
                if g.name == f"tibia{self.prosthesis_side}" or g.name == f"fibula{self.prosthesis_side}":
                    g.delete()
        
        # prosthetic_shank_body.add_site(
        #     name=f"pylon_attachment_site{self.prosthesis_side}",
        #     pos=[0, -socket_length,0], # This site is at the "bottom" of the shank, relative to prosthetic_shank_body's origin
        #     size=[0.01,0.01,0.01],
        #     rgba=[1, 0, 0, 1]
        # )

        # socket_joint_offset = -self.tibia_socket_overlap + 
        socket_joint_offset =(1/3)*self.amputated_tibia_length # Set the socket_joint_offset at (2/3)*socket length 


        prosthetic_shank_body.add_site(
            name=f"pylon_socket_joint{self.prosthesis_side}",
            pos=[0, socket_joint_offset,0], # This site is at the "bottom" of the shank, relative to prosthetic_shank_body's origin
            size=[0.01,0.01,0.01],
            rgba=[1, 0, 0, 1]
        )
        socket_joint_damping_tz = 40 #100 #100 #400 #200 # ty/2
        socket_joint_stiffness_tz = 20000 #100000 #20000 #43500 #30000 #43500 #21750
        socket_joint_damping_tx = 40 #100 #400 #200 # ty/2
        socket_joint_stiffness_tx = 43500 #120000 #43500 #43500 #10000 #21750 # ty/2
        
        if hasattr(self, 'socket_ty_joint_stiffness'):
            socket_joint_stiffness_ty = self.socket_ty_joint_stiffness
            print('socket_joint_stiffness_ty', socket_joint_stiffness_ty)
        else: 
            socket_joint_stiffness_ty= 43500 #10 #43500 #8000 #4350 #43500 #LaPrè, A. K., et al. "Approach for gait analysis in persons with limb loss including residuum and prosthesis socket dynamics." International Journal for Numerical Methods in Biomedical Engineering 34.4 (2018): e2936.
        
        if hasattr(self, 'socket_ty_joint_damping'):
            socket_joint_damping_ty = self.socket_ty_joint_damping
            print('socket_joint_damping_ty', socket_joint_damping_ty)
        else:
            socket_joint_damping_ty = 4 #40 #4 #40 #1 #0  #4 #100 #400 #3000 #100 #5

        socket_joint_stiffness_axial = 10 #1000 #10 #LaPrè, A. K., et al. "Approach for gait analysis in persons with limb loss including residuum and prosthesis socket dynamics." International Journal for Numerical Methods in Biomedical Engineering 34.4 (2018): e2936.
        socket_joint_damping_axial = 2 #300 #5
        socket_joint_stiffness_flexion = 997 #6000 #997 #LaPrè, A. K., et al. "Approach for gait analysis in persons with limb loss including residuum and prosthesis socket dynamics." International Journal for Numerical Methods in Biomedical Engineering 34.4 (2018): e2936.
        socket_joint_damping_flexion = 10
        socket_joint_stiffness_adduction = 623 #6000 #623 #LaPrè, A. K., et al. "Approach for gait analysis in persons with limb loss including residuum and prosthesis socket dynamics." International Journal for Numerical Methods in Biomedical Engineering 34.4 (2018): e2936.
        socket_joint_damping_adduction = 6

        if hasattr(self, 'socket_ty_joint_range'):
            socket_ty_joint_range = self.socket_ty_joint_range
            print('socket_ty_joint_range', socket_ty_joint_range)
        else: 
            socket_ty_joint_range = [-0.02, 0.02]

        # # # # # # More systematic Average over stiffness of all subjects 
        # socket_joint_stiffness_tz = 21000 #5500 
        # socket_joint_damping_tz =  40#ty/2
        # socket_joint_stiffness_tx = 43500 #5500 #ty/2
        # socket_joint_damping_tx =  40 
        # socket_joint_stiffness_ty=  43500 #5500 #LaPrè, A. K., et al. "Approach for gait analysis in persons with limb loss including residuum and prosthesis socket dynamics." International Journal for Numerical Methods in Biomedical Engineering 34.4 (2018): e2936.
        # socket_joint_damping_ty =  40 #
        # socket_joint_stiffness_axial = 60 #LaPrè, A. K., et al. "Approach for gait analysis in persons with limb loss including residuum and prosthesis socket dynamics." International Journal for Numerical Methods in Biomedical Engineering 34.4 (2018): e2936.
        # socket_joint_damping_axial = 6
        # socket_joint_stiffness_flexion = 615  #LaPrè, A. K., et al. "Approach for gait analysis in persons with limb loss including residuum and prosthesis socket dynamics." International Journal for Numerical Methods in Biomedical Engineering 34.4 (2018): e2936.
        # socket_joint_damping_flexion = 6
        # socket_joint_stiffness_adduction = 915 #LaPrè, A. K., et al. "Approach for gait analysis in persons with limb loss including residuum and prosthesis socket dynamics." International Journal for Numerical Methods in Biomedical Engineering 34.4 (2018): e2936.
        # socket_joint_damping_adduction = 9

        # # Add joints to the prosthetic shank body to allow for flexibility
        if hasattr(self, 'socket_joint') and not self.socket_joint:
            pass 
        else:

            if hasattr(self, 'socket_tx_joint') and not self.socket_tx_joint:
                pass
            else:
                prosthetic_shank_body.add_joint(
                    name=f"socket_tx{self.prosthesis_side}", type=mujoco.mjtJoint.mjJNT_SLIDE,
                    pos=[0,socket_joint_offset,0], axis=[1, 0, 0], range=[-0.01, 0.01], # Linear translation X [-0.01]
                    stiffness=socket_joint_stiffness_tx, damping=socket_joint_damping_tx,
                )

            if hasattr(self, 'socket_ty_joint') and not self.socket_ty_joint:
                pass
            else:
                prosthetic_shank_body.add_joint(
                    name=f"socket_ty{self.prosthesis_side}", type=mujoco.mjtJoint.mjJNT_SLIDE,
                    pos=[0,socket_joint_offset,0], axis=[0, 1, 0], range=socket_ty_joint_range, #[-0.02,0.02], #[-0.008,0.01], #[-0.02,0.02], #[-0.038,0.025],
                    stiffness=socket_joint_stiffness_ty, damping=socket_joint_damping_ty, #solref= 0.01#,armature=1,
                    # springref=0.005
                    # range=[-0.038,0.025], #[-0.009,0.001], # Half the range for solimp#[- 0.038,0.025]#LaPrè, A. K., et al. "Approach for gait analysis in persons with limb loss including residuum and prosthesis socket dynamics." International Journal for Numerical Methods in Biomedical Engineering 34.4 (2018): e2936.
                    # range=[-0.034,0.020], #[-0.038,0.023], #[-0.009,0.001], # Half the range for solimp#[- 0.038,0.025]#LaPrè, A. K., et al. "Approach for gait analysis in persons with limb loss including residuum and prosthesis socket dynamics." International Journal for Numerical Methods in Biomedical Engineering 34.4 (2018): e2936.
                    # range=[-0.025,0.015], 
                    # range=[-0.025,0.015], 
                    # range=[-0.030,0.02], 

                    
                    # Linear translation Y [-0.02]
                    # stiffness=socket_joint_stiffness_ty, damping=socket_joint_damping_ty,  
                    # solimp_limit = [0, 0.9, 0.02, 0.5, 1], margin = 0.02 #2 #0.02 #85
                    
                    # solimp_limit = [0, 0.95, 0.04, 0.5, 1], margin = 0.04 #2 #0.02 #85
                    # solimp_limit = [0, 0.95, 0.02, 0.5, 1], margin = 0.02 #2 #0.02 #85
                    # solimp_limit = [0, 0.97, 0.02, 0.5, 1], margin = 0.02 #2 #0.02 #85
                    # solimp_limit = [0, 0.97, 0.015, 0.5, 1], margin = 0.015 #2 #0.02 #85
                    
                    # solimp_limit = [0, 0.92, 0.03, 0.5, 1], margin = 0.03 #2 #0.02 #85
                    # solimp_limit = [0, 0.92, 0.03, 0.5, 1], margin = 0.025 #2 #0.02 #85
                    # solimp_limit = [0, 0.92, 0.025, 0.5, 1], margin = 0.03 #2 #0.02 #85

                    # solimp_limit = [0, 0.85, 0.018, 0.5, 1], margin=0.018,
                    # solimp_limit = [0, 0.92, 0.018, 0.5, 1], margin=0.018,
                    # solimp_limit = [0, 0.92, 0.025, 0.5, 1], margin=0.03,
                    
                    # solimp_limit = [0, 0.85, 0.025, 0.5, 1], margin=0.025,
                    # solimp_limit = [0, 0.85, 0.07, 0.5, 1], margin=0.07,
                    # solimp_limit = [0, 0.88, 0.07, 0.5, 1], margin=0.07,
                    # solimp_limit = [0, 0.88, 0.05, 0.5, 1], margin=0.05,
                    

                    # solimp_limit = [0, 0.88, 0.03, 0.5, 1], margin=0.03,
                    # solimp_limit = [0, 0.88, 0.04, 0.5, 1], margin=0.04,
                    # solimp_limit = [0, 0.88, 0.03, 0.5, 1], margin=0.025,
                    # solimp_limit = [0, 0.88, 0.018, 0.5, 1], margin=0.03,
                    
                    
                    # solimp_limit = [0, 0.85, 0.018, 0.5, 1], margin=0.03,
                    # solimp_limit = [0, 0.85, 0.018, 0.5, 1], margin=0.05,
                    # solimp_limit = [0, 0.85, 0.018, 0.5, 1], margin=0.07,

                    # solimp_limit = [0, 0.85, 0.010, 0.5, 1], margin=0.05,
                    # solimp_limit = [0, 0.96, 0.008, 0.5, 1], margin=0.008,
                    #solimp_limit = [0, 0.88, 0.004, 0.5, 1], margin=0.004,
                    # solimp_limit = [0, 0.96, 0.008, 0.5, 1], margin=0.008,
                    
                    
                    
                    # solimp_limit = [0, 0.96, 0.007, 0.5, 1], margin=0.007,
                    # solimp_limit = [0, 0.96, 0.006, 0.5, 1], margin=0.006,
                    # solimp_limit = [0, 0.94, 0.006, 0.5, 1], margin=0.006,
                    # solimp_limit = [0, 0.92, 0.002, 0.5, 1], margin=0.01,

                    # solimp_limit = [0, 0.96, 0.008, 0.5, 1], margin=0.008 ##### FINAL
                    # solimp_limit = [0, 0.92, 0.008, 0.5, 1], margin=0.08,
                    # solimp_limit = [0, 0.92, 0.008, 0.5, 1], margin=0.05,

                    # solimp_limit = [0, 0.96, 0.008, 0.5, 1], margin=0.008, damping= 0.5
                    # solimp_limit = [0, 0.96, 0.008, 0.5, 1], margin=0.008, damping= 0.2
                    # solimp_limit = [0, 0.96, 0.008, 0.5, 1], margin=0.008, damping= 0.8

                    # solimp_limit = [0, 0.96, 0.008, 0.5, 1], margin=0.008, damping= 2
                    # solimp_limit = [0, 0.96, 0.008, 0.5, 1], margin=0.008, damping= 1.5
                    # solimp_limit = [0, 0.96, 0.008, 0.5, 1], margin=0.008, damping= 3
                    
                    
                    # range=[-0.05,0.04], 
                    # solimp_limit = [0, 0.96, 0.008, 0.5, 1], margin=0.02,# damping= 3
                    # range=[-0.05,0.04],  
                    # solimp_limit = [0, 0.96, 0.02, 0.5, 1], margin=0.02,# damping= 3
                    # range=[-0.05,0.04],  
                    # solimp_limit = [0, 0.96, 0.008, 0.5, 1], margin=0.04,# damping= 3


                    # range=[-0.05,0.04],  
                    # solimp_limit = [0, 0.96, 0.004, 0.5, 1], margin=0.02,# damping= 3
                    # range=[-0.05,0.04],  
                    # solimp_limit = [0, 0.96, 0.008, 0.5, 1], margin=0.025,# damping= 3
                    # range=[-0.15,0.1],  
                    # solimp_limit = [0, 0.96, 0.004, 0.5, 1], margin=0.08,

                    # range=[-0.15,0.1],  
                    # solimp_limit = [0, 0.96, 0.004, 0.5, 1], margin=0.08, stiffness = 10,
                    # range=[-0.15,0.15],  
                    # solimp_limit = [0, 0.96, 0.004, 0.5, 1], margin=0.08, stiffness = 10,
                    # range=[-0.15,0.15],  
                    # solimp_limit = [0, 0.96, 0.004, 0.5, 1], margin=0.1, stiffness = 10,

                    # range=[-0.15,0.15],  
                    # solimp_limit = [0, 0.96, 0.002, 0.5, 1], margin=0.1, stiffness = 10,
                    # range=[-0.15,0.15],  
                    # solimp_limit = [0, 0.96, 0.006, 0.5, 1], margin=0.1, stiffness = 10,
                    # range=[-0.15,0.15],  
                    # solimp_limit = [0, 0.96, 0.004, 0.5, 1], margin=0.15, stiffness = 10,

                )

            if hasattr(self, 'socket_tz_joint') and not self.socket_tz_joint:
                pass
            else:
                prosthetic_shank_body.add_joint(
                    name=f"socket_tz{self.prosthesis_side}", type=mujoco.mjtJoint.mjJNT_SLIDE,
                    pos=[0,socket_joint_offset,0], axis=[0, 0, 1], range=[-0.01, 0.01],# Linear translation Z
                    stiffness=socket_joint_stiffness_tz, damping=socket_joint_damping_tz,
                )

            if hasattr(self, 'socket_flexion_joint') and not self.socket_flexion_joint:
                pass
            else:
                prosthetic_shank_body.add_joint(
                    name=f"socket_flexion{self.prosthesis_side}", type=mujoco.mjtJoint.mjJNT_HINGE,
                    pos=[0,socket_joint_offset,0], axis=[0, 0, 1], range=[-0.174,0.087], #LaPrè, A. K., et al. "Approach for gait analysis in persons with limb loss including residuum and prosthesis socket dynamics." International Journal for Numerical Methods in Biomedical Engineering 34.4 (2018): e2936.
                    #0.1 # Flexion/Extension #socket_pos_relative_to_tibia
                    stiffness=socket_joint_stiffness_flexion, damping=socket_joint_damping_flexion,
                )

            if hasattr(self, 'socket_adduction_joint') and not self.socket_adduction_joint:
                pass
            else:
                prosthetic_shank_body.add_joint(
                    name=f"socket_adduction{self.prosthesis_side}", type=mujoco.mjtJoint.mjJNT_HINGE,
                    pos=[0,socket_joint_offset,0], axis=[1, 0, 0], range=[-0.1, 0.1],# Adduction/Abduction # 0.1
                    stiffness=socket_joint_stiffness_adduction, damping=socket_joint_damping_adduction,
                )

            if hasattr(self, 'socket_rotation_joint') and not self.socket_rotation_joint:
                pass
            else:
                prosthetic_shank_body.add_joint(
                    name=f"socket_rotation{self.prosthesis_side}", type=mujoco.mjtJoint.mjJNT_HINGE,
                    pos=[0,socket_joint_offset,0], axis=[0, 1, 0], range=[-0.35, 0.35], #LaPrè, A. K., et al. "Approach for gait analysis in persons with limb loss including residuum and prosthesis socket dynamics." International Journal for Numerical Methods in Biomedical Engineering 34.4 (2018): e2936.
                    # Angus: Internal/External Rotation # 0.3
                    stiffness=socket_joint_stiffness_axial, damping=socket_joint_damping_axial,
                )

        return prosthetic_shank_body



    
    def adapt_spec_with_prosthesis_adapter(self, spec: mujoco.MjSpec):
        """
        Adapts a MuJoCo MjSpec object by adding a prosthesis and reconnecting
        the original foot bodies to the new prosthetic shank.

        Args:
            spec: The mujoco.MjSpec object to be modified.
        """

        prosthesis_side_str = self.prosthesis_side  # e.g., "_l" for left, "_r" for right

        # Define names for the relevant human bodies
        femur_name = f"femur{prosthesis_side_str}"
        tibia_name = f"tibia{prosthesis_side_str}"
        talus_name = f"talus{prosthesis_side_str}"

        # Find the femur and tibia bodies in the model specification
        femur_body = spec.find_body(femur_name)
        tibia_body = spec.find_body(tibia_name)

        if femur_body is None:
            print(f"Error: '{femur_name}' body not found in the model spec. Cannot add prosthesis.")
            return spec
        if tibia_body is None:
            print(f"Error: '{tibia_name}' body not found in the model spec. Cannot add prosthesis.")
            return spec

        # --- 1. Identify and Detach original foot bodies from the tibia ---
        original_talus_body = None 
        
        # We need to iterate over a copy of the list of children because we're detaching elements,
        # which can modify the list being iterated over.
        children_of_tibia = list(tibia_body.bodies) 
        for child_body in children_of_tibia: 
            if child_body.name == talus_name:
                original_talus_body = child_body
                print(f"Found '{talus_name}' by name as a direct child of '{tibia_name}'.")
                break 
            # This secondary check for ankle joint is a good fallback for cases where
            # the body name might not be exactly 'talus_l' but still represents the talus.
            if any(joint.name == f"ankle_angle{prosthesis_side_str}" for joint in child_body.joints):
                if not original_talus_body: # Only use this if not found by name already
                    original_talus_body = child_body
                    print(f"Found a body by ankle joint '{f'ankle_angle{prosthesis_side_str}'}' as a direct child of '{tibia_name}'. Assuming this is '{talus_name}'.")
                    break 
        
        # Helper function for printing subtree contents
        def print_subtree_contents(body, indent=2):
            # print(f"{' ' * indent}--- Contents of '{body.name}' (ID: {id(body)}) ---")
            # print(f"{' ' * (indent + 2)}Position: {body.pos.tolist() if body.pos is not None else 'N/A'}")
            # print(f"{' ' * (indent + 2)}Quaternion: {body.quat.tolist() if body.quat is not None else 'N/A'}")
            # print(f"{' ' * (indent + 2)}Geometries: {[g.name for g in body.geoms]}")
            # print(f"{' ' * (indent + 2)}Joints: {[j.name for j in body.joints]}")
            # print(f"{' ' * (indent + 2)}Sites: {[s.name for s in body.sites]}")
            # print(f"{' ' * (indent + 2)}Child Bodies: {[b.name for b in body.bodies]}")
            for child in body.bodies:
                print_subtree_contents(child, indent + 2)


        # amputated_tibia_length = 0.4 #0.2
        # tibia_socket_overlap =  0.1

        prosthetic_shank_body= self.calculate_tibia_socket_parameters(spec, original_talus_body.pos) #, amputated_tibia_length, tibia_socket_overlap)

    

        # --- 3. Reconnect the original foot bodies to the prosthetic shank ---
        if original_talus_body:
            # We need to compute the new *relative* position for the copied talus body.
            # The 'original_talus_body.pos' is its position relative to its *old parent (tibia)*.
            # We want to place the new talus at the end of the prosthetic shank.

            # Determine the target attachment point on the prosthetic shank (e.g., the pylon_attachment_site)
            # The site's pos is relative to prosthetic_shank_body.
            pylon_attachment_site = spec.find_site(f"talus_attachment_site_in_pylon{self.prosthesis_side}") #"pylon_attachment_site{prosthesis_side_str}")
            if not pylon_attachment_site:
                print(f"Error: pylon_attachment_site{prosthesis_side_str} not found on prosthetic shank. Cannot attach foot.")
                return spec

            new_talus_pos_relative_to_pylon = list(pylon_attachment_site.pos)


            # This is the helper function to recursively copy a body and its contents
        def copy_body_recursive(source_body, parent_mjbody, target_relative_pos=None, target_relative_quat=None):
            """
            Copies a source_body and its entire subtree (geoms, joints, children, etc.)
            under a new parent_mjbody.

            Args:
                source_body: The MjBody object to copy.
                parent_mjbody: The new MjBody object that will be the parent of the copied body.
                target_relative_pos: Optional. The desired position of the *current* copied body
                                     relative to its *new parent*. This is used only for the initial call
                                     (e.g., for original_talus_body). For recursive calls on children,
                                     it should be None, and source_body.pos will be used directly.
                target_relative_quat: Optional. The desired quaternion of the *current* copied body
                                      relative to its *new parent*. Similar logic to target_relative_pos.
            Returns:
                The newly created MjBody object that is the copy of source_body.
            """

            # Determine the position and quaternion for the new body.
            # Use target_relative_pos/quat for the top-level body if provided.
            # For children in recursive calls, use their original relative positions/quaternions directly.
            body_pos_to_use = target_relative_pos if target_relative_pos is not None else source_body.pos
            body_quat_to_use = target_relative_quat if target_relative_quat is not None else source_body.quat

            new_body = parent_mjbody.add_body(
                name=source_body.name,
                pos=body_pos_to_use,
                quat=body_quat_to_use,
                mocap=source_body.mocap,
                gravcomp=source_body.gravcomp,
            )
            # print(f"  Copied body '{source_body.name}' (Original ID: {id(source_body)}) to new body '{new_body.name}' (New ID: {id(new_body)}) under parent '{parent_mjbody.name}').")
            # print(f"  New body '{new_body.name}' pos relative to new parent: {new_body.pos.tolist()}")

            # Copy inertial properties (if they exist and are correctly structured in MjSpec).
            # Your original code for inertial properties seemed problematic, so ensure this part
            # correctly handles MuJoCo's MjInertial structure if your model explicitly defines it.
            # If your model uses inertiafromgeom="auto" (as in skeleton_muscle.xml's compiler tag),
            # MuJoCo will calculate inertia automatically based on geoms.
            
            # Copy inertia, mass etc.
            new_body.mass = source_body.mass
            new_body.ipos = source_body.ipos
            new_body.fullinertia = source_body.fullinertia  
            # Copy geometries
            for geom in source_body.geoms:
                new_body.add_geom(
                    name=geom.name,
                    type=geom.type,
                    size=geom.size,
                    pos=geom.pos, # Geoms' positions are relative to their parent body, copy directly
                    quat=geom.quat,
                    meshname=geom.meshname,
                    rgba=geom.rgba,
                    contype=geom.contype,
                    conaffinity=geom.conaffinity,
                    condim=geom.condim,
                    group=geom.group,
                    material=geom.material,
                )

            # Copy joints
            for joint in source_body.joints:
                new_body.add_joint(
                    name=joint.name,
                    type=joint.type,
                    pos=joint.pos, # Joints' positions are relative to their parent body, copy directly
                    axis=joint.axis,
                    range=joint.range,
                    stiffness=joint.stiffness,
                    damping=joint.damping,
                    limited=joint.limited,
                    springref=joint.springref,
                )

            # Copy sites
            for site in source_body.sites:
                new_body.add_site(
                    name=site.name,
                    pos=site.pos, # Sites' positions are relative to their parent body, copy directly
                    quat=site.quat,
                    size=site.size,
                    type=site.type,
                    rgba=site.rgba,
                    group=site.group,
                )

            # Recursively copy child bodies.
            # Children's positions/quats are already relative to the *source_body*.
            # When copying them under the *new_body*, their pos/quat remain the same.
            for child_source_body in source_body.bodies:
                copy_body_recursive(child_source_body, new_body) # NO target_relative_pos/quat for children
            return new_body

        # --- 3. Reconnect the original foot bodies to the prosthetic shank ---
        if original_talus_body:
            # Determine the target attachment point on the prosthetic shank.
            pylon_attachment_site = spec.find_site(f"talus_attachment_site_in_pylon{self.prosthesis_side}") #spec.find_site(f"pylon_attachment_site{prosthesis_side_str}")
            if not pylon_attachment_site:
                print(f"Error: pylon_attachment_site{prosthesis_side_str} not found on prosthetic shank. Cannot attach foot.")
                return spec

            # The new position of the talus body's origin will be directly at the pylon_attachment_site
            # relative to the prosthetic_shank_body.
            new_talus_pos_relative_to_pylon = list(pylon_attachment_site.pos)

            # IMPORTANT DEBUGGING STEP: Print the state of original_talus_body right before the copy
            # print(f"\n--- State of original_talus_body (ID: {id(original_talus_body)}) before copy_body_recursive call ---")
            # print(f"Name: {original_talus_body.name}, Pos: {original_talus_body.pos.tolist()}, Quat: {original_talus_body.quat.tolist()}")
            # print(f"It has {len(original_talus_body.bodies)} children.")
            # print("------------------------------------------------------------------------------------\n")

            # Call the recursive copy function.
            # Pass the calculated new_talus_pos_relative_to_pylon as the target position
            # for the root of the copied subtree.
            reparented_talus_body = copy_body_recursive(
                original_talus_body,
                prosthetic_shank_body,
                target_relative_pos=np.array(new_talus_pos_relative_to_pylon),# + np.array([0,self.tibia_socket_overlap,0])),
                target_relative_quat=original_talus_body.quat # Assuming no change in orientation for the foot
            )

            # print(f"Recreated and reconnected '{reparented_talus_body.name}' (ID: {id(reparented_talus_body)}) "
            #       f"and its entire foot subtree to 'pylon_socket{prosthesis_side_str}' (ID: {id(prosthetic_shank_body)}).")

            # Verification of the NEWLY RECREATED body
            # print(f"\n--- Verifying contents of '{reparented_talus_body.name}' (ID: {id(reparented_talus_body)}) after re-creation ---")
            # print(f"  New Parent (by reference): '{prosthetic_shank_body.name}'")
            # print(f"  Its position relative to new parent: {reparented_talus_body.pos.tolist()}")
            # print(f"  Its quaternion relative to new parent: {reparented_talus_body.quat.tolist()}")
            # print(f"  Geometries: {[g.name for g in reparented_talus_body.geoms]}")
            # print(f"  Joints: {[j.name for j in reparented_talus_body.joints]}")
            # print(f"  Child Bodies: {[b.name for b in reparented_talus_body.bodies]}")

            print_subtree_contents(reparented_talus_body)
            # print("--------------------------------------------------\n")

            # Check for the *original* talus body (should still exist as a Python object, but be detached)
            print(f"DEBUG: Original talus body object ('{original_talus_body.name}', ID: {id(original_talus_body)}) still exists but is now detached from the MjSpec. Its parent in the MjSpec hierarchy is effectively None.")


        else:
            print("CRITICAL ERROR: Talus body was not found after initial search, so no foot bodies could be reconnected.")

        # This detach block should be outside the `if original_talus_body:` to ensure it runs
        # even if an error occurred earlier, but only if original_talus_body was successfully found.
        # It's good that you have it here for cleanup.
        if original_talus_body:
            # print(f"\n--- Identifying original_talus_body (after finding loop) ---")
            # print(f"Variable 'original_talus_body' (ID: {id(original_talus_body)}) points to MjBody object named: '{original_talus_body.name}'")
            # print(f"Expected name: '{talus_name}'")
            # print(f"Is it the expected body? {original_talus_body.name == talus_name}")
            # print(f"Its current parent (before detach): '{tibia_name}' (confirmed by direct child search of '{tibia_name}.bodies')")

            # print(f"\n--- Full subtree of '{original_talus_body.name}' before detachment ---")
            print_subtree_contents(original_talus_body)
            # print("-------------------------------------------\n")

            spec.detach_body(original_talus_body)
            print(f"Detached '{original_talus_body.name}' (ID: {id(original_talus_body)}) from its original parent '{tibia_name}'. This object is now detached from the MjSpec hierarchy.")
        else:
            print(f"CRITICAL ERROR: Talus body (neither by name '{talus_name}' nor by ankle joint) "
                  f"was NOT found as a direct child of '{tibia_name}'. Cannot proceed with re-parenting.")
            return spec

        return spec
    



    # def adapt_spec_with_prosthesis_adapter_old(self, spec: mujoco.MjSpec):
    #     """
    #     Adapts a MuJoCo MjSpec object by adding a prosthesis and reconnecting
    #     the original foot bodies to the new prosthetic shank.

    #     Args:
    #         spec: The mujoco.MjSpec object to be modified.
    #     """

    #     prosthesis_side_str = self.prosthesis_side  # e.g., "_l" for left, "_r" for right

    #     # Define names for the relevant human bodies
    #     femur_name = f"femur{prosthesis_side_str}"
    #     tibia_name = f"tibia{prosthesis_side_str}"
    #     talus_name = f"talus{prosthesis_side_str}"

    #     # Find the femur and tibia bodies in the model specification
    #     femur_body = spec.find_body(femur_name)
    #     tibia_body = spec.find_body(tibia_name)

    #     if femur_body is None:
    #         print(f"Error: '{femur_name}' body not found in the model spec. Cannot add prosthesis.")
    #         return spec
    #     if tibia_body is None:
    #         print(f"Error: '{tibia_name}' body not found in the model spec. Cannot add prosthesis.")
    #         return spec

    #     # --- 1. Identify and Detach original foot bodies from the tibia ---
    #     original_talus_body = None 
        
    #     # We need to iterate over a copy of the list of children because we're detaching elements,
    #     # which can modify the list being iterated over.
    #     children_of_tibia = list(tibia_body.bodies) 
    #     for child_body in children_of_tibia: 
    #         if child_body.name == talus_name:
    #             original_talus_body = child_body
    #             print(f"Found '{talus_name}' by name as a direct child of '{tibia_name}'.")
    #             break 
    #         # This secondary check for ankle joint is a good fallback for cases where
    #         # the body name might not be exactly 'talus_l' but still represents the talus.
    #         if any(joint.name == f"ankle_angle{prosthesis_side_str}" for joint in child_body.joints):
    #             if not original_talus_body: # Only use this if not found by name already
    #                 original_talus_body = child_body
    #                 print(f"Found a body by ankle joint '{f'ankle_angle{prosthesis_side_str}'}' as a direct child of '{tibia_name}'. Assuming this is '{talus_name}'.")
    #                 break 
        
    #     # Get talus body postion shift along y-axis # This should be total leg length even with amputation and prosthesis 
    #     original_talus_pos_y = original_talus_body.pos[1]
    #     # Helper function for printing subtree contents
    #     def print_subtree_contents(body, indent=2):
    #         # print(f"{' ' * indent}--- Contents of '{body.name}' (ID: {id(body)}) ---")
    #         # print(f"{' ' * (indent + 2)}Position: {body.pos.tolist() if body.pos is not None else 'N/A'}")
    #         # print(f"{' ' * (indent + 2)}Quaternion: {body.quat.tolist() if body.quat is not None else 'N/A'}")
    #         # print(f"{' ' * (indent + 2)}Geometries: {[g.name for g in body.geoms]}")
    #         # print(f"{' ' * (indent + 2)}Joints: {[j.name for j in body.joints]}")
    #         # print(f"{' ' * (indent + 2)}Sites: {[s.name for s in body.sites]}")
    #         # print(f"{' ' * (indent + 2)}Child Bodies: {[b.name for b in body.bodies]}")
    #         for child in body.bodies:
    #             print_subtree_contents(child, indent + 2)


    #     # Get talus body position shift along y-axis
    #     original_talus_pos_y = original_talus_body.pos[1]

    #     # Define the desired vertical offset for the top of the prosthetic socket
    #     # relative to the tibia's origin (knee joint).
    #     # This value should be negative to move the socket downwards.
    #     # For example, -0.05 means the socket starts 5 cm below the knee.
    #     socket_top_offset_y = -0.2 #-0.05 # Adjust this value as desired for your offset

    #     # Dynamically calculate the required length of the prosthetic shank (pylon).
    #     # This length bridges the gap between the socket's starting point
    #     # and the original ankle's vertical position.
    #     # The equation ensures: (Tibia Y + socket_top_offset_y) - shank_length = Tibia Y + original_talus_pos_y
    #     shank_length = socket_top_offset_y - original_talus_pos_y
    #     # Example: If socket_top_offset_y = -0.05 and original_talus_pos_y = -0.45 (0.45m below knee),
    #     # then shank_length = -0.05 - (-0.45) = 0.40.

    #     shank_radius = 0.04 # Keep your existing radius
    #     pylon_mass = 0.3 # Like in Transtibial_Left_Prosthesis
    #     pylon_inertia = [0.0136, 0.0021,0.0136, 0, 0,0]

    #     # Position of the prosthetic_shank_body's origin relative to the tibia.
    #     # This applies the desired offset for the socket's top.
    #     adapter_pos_relative_to_tibia = np.array([0, socket_top_offset_y, 0])
    #     pylon_relative_inertia_pos = np.array([0,0.0491,0])
    #     pylon_inertia_pos = adapter_pos_relative_to_tibia - pylon_relative_inertia_pos


    #     # Scale tibia mass, fullinertia and center of mass position 
    #     # --- NEW: Scale Tibia Mass based on Amputation Ratio ---
    #     original_tibia_mass = tibia_body.mass
    #     original_tibia_fullinertia = list(tibia_body.fullinertia) # Make a copy

    #     original_tibia_length = abs(original_talus_pos_y) # Distance from tibia origin (knee) to talus origin (ankle)
    
    #     if original_tibia_length == 0:
    #         print(f"Warning: Original tibia length calculated as zero for {tibia_name}. Mass scaling skipped.")
    #     else:
    #         # Calculate the remaining length of the tibia (from knee to socket top)
    #         remaining_tibia_length = abs(socket_top_offset_y) # Distance from tibia origin (knee) to socket top


    #     amputation_ratio = remaining_tibia_length / original_tibia_length

    #     print(f"Original {tibia_name} length: {original_tibia_length:.4f} m")
    #     print(f"Remaining {tibia_name} length (to socket top): {remaining_tibia_length:.4f} m")
    #     print(f"Amputation ratio for {tibia_name}: {amputation_ratio:.4f}")

    #     # Scale the mass of the remaining tibia
    #     tibia_body.mass = original_tibia_mass * amputation_ratio
    #     print(f"Scaled {tibia_name} mass from {original_tibia_mass:.4f} kg to {tibia_body.mass:.4f} kg")

    #     # Scale the inertia. Inertia scales by (mass_ratio) * (length_ratio)^2 for simple cases
    #     # or more accurately, if we assume a cylinder, by (mass_ratio) * (length_ratio)^2.
    #     # For simplicity and a first pass, we can scale by the mass ratio for fullinertia as well.
    #     # A more precise scaling might involve re-calculating the inertia tensor based on the new geometry.
    #     # For now, let's scale linearly with mass ratio, which is common if the shape isn't drastically altered.
    #     tibia_body.fullinertia = [val *amputation_ratio**3 for val in original_tibia_fullinertia]
    #     print(f"Scaled {tibia_name} fullinertia by factor {amputation_ratio:.4f}")
    #     print(f"Full inertia is {tibia_body.fullinertia}")

    

    #     # Move tibia ipos upward also by ratio 
    #     tibia_body_ipos = tibia_body.ipos
    #     tibia_body.ipos = [p*amputation_ratio for p in tibia_body_ipos]
    #     print(f'tibia_body.ipos: {tibia_body.ipos}')

    #     tibia_body.add_site(
    #         name=f"tibia_org_COM{prosthesis_side_str}",
    #         pos=tibia_body_ipos, # This site is at the "bottom" of the shank, relative to prosthetic_shank_body's origin
    #         size=[0.01,0.01,0.01],
    #         rgba=[0, 1, 0, 1]
    #     )

    #     tibia_body.add_site(
    #         name=f"tibia_COM{prosthesis_side_str}",
    #         pos=tibia_body.ipos, # This site is at the "bottom" of the shank, relative to prosthetic_shank_body's origin
    #         size=[0.01,0.01,0.01],
    #         rgba=[0, 1, 0, 1]
    #     )


    #     prosthetic_shank_body = tibia_body.add_body(
    #         name=f"pylon_socket{prosthesis_side_str}",
    #         pos=adapter_pos_relative_to_tibia, # Use the calculated offset here
    #         mass = pylon_mass,
    #         fullinertia = pylon_inertia,
    #         ipos = pylon_inertia_pos,
    #     )
       
    #     prosthetic_shank_body.add_site(
    #         name=f"pylon_COM{prosthesis_side_str}",
    #         pos=pylon_inertia_pos, # This site is at the "bottom" of the shank, relative to prosthetic_shank_body's origin
    #         size=[0.01,0.01,0.01],
    #         rgba=[0, 1, 0, 1]
    #     )

    #     print(f"Created new prosthetic shank body: '{prosthetic_shank_body.name}' (ID: {id(prosthetic_shank_body)}). Its parent is: '{tibia_name}'")

    #     prosthetic_shank_body.add_geom(
    #         name=f"pylon_socket_geom{prosthesis_side_str}",
    #         type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    #         size=[shank_radius,shank_length/4,shank_radius],#, shank_length],
    #         pos=[0, socket_top_offset_y-shank_length/4 ,0], # Center of the cylinder, relative to prosthetic_shank_body's origin
    #         euler=[1.571, 0, 0], # Rotate to be vertical if it's currently horizontal
    #         rgba=[0.5, 0.5, 0.5, 1]
    #     )
        
    #     prosthetic_shank_body.add_site(
    #         name=f"pylon_attachment_site{prosthesis_side_str}",
    #         pos=[0, -shank_length,0], # This site is at the "bottom" of the shank, relative to prosthetic_shank_body's origin
    #         size=[0.01,0.01,0.01],
    #         rgba=[1, 0, 0, 1]
    #     )

    #     # Add joints to the prosthetic shank body to allow for flexibility
    #     prosthetic_shank_body.add_joint(
    #         name=f"socket_flexion{prosthesis_side_str}", type=mujoco.mjtJoint.mjJNT_HINGE,
    #         pos=[0, socket_top_offset_y, 0], axis=[0, 0, 1], range=[-1, 1], # Flexion/Extension
    #     )
    #     prosthetic_shank_body.add_joint(
    #         name=f"socket_adduction{prosthesis_side_str}", type=mujoco.mjtJoint.mjJNT_HINGE,
    #         pos=[0, socket_top_offset_y, 0], axis=[1, 0, 0], range=[-0.1, 0.1], # Adduction/Abduction
    #     )
    #     prosthetic_shank_body.add_joint(
    #         name=f"socket_rotation{prosthesis_side_str}", type=mujoco.mjtJoint.mjJNT_HINGE,
    #         pos=[0, socket_top_offset_y, 0], axis=[0, 1, 0], range=[-0.3, 0.3], # Internal/External Rotation
    #     )
    #     prosthetic_shank_body.add_joint(
    #         name=f"socket_tx{prosthesis_side_str}", type=mujoco.mjtJoint.mjJNT_SLIDE,
    #         pos=[0, socket_top_offset_y, 0], axis=[1, 0, 0], range=[-0.02, 0.02], # Linear translation X
    #     )
    #     prosthetic_shank_body.add_joint(
    #         name=f"socket_ty{prosthesis_side_str}", type=mujoco.mjtJoint.mjJNT_SLIDE,
    #         pos=[0, socket_top_offset_y, 0], axis=[0, 1, 0], range=[-0.1, 0.1], # Linear translation Y
    #     )
    #     prosthetic_shank_body.add_joint(
    #         name=f"socket_tz{prosthesis_side_str}", type=mujoco.mjtJoint.mjJNT_SLIDE,
    #         pos=[0, socket_top_offset_y, 0], axis=[0, 0, 1], range=[-0.02, 0.02], # Linear translation Z
    #     )

    #     # --- 3. Reconnect the original foot bodies to the prosthetic shank ---
    #     if original_talus_body:
    #         # We need to compute the new *relative* position for the copied talus body.
    #         # The 'original_talus_body.pos' is its position relative to its *old parent (tibia)*.
    #         # We want to place the new talus at the end of the prosthetic shank.

    #         # Determine the target attachment point on the prosthetic shank (e.g., the pylon_attachment_site)
    #         # The site's pos is relative to prosthetic_shank_body.
    #         pylon_attachment_site = spec.find_site(f"pylon_attachment_site{prosthesis_side_str}")
    #         if not pylon_attachment_site:
    #             print(f"Error: pylon_attachment_site{prosthesis_side_str} not found on prosthetic shank. Cannot attach foot.")
    #             return spec

    #         new_talus_pos_relative_to_pylon = list(pylon_attachment_site.pos)


    #         # This is the helper function to recursively copy a body and its contents
    #     def copy_body_recursive(source_body, parent_mjbody, target_relative_pos=None, target_relative_quat=None):
    #         """
    #         Copies a source_body and its entire subtree (geoms, joints, children, etc.)
    #         under a new parent_mjbody.

    #         Args:
    #             source_body: The MjBody object to copy.
    #             parent_mjbody: The new MjBody object that will be the parent of the copied body.
    #             target_relative_pos: Optional. The desired position of the *current* copied body
    #                                  relative to its *new parent*. This is used only for the initial call
    #                                  (e.g., for original_talus_body). For recursive calls on children,
    #                                  it should be None, and source_body.pos will be used directly.
    #             target_relative_quat: Optional. The desired quaternion of the *current* copied body
    #                                   relative to its *new parent*. Similar logic to target_relative_pos.
    #         Returns:
    #             The newly created MjBody object that is the copy of source_body.
    #         """

    #         # Determine the position and quaternion for the new body.
    #         # Use target_relative_pos/quat for the top-level body if provided.
    #         # For children in recursive calls, use their original relative positions/quaternions directly.
    #         body_pos_to_use = target_relative_pos if target_relative_pos is not None else source_body.pos
    #         body_quat_to_use = target_relative_quat if target_relative_quat is not None else source_body.quat

    #         new_body = parent_mjbody.add_body(
    #             name=source_body.name,
    #             pos=body_pos_to_use,
    #             quat=body_quat_to_use,
    #             mocap=source_body.mocap,
    #             gravcomp=source_body.gravcomp,
    #         )
    #         # print(f"  Copied body '{source_body.name}' (Original ID: {id(source_body)}) to new body '{new_body.name}' (New ID: {id(new_body)}) under parent '{parent_mjbody.name}').")
    #         # print(f"  New body '{new_body.name}' pos relative to new parent: {new_body.pos.tolist()}")

    #         # Copy inertial properties (if they exist and are correctly structured in MjSpec).
    #         # Your original code for inertial properties seemed problematic, so ensure this part
    #         # correctly handles MuJoCo's MjInertial structure if your model explicitly defines it.
    #         # If your model uses inertiafromgeom="auto" (as in skeleton_muscle.xml's compiler tag),
    #         # MuJoCo will calculate inertia automatically based on geoms.
            
    #         # Copy inertia, mass etc.
    #         new_body.mass = source_body.mass
    #         new_body.ipos = source_body.ipos
    #         new_body.fullinertia = source_body.fullinertia  
    #         # Copy geometries
    #         for geom in source_body.geoms:
    #             new_body.add_geom(
    #                 name=geom.name,
    #                 type=geom.type,
    #                 size=geom.size,
    #                 pos=geom.pos, # Geoms' positions are relative to their parent body, copy directly
    #                 quat=geom.quat,
    #                 meshname=geom.meshname,
    #                 rgba=geom.rgba,
    #                 contype=geom.contype,
    #                 conaffinity=geom.conaffinity,
    #                 condim=geom.condim,
    #                 group=geom.group,
    #                 material=geom.material,
    #             )

    #         # Copy joints
    #         for joint in source_body.joints:
    #             new_body.add_joint(
    #                 name=joint.name,
    #                 type=joint.type,
    #                 pos=joint.pos, # Joints' positions are relative to their parent body, copy directly
    #                 axis=joint.axis,
    #                 range=joint.range,
    #                 stiffness=joint.stiffness,
    #                 damping=joint.damping,
    #                 limited=joint.limited,
    #                 springref=joint.springref,
    #             )

    #         # Copy sites
    #         for site in source_body.sites:
    #             new_body.add_site(
    #                 name=site.name,
    #                 pos=site.pos, # Sites' positions are relative to their parent body, copy directly
    #                 quat=site.quat,
    #                 size=site.size,
    #                 type=site.type,
    #                 rgba=site.rgba,
    #                 group=site.group,
    #             )

    #         # Recursively copy child bodies.
    #         # Children's positions/quats are already relative to the *source_body*.
    #         # When copying them under the *new_body*, their pos/quat remain the same.
    #         for child_source_body in source_body.bodies:
    #             copy_body_recursive(child_source_body, new_body) # NO target_relative_pos/quat for children
    #         return new_body

    #     # --- 3. Reconnect the original foot bodies to the prosthetic shank ---
    #     if original_talus_body:
    #         # Determine the target attachment point on the prosthetic shank.
    #         pylon_attachment_site = spec.find_site(f"pylon_attachment_site{prosthesis_side_str}")
    #         if not pylon_attachment_site:
    #             print(f"Error: pylon_attachment_site{prosthesis_side_str} not found on prosthetic shank. Cannot attach foot.")
    #             return spec

    #         # The new position of the talus body's origin will be directly at the pylon_attachment_site
    #         # relative to the prosthetic_shank_body.
    #         new_talus_pos_relative_to_pylon = list(pylon_attachment_site.pos)

    #         # IMPORTANT DEBUGGING STEP: Print the state of original_talus_body right before the copy
    #         # print(f"\n--- State of original_talus_body (ID: {id(original_talus_body)}) before copy_body_recursive call ---")
    #         # print(f"Name: {original_talus_body.name}, Pos: {original_talus_body.pos.tolist()}, Quat: {original_talus_body.quat.tolist()}")
    #         # print(f"It has {len(original_talus_body.bodies)} children.")
    #         # print("------------------------------------------------------------------------------------\n")

    #         # Call the recursive copy function.
    #         # Pass the calculated new_talus_pos_relative_to_pylon as the target position
    #         # for the root of the copied subtree.
    #         reparented_talus_body = copy_body_recursive(
    #             original_talus_body,
    #             prosthetic_shank_body,
    #             target_relative_pos=np.array(new_talus_pos_relative_to_pylon),
    #             target_relative_quat=original_talus_body.quat # Assuming no change in orientation for the foot
    #         )

    #         # print(f"Recreated and reconnected '{reparented_talus_body.name}' (ID: {id(reparented_talus_body)}) "
    #         #       f"and its entire foot subtree to 'pylon_socket{prosthesis_side_str}' (ID: {id(prosthetic_shank_body)}).")

    #         # Verification of the NEWLY RECREATED body
    #         # print(f"\n--- Verifying contents of '{reparented_talus_body.name}' (ID: {id(reparented_talus_body)}) after re-creation ---")
    #         # print(f"  New Parent (by reference): '{prosthetic_shank_body.name}'")
    #         # print(f"  Its position relative to new parent: {reparented_talus_body.pos.tolist()}")
    #         # print(f"  Its quaternion relative to new parent: {reparented_talus_body.quat.tolist()}")
    #         # print(f"  Geometries: {[g.name for g in reparented_talus_body.geoms]}")
    #         # print(f"  Joints: {[j.name for j in reparented_talus_body.joints]}")
    #         # print(f"  Child Bodies: {[b.name for b in reparented_talus_body.bodies]}")

    #         print_subtree_contents(reparented_talus_body)
    #         # print("--------------------------------------------------\n")

    #         # Check for the *original* talus body (should still exist as a Python object, but be detached)
    #         print(f"DEBUG: Original talus body object ('{original_talus_body.name}', ID: {id(original_talus_body)}) still exists but is now detached from the MjSpec. Its parent in the MjSpec hierarchy is effectively None.")


    #     else:
    #         print("CRITICAL ERROR: Talus body was not found after initial search, so no foot bodies could be reconnected.")

    #     # This detach block should be outside the `if original_talus_body:` to ensure it runs
    #     # even if an error occurred earlier, but only if original_talus_body was successfully found.
    #     # It's good that you have it here for cleanup.
    #     if original_talus_body:
    #         # print(f"\n--- Identifying original_talus_body (after finding loop) ---")
    #         # print(f"Variable 'original_talus_body' (ID: {id(original_talus_body)}) points to MjBody object named: '{original_talus_body.name}'")
    #         # print(f"Expected name: '{talus_name}'")
    #         # print(f"Is it the expected body? {original_talus_body.name == talus_name}")
    #         # print(f"Its current parent (before detach): '{tibia_name}' (confirmed by direct child search of '{tibia_name}.bodies')")

    #         # print(f"\n--- Full subtree of '{original_talus_body.name}' before detachment ---")
    #         print_subtree_contents(original_talus_body)
    #         # print("-------------------------------------------\n")

    #         spec.detach_body(original_talus_body)
    #         print(f"Detached '{original_talus_body.name}' (ID: {id(original_talus_body)}) from its original parent '{tibia_name}'. This object is now detached from the MjSpec hierarchy.")
    #     else:
    #         print(f"CRITICAL ERROR: Talus body (neither by name '{talus_name}' nor by ankle joint) "
    #               f"was NOT found as a direct child of '{tibia_name}'. Cannot proceed with re-parenting.")
    #         return spec

    #     return spec







    def scale_foot_to_SACH_keep_distribution(self, spec: mujoco.MjSpec):
        """
        Scales the mass and fullinertia of specified foot bodies to a target total mass
        (SACH_total_mass) while maintaining the original mass distribution proportions
        among the foot segments.

        Args:
            spec: The MuJoCo MjSpec object representing the model.
        """
        SACH_total_mass = 0.575  # kg WITHOUT ADAPTER

        # Initialize dictionaries to store original mass and inertia
        mass_dict = {}
        fullinertia_dict = {}

        # Define the base names of the foot bodies
        foot_body_base_names = ['talus', 'calcn', 'toes']

        # Append prosthesis_side to each foot body name
        # Assuming self.prosthesis_side is defined (e.g., '_r' or '_l')
        foot_body_names = [foot_body_base_name + self.prosthesis_side for foot_body_base_name in foot_body_base_names]

        # Get original mass and fullinertia of these segments
        for b in spec.bodies:
            if b.name in foot_body_names:
                mass_dict[b.name] = b.mass
                fullinertia_dict[b.name] = b.fullinertia

        # Calculate the sum of all original masses in the foot
        foot_mass_total_original = 0.0
        for name in foot_body_names:
            if name in mass_dict:  # Ensure the body was found and has a mass entry
                foot_mass_total_original += mass_dict[name]
            else:
                print(f"Warning: Body '{name}' not found in spec.bodies or mass_dict. Skipping.")


        # Handle the case where no foot bodies were found or total mass is zero
        if foot_mass_total_original == 0:
            print("Error: Total original foot mass is zero. Cannot scale.")
            return

        # Calculate the mass ratios for each foot segment
        foot_segment_ratios = {}
        for name in foot_body_names:
            if name in mass_dict:
                foot_segment_ratios[name] = mass_dict[name] / foot_mass_total_original
            else:
                foot_segment_ratios[name] = 0.0 # Should not happen if the above check is robust

        # Scale the mass and fullinertia for each foot segment
        for b in spec.bodies:
            if b.name in foot_body_names:
                # Calculate the new mass based on the SACH total mass and original ratio
                new_mass = SACH_total_mass * foot_segment_ratios[b.name]
                b.mass = new_mass

                # Scale the fullinertia. Inertia scales proportionally to mass for a similar shape.
                # This assumes that the shape and density distribution within each segment remains
                # similar, only the overall mass changes.
                if b.name in fullinertia_dict and mass_dict[b.name] != 0:
                    # Calculate the scaling factor for inertia
                    inertia_scaling_factor = new_mass / mass_dict[b.name]
                    b.fullinertia = fullinertia_dict[b.name]*inertia_scaling_factor #[val * inertia_scaling_factor for val in fullinertia_dict[b.name]]
                elif b.name in fullinertia_dict and mass_dict[b.name] == 0:
                    # If original mass was zero but inertia existed, set new inertia to zero
                    b.fullinertia = [0.0] * len(fullinertia_dict[b.name])
                else:
                    # Handle cases where original fullinertia was not found (e.g., if it was zero or undefined)
                    print(f"Warning: Original fullinertia for {b.name} not found or was zero. Cannot scale inertia proportionally.")
                    # You might choose to set it to zero, or a default small value, or handle as per your model's needs.
                    # For now, we'll leave it as is if it was not in the dict.
                    # If you want to explicitly set to zero if not found:
                    # b.fullinertia = [0.0] * 6 # Assuming fullinertia is a 6-element list/array

                # Add site at mass center and make it bigger and black to visualize 
                # Check if a site with this name already exists to avoid duplicates if function is called multiple times
                site_name = f"com_site_{b.name}"
                site_exists = any(s.name == site_name for s in b.sites)
                if not site_exists:
                    b.add_site(
                        name=site_name,
                        size=[0.015, 0.015, 0.015], # Make it noticeably big (e.g., 1.5 cm radius)
                        rgba=[0.0, 0.0, 0.0, 1.0], # Black color, fully opaque
                        pos=b.ipos # Place the site at the body's center of mass (which is its 'pos' in its parent frame)
                    )
                    #b.sites.append(new_site)
                else:
                    print(f"Site '{site_name}' already exists for body '{b.name}'. Skipping addition.")



        return spec 








    def add_SACHFoot_properties(self, spec: mujoco.MjSpec):
        # Replace model foot properties with SACH Foot properties:
        # Total foot length 
        # Toe length
        # Toe joint (mtp_angle) position, axis, stiffness
        # ankle joint position, axis, stiffness 
        
        # Take out subtalar joint

        # Toe body mass, center of mass, inertia
        # Talus body mass, center of mass, inertia
        # Calcn body mass, center of mass, inertia  

        # Total foot length: Keep original foot size

        # Toe length: Keep original toe length? 

        # The SACH Foot has small masses and inertias so adapt the boundmass and boundinertia
        spec.compiler.boundmass = 0.00001
        spec.compiler.boundinertia = 0.00001

        # Talus, Calcn, Toe:  mass, center of mass and inertia --> Scale original mass and inertia down 
        self.scale_foot_to_SACH_keep_distribution(spec)

        if hasattr(self, 'remove_joint_names'): 
            print(f'Joints to remove: {self.remove_joint_names}')
            remove_joint_names = [name + self.prosthesis_side for name in self.remove_joint_names]
            self.remove_joint(spec, remove_joint_names)
            self.remove_equality(spec, remove_joint_names)
        if hasattr(self, 'joint_stiffness'): 
            # joint_stiffness = self.joint_stiffness
            # joint_names =  list(joint_stiffness.keys())

            # print(f'Increase stiffness of joint {self.stiffen_and_dampen_joints} to {self.joint_stiffness}')
            # print(f'Increase damping of joint {self.stiffen_and_dampen_joints} to {self.joint_damping}')
            self.increase_joint_stiffness_each_joint(spec)
        if hasattr(self, 'joint_damping'):
            self.increase_joint_damping_each_joint(spec)


        self.remove_site_actuator_tendon(spec)

       
        # # joint force sensor site name only for evaluation
        # if hasattr(self, 'add_sensors') and self.add_sensors:
        #     joint_force_sensor_site_name = "hip_mimic"
        #     self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
        #     self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
        #     self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
        #     self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")
        #     joint_force_sensor_site_name = "knee_mimic"
        #     self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
        #     self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
        #     self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
        #     self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")
        #     joint_force_sensor_site_name = "foot_mimic"
        #     self.add_force_sensor(spec, f"left_{joint_force_sensor_site_name}")
        #     self.add_force_sensor(spec, f"right_{joint_force_sensor_site_name}")
        #     self.add_torque_sensor(spec, f"left_{joint_force_sensor_site_name}")
        #     self.add_torque_sensor(spec, f"right_{joint_force_sensor_site_name}")

        return spec 



    # def _add_2_box_per_foot_to_spec(self, spec: mujoco.MjSpec):
    #     # find foot and attach box
    #     alpha_box_feet = 0.5
    #     scaling  = 1
    #     toe_l = spec.find_body("toes_l")
    #     size_foot = np.array([0.09, 0.03, 0.05])* scaling #np.array([0.100, 0.03, 0.05])* scaling
    #     size_toes = np.array([0.041, 0.03, 0.048]) * scaling 
    #     # size_toes = np.array([0.045, 0.03, 0.05]) * scaling 
    #     pos_foot = np.array([0.085, 0.019, -0.01]) * scaling
    #     pos_toes = np.array([0.035, 0.019, 0.01]) * scaling
    #     # Flip the z-axis for the mirrored positions and reassemble
    #     pos_foot_l = np.concatenate([pos_foot[:2], [-pos_foot[2]]])
    #     pos_toes_l = np.concatenate([pos_toes[:2], [-pos_toes[2]]])
    #     euler_foot = [0.0, 0.15, 0.0] #[0.0, 0.15, 0.0]
    #     euler_toes = [0.0, 0.15, 0.0] #[0.0, 0.15, 0.0]
    #     # size = np.array([0.112, 0.03, 0.05]) * scaling
    #     # pos = np.array([-0.09, 0.019, 0.0]) * scaling
    #     toe_l.add_geom(name="toes_box_l", type=mujoco.mjtGeom.mjGEOM_BOX, size=size_toes, pos=pos_toes_l,
    #                    rgba=[0, 1, 0, alpha_box_feet], euler=euler_toes)
    #     toe_r = spec.find_body("toes_r")
    #     toe_r.add_geom(name="toes_box_r", type=mujoco.mjtGeom.mjGEOM_BOX, size=size_toes, pos=pos_toes,
    #                    rgba=[0, 1, 0, alpha_box_feet], euler=[a*-1 for a in euler_toes])
        
    #     calcn_l = spec.find_body("calcn_l")
    #     calcn_r = spec.find_body("calcn_r")
    #     calcn_l.add_geom(name="foot_box_l", type=mujoco.mjtGeom.mjGEOM_BOX, size=size_foot, pos=pos_foot_l,
    #                    rgba=[1, 0, 0, alpha_box_feet], euler=euler_foot)
    #     calcn_r.add_geom(name="foot_box_r", type=mujoco.mjtGeom.mjGEOM_BOX, size=size_foot, pos=pos_foot,
    #                    rgba=[1, 0, 0, alpha_box_feet], euler=[a*-1 for a in euler_foot])
        

    #     # # make true foot uncollidable
    #     # foot_geoms = ["r_foot", "r_bofoot", "l_foot", "l_bofoot"]
    #     # for g in spec.geoms:
    #     #     if g.name in foot_geoms:
    #     #         g.contype = 0
    #     #         g.conaffinity = 0

    #     for g in spec.geoms:
    #         g.contype = 0
    #         g.conaffinity = 0

    #     # --- define contacts between feet and floor --
    #     spec.add_pair(geomname1="floor", geomname2="foot_box_r")
    #     spec.add_pair(geomname1="floor", geomname2="foot_box_l")
    #     spec.add_pair(geomname1="floor", geomname2="toes_box_r")
    #     spec.add_pair(geomname1="floor", geomname2="toes_box_l")

    #     return spec
    
    @info_property
    def foot_geom_names(self):
        return ['foot_box_r','toes_box_r', 'foot_box_l', 'toes_box_l']


    def _get_observation_specification(self, spec: mujoco.MjSpec):
        """
        Getter for the observation space specification.
        Args:
            spec (MjSpec): Specification of the environment.
        Returns:
            List[ObservationType]: List of observation space specification.
        """


        if self.add_pos_ori_to_observation:
            if hasattr(self, 'prosthesis_body_position_range'):
                rand_pos_body_names=[]
                observation_spec_body_pos = []
                # if OmegaConf.is_dict(self.prosthesis_body_position_range): # when loading with orbax
                #     prosthesis_body_position_range_dict = OmegaConf.to_container(self.prosthesis_body_position_range, resolve=True)
                #     rand_pos_body_names = list(prosthesis_body_position_range_dict.keys())
                if isinstance(self.prosthesis_body_position_range, dict):
                    rand_pos_body_names = list(self.prosthesis_body_position_range.keys())
                    # rand_pos_body_names.append(self.prosthesis_body_position_range.keys())
                for b in rand_pos_body_names:
                    b = b + self.prosthesis_side  # Append prosthesis side to body names 
                    observation_spec_body_pos.append(ObservationType.ModelBodyPos(f"pos_{b}", xml_name=b))

            if hasattr(self, 'prosthesis_body_orientation_range'):
                rand_ori_body_names=[]
                observation_spec_body_quat = []
                # if OmegaConf.is_dict(self.prosthesis_body_orientation_range): # when loading with orbax
                #     prosthesis_body_orientation_range_dict = OmegaConf.to_container(self.prosthesis_body_orientation_range, resolve=True)
                #     rand_ori_body_names = list(prosthesis_body_orientation_range_dict.keys())
                if isinstance(self.prosthesis_body_orientation_range, dict):
                    rand_ori_body_names= list(self.prosthesis_body_orientation_range.keys())
                for b in rand_ori_body_names:    
                    b = b + self.prosthesis_side  # Append prosthesis side to body names
                    observation_spec_body_quat.append(ObservationType.ModelBodyRot(f"quat_{b}", xml_name=b))

            # if hasattr(self, 'amputation_height_range'):
            #     rand_pos_amp_body_pos = []
            #     observation_spec_amp_body_pos = []
            #     # if 


        joint_names = []
        for j in spec.joints: 
            joint_names.append(j.name)

        # print(f"Joint names: {joint_names}")
        # print(f"Number of joints: {len(joint_names)}")

        if 'root' in joint_names: 
            joint_names.remove('root')

        observation_spec_joint_pos = []
        observation_spec_joint_vel = []

        for j in joint_names:
            observation_spec_joint_pos.append(ObservationType.JointPos(f"q_{j}", xml_name=j))
            observation_spec_joint_vel.append(ObservationType.JointVel(f"dq_{j}", xml_name=j))

        # if self.reward_type == 'TargetVelocityGoalReward': #'LocomotionReward':
        #     info_props = {}
        #     info_props["upper_body_xml_name"] = 'root'
        #     info_props["root_free_joint_xml_name"] = 'root'
        #     info_props["goal_visualization_arrow_offset"] = 0
        #     max_x_vel = 1.2 
        #     max_y_vel = 0 
        #     max_yaw_vel = 0
        #     observation_spec = [GoalRandomRootVelocity(info_props, max_x_vel, max_y_vel, max_yaw_vel), ObservationType.FreeJointPosNoXY("q_root", xml_name="root")] + observation_spec_joint_pos + observation_spec_joint_vel
        # else: 
        # # print(f"Observation spec joint pos: {observation_spec_joint_pos}")
        # # print(f"Observation spec joint vel: {observation_spec_joint_vel}")
        observation_spec = [  # ------------- JOINT POS -------------
                                ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),

                                ] + observation_spec_joint_pos + observation_spec_joint_vel
        
        if self.add_pos_ori_to_observation:
            if hasattr(self, 'prosthesis_body_position_range'):
                observation_spec += observation_spec_body_pos 
            if hasattr(self, 'prosthesis_body_position_range'):
                observation_spec += observation_spec_body_quat
        # print("Obs_spec length:", len(observation_spec))
        return observation_spec


        # # FIRST CODE 
        # joint_names = []
        # for j in spec.joints: 
        #     joint_names.append(j.name)

        # # print(f"Joint names: {joint_names}")
        # # print(f"Number of joints: {len(joint_names)}")

        # if 'root' in joint_names: 
        #     joint_names.remove('root')
        # # print(f"Joint names without root: {joint_names}")

        # observation_spec_joint_pos = []
        # observation_spec_joint_vel = []

        # for j in joint_names:
        #     observation_spec_joint_pos.append(ObservationType.JointPos(f"q_{j}", xml_name=j))
        #     observation_spec_joint_vel.append(ObservationType.JointVel(f"dq_{j}", xml_name=j))

        # # if self.reward_type == 'TargetVelocityGoalReward': #'LocomotionReward':
        # #     info_props = {}
        # #     info_props["upper_body_xml_name"] = 'root'
        # #     info_props["root_free_joint_xml_name"] = 'root'
        # #     info_props["goal_visualization_arrow_offset"] = 0
        # #     max_x_vel = 1.2 
        # #     max_y_vel = 0 
        # #     max_yaw_vel = 0
        # #     observation_spec = [GoalRandomRootVelocity(info_props, max_x_vel, max_y_vel, max_yaw_vel), ObservationType.FreeJointPosNoXY("q_root", xml_name="root")] + observation_spec_joint_pos + observation_spec_joint_vel
        # # else: 
        # # # print(f"Observation spec joint pos: {observation_spec_joint_pos}")
        # # # print(f"Observation spec joint vel: {observation_spec_joint_vel}")
        # observation_spec = [  # ------------- JOINT POS -------------
        #                     ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
        #                     # --- lower limb right ---
        #                     # ObservationType.JointPos("q_hip_flexion_r", xml_name="hip_flexion_r"),
        #                     # ObservationType.JointPos("q_hip_adduction_r", xml_name="hip_adduction_r"),
        #                     # ObservationType.JointPos("q_hip_rotation_r", xml_name="hip_rotation_r"),
        #                     # ObservationType.JointPos("q_knee_angle_r", xml_name="knee_angle_r"),
        #                     # ObservationType.JointPos("q_ankle_angle_r", xml_name="ankle_angle_r"),
        #                     # ObservationType.JointPos("q_subtalar_angle_r", xml_name="subtalar_angle_r"),
        #                     # ObservationType.JointPos("q_mtp_angle_r", xml_name="mtp_angle_r"),
        #                     # # --- lower limb left ---
        #                     # ObservationType.JointPos("q_hip_flexion_l", xml_name="hip_flexion_l"),
        #                     # ObservationType.JointPos("q_hip_adduction_l", xml_name="hip_adduction_l"),
        #                     # ObservationType.JointPos("q_hip_rotation_l", xml_name="hip_rotation_l"),
        #                     # ObservationType.JointPos("q_knee_angle_l", xml_name="knee_angle_l"),
        #                     # ObservationType.JointPos("q_ankle_angle_l", xml_name="ankle_angle_l"),
        #                     # ObservationType.JointPos("q_subtalar_angle_l", xml_name="subtalar_angle_l"),
        #                     # ObservationType.JointPos("q_mtp_angle_l", xml_name="mtp_angle_l"),
        #                     # # --- lumbar ---
        #                     # ObservationType.JointPos("q_lumbar_extension", xml_name="lumbar_extension"),
        #                     # ObservationType.JointPos("q_lumbar_bending", xml_name="lumbar_bending"),
        #                     # ObservationType.JointPos("q_lumbar_rotation", xml_name="lumbar_rotation"),
        #                     # # --- upper body right ---
        #                     # ObservationType.JointPos("q_arm_flex_r", xml_name="arm_flex_r"),
        #                     # ObservationType.JointPos("q_arm_add_r", xml_name="arm_add_r"),
        #                     # ObservationType.JointPos("q_arm_rot_r", xml_name="arm_rot_r"),
        #                     # ObservationType.JointPos("q_elbow_flex_r", xml_name="elbow_flex_r"),
        #                     # ObservationType.JointPos("q_pro_sup_r", xml_name="pro_sup_r"),
        #                     # ObservationType.JointPos("q_wrist_flex_r", xml_name="wrist_flex_r"),
        #                     # ObservationType.JointPos("q_wrist_dev_r", xml_name="wrist_dev_r"),
        #                     # # --- upper body left ---
        #                     # ObservationType.JointPos("q_arm_flex_l", xml_name="arm_flex_l"),
        #                     # ObservationType.JointPos("q_arm_add_l", xml_name="arm_add_l"),
        #                     # ObservationType.JointPos("q_arm_rot_l", xml_name="arm_rot_l"),
        #                     # ObservationType.JointPos("q_elbow_flex_l", xml_name="elbow_flex_l"),
        #                     # ObservationType.JointPos("q_pro_sup_l", xml_name="pro_sup_l"),
        #                     # ObservationType.JointPos("q_wrist_flex_l", xml_name="wrist_flex_l"),
        #                     # ObservationType.JointPos("q_wrist_dev_l", xml_name="wrist_dev_l"),

        #                     # # ------------- JOINT VEL -------------
        #                     # ObservationType.FreeJointVel("dq_root", xml_name="root"),
        #                     # # --- lower limb right ---
        #                     # ObservationType.JointVel("dq_hip_flexion_r", xml_name="hip_flexion_r"),
        #                     # ObservationType.JointVel("dq_hip_adduction_r", xml_name="hip_adduction_r"),
        #                     # ObservationType.JointVel("dq_hip_rotation_r", xml_name="hip_rotation_r"),
        #                     # ObservationType.JointVel("dq_knee_angle_r", xml_name="knee_angle_r"),
        #                     # ObservationType.JointVel("dq_ankle_angle_r", xml_name="ankle_angle_r"),
        #                     # ObservationType.JointVel("dq_subtalar_angle_r", xml_name="subtalar_angle_r"),
        #                     # ObservationType.JointVel("dq_mtp_angle_r", xml_name="mtp_angle_r"),
        #                     # # --- lower limb left ---
        #                     # ObservationType.JointVel("dq_hip_flexion_l", xml_name="hip_flexion_l"),
        #                     # ObservationType.JointVel("dq_hip_adduction_l", xml_name="hip_adduction_l"),
        #                     # ObservationType.JointVel("dq_hip_rotation_l", xml_name="hip_rotation_l"),
        #                     # ObservationType.JointVel("dq_knee_angle_l", xml_name="knee_angle_l"),
        #                     # ObservationType.JointVel("dq_ankle_angle_l", xml_name="ankle_angle_l"),
        #                     # ObservationType.JointVel("dq_subtalar_angle_l", xml_name="subtalar_angle_l"),
        #                     # ObservationType.JointVel("dq_mtp_angle_l", xml_name="mtp_angle_l"),
        #                     # # --- lumbar ---
        #                     # ObservationType.JointVel("dq_lumbar_extension", xml_name="lumbar_extension"),
        #                     # ObservationType.JointVel("dq_lumbar_bending", xml_name="lumbar_bending"),
        #                     # ObservationType.JointVel("dq_lumbar_rotation", xml_name="lumbar_rotation"),
        #                     # # --- upper body right ---
        #                     # ObservationType.JointVel("dq_arm_flex_r", xml_name="arm_flex_r"),
        #                     # ObservationType.JointVel("dq_arm_add_r", xml_name="arm_add_r"),
        #                     # ObservationType.JointVel("dq_arm_rot_r", xml_name="arm_rot_r"),
        #                     # ObservationType.JointVel("dq_elbow_flex_r", xml_name="elbow_flex_r"),
        #                     # ObservationType.JointVel("dq_pro_sup_r", xml_name="pro_sup_r"),
        #                     # ObservationType.JointVel("dq_wrist_flex_r", xml_name="wrist_flex_r"),
        #                     # ObservationType.JointVel("dq_wrist_dev_r", xml_name="wrist_dev_r"),
        #                     # # --- upper body left ---
        #                     # ObservationType.JointVel("dq_arm_flex_l", xml_name="arm_flex_l"),
        #                     # ObservationType.JointVel("dq_arm_add_l", xml_name="arm_add_l"),
        #                     # ObservationType.JointVel("dq_arm_rot_l", xml_name="arm_rot_l"),
        #                     # ObservationType.JointVel("dq_elbow_flex_l", xml_name="elbow_flex_l"),
        #                     # ObservationType.JointVel("dq_pro_sup_l", xml_name="pro_sup_l"),
        #                     # ObservationType.JointVel("dq_wrist_flex_l", xml_name="wrist_flex_l"),
        #                     # ObservationType.JointVel("dq_wrist_dev_l", xml_name="wrist_dev_l")
        #                     ] + observation_spec_joint_pos + observation_spec_joint_vel
        # # print("Obs_spec length:", len(observation_spec))
        # return observation_spec
    


    def _get_action_specification(self, spec: mujoco. MjSpec):
        """
        Getter for the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[str]: List of action space specification.
        """

        action_spec = []
        for m in spec.actuators:
            action_spec.append(m.name)

        # print(f"Action spec: {action_spec}")

        # action_spec = ["mot_lumbar_ext", "mot_lumbar_bend", "mot_lumbar_rot", "mot_shoulder_flex_r",
        #                "mot_shoulder_add_r", "mot_shoulder_rot_r", "mot_elbow_flex_r", "mot_pro_sup_r",
        #                "mot_wrist_flex_r", "mot_wrist_dev_r", "mot_shoulder_flex_l", "mot_shoulder_add_l",
        #                "mot_shoulder_rot_l", "mot_elbow_flex_l", "mot_pro_sup_l", "mot_wrist_flex_l",
        #                "mot_wrist_dev_l", "mot_hip_flexion_r", "mot_hip_adduction_r", "mot_hip_rotation_r",
        #                "mot_knee_angle_r", "mot_ankle_angle_r", "mot_subtalar_angle_r", "mot_mtp_angle_r",
        #                "mot_hip_flexion_l", "mot_hip_adduction_l", "mot_hip_rotation_l", "mot_knee_angle_l",
        #                "mot_ankle_angle_l", "mot_subtalar_angle_l", "mot_mtp_angle_l"]

        return action_spec
    

    def _calculate_cylinder_radius(self, body_mass, cylinder_inertia_y):
        
        radius = (2*cylinder_inertia_y/body_mass)

        return radius
    
    # def _calculate_org_cylinder_height(self, body_mass,radius, cylinder_inertia):

    #     height = np.sqrt((12*cylinder_inertia/body_mass) - 3*radius**2)

    #     return height
    
    def _calculate_cylinder_inertia_xorz(self, body_mass, radius, height):
        inertia = (1/12)*body_mass * (3*radius**2 + height**2)
        return inertia 
    

    def _replace_joint(self, spec, body_name, new_joint_type, axis):
        body = spec.find_body(body_name)
        body.joints[0].type = new_joint_type #mujoco.mjtJoint.mjJNT_SLIDE
        body.joints[0].axis = axis
        return spec




class MjxSkeletonMuscleProsthesisRandObs(MjxSkeletonMuscleProsthesis):
    """
    Mjx version of SkeletonMuscle with specs for adding a prosthesis.
    """

    mjx_enabled = True

    def __init__(self, timestep: float = 0.002, n_substeps: int = 5, **kwargs):
        """
        Constructor for MjxSkeletonMuscleProsthesis.
        Args:
            timestep (float): The time step for the simulation.
            n_substeps (int): The number of substeps for the simulation.
            **kwargs: Additional keyword arguments for configuration.
        Raises:
            ValueError: If required arguments are missing.
        """
        super().__init__(timestep=timestep, n_substeps=n_substeps,
                        #  model_option_conf=model_option_conf, spec=spec,
                           **kwargs)

    
    def _get_observation_specification(self, spec: mujoco.MjSpec):
        """
        Getter for the observation space specification.
        Args:
            spec (MjSpec): Specification of the environment.
        Returns:
            List[ObservationType]: List of observation space specification.
        """
        pylon_name = f"pylon_socket{self.prosthesis_side}"
        talus_name = f"talus{self.prosthesis_side}"
        rand_body_names = [pylon_name, talus_name]

        joint_names = []
        for j in spec.joints: 
            joint_names.append(j.name)

        # print(f"Joint names: {joint_names}")
        # print(f"Number of joints: {len(joint_names)}")

        if 'root' in joint_names: 
            joint_names.remove('root')
        # print(f"Joint names without root: {joint_names}")

        observation_spec_joint_pos = []
        observation_spec_joint_vel = []
        observation_spec_body_pos = []
        observation_spec_body_quat = []

        for j in joint_names:
            observation_spec_joint_pos.append(ObservationType.JointPos(f"q_{j}", xml_name=j))
            observation_spec_joint_vel.append(ObservationType.JointVel(f"dq_{j}", xml_name=j))

        for b in rand_body_names: 
            observation_spec_body_pos.append(ObservationType.ModelBodyPos(f"pos_{b}", xml_name=b))
            observation_spec_body_quat.append(ObservationType.ModelBodyRot(f"quat_{b}", xml_name=b))

        # if self.reward_type == 'TargetVelocityGoalReward': #'LocomotionReward':
        #     info_props = {}
        #     info_props["upper_body_xml_name"] = 'root'
        #     info_props["root_free_joint_xml_name"] = 'root'
        #     info_props["goal_visualization_arrow_offset"] = 0
        #     max_x_vel = 1.2 
        #     max_y_vel = 0 
        #     max_yaw_vel = 0
        #     observation_spec = [GoalRandomRootVelocity(info_props, max_x_vel, max_y_vel, max_yaw_vel), ObservationType.FreeJointPosNoXY("q_root", xml_name="root")] + observation_spec_joint_pos + observation_spec_joint_vel
        # else: 
        # # print(f"Observation spec joint pos: {observation_spec_joint_pos}")
        # # print(f"Observation spec joint vel: {observation_spec_joint_vel}")
        observation_spec = [  # ------------- JOINT POS -------------
                            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),

                            ] + observation_spec_joint_pos + observation_spec_joint_vel + observation_spec_body_pos + observation_spec_body_quat
        # print("Obs_spec length:", len(observation_spec))
        return observation_spec
    


# class MjxSkeletonMuscleProsthesisRandPylonObs(MjxSkeletonMuscleProsthesis):
#     """
#     Mjx version of SkeletonMuscle with specs for adding a prosthesis.
#     """

#     mjx_enabled = True

#     def __init__(self, timestep: float = 0.002, n_substeps: int = 5, **kwargs):
#         """
#         Constructor for MjxSkeletonMuscleProsthesis.
#         Args:
#             timestep (float): The time step for the simulation.
#             n_substeps (int): The number of substeps for the simulation.
#             **kwargs: Additional keyword arguments for configuration.
#         Raises:
#             ValueError: If required arguments are missing.
#         """
#         super().__init__(timestep=timestep, n_substeps=n_substeps,
#                         #  model_option_conf=model_option_conf, spec=spec,
#                            **kwargs)

    
#     def _get_observation_specification(self, spec: mujoco.MjSpec):
#         """
#         Getter for the observation space specification.
#         Args:
#             spec (MjSpec): Specification of the environment.
#         Returns:
#             List[ObservationType]: List of observation space specification.
#         """
#         pylon_name = f"pylon_socket{self.prosthesis_side}"
#         # talus_name = f"talus{self.prosthesis_side}"
#         rand_body_names = [pylon_name]#, talus_name]

#         joint_names = []
#         for j in spec.joints: 
#             joint_names.append(j.name)

#         # print(f"Joint names: {joint_names}")
#         # print(f"Number of joints: {len(joint_names)}")

#         if 'root' in joint_names: 
#             joint_names.remove('root')
#         # print(f"Joint names without root: {joint_names}")

#         observation_spec_joint_pos = []
#         observation_spec_joint_vel = []
#         observation_spec_body_pos = []
#         observation_spec_body_quat = []

#         for j in joint_names:
#             observation_spec_joint_pos.append(ObservationType.JointPos(f"q_{j}", xml_name=j))
#             observation_spec_joint_vel.append(ObservationType.JointVel(f"dq_{j}", xml_name=j))

#         for b in rand_body_names: 
#             observation_spec_body_pos.append(ObservationType.ModelBodyPos(f"pos_{b}", xml_name=b))
#             observation_spec_body_quat.append(ObservationType.ModelBodyRot(f"quat_{b}", xml_name=b))

#         observation_spec = [  # ------------- JOINT POS -------------
#                             ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),

#                             ] + observation_spec_joint_pos + observation_spec_joint_vel + observation_spec_body_pos + observation_spec_body_quat
#         # print("Obs_spec length:", len(observation_spec))
#         return observation_spec
    

    
# class MjxSkeletonMuscleProsthesisRandTalusObs(MjxSkeletonMuscleProsthesis):
#     """
#     Mjx version of SkeletonMuscle with specs for adding a prosthesis.
#     """

#     mjx_enabled = True

#     def __init__(self, timestep: float = 0.002, n_substeps: int = 5, **kwargs):
#         """
#         Constructor for MjxSkeletonMuscleProsthesis.
#         Args:
#             timestep (float): The time step for the simulation.
#             n_substeps (int): The number of substeps for the simulation.
#             **kwargs: Additional keyword arguments for configuration.
#         Raises:
#             ValueError: If required arguments are missing.
#         """
#         super().__init__(timestep=timestep, n_substeps=n_substeps,
#                         #  model_option_conf=model_option_conf, spec=spec,
#                            **kwargs)

    
#     def _get_observation_specification(self, spec: mujoco.MjSpec):
#         """
#         Getter for the observation space specification.
#         Args:
#             spec (MjSpec): Specification of the environment.
#         Returns:
#             List[ObservationType]: List of observation space specification.
#         """
#         # pylon_name = f"pylon_socket{self.prosthesis_side}"
#         talus_name = f"talus{self.prosthesis_side}"
#         rand_body_names = [talus_name]

#         joint_names = []
#         for j in spec.joints: 
#             joint_names.append(j.name)

#         # print(f"Joint names: {joint_names}")
#         # print(f"Number of joints: {len(joint_names)}")

#         if 'root' in joint_names: 
#             joint_names.remove('root')
#         # print(f"Joint names without root: {joint_names}")

#         observation_spec_joint_pos = []
#         observation_spec_joint_vel = []
#         observation_spec_body_pos = []
#         observation_spec_body_quat = []

#         for j in joint_names:
#             observation_spec_joint_pos.append(ObservationType.JointPos(f"q_{j}", xml_name=j))
#             observation_spec_joint_vel.append(ObservationType.JointVel(f"dq_{j}", xml_name=j))

#         for b in rand_body_names: 
#             observation_spec_body_pos.append(ObservationType.ModelBodyPos(f"pos_{b}", xml_name=b))
#             observation_spec_body_quat.append(ObservationType.ModelBodyRot(f"quat_{b}", xml_name=b))

#         observation_spec = [  # ------------- JOINT POS -------------
#                             ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),

#                             ] + observation_spec_joint_pos + observation_spec_joint_vel + observation_spec_body_pos + observation_spec_body_quat
#         # print("Obs_spec length:", len(observation_spec))
#         return observation_spec