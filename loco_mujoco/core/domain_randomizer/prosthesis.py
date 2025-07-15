import jax 
import jax.numpy as jnp
import numpy as np
from loco_mujoco.core.domain_randomizer import DomainRandomizer
from flax import struct

from loco_mujoco.core.utils.backend import assert_backend_is_supported
from mujoco.mjx import Data, Model
from mujoco import MjData, MjModel
import mujoco
import mujoco.mjx as mjx

from typing import Any, Union, Tuple
from types import ModuleType


from scipy.spatial.transform import Rotation as R
from jax.scipy.spatial.transform import Rotation as jaxR



## ! Update_observation and update_action only take input and pass output without any changes! (Can be used to add noise in the future) 
@struct.dataclass
class ProsthesisRandomizerState:
    """
    Represents the state of the default randomizer.

    """
    prosthesis_joint_stiffness: Union[np.ndarray, jax.Array]
    prosthesis_dof_damping: Union[np.ndarray, jax.Array]
    prosthesis_body_position: Union[np.ndarray, jax.Array]
    prosthesis_body_orientation: Union[np.ndarray, jax.Array]
    prosthesis_socket_joint_value: Union[np.ndarray, jax.Array]


    # geom_friction: Union[np.ndarray, jax.Array]
    # geom_stiffness: Union[np.ndarray, jax.Array]
    # geom_damping: Union[np.ndarray, jax.Array]
    # base_mass_to_add: float
    # com_displacement: Union[np.ndarray, jax.Array]
    # link_mass_multipliers: Union[np.ndarray, jax.Array]
    # joint_friction_loss: Union[np.ndarray, jax.Array]
    # dof_damping: Union[np.ndarray, jax.Array]
    # joint_armature: Union[np.ndarray, jax.Array]



class ProsthesisRandomizer(DomainRandomizer):
    """
    Randomize prosthesis parameters in environment 

    Gives options to randomize 
    - Stiffness and damping of amputated limb prosthesis
    - #########Mass of amputated limb prosthesis? Intertia? ######
    - Orientation of amputated limb prosthesis
    - Position of amputated limb prosthesis
     
    
    """
    def __init__(self, env, **kwargs): 
        self._init_prosthesis_joint_stiffness = None 
        self._init_prosthesis_dof_damping = None
        self._init_prosthesis_body_position = None
        self._init_prosthesis_body_orientation = None
        self._init_prosthesis_socket_joint_value = None
        self._init_prosthesis_socket_joint_springref = None
        self.init_talus_pos = None
        super().__init__(env, **kwargs)


    # def init_state(self,
    #                env: Any,
    #                key: Any,
    #                model: Union[MjModel, Model],
    #                data: Union[MjData, Data],
    #                backend: ModuleType) -> ProsthesisRandomizerState:
    #     """
    #     Initialize the randomizer state.

    #     Args:
    #         env (Any): The environment instance.
    #         key (Any): Random seed key.
    #         model (Union[MjModel, Model]): The simulation model.
    #         data (Union[MjData, Data]): The simulation data.
    #         backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

    #     Returns:
    #         DefaultRandomizerState: The initialized randomizer state.

    #     """
    #     self._body_indices = {}
    #     self._joint_indices = {}
    #     self._dof_indices = {}

    #     # get stiffness and damping for randomization_joints 
    #     prosthesis_side = self.rand_conf["prosthesis_side"]
    #     joint_names = self.rand_conf["randomization_joints"]

    #     assert prosthesis_side in ["left_side", "right_side"], f"Invalid prosthesis side: {prosthesis_side}. Expected 'left' or 'right'."

    #     if prosthesis_side == "left_side":
    #         joint_names = [name + '_l' for name in joint_names]
    #     elif prosthesis_side == "right_side":
    #         joint_names = [name + '_r' for name in joint_names]

    #     self.joint_names_side = joint_names

    #     # get joint stiffness and damping
    #     # Get mujoco joint indices for the prosthesis joints
    #     for joint_name in joint_names:
    #         # if joint_name in model.joint_names:
    #         idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    #         if idx != -1:
    #             self._joint_indices[joint_name] = idx
        
    #     # Initialize dof_indices
    #     for joint_name, idx in self._joint_indices.items():
    #         dof_adr = model.jnt_dofadr[idx]
    #         self._dof_indices[joint_name] = dof_adr

    #     # For bodies add the correct side and then get positon and orientation 
    #     if prosthesis_side == "left_side":
    #         body_names = [name + '_l' for name in self.rand_conf["randomization_bodies"]]
    #     elif prosthesis_side == "right_side":
    #         body_names = [name + '_r' for name in self.rand_conf["randomization_bodies"]]

    #     for body_name in body_names:
    #         idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    #         self._body_indices[body_name] = idx

    #     return ProsthesisRandomizerState(prosthesis_joint_stiffness=backend.array([10.0] * len(self._joint_indices.values())),
    #                                   prosthesis_dof_damping=backend.array([1.0] * len(self._dof_indices.values())), 
    #                                     # prosthesis_body_position=backend.array([[0.0, 0.0, 0.0]] * len(self._body_indices.values())),
    #                                     # prosthesis_body_orientation=backend.array([[1.0, 0.0, 0.0, 0.0]] * len(self._body_indices.values())
    #                                   )
    
    # def init_state(self,
    #                env: Any,
    #                key: Any,
    #                model: Union[MjModel, Model],
    #                data: Union[MjData, Data],
    #                backend: ModuleType) -> ProsthesisRandomizerState:
    #     """
    #     Initialize the randomizer state.

    #     Args:
    #         env (Any): The environment instance.
    #         key (Any): Random seed key.
    #         model (Union[MjModel, Model]): The simulation model.
    #         data (Union[MjData, Data]): The simulation data.
    #         backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

    #     Returns:
    #         DefaultRandomizerState: The initialized randomizer state.

    #     """
    #     self._body_indices = {}
    #     self._joint_indices = {}
    #     self._dof_indices = {}

    #     # get stiffness and damping for randomization_joints 
    #     prosthesis_side = self.rand_conf["prosthesis_side"]
    #     joint_names = self.rand_conf["randomization_joints"]

    #     assert prosthesis_side in ["left_side", "right_side"], f"Invalid prosthesis side: {prosthesis_side}. Expected 'left' or 'right'."

    #     if prosthesis_side == "left_side":
    #         joint_names = [name + '_l' for name in joint_names]
    #     elif prosthesis_side == "right_side":
    #         joint_names = [name + '_r' for name in joint_names]

    #     self.joint_names_side = joint_names

    #     # get joint stiffness and damping
    #     # Get mujoco joint indices for the prosthesis joints
    #     for joint_name in joint_names:
    #         # if joint_name in model.joint_names:
    #         idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    #         if idx != -1:
    #             self._joint_indices[joint_name] = idx
        
    #     # Initialize dof_indices
    #     for joint_name, idx in self._joint_indices.items():
    #         dof_adr = model.jnt_dofadr[idx]
    #         self._dof_indices[joint_name] = dof_adr

    #     # Get stiffness and damping for those joints
    #     prosthesis_joint_stiffness = backend.array([model.jnt_stiffness[idx] for idx in self._joint_indices.values()])
    #     prosthesis_dof_damping = backend.array([model.dof_damping[idx] for idx in self._dof_indices.values()])


    #     # For bodies add the correct side and then get positon and orientation 
    #     if prosthesis_side == "left_side":
    #         body_names = [name + '_l' for name in self.rand_conf["randomization_bodies"]]
    #     elif prosthesis_side == "right_side":
    #         body_names = [name + '_r' for name in self.rand_conf["randomization_bodies"]]

    #     self.body_names_side = body_names
    #     # Get body indices for the prosthesis bodies
    #     # self.body_indices = []
    #     for body_name in body_names:
    #         idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    #         self._body_indices[body_name] = idx
    #         # else:
    #         #     raise ValueError(f"Body '{body_name}' not found in model.body_names.")
    #     # # Get position and orientation for those bodies
    #     prosthesis_body_position = backend.array([model.body_pos[idx] for idx in self._body_indices.values()])
    #     prosthesis_body_orientation = backend.array([model.body_quat[idx] for idx in self._body_indices.values()])

    #     # assert_backend_is_supported(backend)

            
    #     return ProsthesisRandomizerState(prosthesis_joint_stiffness=prosthesis_joint_stiffness,
    #                                   prosthesis_dof_damping=prosthesis_dof_damping,
    #                                   prosthesis_body_position=prosthesis_body_position,
    #                                   prosthesis_body_orientation=prosthesis_body_orientation,


    #                                 #   base_mass_to_add=0.0,
    #                                 #   com_displacement=backend.array([0.0, 0.0, 0.0]),
    #                                 #   link_mass_multipliers=backend.array([1.0] * (model.nbody-1)), #exclude worldbody
    #                                 #   joint_friction_loss=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
    #                                 #   dof_damping=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
    #                                 #   joint_armature=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
    #                                   )

    def init_state(self,
                   env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType) -> ProsthesisRandomizerState:
        """
        Initialize the randomizer state.

        Args:
            env (Any): The environment instance.
            key (Any): Random seed key.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            DefaultRandomizerState: The initialized randomizer state.

        """
        self._body_pos_indices = {}
        self._body_quat_indices = {}
        self._joint_indices = {}
        self._dof_indices = {}
        self._socket_joint_indices = {}
        
        # self._feet_geom_solref_indices = {}

        # get stiffness and damping for randomization_joints 
        prosthesis_side = self.rand_conf["prosthesis_side"]
        if "randomization_joint_names" in self.rand_conf:
            joint_names = self.rand_conf["randomization_joint_names"]
        elif "prosthesis_joint_stiffness_range" in self.rand_conf:
            self.stiffness_dict = self.rand_conf["prosthesis_joint_stiffness_range"]
            joint_names = list(self.stiffness_dict.keys())
        if "randomization_dof_names" in self.rand_conf:
            dof_names = self.rand_conf["randomization_dof_names"]
        elif "prosthesis_dof_damping_range" in self.rand_conf:
            self.damping_dict = self.rand_conf["prosthesis_dof_damping_range"]
            dof_names = list(self.damping_dict.keys())
        if "socket_joint_range" in self.rand_conf: 
            self.socket_joint_range_dict = self.rand_conf["socket_joint_range"]
            socket_joint_names = list(self.socket_joint_range_dict)
        if "randomization_body_position_names" in self.rand_conf:
            body_pos_names = self.rand_conf["randomization_body_position_names"]
        elif "prosthesis_body_position_range" in self.rand_conf:
            # New format: check if it's a nested dict
            position_range_config = self.rand_conf["prosthesis_body_position_range"]
            if isinstance(position_range_config, dict) and \
               all(isinstance(v, dict) for v in position_range_config.values()):
                self.is_position_range_nested = True
                body_pos_names = list(position_range_config.keys())
                self.prosthesis_body_position_range_dict = position_range_config # Store for later use in sampling
            else:
                # This would be the old flat format but defined under "prosthesis_body_position_range" key
                # It's ambiguous without explicit names.
                print("Warning: 'prosthesis_body_position_range' is not nested and 'randomization_body_position_names' is not found for position randomization. No bodies will be randomized for position.")
                body_pos_names = [] # Default to empty if names not specified explicitly
        if "randomization_body_orientation_names" in self.rand_conf:
            body_quat_names = self.rand_conf["randomization_body_orientation_names"]
        elif "prosthesis_body_orientation_range" in self.rand_conf: 
            quat_range_config = self.rand_conf["prosthesis_body_orientation_range"]
            if isinstance(quat_range_config, dict) and all(isinstance(v, dict) for v in quat_range_config.values()):
                self.is_orientation_range_nested = True
                body_quat_names = list(quat_range_config.keys())
                self.prosthesis_body_quat_range_dict = quat_range_config # Store for later use in sampling
            else:
                # This would be the old flat format but defined under "prosthesis_body_position_range" key
                # It's ambiguous without explicit names.
                print("Warning: 'prosthesis_body_orientation_range' is not nested and 'randomization_body_orientation_names' is not found for orientation randomization. No bodies will be randomized for orientation.")
                body_quat_names = [] # Default to empty if names not specified explicitly
            
        # foot_geom_names = self.rand_conf["randomization_foot_geom_solref_names"]
        

        assert prosthesis_side in ["left_side", "right_side"], f"Invalid prosthesis side: {prosthesis_side}. Expected 'left' or 'right'."


        if prosthesis_side == "left_side":
            self.prosthesis_side_str = '_l'
        elif prosthesis_side == "right_side":
            self.prosthesis_side_str = '_r'

        joint_names = [name + self.prosthesis_side_str for name in joint_names]
        dof_names = [name + self.prosthesis_side_str for name in dof_names]
        body_pos_names = [name + self.prosthesis_side_str for name in body_pos_names]
        body_quat_names = [name + self.prosthesis_side_str for name in body_quat_names]
        socket_joint_names = [name + self.prosthesis_side_str for name in socket_joint_names]
        # foot_geom_names = [name + '_l' for name in foot_geom_names] + [name + '_r' for name in foot_geom_names]


        # Get mujoco joint indices for the joints
        valid_joint_names = []
        valid_stiffness_dict = {}
        for joint_name in joint_names:
            # if joint_name in model.joint_names:
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if idx != -1:
                self._joint_indices[joint_name] = idx
                valid_joint_names.append(joint_name)
                joint_name = joint_name.replace(self.prosthesis_side_str,'')
                if joint_name in self.stiffness_dict:
                    valid_stiffness_dict[joint_name] = self.stiffness_dict[joint_name]
        joint_names = valid_joint_names
        self.stiffness_dict = valid_stiffness_dict

        # Get stiffness and damping for those joints
        prosthesis_joint_stiffness = {}
        for joint_name in joint_names:
            prosthesis_joint_stiffness[joint_name] = backend.array([model.jnt_stiffness[self._joint_indices[joint_name]]])

        # prosthesis_joint_stiffness = backend.array([model.jnt_stiffness[idx] for idx in self._joint_indices.values()])
        
        # Get mujoco dof indices for the joints 
        valid_dof_names = []
        valid_damping_dict = {}
        for dof_name in dof_names:
            # if dof_name in model.dof_names:
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, dof_name)
            if idx != -1:
                idx =  model.jnt_dofadr[idx]
                self._dof_indices[dof_name] = idx
                valid_dof_names.append(dof_name)
                dof_name = dof_name.replace(self.prosthesis_side_str,'')
                if dof_name in self.damping_dict:
                    valid_damping_dict[dof_name] = self.damping_dict[dof_name]
        dof_names = valid_dof_names
        self.damping_dict = valid_damping_dict

        # Get stiffness and damping for those dofs
        prosthesis_dof_damping = {}
        for dof_name in dof_names:
            prosthesis_dof_damping[dof_name] = backend.array([model.dof_damping[self._dof_indices[dof_name]]])
        #prosthesis_dof_damping = backend.array([model.dof_damping[idx] for idx in self._dof_indices.values()])


        # # Get body indices for the prosthesis bodies
        # # self.body_indices = []
        # for body_pos_name in body_pos_names:
        #     idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_pos_name)
        #     self._body_pos_indices[body_pos_name] = idx

        self.randomized_pos_body_names = [] # Store names of bodies that will *actually* be randomized for position
        for body_name in body_pos_names:
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if idx == -1:
                print(f"Warning: Body '{body_name}' (for position) not found in model. Skipping randomization for it.")
            else:
                self._body_pos_indices[body_name] = idx
                self.randomized_pos_body_names.append(body_name)

        self.randomized_quat_body_names = []
        for body_quat_name in body_quat_names:
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_quat_name)
            if idx == -1: 
                print(f"Warning: Body '{body_quat_name}' (for orientation) not found in model. Skipping randomization for it.")
            else: 
                self._body_quat_indices[body_quat_name] = idx
                self.randomized_quat_body_names.append(body_quat_name)
            
        # # Get position and orientation for those bodies
        prosthesis_body_position ={}
        prosthesis_body_orientation ={}

        # initial_prosthesis_body_position_list = []
        # for body_name in self.randomized_pos_body_names:
        #     idx = self._body_pos_indices[body_name] # We know it exists from the loop above
        #     initial_prosthesis_body_position_list.append(model.body_pos[idx])
        # #self._init_prosthesis_body_position = self.backend.array(initial_prosthesis_body_position_list)

        for body_name in body_pos_names:
            prosthesis_body_position[body_name] = backend.array([model.body_pos[self._body_pos_indices[body_name]]])
        for body_name in body_quat_names:
            prosthesis_body_orientation[body_name] = backend.array([model.body_quat[self._body_quat_indices[body_name]]])
        
            
        # prosthesis_body_position = backend.array([model.body_pos[idx] for idx in self._body_pos_indices.values()])
        # prosthesis_body_orientation = backend.array([model.body_quat[idx] for idx in self._body_quat_indices.values()])


        # Get mujoco joint indices for the socket joints
        valid_socket_joint_names = []
        valid_socket_joint_range_dict = {}
        for joint_name in socket_joint_names:
            # if joint_name in model.joint_names:
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if idx != -1:
                idx =  model.jnt_qposadr[idx]
                self._socket_joint_indices[joint_name] = idx
                valid_socket_joint_names.append(joint_name)
                joint_name = joint_name.replace(self.prosthesis_side_str,'')
                if joint_name in self.socket_joint_range_dict:
                    valid_socket_joint_range_dict[joint_name] = self.socket_joint_range_dict[joint_name]
        socket_joint_names = valid_socket_joint_names
        self.socket_joint_range_dict = valid_socket_joint_range_dict

        
        prosthesis_socket_joint_value = {}
        for joint_name in socket_joint_names:
            prosthesis_socket_joint_value[joint_name] = backend.array([model.qpos0[self._socket_joint_indices[joint_name]]])
            # prosthesis_socket_joint_value[joint_name] = backend.array([data.qpos[self._socket_joint_indices[joint_name]]])


        ###### Indices for tibia, socket, talus for initialization and parameter adaption for different socket_ty 
        if 'socket_ty'+self.prosthesis_side_str in self._socket_joint_indices:
            tibia_name = 'tibia' + self.prosthesis_side_str
            socket_name = 'pylon_socket' + self.prosthesis_side_str
            talus_name = 'talus' + self.prosthesis_side_str
            socket_joints = ['socket_flexion', 'socket_adduction', 'socket_rotation', 'socket_tx', 'socket_ty', 'socket_tz']
            socket_joints = [j + self.prosthesis_side_str for j in socket_joints]
            socket_ty_name = 'socket_ty' + self.prosthesis_side_str

            # Iterate over socket joints and find them in model and get their pos
            # socket_joint_indices = {}
            self.joint_pos = {}
            self.valid_socket_joints = {}
            for j in socket_joints: 
                idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                if idx != -1: 
                    self.joint_pos[j] = model.jnt_pos[idx]
                    # socket_joint_indices[j] =idx
                    self.valid_socket_joints[j] = j

            # Get indices
            self._tibia_idx= mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tibia_name)
            self._socket_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, socket_name)
            self._talus_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, talus_name)

            # Socket ty index
            socket_ty_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, socket_ty_name)
            self._socket_ty_idx =  model.jnt_qposadr[socket_ty_idx]

        # assert_backend_is_supported(backend)

        # # Foot geom indices
        # for foot_geom_name in foot_geom_names:
        #     idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, foot_geom_name)
        #     if idx != -1:
        #         if foot_geom_name.endswith("_l") and prosthesis_side == "left_side":
        #             self._feet_geom_solref_indices["prosthesis_side"] = model.geom_solref[idx]
        #         elif foot_geom_name.endswith("_r") and prosthesis_side == "right_side":
        #             self._feet_geom_solref_indices["prosthesis_side"] = model.geom_solref[idx]
        #         else:
        #             self._feet_geom_solref_indices["other_side"] = model.geom_solref[idx]
        #         # self._feet_geom_solref_indices[foot_geom_name] = model.geom_solref[idx]
        
        # feet_geom_solref = backend.array([model.geom_solref[idx] for idx in self._feet_geom_solref_indices.values()])

        # Map feet geom names to indices, and match side for solref assignment
        # Example: feet_geom_solref_range: {'prosthesis_side': [0.01, 0.1], 'other_side': [0.01, 0.1]}
        # randomization_feet_geom_names: ['foot_box']
        # foot_geom_names will be ['foot_box_l', 'foot_box_r']
        # We want to know which index is prosthesis_side and which is other_side

        # # Build a mapping from geom name to side
        # self._feet_geom_side_map = {}
        # for name in foot_geom_names:
        #     if prosthesis_side == "left_side" and name.endswith("_l"):
        #         self._feet_geom_side_map[name] = "prosthesis_side"
        #     elif prosthesis_side == "right_side" and name.endswith("_r"):
        #         self._feet_geom_side_map[name] = "prosthesis_side"
        #     else:
        #         self._feet_geom_side_map[name] = "other_side"

            
        return ProsthesisRandomizerState(prosthesis_joint_stiffness=prosthesis_joint_stiffness,
                                      prosthesis_dof_damping=prosthesis_dof_damping,
                                      prosthesis_body_position=prosthesis_body_position,
                                      prosthesis_body_orientation=prosthesis_body_orientation,
                                      prosthesis_socket_joint_value= prosthesis_socket_joint_value
                                    #   feet_geom_solref= feet_geom_solref


                                    #   base_mass_to_add=0.0,
                                    #   com_displacement=backend.array([0.0, 0.0, 0.0]),
                                    #   link_mass_multipliers=backend.array([1.0] * (model.nbody-1)), #exclude worldbody
                                    #   joint_friction_loss=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
                                    #   dof_damping=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
                                    #   joint_armature=backend.array([0.0] * (model.nv-6)), #exclude freejoint 6 dofs
                                      )



    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the randomizer, applying domain randomization.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.

        """
        assert_backend_is_supported(backend)
        domain_randomizer_state = carry.domain_randomizer_state

        # jax.debug.print("carry.domain_randomizer_state RESET")

        # Working print 
        # jax.debug.print("domain_randomizer_state org reset: {domain_randomizer_state}", domain_randomizer_state=domain_randomizer_state)

        if backend == np and self._init_prosthesis_joint_stiffness is None:
            self._init_prosthesis_joint_stiffness = model.jnt_stiffness.copy()
            self._init_prosthesis_dof_damping = model.dof_damping.copy()
            self._init_prosthesis_body_position = model.body_pos.copy()
            self._init_prosthesis_body_orientation = model.body_quat.copy()
            self._init_prosthesis_socket_joint_value = model.qpos0.copy()
            self._init_prosthesis_socket_joint_springref = model.qpos_spring.copy()
            # self._init_feet_geom_solref = model.geom_solref.copy()
        # elif backend == jnp:
        #     self._init_prosthesis_joint_stiffness = jnp.array(model.jnt_stiffness)
        #     self._init_prosthesis_dof_damping = jnp.array(model.dof_damping)
        #     self._init_prosthesis_body_position = jnp.array(model.body_pos)
        #     self._init_prosthesis_body_orientation = jnp.array(model.body_quat)


        prosthesis_joint_stiffness, carry = self._sample_joint_stiffness(model, carry, backend)
        prosthesis_dof_damping, carry = self._sample_dof_damping(model, carry, backend)
        prosthesis_body_position, carry = self._sample_geom_position(model, carry, backend)
        prosthesis_body_orientation, carry = self._sample_joint_orientation(model, carry, backend)
        prosthesis_socket_joint_value, carry = self._sample_socket_joint_value(model, data, carry, backend)
        # jax.debug.print('prosthesis_socket_joint_value: {value}', value = prosthesis_socket_joint_value)
        # feet_geom_solref, carry = self._sample_feet_geom_solref(model, carry, backend)


        # print("prosthesis_joint_stiffness: ", prosthesis_joint_stiffness)
        # print("prosthesis_dof_damping: ", prosthesis_dof_damping)

        # jax.debug.print("prosthesis_joint_stiffness: {prosthesis_joint_stiffness}", prosthesis_joint_stiffness=prosthesis_joint_stiffness)
        # jax.debug.print("prosthesis_dof_damping: {prosthesis_dof_damping}", prosthesis_dof_damping=prosthesis_dof_damping)
        # jax.debug.print("prosthesis_body_position: {prosthesis_body_position}", prosthesis_body_position=prosthesis_body_position)
        # jax.debug.print("prosthesis_body_orientation: {prosthesis_body_orientation}", prosthesis_body_orientation=prosthesis_body_orientation)

        carry = carry.replace(domain_randomizer_state=domain_randomizer_state.replace(
                prosthesis_joint_stiffness=prosthesis_joint_stiffness,
                prosthesis_dof_damping=prosthesis_dof_damping,
                prosthesis_body_position=prosthesis_body_position,
                prosthesis_body_orientation=prosthesis_body_orientation,
                prosthesis_socket_joint_value= prosthesis_socket_joint_value
                # feet_geom_solref = feet_geom_solref
                ))
        
        # jax.debug.print("carry.domain_randomizer_state end of reset: {carry}", carry=carry)

        return data, carry


    def update(self,
               env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType) -> Tuple[Union[MjModel, Model], Union[MjData, Data], Any]:
        """
        Update the randomizer by applying the state changes to the model.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjModel, Model], Union[MjData, Data], Any]: The updated simulation model, data, and carry.

        """

        assert_backend_is_supported(backend)

        domrand_state = carry.domain_randomizer_state
        # jax.debug.print("domrand_state in update: {domrand_state}", domrand_state=domrand_state)
        # print("domrand_state in update: ", domrand_state)
        if self.rand_conf["randomize_prosthesis_body_orientation"]:
            # print('model.body_pos before: ', model.body_pos)
            if backend == jnp: 
                body_names = list(domrand_state.prosthesis_body_orientation.keys())
                body_values = list(domrand_state.prosthesis_body_orientation.values())
                body_names = [name + self.prosthesis_side_str for name in body_names]
                body_indices = [self._body_quat_indices[name] for name in body_names]

                body_indices = jnp.array(body_indices)
                body_values = jnp.array(body_values)

                body_quat = model.body_quat.at[jnp.array(body_indices)].set(body_values)
            elif backend == np: 
                body_quat = self._init_prosthesis_body_orientation.copy()
                for body_name, value in domrand_state.prosthesis_body_orientation.items():
                    body_name = body_name + self.prosthesis_side_str
                    idx = self._body_quat_indices[body_name]
                    body_quat[idx] = value

            model = self._set_attribute_in_model(model, "body_quat", body_quat, backend)
        
        # dof_indices = list(self._dof_indices.values())
        # jnt_indices = list(self._joint_indices.values())
        # body_pos_indices = list(self._body_pos_indices.values())
        # body_quat_indices = list(self._body_quat_indices.values())
        # feet_geom_solref_indices = list(self._feet_geom_solref_indices.values())


        # if backend == jnp:
        #     # # Use JAX for randomization
        #     # # Unpack joint stiffness dictionary into indices and values
        #     # joint_names = list(domrand_state.prosthesis_joint_stiffness.keys())
        #     # joint_values = list(domrand_state.prosthesis_joint_stiffness.values())
        #     # joint_indices = [self._joint_indices[name] for name in joint_names]

        #     # # Convert to jnp arrays
        #     # joint_indices = jnp.array(joint_indices)
        #     # joint_values = jnp.array(joint_values)

        #     # # Apply to model
        #     # jnt_stiffness = model.jnt_stiffness.at[joint_indices].set(joint_values)
        #     # # jnt_stiffness = model.jnt_stiffness.at[jnp.array(jnt_indices)].set(domrand_state.prosthesis_joint_stiffness)
        #     # dof_damping = model.dof_damping.at[jnp.array(dof_indices)].set(domrand_state.prosthesis_dof_damping)
            
        #     # body_pos = model.body_pos.at[jnp.array(body_pos_indices)].set(domrand_state.prosthesis_body_position) #model.body_pos[jnp.array(body_pos_indices)])# + domrand_state.prosthesis_body_position)
        #     body_quat = model.body_quat.at[jnp.array(body_quat_indices)].set(domrand_state.prosthesis_body_orientation) #model.body_quat[jnp.array(body_quat_indices)])# + domrand_state.prosthesis_body_orientation)
        #     # feet_geom_solref = model.geom_solref.at[jnp.array(feet_geom_solref_indices)].set(domrand_state.feet_geom_solref)
        # else:
        #     # dof_damping = self._init_prosthesis_dof_damping.copy()
        #     # dof_damping[dof_indices] = domrand_state.prosthesis_dof_damping
        #     # jnt_stiffness = self._init_prosthesis_joint_stiffness.copy()
        #     # for joint_name, value in domrand_state.prosthesis_joint_stiffness.items():
        #     #     idx = self._joint_indices[joint_name]
        #     #     jnt_stiffness[idx] = value
        #     # jnt_stiffness[jnt_indices] = domrand_state.prosthesis_joint_stiffness
            
        #     # body_pos = self._init_prosthesis_body_position.copy()
        #     # body_pos[body_pos_indices] = domrand_state.prosthesis_body_position
        #     body_quat = self._init_prosthesis_body_orientation.copy()
        #     body_quat[body_quat_indices] = domrand_state.prosthesis_body_orientation
        #     # feet_geom_solref = self._init_feet_geom_solref.copy()
        #     # feet_geom_solref[feet_geom_solref_indices] = domrand_state.feet_geom_solref


        # jax.debug.print("model.jnt_stiffness before: {jnt_stiffness}", jnt_stiffness=model.jnt_stiffness)
        # jax.debug.print("model.dof_damping before: {dof_damping}", dof_damping=model.dof_damping)
        # jax.debug.print("model.body_pos before: {body_pos}", body_pos=model.body_pos)
        # # jax.debug.print("model.body_quat before: {body_quat}", body_quat=model.body_quat)

        if self.rand_conf["randomize_prosthesis_dof_damping"]:
            if backend == jnp: 
                dof_names = list(domrand_state.prosthesis_dof_damping.keys())
                dof_values = list(domrand_state.prosthesis_dof_damping.values())
                dof_indices = [self._dof_indices[name] for name in dof_names]
                # Convert to jnp arrays
                dof_indices = jnp.array(dof_indices)
                dof_values = jnp.array(dof_values)

                # Apply to model
                dof_damping = model.dof_damping.at[dof_indices].set(dof_values)
            else: 
                dof_damping = self._init_prosthesis_dof_damping.copy()
                for dof_name, value in domrand_state.prosthesis_dof_damping.items():
                    idx = self._dof_indices[dof_name]
                    dof_damping[idx] = value
            
            model = self._set_attribute_in_model(model, "dof_damping", dof_damping, backend)
        # print('model.damping', model.dof_damping)
        
        if self.rand_conf["randomize_prosthesis_joint_stiffness"]:

            if backend == jnp:
                # Use JAX for randomization
                # Unpack joint stiffness dictionary into indices and values
                joint_names = list(domrand_state.prosthesis_joint_stiffness.keys())
                joint_values = list(domrand_state.prosthesis_joint_stiffness.values())
                joint_indices = [self._joint_indices[name] for name in joint_names]

                # Convert to jnp arrays
                joint_indices = jnp.array(joint_indices)
                joint_values = jnp.array(joint_values)

                # Apply to model
                jnt_stiffness = model.jnt_stiffness.at[joint_indices].set(joint_values)
            else: 
                jnt_stiffness = self._init_prosthesis_joint_stiffness.copy()
                for joint_name, value in domrand_state.prosthesis_joint_stiffness.items():
                    idx = self._joint_indices[joint_name]
                    jnt_stiffness[idx] = value

            model = self._set_attribute_in_model(model, "jnt_stiffness", jnt_stiffness, backend)
        # print('model.stiffness', model.jnt_stiffness)

        # # if self.rand_conf["randomize_prosthesis_body_position"]:
        # #     model = self._set_attribute_in_model(model, "body_pos", body_pos, backend)
        # if self.rand_conf["randomize_prosthesis_body_orientation"]:
        #     model = self._set_attribute_in_model(model, "body_quat", body_quat, backend)
        # # if self.rand_conf["randomize_prosthesis_foot_geom_solref"]:
        # #     model = self._set_attribute_in_model(model, "geom_solref", feet_geom_solref, backend)


        if self.rand_conf["randomize_prosthesis_socket_joint"]:
            # jax.debug.print('data.qpos before: {qpos}', qpos = data.qpos)
            # print('qpos before:', data.qpos)
            if backend == jnp:
                # Use JAX for randomization
                # Unpack joint stiffness dictionary into indices and values
                joint_names = list(domrand_state.prosthesis_socket_joint_value.keys())
                joint_values = list(domrand_state.prosthesis_socket_joint_value.values())
                joint_indices = [self._socket_joint_indices[name] for name in joint_names]

                # Convert to jnp arrays
                joint_indices = jnp.array(joint_indices)
                joint_values = jnp.array(joint_values)

                #jax.debug.print('joint_values: {joint_values}', joint_values=joint_values)

                
                # Apply to data
                all_joint_values = data.qpos.at[joint_indices].set(joint_values)

                data = data.replace(qpos=all_joint_values) 
            elif backend == np: 
                all_joint_values = self._init_prosthesis_socket_joint_value.copy()
                for joint_name, value in domrand_state.prosthesis_socket_joint_value.items():
                    idx = self._socket_joint_indices[joint_name]
                    all_joint_values[idx] = value

                data.qpos = all_joint_values
            # model = self._set_attribute_in_model(model, "qpos0", all_joint_values, backend)
            # jax.debug.print('data.qpos after: {qpos}', qpos = data.qpos)

            # Set qpos depeding on randomization
            updated_springref, carry = self._set_joint_springref(model,domrand_state.prosthesis_socket_joint_value,carry, backend)
            # jax.debug.print('model_updated_qpos0 before: {springref}', springref=model.qpos_spring)
            model = self._set_attribute_in_model(model, 'qpos_spring', updated_springref, backend)
            # jax.debug.print('model_updated_qpos0 after: {springref}', springref=model.qpos_spring)

            # if backend == np:
            #     mujoco.mj_forward(model, data)
            # elif backend == jnp:
            #     mjx.forward(model, data)
            if 'socket_ty'+self.prosthesis_side_str in self._socket_joint_indices:

                if not np.any(self.init_talus_pos):
                    # jax.debug.print("IN LOOOOOOPPPPP")
                    self.init_talus_pos = model.body_pos[self._talus_idx].copy()
                print(f"init_talus_pos:", self.init_talus_pos)
                
                talus_offset_y = domrand_state.prosthesis_socket_joint_value[f"socket_ty"+self.prosthesis_side_str]
                # jax.debug.print("pos_y: {pos_y}", pos_y = pos_y)
                # talus_pos = self._init_prosthesis_body_position[self._talus_idx].copy() #model.body_pos[self._talus_idx].copy()
                # jax.debug.print("talus_pos: {talus_pos}", talus_pos = talus_pos)
                new_talus_pos = self.init_talus_pos - backend.array([0,talus_offset_y,0])

                if not self.rand_conf["randomize_prosthesis_body_position"]:
                    # jax.debug.print("new talus pos: {new_talus_pos}", new_talus_pos=new_talus_pos) 
                    if backend == np: 
                        body_pos = self._init_prosthesis_body_position.copy()
                        body_pos[self._talus_idx] = new_talus_pos
                    elif backend == jnp: 
                        body_pos = model.body_pos.at[jnp.array(self._talus_idx)].set(new_talus_pos)
                    # jax.debug.print("body_pos: {body_pos}", body_pos=body_pos)
                    model = self._set_attribute_in_model(model, "body_pos", body_pos, backend)


            # # # print('qpos after:', data.qpos)
            # #### Adapt tibia and socket mass, inertia, and center of mass 
            if "adapt_tibia_socket_parameters" in self.rand_conf:
                if 'socket_ty'+self.prosthesis_side_str in self._socket_joint_indices and self.rand_conf["adapt_tibia_socket_parameters"]:
                # print('socket_ty in rand')
                # Adapt the tibia or prosthesis properties depending on length
                    body_mass, body_inertia, body_center_of_mass = self._adapt_tibia_socket_parameters(model, data, carry, backend)

                    model = self._set_attribute_in_model(model, "body_mass", body_mass, backend)
                    model = self._set_attribute_in_model(model, "body_inertia", body_inertia, backend)
                    model = self._set_attribute_in_model(model, "body_ipos", body_center_of_mass, backend)



        if self.rand_conf["randomize_prosthesis_body_position"]:
            # print('model.body_pos before: ', model.body_pos)
            if backend == jnp: 
                body_names = list(domrand_state.prosthesis_body_position.keys())
                body_values = list(domrand_state.prosthesis_body_position.values())
                body_names = [name + self.prosthesis_side_str for name in body_names]
                body_indices = [self._body_pos_indices[name] for name in body_names]

                body_indices = jnp.array(body_indices)
                body_values = jnp.array(body_values)

                body_pos = model.body_pos.at[jnp.array(body_indices)].set(body_values)
            elif backend == np: 
                body_pos = self._init_prosthesis_body_position.copy()
                for body_name, value in domrand_state.prosthesis_body_position.items():
                    body_name = body_name + self.prosthesis_side_str
                    idx = self._body_pos_indices[body_name]
                    body_pos[idx] = value

            # If talus height needs to be adapted depending on socket_ty (so if socket_ty in randomize_prosthesis_socket_joint)
            if hasattr(self, '_talus_idx'):
                # jax.debug.print("new talus pos: {new_talus_pos}", new_talus_pos=new_talus_pos) 
                talus_offset_array = backend.array([0,talus_offset_y,0])
                if backend == np: 
                    body_pos[self._talus_idx] -= talus_offset_array
                elif backend == jnp: 
                    body_pos = body_pos.at[jnp.array(self._talus_idx)].set(body_pos.at[jnp.array(self._talus_idx)].get() - talus_offset_array)

            model = self._set_attribute_in_model(model, "body_pos", body_pos, backend)
            print('model.body_pos after: ', model.body_pos)




        # jax.debug.print("model.jnt_stiffness: {jnt_stiffness}", jnt_stiffness=model.jnt_stiffness)
        # jax.debug.print("model.dof_damping: {dof_damping}", dof_damping=model.dof_damping)
        # jax.debug.print("model.body_pos after: {body_pos}", body_pos=model.body_pos)
        # jax.debug.print("model.body_quat: {body_quat}", body_quat=model.body_quat)
        # print("model.jnt_stiffness: ", model.jnt_stiffness)
        # print("model.dof_damping: ", model.dof_damping)
        # print("model.body_pos: ", model.body_pos)
        # print("model.body_quat: ", model.body_quat)

        return model, data, carry




    def _sample_joint_stiffness(self, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """ Samples the joint stiffness parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized joint stiffness and carry.

        """
        assert_backend_is_supported(backend)
        sampled_stiffness = {}
        
        # stiffness_dict = self.rand_conf["prosthesis_joint_stiffness_range"]
        sampled_stiffness = {joint_name+self.prosthesis_side_str: 0.0 for joint_name in self.stiffness_dict}

        if self.rand_conf["randomize_prosthesis_joint_stiffness"]:
            # stiffness_dict = self.rand_conf["prosthesis_joint_stiffness_range"]
            # sampled_stiffness = {joint_name+prosthesis_side_str : 0.0 for joint_name in stiffness_dict}

            if backend == jnp:
                key = carry.key
                key, _k = jax.random.split(key)
                rand_values = jax.random.uniform(_k, shape=(len(self.stiffness_dict),))
                carry = carry.replace(key=key)
            else:
                rand_values = np.random.uniform(size=(len(self.stiffness_dict),))

            for i, (joint_name, (stiff_min, stiff_max)) in enumerate(self.stiffness_dict.items()):
                joint_name = joint_name + self.prosthesis_side_str
                stiffness_val = stiff_min + (stiff_max - stiff_min) * rand_values[i]
                sampled_stiffness[joint_name] = stiffness_val

        else:
            # Use existing model stiffness for defined joints
            for i, (joint_name, (stiff_min, stiff_max)) in enumerate(self.stiffness_dict.items()):
                joint_name = joint_name + self.prosthesis_side_str
            # for joint_name in self.rand_conf.get("randomization_joint_names", []):
            #     joint_name = joint_name + prosthesis_side_str
                if joint_name not in self._joint_indices:
                    raise KeyError(f"Joint '{joint_name}' not found in joint indices.")
                idx = self._joint_indices[joint_name]
                if backend == np:
                    value = model.jnt_stiffness[idx]
                else:
                    value = model.jnt_stiffness.at[idx].get()
                sampled_stiffness[joint_name] = value

        return sampled_stiffness, carry


    def _sample_dof_damping(self, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Samples the joint damping parameters.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized joint damping and carry.

        """

        assert_backend_is_supported(backend)
        sampled_damping = {}

        sampled_damping = {dof_name+self.prosthesis_side_str: 0.0 for dof_name in self.damping_dict}

        if self.rand_conf["randomize_prosthesis_dof_damping"]:
            # damping_min, damping_max = self.rand_conf["prosthesis_dof_damping_range"]
            # n_dofs = len(self._dof_indices.values()) 

            if backend == jnp:
                key = carry.key
                key, _k = jax.random.split(key)
                rand_values = jax.random.uniform(_k, shape=(len(self.damping_dict),))
                carry = carry.replace(key=key)
            else:
                rand_values = np.random.uniform(size=(len(self.damping_dict),))
            
            for i, (dof_name, (damp_min, damp_max)) in enumerate(self.damping_dict.items()):
                dof_name = dof_name + self.prosthesis_side_str
                damping_val = damp_min + (damp_max - damp_min) * rand_values[i]
                sampled_damping[dof_name] = damping_val

        else:
            # Use existing model damping for defined dofs
            for i, (dof_name, (damp_min, damp_max)) in enumerate(self.damping_dict.items()):
                dof_name = dof_name + self.prosthesis_side_str
                if dof_name not in self._dof_indices:
                    raise KeyError(f"DOF '{dof_name}' not found in dof indices.")
                idx = self._dof_indices[dof_name]
                if backend == np:
                    value = model.dof_damping[idx]
                else:
                    value = model.dof_damping.at[idx].get()
                sampled_damping[dof_name] = value

        return sampled_damping, carry


    def _sample_geom_position(self, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Sample random position for the amputated limb prosthesis.
        
        Args:
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).
        
        Returns:
            ndarray: Randomly sampled position.
        """
        assert_backend_is_supported(backend)

        # If no bodies are configured for position randomization, return initial state directly
        if not self.randomized_pos_body_names:
            # Return the original initial positions, as no randomization is applied
            return self._init_prosthesis_body_position, carry

        sampled_offsets_for_bodies = {}
        # --- Determine sampling parameters based on configuration format ---
        if hasattr(self, 'is_position_range_nested') and self.is_position_range_nested:
            # New format: per-body ranges
            position_min_flat = []
            position_max_flat = []
            directions_flat_list = [] # List to store 'x','y','z' for each flattened dimension
            
            # This list will store the specific ranges for each body, in the order of self.randomized_pos_body_names
            per_body_ranges_info = {}

            for body_name in self.randomized_pos_body_names:
                body_name = body_name.replace(self.prosthesis_side_str, '')
                body_range = self.prosthesis_body_position_range_dict[body_name]
                directions = list(body_range.keys())
                
                body_min_coords = backend.array([body_range[axis][0] for axis in directions])
                body_max_coords = backend.array([body_range[axis][1] for axis in directions])

                per_body_ranges_info[body_name] = {
                    'min': body_min_coords,
                    'max': body_max_coords,
                    'directions': directions # e.g., ['x', 'z']
                }
                
            #     # For overall sampling, we need to know the total number of dimensions
            #     for axis in directions:
            #         position_min_flat.append(body_range[axis][0])
            #         position_max_flat.append(body_range[axis][1])
            #         directions_flat_list.append(axis) # e.g., ['x', 'z', 'y', 'x'] if two bodies

            # position_min = backend.array(position_min_flat)
            # position_max = backend.array(position_max_flat)
            # n_total_coordinates_to_sample = len(directions_flat_list) # Total dimensions across all bodies

            # if backend == jnp:
            #     key = carry.key
            # else:
            #     pass # NumPy random does not need a key

            # `per_body_ranges_info` is already a dictionary, so we'll iterate through it.
            # No need for `position_min_flat`, `position_max_flat`, `directions_flat_list`
            # or `n_total_coordinates_to_sample` in this per-body sampling approach.
            # The original `per_body_ranges_info` is already what we need to iterate.
            
            # The input `per_body_ranges_info` is assumed to be a dictionary where:
            # per_body_ranges_info[body_name] = {'min': backend.array([min_vals]), 'max': backend.array([max_vals]), 'directions': ['x', 'z']}

            for body_name, body_info in per_body_ranges_info.items():
                # Ensure the full MuJoCo body name exists in the model
                # This mapping might have happened in _get_body_id_map, but we need the MuJoCo ID for initial pos
                full_mujoco_body_name = body_name + self.prosthesis_side_str
                body_id = self._body_pos_indices[full_mujoco_body_name]
                if body_id == -1:
                    print(f"Warning: Body '{full_mujoco_body_name}' not found for randomization. Skipping.")
                    continue

                directions = body_info['directions']
                body_min_coords = body_info['min'] # Already backend.array from initial construction
                body_max_coords = body_info['max'] # Already backend.array from initial construction
                num_coords_for_body = len(directions)

                if num_coords_for_body == 0:
                    print(f"Warning: No directions specified for body '{body_name}'. Skipping randomization.")
                    continue

                # We sample interpolation values specifically for this body's coordinates
                if backend == jnp:
                    key = carry.key
                    key, _k = jax.random.split(key) # Split key for each body if JAX
                    interpolation_for_body = jax.random.uniform(_k, shape=(num_coords_for_body,))
                else: # numpy
                    interpolation_for_body = np.random.uniform(size=(num_coords_for_body,))

                # Calculate the sampled offsets (deltas) for this body's specific coordinates
                sampled_offsets_interp_for_body = body_min_coords + (body_max_coords - body_min_coords) * interpolation_for_body

                # Initialize the 3D offset vector for the current body (defaults to zeros)
                current_body_offset_vector = backend.zeros(3)

                # Assign the sampled values to the correct x,y,z components
                for j, direction in enumerate(directions):
                    coord_idx = -1
                    if direction == 'x':
                        coord_idx = 0
                    elif direction == 'y':
                        coord_idx = 1
                    elif direction == 'z':
                        coord_idx = 2

                    if coord_idx != -1: # Valid direction
                        if backend == jnp:
                            current_body_offset_vector = current_body_offset_vector.at[coord_idx].set(sampled_offsets_interp_for_body[j])
                        else: # numpy
                            current_body_offset_vector[coord_idx] = sampled_offsets_interp_for_body[j]

                # # Store the reconstructed 3D offset vector for this body in the dictionary
                # sampled_offsets_for_bodies[body_name] = current_body_offset_vector 

                if backend == np:
                    original_positions =  model.body_pos[body_id].copy()
                elif backend == jnp: 
                    original_positions = model.body_pos.at[body_id].get()

                # Store the reconstructed 3D offset vector for this body in the dictionary
                sampled_offsets_for_bodies[body_name] = current_body_offset_vector + original_positions

            # At this point, sampled_offsets_for_bodies contains {body_name: [dx, dy, dz]} for all randomized bodies.
            # Now, populate sampled_positions using these offsets and initial body positions.
            sampled_position_offsets = sampled_offsets_for_bodies  # Renaming for clarity based on original intent

        if self.rand_conf["randomize_prosthesis_body_position"]:
            # sampled_position_offsets now contains the randomized delta (offset)
            # from the initial position for each *randomized* body.
            # We add these offsets to the initial positions.
            sampled_position = sampled_position_offsets
        else:
            # # No randomization, keep initial position
            # sampled_position = self._init_prosthesis_body_position
            if backend == np:
                sampled_position = model.body_pos[list(self._body_pos_indices.values())].copy()
            elif backend == jnp:
                # JAX does not support list indexing with .at[].get(), so use jnp.array for indices
                indices = jnp.array(list(self._body_pos_indices.values()))
                sampled_position = model.body_pos.at[indices].get()

        return sampled_position, carry



    # # ORG CORRECT
    # def _sample_geom_position(self, model: Union[MjModel, Model],
    #                           carry: Any,
    #                           backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
    #     """
    #     Sample random position for the amputated limb prosthesis.
        
    #     Args:
    #         backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).
        
    #     Returns:
    #         ndarray: Randomly sampled position.
    #     """
    #     assert_backend_is_supported(backend)

    #     # Adapt prosthesis_body_position_range: {'x': [-0.1, 0.1], 'y': [-0.1, 0.1], 'z': [-0.001, 0.001]}
    #     position_range = self.rand_conf["prosthesis_body_position_range"]
    #     directions = list(position_range.keys())

    #     # position_range = self.rand_conf["prosthesis_body_position_range"]
    #     # directions = self.rand_conf["prosthesis_body_position_direction"]
    #     # Build min/max arrays in the order of directions
    #     position_min = np.array([position_range[axis][0] for axis in directions])
    #     position_max = np.array([position_range[axis][1] for axis in directions])
    #     n_bodies = len(self._body_pos_indices.values())

    #     ncoordinates = len(directions) #self.rand_conf["prosthesis_body_position_direction"])

        
    #     if backend == jnp:
    #         key = carry.key
    #         key, _k = jax.random.split(key)
    #         interpolation = jax.random.uniform(_k, shape=(n_bodies, ncoordinates))
    #         carry = carry.replace(key=key)
    #     else:
    #         interpolation = np.random.uniform(size=(n_bodies, ncoordinates))

    #     sampled_position_interp = position_min + (position_max - position_min) * interpolation
    #     if self.rand_conf["randomize_prosthesis_body_position"]:
    #         if backend == np:
    #             sampled_position = self._init_prosthesis_body_position[list(self._body_pos_indices.values())].copy()
    #             for i in range(ncoordinates):
    #                 if directions[i] == 'x':
    #                     sampled_position[:, 0] = sampled_position_interp[:, i] + sampled_position[:,0]
    #                 elif directions[i] == 'y':
    #                     sampled_position[:, 1] = sampled_position_interp[:, i] + sampled_position[:,1]
    #                 elif directions[i] == 'z':
    #                     sampled_position[:, 2] = sampled_position_interp[:, i] + sampled_position[:,2]
    #             # if 'x' in self.rand_conf["prosthesis_body_position_direction"]:
    #             #     sampled_position[:, 0] = sampled_position_interp[:, i]
    #             #     i += 1
    #             # if 'y' in self.rand_conf["prosthesis_body_position_direction"]:
    #             #     sampled_position[:, 1] = sampled_position_interp[:, i]
    #             #     i += 1
    #             # if 'z' in self.rand_conf["prosthesis_body_position_direction"]:
    #             #     sampled_position[:, 2] = sampled_position_interp[:, i]
    #             #     i += 1
    #         elif backend == jnp:
    #             # JAX does not support list indexing with .at[].get(), so use jnp.array for indices
    #             indices = jnp.array(list(self._body_pos_indices.values()))
    #             sampled_position = model.body_pos.at[indices].get()
    #             def update_position(pos, interp):
    #                 for i in range(ncoordinates):
    #                     if directions[i] == 'x':
    #                         pos = pos.at[..., 0].set(interp[..., i] + pos[..., 0])
    #                     elif directions[i] == 'y':
    #                         pos = pos.at[..., 1].set(interp[..., i] + pos[..., 1])
    #                     elif directions[i] == 'z':
    #                         pos = pos.at[..., 2].set(interp[..., i] + pos[..., 2])
    #                 return pos

    #             sampled_position = jax.vmap(update_position)(sampled_position, sampled_position_interp)
    #     else:
    #         # No randomization, keep initial position
    #         if backend == np:
    #             sampled_position = model.body_pos[list(self._body_pos_indices.values())].copy()
    #         elif backend == jnp:
    #             # JAX does not support list indexing with .at[].get(), so use jnp.array for indices
    #             indices = jnp.array(list(self._body_pos_indices.values()))
    #             sampled_position = model.body_pos.at[indices].get()
    #         # sampled_position = self._init_prosthesis_body_position[list(self._body_indices.values())].copy()
    #     return sampled_position, carry


    def _sample_joint_orientation(self, model: Union[MjModel, Model],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Sample random orientation for the amputated limb prosthesis.
        
        Args:
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).
        
        Returns:
            ndarray: Randomly sampled orientation.
        """
        assert_backend_is_supported(backend)

        if not self.randomized_quat_body_names:
            return self._init_prosthesis_body_orientation, carry 
        
        sampled_orientation_for_bodies = {}

        if hasattr(self, 'is_orientation_range_nested') and self.is_orientation_range_nested:
            per_body_ranges_info = {}

            for body_name in self.randomized_quat_body_names:
                body_name = body_name.replace(self.prosthesis_side_str, '')
                body_range = self.prosthesis_body_quat_range_dict[body_name]
                directions = list(body_range.keys())
                
                body_min_coords = backend.array([body_range[axis][0] for axis in directions])
                body_max_coords = backend.array([body_range[axis][1] for axis in directions])

                per_body_ranges_info[body_name] = {
                    'min': body_min_coords,
                    'max': body_max_coords,
                    'directions': directions # e.g., ['x', 'z']
                }

            for body_name, body_info in per_body_ranges_info.items():
                # Ensure the full MuJoCo body name exists in the model
                # This mapping might have happened in _get_body_id_map, but we need the MuJoCo ID for initial pos
                full_mujoco_body_name = body_name + self.prosthesis_side_str
                body_id = self._body_quat_indices[full_mujoco_body_name]
                if body_id == -1:
                    print(f"Warning: Body '{full_mujoco_body_name}' not found for randomization. Skipping.")
                    continue

                directions = body_info['directions']
                body_min_coords = body_info['min'] # Already backend.array from initial construction
                body_max_coords = body_info['max'] # Already backend.array from initial construction
                num_coords_for_body = len(directions)

                if num_coords_for_body == 0:
                    print(f"Warning: No directions specified for body '{body_name}'. Skipping randomization.")
                    continue

                if backend == jnp:
                    key = carry.key
                    key, _k = jax.random.split(key) # Split key for each body if JAX
                    interpolation_for_body = jax.random.uniform(_k, shape=(num_coords_for_body,))
                else: # numpy
                    interpolation_for_body = np.random.uniform(size=(num_coords_for_body,))

                # Calculate the sampled offsets (deltas) for this body's specific coordinates
                sampled_offsets_interp_for_body = body_min_coords + (body_max_coords - body_min_coords) * interpolation_for_body

                if backend == np:
                    init_orientation = self._init_prosthesis_body_orientation[body_id].copy() #self._init_prosthesis_body_orientation[list(self._body_quat_indices.values())].copy() # Format: (w,x,y,z)
                    # Convert (w, x, y, z) to (x, y, z, w) for rotation calculations
                    init_orientation_sorted = np.concatenate(
                        [init_orientation[1:4], init_orientation[0:1]], axis=0
                    ) #Format: (x,y,z,w)
                    init_rotation = R.from_quat(init_orientation_sorted)
                elif backend == jnp:
                    # JAX does not support list indexing with .at[].get(), so use jnp.array for indices
                    indices = jnp.array(body_id) #self._body_quat_indices.values()))
                    init_orientation = model.body_quat.at[indices].get()
                    # Convert (w, x, y, z) to (x, y, z, w) for rotation calculations
                    init_orientation_sorted = jnp.concatenate(
                        [init_orientation[...,1:4], init_orientation[...,0:1]], axis=0
                    ) #Format: (x,y,z,w)
                    init_rotation = jaxR.from_quat(init_orientation_sorted)
                
                # jax.debug.print("init_orientation: {orientation_euler}", orientation_euler=init_orientation)
                # jax.debug.print("init_orientation_sorted: {orientation_euler}", orientation_euler=init_orientation_sorted)
                
                orientation_euler = init_rotation.as_euler('xyz', degrees=False)
                for j, direction in enumerate(directions):
                    coord_idx = -1
                    if direction == 'x':
                        coord_idx = 0
                    elif direction == 'y':
                        coord_idx = 1
                    elif direction == 'z':
                        coord_idx = 2

                    if coord_idx != -1: # Valid direction
                        if backend == jnp:
                            # orientation_euler = orientation_euler.at[...,coord_idx].set(sampled_offsets_interp_for_body[...,j]+orientation_euler[...,coord_idx])
                            orientation_euler = orientation_euler.at[coord_idx].set(sampled_offsets_interp_for_body[j]+orientation_euler[coord_idx])
                        else: # numpy
                            # orientation_euler[:,coord_idx] = sampled_offsets_interp_for_body[j] + orientation_euler[:,coord_idx]
                            orientation_euler[coord_idx] = sampled_offsets_interp_for_body[j] + orientation_euler[coord_idx]

                # jax.debug.print("sampled_offsets_interp_for_body: {sampled_offsets_interp_for_body}", sampled_offsets_interp_for_body=sampled_offsets_interp_for_body)
                # jax.debug.print("orientation_euler: {orientation_euler}", orientation_euler=orientation_euler)

                # if self.rand_conf["randomize_prosthesis_body_orientation"]:
                #     for i in range(len(directions)):
                #         if backend == np:
                #             if directions[i] == 'x':
                #                 orientation_euler[:, 0] = sampled_offsets_interp_for_body[:, i] + orientation_euler[:, 0]
                #             elif directions[i] == 'y':
                #                 orientation_euler[:, 1] = sampled_offsets_interp_for_body[:, i] + orientation_euler[:, 1]
                #             elif directions[i] == 'z':
                #                 orientation_euler[:, 2] = sampled_offsets_interp_for_body[:, i] + orientation_euler[:, 2]
                #         elif backend == jnp:
                #             def update_orientation(euler, interp):
                #                 for i in range(num_coords_for_body):
                #                     if directions[i] == 'x':
                #                         euler = euler.at[..., 0].set(interp[..., i] + euler[..., 0])
                #                         # euler = euler.at[..., 0].set(interp[i] + orientation_euler[0])
                #                     elif directions[i] == 'y':
                #                         euler = euler.at[..., 1].set(interp[..., i] + euler[..., 1])
                #                         # euler = euler.at[..., 1].set(interp[i] + orientation_euler[1])
                #                     elif directions[i] == 'z':
                #                         euler = euler.at[..., 2].set(interp[..., i] + euler[..., 2])
                #                         # euler = euler.at[..., 2].set(interp[i] + orientation_euler[2])
                #                 return euler
                            

                #             orientation_euler = jax.vmap(update_orientation)(orientation_euler, sampled_offsets_interp_for_body)

                # sampled_orientation = sampled_rotation.as_quat()  # Convert back to (w, x, y, z) format
                
                if self.rand_conf["randomize_prosthesis_body_orientation"]:
                    if backend == np:
                        sampled_rotation = R.from_euler('xyz', orientation_euler, degrees=False)
                        sampled_orientation_unsorted = sampled_rotation.as_quat() 
                        sampled_orientation_sort = np.concatenate(
                            [sampled_orientation_unsorted[:, 3:4], sampled_orientation_unsorted[:, 0:3]], axis=1
                        )
                    elif backend == jnp:
                        sampled_rotation = jaxR.from_euler('xyz', orientation_euler, degrees=False)
                        sampled_orientation_unsorted = sampled_rotation.as_quat()
                        sampled_orientation_sort = jnp.concatenate(
                            [sampled_orientation_unsorted[3:4], sampled_orientation_unsorted[ 0:3]], axis=0
                        )
                        # jax.debug.print("sampled_orientation_unsorted: {sampled_orientation_unsorted}", sampled_orientation_unsorted=sampled_orientation_unsorted)
                        # jax.debug.print("sampled_orientation_sort: {orientation_euler}", orientation_euler=sampled_orientation_sort)

                        # sampled_orientation_var = init_orientation
                        # sampled_orientation_var = sampled_orientation_var.at[...,0].set(sampled_orientation_sort[...,0])
                        # sampled_orientation_var = sampled_orientation_var.at[...,1].set(sampled_orientation_sort[...,1])
                        # sampled_orientation_var = sampled_orientation_var.at[...,2].set(sampled_orientation_sort[...,2])
                        # sampled_orientation_var = sampled_orientation_var.at[...,3].set(sampled_orientation_sort[...,3])

                    sampled_orientation_for_bodies[body_name] = sampled_orientation_sort #sampled_orientation_sort#.squeeze(0) 
                    
            sampled_orientation_randomized = sampled_orientation_for_bodies    
            # jax.debug.print("sampled_orientation_for_bodies: {orientation_euler}", orientation_euler=sampled_orientation_for_bodies)

        if self.rand_conf["randomize_prosthesis_body_orientation"]:
            sampled_orientation = sampled_orientation_randomized
        else: 
            if backend == np: 
                sampled_orientation = model.body_quat[list(self._body_quat_indices.values())].copy()
            elif backend == jnp: 
                indices = jnp.array(list(self._body_quat_indices.values()))
                sampled_orientation = model.body_quat.at[indices].get()

        
        return sampled_orientation, carry 





    # def _sample_joint_orientation_old(self, model: Union[MjModel, Model],
    #                           carry: Any,
    #                           backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
    #     """
    #     Sample random orientation for the amputated limb prosthesis.
        
    #     Args:
    #         backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).
        
    #     Returns:
    #         ndarray: Randomly sampled orientation.
    #     """
    #     assert_backend_is_supported(backend)

    #     orientation_range = self.rand_conf["prosthesis_body_orientation_range"]

    #     directions= list(orientation_range.keys())
    #     # directions = self.rand_conf["prosthesis_body_orientation_direction"]
    #     # Build min/max arrays in the order of directions
    #     orientation_min = np.array([orientation_range[axis][0] for axis in directions])
    #     orientation_max = np.array([orientation_range[axis][1] for axis in directions])
    #     n_bodies = len(self._body_quat_indices.values())
    #     ndirections = len(directions) #self.rand_conf["prosthesis_body_orientation_direction"]) 
        
    #     if backend == jnp:
    #         key = carry.key
    #         key, _k = jax.random.split(key)
    #         interpolation = jax.random.uniform(_k, shape=(n_bodies, ndirections))
    #         carry = carry.replace(key=key)
    #     else:
    #         interpolation = np.random.uniform(size=(n_bodies, ndirections))
    #     sampled_orientation_euler = orientation_min + (orientation_max - orientation_min) * interpolation
        
    #     if backend == np:
    #         init_orientation = self._init_prosthesis_body_orientation[list(self._body_quat_indices.values())].copy() # Format: (w,x,y,z)
    #         # Convert (w, x, y, z) to (x, y, z, w) for rotation calculations
    #         init_orientation_sorted = np.concatenate(
    #             [init_orientation[:, 1:4], init_orientation[:, 0:1]], axis=1
    #         ) #Format: (x,y,z,w)
    #         init_rotation = R.from_quat(init_orientation_sorted)
    #     elif backend == jnp:
    #         # JAX does not support list indexing with .at[].get(), so use jnp.array for indices
    #         indices = jnp.array(list(self._body_quat_indices.values()))
    #         init_orientation = model.body_quat.at[indices].get()
    #         # Convert (w, x, y, z) to (x, y, z, w) for rotation calculations
    #         init_orientation_sorted = jnp.concatenate(
    #             [init_orientation[:, 1:4], init_orientation[:, 0:1]], axis=1
    #         ) #Format: (x,y,z,w)
    #         init_rotation = jaxR.from_quat(init_orientation_sorted)
        
        
    #     orientation_euler = init_rotation.as_euler('xyz', degrees=False)

    #     if self.rand_conf["randomize_prosthesis_body_orientation"]:
    #         for i in range(len(directions)):
    #             if backend == np:
    #                 if directions[i] == 'x':
    #                     orientation_euler[:, 0] = sampled_orientation_euler[:, i] + orientation_euler[:, 0]
    #                 elif directions[i] == 'y':
    #                     orientation_euler[:, 1] = sampled_orientation_euler[:, i] + orientation_euler[:, 1]
    #                 elif directions[i] == 'z':
    #                     orientation_euler[:, 2] = sampled_orientation_euler[:, i] + orientation_euler[:, 2]
    #             elif backend == jnp:
    #                 def update_orientation(euler, interp):
    #                     for i in range(ndirections):
    #                         if directions[i] == 'x':
    #                             euler = euler.at[..., 0].set(interp[..., i] + euler[..., 0])
    #                             # euler = euler.at[..., 0].set(interp[i] + orientation_euler[0])
    #                         elif directions[i] == 'y':
    #                             euler = euler.at[..., 1].set(interp[..., i] + euler[..., 1])
    #                             # euler = euler.at[..., 1].set(interp[i] + orientation_euler[1])
    #                         elif directions[i] == 'z':
    #                             euler = euler.at[..., 2].set(interp[..., i] + euler[..., 2])
    #                             # euler = euler.at[..., 2].set(interp[i] + orientation_euler[2])
    #                     return euler
                    

    #                 orientation_euler = jax.vmap(update_orientation)(orientation_euler, sampled_orientation_euler)


    #     # sampled_orientation = sampled_rotation.as_quat()  # Convert back to (w, x, y, z) format
    #     if self.rand_conf["randomize_prosthesis_body_orientation"]:
    #         if backend == np:
    #             sampled_rotation = R.from_euler('xyz', orientation_euler, degrees=False)
    #             sampled_orientation_unsorted = sampled_rotation.as_quat() 
    #             sampled_orientation = np.concatenate(
    #                 [sampled_orientation_unsorted[:, 3:4], sampled_orientation_unsorted[:, 0:3]], axis=1
    #             )
    #         elif backend == jnp:
    #             sampled_rotation = jaxR.from_euler('xyz', orientation_euler, degrees=False)
    #             sampled_orientation_unsorted = sampled_rotation.as_quat()
    #             sampled_orientation = jnp.concatenate(
    #                 [sampled_orientation_unsorted[..., 3:4], sampled_orientation_unsorted[..., 0:3]], axis=1
    #             )
    #             # Convert back to jnp array for consistency
    #             # jax.debug.print("sampled_orientation: {sampled_orientation}", sampled_orientation = sampled_orientation)
    #             # jax.debug.print("sampled_orientation_euler: {orientation_euler}", orientation_euler = orientation_euler)
    #             # jax.debug.print("init orientation: {init_orientation}", init_orientation = init_orientation)
    #         # if backend == np:
    #         #     # Loop over each body and convert euler to quat in-place
    #         #     for i in range(sampled_orientation.shape[0]):
    #         #         mujoco.mju_euler2Quat(sampled_orientation[i], orientation_euler[i], 'xyz')
    #         #         sampled_orientation[i] = sampled_orientation[i]
    #         # else:
    #         #     # For JAX, use vmap with a function that returns a new quaternion
    #         #     def euler2quat(eul):
    #         #         quat = jnp.zeros(4)
    #         #         quat = quat.at[:].set(jnp.array(mujoco.mju_euler2Quat(np.zeros(4), np.array(eul), 'xyz')))
    #         #         return quat
    #         #     sampled_orientation = jax.vmap(euler2quat)(orientation_euler)
    #     else:
    #         # No randomization, keep initial orientation
    #         if backend == jnp: 
    #             indices = jnp.array(list(self._body_quat_indices.values()))
    #             sampled_orientation = model.body_quat.at[indices].get()
    #         elif backend == np:
    #             sampled_orientation = model.body_quat[list(self._body_quat_indices.values())].copy() #self._init_prosthesis_body_orientation[list(self._body_indices.values())].copy()

    #     return sampled_orientation, carry
    
    

    def _sample_socket_joint_value(self, model: Union[MjModel, Model], data: Union[MjData, Data],
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        
        assert_backend_is_supported(backend)
        sampled_socket_joint_value = {}

        sampled_socket_joint_value = {joint_name+self.prosthesis_side_str: 0.0 for joint_name in self.socket_joint_range_dict}

        if self.rand_conf["randomize_prosthesis_socket_joint"]:

            if backend == jnp:
                key = carry.key
                key, _k = jax.random.split(key)
                rand_values = jax.random.uniform(_k, shape=(len(self.socket_joint_range_dict),))
                carry = carry.replace(key=key)

                for i, (joint_name, (joint_min, joint_max)) in enumerate(self.socket_joint_range_dict.items()):
                    joint_name = joint_name + self.prosthesis_side_str
                    # joint_value = joint_min + (joint_max - joint_min) * rand_values[...,i]
                    joint_value = joint_min + (joint_max - joint_min) * rand_values[i]
                    sampled_socket_joint_value[joint_name] = joint_value

            else:
                rand_values = np.random.uniform(size=(len(self.socket_joint_range_dict),))
            
                for i, (joint_name, (joint_min, joint_max)) in enumerate(self.socket_joint_range_dict.items()):
                    joint_name = joint_name + self.prosthesis_side_str
                    joint_value = joint_min + (joint_max - joint_min) * rand_values[i]
                    sampled_socket_joint_value[joint_name] = joint_value

        else:
            # Use existing model damping for defined dofs
            for i, (joint_name, (joint_min, joint_max)) in enumerate(self.socket_joint_range_dict.items()):
                joint_name = joint_name + self.prosthesis_side_str
                if joint_name not in self._socket_joint_indices:
                    raise KeyError(f"Socket joint '{joint_name}' not found in dof indices.")
                idx = self._socket_joint_indices[joint_name]
                if backend == np:
                    #value = data.qpos[:,idx]
                    value = data.qpos[idx]
                else:
                    # value = data.qpos.at[...,idx].get()
                    value = data.qpos.at[idx].get()
                sampled_socket_joint_value[joint_name] = value

        return sampled_socket_joint_value, carry
    

    def _set_joint_springref(self, model: Union[MjModel, Model], sampled_socket_joint_value,
                              carry: Any,
                              backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        
        assert_backend_is_supported(backend)

        if backend == jnp:
            updated_springref = model.qpos_spring.at[...].get()
        else: # np backend
            updated_springref = self._init_prosthesis_socket_joint_springref.copy() # Direct reference, modifications will be in-place

        for joint_name, joint_value in sampled_socket_joint_value.items():
            
            # Get the starting index of this joint's position in qpos0
            # For slide/hinge joints, this will be a single index.
            # For ball/free joints, it will be the start of their quaternion/position array.
            qpos_idx = self._socket_joint_indices[joint_name]

            if backend == jnp:
                # JAX: Use .at[idx].set() for immutable array updates
                # For slide/hinge, it's a single value.
                updated_springref = updated_springref.at[qpos_idx].set(joint_value)
            else: # np backend
                # NumPy: Direct assignment to modify in-place
                updated_springref[qpos_idx] = joint_value

        # # For JAX, if model.qpos0 is truly immutable and needs to be returned
        # # as part of a new model, you'd do that here.
        # # In many JAX contexts with MuJoCo, model objects might be passed around
        # # and mutations handled by the framework (e.g., Brax's `System` object).
        # # Assuming we need to update model.qpos0 directly for the numpy case
        # # and potentially return a new model.qpos0 for JAX.
        # if backend == jnp:
        #     # If the model object itself is meant to be immutable in JAX,
        #     # you might return a new model object or just the updated_qpos0
        #     # for further processing. For simplicity here, we assume that
        #     # we're conceptually updating a component of the model.
        #     # Depending on your specific JAX/MuJoCo integration, `model`
        #     # might need to be re-created or a specific `model.replace()`
        #     # method used. For now, we'll return the updated array.
        #     model.qpos0 = updated_qpos0 # This might still be an impure operation for pure JAX functions.
        #     return updated_qpos0, carry
        # else:
        #     # For NumPy, the modification was in-place.
        #     return model.qpos0, carry
        return updated_springref, carry 



    def _adapt_tibia_socket_parameters(self, model: Union[MjModel, Model], data: Union[MjData, Data],
                                       carry: Any, backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
          
        # tibia_name = 'tibia' + self.prosthesis_side_str
        # socket_name = 'pylon_socket' + self.prosthesis_side_str
        # talus_name = 'talus' + self.prosthesis_side_str
        # socket_joints = ['socket_flexion', 'socket_adduction', 'socket_rotation', 'socket_tx', 'socket_ty', 'socket_tz']
        # socket_joints = [j + self.prosthesis_side_str for j in socket_joints]

        # # Iterate over socket joints and find them in model and get their pos
        # socket_joint_indices = {}
        # joint_pos = {}
        # valid_socket_joints = {}
        # for j in socket_joints: 
        #     idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        #     if idx != -1: 
        #         joint_pos[j] = model.jnt_pos[idx]
        #         socket_joint_indices[j] =idx
        #         valid_socket_joints[j] = j

        # Check if all elements are the same
        pos_y = []
        for j in self.valid_socket_joints:
            pos_y.append(self.joint_pos[j][1])

        different_values_in_list = len(set(pos_y))
        
        assert different_values_in_list ==1,"Socket joints are not all in the same position"

        # Get tibia data
        # tibia_idx= mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tibia_name)
        org_tibia_mass = model.body_mass[self._tibia_idx].copy()
        org_tibia_inertia = model.body_inertia[self._tibia_idx].copy()
        org_tibia_iquat =model.body_iquat[self._tibia_idx].copy()
        org_tibia_center_of_mass = model.body_ipos[self._tibia_idx].copy()
    

        # Get socket data 
        # socket_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, socket_name)
        org_socket_mass = model.body_mass[self._socket_idx].copy()
        org_socket_inertia = model.body_inertia[self._socket_idx].copy()
        org_socket_iquat = model.body_iquat[self._socket_idx].copy()
        org_socket_center_of_mass = model.body_ipos[self._socket_idx].copy()
        # org_socket_pos = model.body_pos[self._socket_idx]
        socket_top_pos_in_tibia = model.body_pos[self._socket_idx].copy()
        
        socket_jnt_pos_in_socket = pos_y[0] # Position where overlap between tibia and socket end 

        jax.debug.print('org_tibia_mass: {org_tibia_mass}', org_tibia_mass=org_tibia_mass)
        jax.debug.print('org_socket_mass: {org_socket_mass}', org_socket_mass=org_socket_mass)
       
        org_tibia_quat_sort = backend.concatenate([org_tibia_iquat[1:4], org_tibia_iquat[0:1]],axis=0)
        org_socket_quat_sort = backend.concatenate([org_socket_iquat[1:4], org_socket_iquat[0:1]],axis=0)

        if backend == jnp: 
            tibia_rot = jaxR.from_quat(org_tibia_quat_sort)
            socket_rot = jaxR.from_quat(org_socket_quat_sort)
        elif backend == np: 
            tibia_rot = R.from_quat(org_tibia_quat_sort)
            socket_rot = R.from_quat(org_socket_quat_sort)

        # jax.debug.print('org_tibia_inertia: {org_tibia_inertia}', org_tibia_inertia=org_tibia_inertia)
        jax.debug.print('org_socket_inertia: {org_socket_inertia}', org_socket_inertia=org_socket_inertia)

        org_tibia_inertia_rot= tibia_rot.apply(org_tibia_inertia)
        org_socket_inertia_rot = socket_rot.apply(org_socket_inertia)
        
        # jax.debug.print('org_tibia_inertia_rot: {org_tibia_inertia_rot}', org_tibia_inertia_rot=org_tibia_inertia_rot)
        jax.debug.print('org_socket_inertia_rot: {org_socket_inertia_rot}', org_socket_inertia_rot=org_socket_inertia_rot)

        

        # Tibia length before changes through socket_ty 
        tibia_length = backend.abs(socket_top_pos_in_tibia[1] + socket_jnt_pos_in_socket)
        # talus_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, talus_name)
        socket_length = backend.abs(model.body_pos[self._talus_idx][1])# top until talus pos 


        # if socket_ty positive (moving up): decrease tibia and increase socket 
        # If socket_ty negative (moving down): incirease tibia and decrease socket

        # assert model.qpos0[self._socket_ty_idx] < tibia_length  
        # 
        # socket_ty_name = 'socket_ty' + self.prosthesis_side_str
        # socket_ty_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, socket_ty_name)
        # socket_ty_idx =  model.jnt_qposadr[socket_ty_idx]
        tibia_ratio = tibia_length -  model.qpos0[self._socket_ty_idx] / tibia_length
        socket_ratio = socket_length + model.qpos0[self._socket_ty_idx] /socket_length


        # Update values depending on level of amputation
        tibia_mass = org_tibia_mass * tibia_ratio
        socket_mass = org_socket_mass * socket_ratio

        jax.debug.print('org_tibia_mass: {org_tibia_mass}', org_tibia_mass=org_tibia_mass)
        jax.debug.print('org_socket_mass: {org_socket_mass}', org_socket_mass=org_socket_mass)


        # tibia_inertia_rot = org_tibia_inertia_rot
        tibia_radius = self._calculate_cylinder_radius(org_tibia_mass, org_tibia_inertia_rot[1])
        # tibia_inertia_rot[1] = org_tibia_inertia_rot[1] * tibia_ratio
        # tibia_inertia_rot[0] = self._calculate_cylinder_inertia_xorz(org_tibia_mass, tibia_radius, tibia_length -  data.qpos[self._socket_ty_idx])
        # tibia_inertia_rot[2] = tibia_inertia_rot[0]

        # tibia_inertia = tibia_rot.apply(tibia_inertia_rot, inverse = True)
        

        # socket_inertia_rot = []
        socket_radius = self._calculate_cylinder_radius(org_socket_mass, org_socket_inertia_rot[1])
        # socket_inertia_rot[1] = org_socket_inertia_rot[1] * socket_ratio
        # socket_inertia_rot[0] = self._calculate_cylinder_inertia_xorz(org_socket_mass, socket_radius, socket_length[1] + data.qpos[self._socket_ty_idx])
        # socket_inertia_rot[2] = socket_inertia_rot[0]

        tibia_inertia_rot_x_z= self._calculate_cylinder_inertia_xorz(org_tibia_mass, tibia_radius, tibia_length -  data.qpos[self._socket_ty_idx])

        socket_inertia_rot_x_z=self._calculate_cylinder_inertia_xorz(org_socket_mass, socket_radius, socket_length + data.qpos[self._socket_ty_idx])


        # socket_inertia = socket_rot.apply(socket_inertia_rot, inverse = True)

        if backend == jnp: 
            # tibia_inertia = org_tibia_inertia
            # tibia_inertia = tibia_inertia.at[1].set(org_tibia_inertia[1] * tibia_ratio)
            # tibia_inertia = 
            tibia_inertia_rot = org_tibia_inertia_rot
            tibia_inertia_rot = tibia_inertia_rot.at[1].set(org_tibia_inertia_rot[1] * tibia_ratio)
            tibia_inertia_rot = tibia_inertia_rot.at[jnp.array([0])].set(tibia_inertia_rot_x_z)
            tibia_inertia_rot = tibia_inertia_rot.at[jnp.array([2])].set(tibia_inertia_rot_x_z)

            socket_inertia_rot = org_socket_inertia_rot
            socket_inertia_rot = socket_inertia_rot.at[1].set(org_socket_inertia_rot[1] * socket_ratio)
            socket_inertia_rot = socket_inertia_rot.at[jnp.array([0])].set(socket_inertia_rot_x_z)
            socket_inertia_rot = socket_inertia_rot.at[jnp.array([2])].set(socket_inertia_rot_x_z)

            jax.debug.print('socket_inertia_rot: {socket_inertia_rot}', socket_inertia_rot=socket_inertia_rot)

            tibia_center_of_mass = org_tibia_center_of_mass
            tibia_center_of_mass= tibia_center_of_mass.at[1].set(org_tibia_center_of_mass[1] * tibia_ratio)

            socket_center_of_mass = org_socket_center_of_mass
            socket_center_of_mass= socket_center_of_mass.at[1].set(org_socket_center_of_mass[1] * socket_ratio)

        elif backend == np: 
            # tibia_inertia = org_tibia_inertia
            # tibia_inertia[1] = org_tibia_inertia * tibia_ratio


            tibia_inertia_rot = org_tibia_inertia_rot
            tibia_inertia_rot[1] = org_tibia_inertia_rot[1] * tibia_ratio
            tibia_inertia_rot[0] = tibia_inertia_rot_x_z
            tibia_inertia_rot[2] = tibia_inertia_rot_x_z

            # jax.debug.print('tibia_inertia_rot: {tibia_inertia_rot} ', tibia_inertia_rot=tibia_inertia_rot)



            socket_inertia_rot = org_socket_inertia_rot
            socket_inertia_rot[1] = org_socket_inertia_rot[1] * socket_ratio
            socket_inertia_rot[0] = socket_inertia_rot_x_z
            socket_inertia_rot[2] = socket_inertia_rot_x_z

            jax.debug.print('socket_inertia_rot: {socket_inertia_rot}', socket_inertia_rot=socket_inertia_rot)

        
            tibia_center_of_mass = org_tibia_center_of_mass
            tibia_center_of_mass[1] = org_tibia_center_of_mass[1] * tibia_ratio

            # socket_inertia = org_socket_inertia
            # socket_inertia[1] = org_socket_inertia *socket_ratio

            # socket_inertia = org_socket_inertia * socket_ratio
            socket_center_of_mass = org_socket_center_of_mass
            socket_center_of_mass[1] = org_socket_center_of_mass[1] *socket_ratio

        
        tibia_inertia_temp = backend.abs(tibia_rot.inv().apply(tibia_inertia_rot))
        socket_inertia_temp = backend.abs(socket_rot.inv().apply(socket_inertia_rot))

        jax.debug.print('socket_inertia_temp: {socket_inertia_temp}', socket_inertia_temp=socket_inertia_temp)
        # jax.debug.print('tibia_inertia_temp: {tibia_inertia_temp}', tibia_inertia_temp)

        tibia_inertia = tibia_center_of_mass
        socket_inertia = socket_center_of_mass
        
        if backend == jnp: 
            tibia_inertia = tibia_inertia.at[...,0].set(tibia_inertia_temp[0,...])
            tibia_inertia = tibia_inertia.at[...,1].set(tibia_inertia_temp[1,...])
            tibia_inertia = tibia_inertia.at[...,2].set(tibia_inertia_temp[2,...])

            socket_inertia = socket_inertia.at[...,0].set(socket_inertia_temp[0,...])
            socket_inertia = socket_inertia.at[...,1].set(socket_inertia_temp[1,...])
            socket_inertia = socket_inertia.at[...,2].set(socket_inertia_temp[2,...])
        elif backend == np: 
            tibia_inertia = tibia_inertia_temp
            socket_inertia = socket_inertia_temp


        
        # jax.debug.print('tibia_inertia: {tibia_inertia}', tibia_inertia=tibia_inertia)
        jax.debug.print('socket_inertia: {socket_inertia}', socket_inertia=socket_inertia)

        if backend == jnp: 
            modified_body_idx = jnp.array([self._tibia_idx, self._socket_idx])
            modified_mass = jnp.array([tibia_mass, socket_mass])
            modified_inertia = jnp.array([tibia_inertia, socket_inertia])
            modified_center_of_mass = jnp.array([tibia_center_of_mass, socket_center_of_mass])


            all_mass = model.body_mass.at[modified_body_idx].set(modified_mass)
            all_inertia = model.body_inertia.at[modified_body_idx].set(modified_inertia)
            all_center_of_mass = model.body_ipos.at[modified_body_idx].set(modified_center_of_mass)
            # all_mass = model.body_mass.at[...,self._tibia_idx].set(tibia_mass)
            # all_mass = model.body_mass.at[...,self._socket_idx].set(socket_mass)
            # all_inertia = model.body_inertia.at[...,self._tibia_idx].set(tibia_inertia)
            # all_inertia = model.body_inertia.at[...,self._socket_idx].set(socket_inertia)
            # all_center_of_mass = model.body_ipos.at[...,self._tibia_idx].set(tibia_center_of_mass)
            # all_center_of_mass = model.body_ipos.at[...,self._socket_idx].set(socket_center_of_mass)

        elif backend == np: 
            all_mass = model.body_mass.copy()
            all_mass[self._tibia_idx] = tibia_mass
            all_mass[self._socket_idx] = socket_mass

            all_inertia = model.body_inertia.copy()
            all_inertia[self._tibia_idx] = tibia_inertia
            all_inertia[self._socket_idx] = socket_inertia

            all_center_of_mass = model.body_ipos.copy()
            all_center_of_mass[self._tibia_idx] = tibia_center_of_mass
            all_center_of_mass[self._socket_idx] = socket_center_of_mass


        return all_mass, all_inertia, all_center_of_mass


















          











    # def _sample_feet_geom_solref(self, model: Union[MjModel, Model],
    #                           carry: Any,
    #                           backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
    #     """ Sample random solref for the feet geoms of the amputated limb prosthesis.
    #     Args:
    #         model (Union[MjModel, Model]): The simulation model.
    #         carry (Any): Carry instance with additional state information.
    #         backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).
    #     Returns:
    #         Tuple[Union[np.ndarray, jnp.ndarray], Any]: The randomized feet geom solref and carry.
    #     """
    #     assert_backend_is_supported(backend)
    #     if self.rand_conf["randomize_feet_geom_solref"]:
    #         solref_range = self.rand_conf["feet_geom_solref_range"]
    #         geom_sides = list(solref_range.keys())  # ['prosthesis_side', 'other_side']

    #         # Build solref_min and solref_max arrays in the order of indices in self._feet_geom_solref_indices
    #         solref_min = []
    #         solref_max = []
    #         for side_key in self._feet_geom_solref_indices.keys():
    #             solref_min.append(solref_range[side_key][0])
    #             solref_max.append(solref_range[side_key][1])
    #         solref_min = np.array(solref_min)
    #         solref_max = np.array(solref_max)

    #         n_geoms = len(self._feet_geom_solref_indices.values())
    #         if backend == jnp:
    #             key = carry.key
    #             key, _k = jax.random.split(key)
    #             interpolation = jax.random.uniform(_k, shape=(n_geoms, 2))
    #             carry = carry.replace(key=key)
    #         else:
    #             interpolation = np.random.uniform(size=(n_geoms, 2))
    #     # if self.rand_conf["randomize_prosthesis_foot_geom_solref"]:
    #         sampled_solref = solref_min + (solref_max - solref_min) * interpolation
    #     else:
    #         if backend == np: 
    #             sampled_solref = model.geom_solref[list(self._feet_geom_solref_indices.values())].copy()
    #         elif backend == jnp:
    #             indices = jnp.array(list(self._feet_geom_solref_indices.values()))
    #             sampled_solref = model.geom_solref.at[indices].get()


    #     return sampled_solref, carry
    


    def update_action(self,
                      env: Any,
                      action: Union[np.ndarray, jnp.ndarray],
                      model: Union[MjModel, Model],
                      data: Union[MjData, Data],
                      carry: Any,
                      backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Update the action with randomization effects.

        Args:
            env (Any): The environment instance.
            action (Union[np.ndarray, jnp.ndarray]): The action to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The updated action and carry.

        """
        assert_backend_is_supported(backend)
        return action, carry
    
    

    def update_observation(self,
                           env: Any,
                           obs: Union[np.ndarray, jnp.ndarray],
                           model: Union[MjModel, Model],
                           data: Union[MjData, Data],
                           carry: Any,
                           backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:

        """
            Update the observation with randomization effects.

            Args:
                env (Any): The environment instance.
                obs (Union[np.ndarray, jnp.ndarray]): The observation to be updated.
                model (Union[MjModel, Model]): The simulation model.
                data (Union[MjData, Data]): The simulation data.
                carry (Any): Carry instance with additional state information.
                backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

            Returns:
                Tuple[Union[np.ndarray, jnp.ndarray], Any]: The updated observation and carry.

            """
        
        assert_backend_is_supported(backend)
        return obs, carry
    


    def _calculate_cylinder_radius(self, body_mass, cylinder_inertia_y):
        
        radius = (2*cylinder_inertia_y/body_mass)

        return radius
    
    # def _calculate_org_cylinder_height(self, body_mass,radius, cylinder_inertia):

    #     height = np.sqrt((12*cylinder_inertia/body_mass) - 3*radius**2)

    #     return height
    
    def _calculate_cylinder_inertia_xorz(self, body_mass, radius, height):
        inertia = (1/12)*body_mass * (3*radius**2 + height**2)
        return inertia 
    

