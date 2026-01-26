from types import ModuleType
from typing import Any, Dict, Tuple, Union
import jax
import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model
from flax import struct
import numpy as np
import jax.numpy as jnp
from jax._src.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.core.reward.base import Reward
from loco_mujoco.core.observations.base import ObservationType
from loco_mujoco.core.utils import backend, mj_jntname2qposid, mj_jntname2qvelid, mj_jntid2qposid, mj_jntid2qvelid
from loco_mujoco.core.utils.math import calculate_relative_site_quatities, quaternion_angular_distance
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast
# from loco_mujoco.core.reward.utils import out_of_bounds_action_cost
from loco_mujoco.core.reward import MimicReward
from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction



def check_traj_provided(method):
    """
    Decorator to check if trajectory handler is None. Raises ValueError if not provided.
    """
    def wrapper(self, *args, **kwargs):
        env = kwargs.get('env', None) if 'env' in kwargs else args[5]  # Assumes 'env' is the 6th positional argument
        if getattr(env, "th") is None:
            raise ValueError("TrajectoryHandler not provided, but required for trajectory-based rewards.")
        return method(self, *args, **kwargs)
    return wrapper


@struct.dataclass
class NaturalRewardState:
    """
    State of NaturalReward.
    """
    last_qvel: Union[np.ndarray, jnp.ndarray]
    last_action: Union[np.ndarray, jnp.ndarray]
    last_qfrc_actuator: Union[np.ndarray, jnp.ndarray]
    qfrc_actuator_history: Union[np.ndarray, jnp.ndarray]  # Shape: (10, n_actuators)

    
class MimicRewardEmergenceNatural(MimicReward):
    """
    Inspired from paper: Schumacher, Pierre, et al. "Emergence of natural and robust bipedal walking by learning from biologically plausible objectives." iScience 28.4 (2025).

    r = rvel + ceffort + cpain

    Combined with DeepMimic reward function that computes the reward based on the deviation from the trajectory. The reward is
    computed as the negative exponential of the squared difference between the current state and the trajectory state.
    The reward is computed for the joint positions, joint velocities, relative site positions,
    relative site orientations, and relative site velocities. These sites are specified in the environment properties
    and are placed at key points on the body to mimic the motion of the body.
    """

    def __init__(self, env: Any,
                sites_for_mimic=None,
                joints_for_mimic=None,
                **kwargs):
        """
        Initialize the DeepMimic reward function.

        Args:
            env (Any): Environment instance.
            sites_for_mimic (List[str], optional): List of site names to mimic. Defaults to None, taking all.
            joints_for_mimic (List[str], optional): List of joint names to mimic. Defaults to None, taking all.
            **kwargs (Any): Additional keyword arguments.

        """

        super().__init__(env,sites_for_mimic, joints_for_mimic, **kwargs)

        # reward coefficients
        # DeepMimic
        self._qpos_w_exp = kwargs.get("qpos_w_exp", 10.0)
        self._qvel_w_exp = kwargs.get("qvel_w_exp", 2.0)
        self._rpos_w_exp = kwargs.get("rpos_w_exp", 100.0)
        self._rquat_w_exp = kwargs.get("rquat_w_exp", 10.0)
        self._rvel_w_exp = kwargs.get("rvel_w_exp", 0.1)
        self._qpos_w_sum = kwargs.get("qpos_w_sum", 0.0)
        self._qvel_w_sum = kwargs.get("qvel_w_sum", 0.0)
        self._rpos_w_sum = kwargs.get("rpos_w_sum", 0.5)
        self._rquat_w_sum = kwargs.get("rquat_w_sum", 0.3)
        self._rvel_w_sum = kwargs.get("rvel_w_sum", 0.0)
        

        # Emergence of natural walking
        self._action_threshold = kwargs.get("action_threshold", 0.15)
        self._action_out_of_bounds_coeff = kwargs.get("action_out_of_bounds_coeff", 0.01)
        self._action_coeff = kwargs.get("action_coeff", 0.0)
        self._action_muscle_coeff = kwargs.get("action_muscle_coeff", 0.0)
        self._action_motor_coeff = kwargs.get("action_motor_coeff", 0.0)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 0.0)
        self._torque_at_limit_coeff = kwargs.get("torque_at_limit_coeff", 0.0)
        self._joint_limit_threshold_slide = kwargs.get("joint_limit_threshold_slide", 0.003)
        self._joint_limit_threshold_hinge = kwargs.get("joint_limit_threshold_hinge", 0.05)
        self._grf_threshold = kwargs.get("grf_threshold", 1.4) # arbitrary value for now
        self._grf_coeff = kwargs.get("grf_coeff", 0.0)
        self._joint_torque_vel_arm_coeff = kwargs.get("joint_torque_vel_arm_coeff", 0.0)
        self._torque_at_limit_direction_coeff = kwargs.get("torque_at_limit_direction_coeff", 0.0)
        
        

        # Lateral range coeff
        self._lateral_range_coeff = kwargs.get("lateral_range_coeff", 0.0)
        self._lateral_pos_reward_range = kwargs.get("lateral_pos_reward_range", 0.5)

        # Target velocity and body 
        self._target_velocity = kwargs.get("target_velocity", 1.1)  # m/s
        self._target_body = kwargs.get("target_body", "pelvis")
        self._vel_coeff = kwargs.get("vel_coeff", 1.0)
        self._vel_omega = kwargs.get("vel_omega", 1)

        # Target velocity and body in 0 
        self._target_velocity_z = kwargs.get("target_velocity_z", 0.0)  # m/s
        self._target_body_z = kwargs.get("target_body_z", "pelvis")
        self._vel_coeff_z = kwargs.get("vel_coeff_z", 1.0)
        self._vel_omega_z = kwargs.get("vel_omega_z", 50)



        # Arm 
        arm_joint_names_general = ["arm_flex", "arm_add", "arm_rot", "elbow_flex", "pro_sup", "wrist_flex", "wrist_dev"]
        model = env._model
        self.muscle_actuator_indices = np.where(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
        self.motor_actuator_indices = np.where(model.actuator_dyntype != mujoco.mjtDyn.mjDYN_MUSCLE)[0]
        # jax.debug.print("Muscle actuator indices: {muscle_indices}", muscle_indices=self.muscle_actuator_indices)
        # jax.debug.print("Motor actuator indices: {motor_indices}", motor_indices=self.motor_actuator_indices)

        if not env._disable_arms:
            self._arm_joint_names = [n + "_l" for n in arm_joint_names_general] + [n + "_r" for n in arm_joint_names_general]
            self._arm_joint_ind = [np.array(mj_jntname2qvelid(name, model)) for name in  self._arm_joint_names] #np.array(mj_jntname2qvelid(self._arm_joint_names, model))
            self._arm_joint_qvel_mask = np.zeros(model.nv,dtype=bool)
            self._arm_joint_qvel_mask[self._arm_joint_ind] = True 
        else: 
            self._arm_joint_names = []
            self._arm_joint_ind = []
            self._arm_joint_qvel_mask = np.zeros(model.nv,dtype=bool)

        # self._arm_joint_qvel_mask = np.zeros(model.nv,dtype=bool)
        # self._arm_joint_qvel_mask[self._arm_joint_ind] = True 
        # self._arm_root_joint_qvel_mask[self._free_joint_qvel_ind] = True 

        # rel_site_names = self._info_props["sites_for_mimic"] if sites_for_mimic is None else sites_for_mimic
        # self._rel_site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        #                                for name in rel_site_names])
        # self._rel_body_ids = np.array([model.site_bodyid[site_id] for site_id in self._rel_site_ids])
        
        # get main body name of the environment
        self.main_body_name = self._info_props["upper_body_xml_name"]
        self.main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.main_body_name)
        rel_site_names = self._info_props["sites_for_mimic"] if sites_for_mimic is None else sites_for_mimic
        self._rel_site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                                       for name in rel_site_names])
        self._rel_body_ids = np.array([model.site_bodyid[site_id] for site_id in self._rel_site_ids])

        # determine qpos and qvel indices
        quat_in_qpos = []
        qpos_ind = []
        qvel_ind = []
        for i in range(model.njnt):
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joints_for_mimic is None or jnt_name in joints_for_mimic:
                qposid = mj_jntid2qposid(i, model)
                qvelid = mj_jntid2qvelid(i, model)
                qpos_ind.append(qposid)
                qvel_ind.append(qvelid)
                if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    quat_in_qpos.append(qposid[3:])
        self._qpos_ind = np.concatenate(qpos_ind)
        self._qvel_ind = np.concatenate(qvel_ind)
        if quat_in_qpos:
            quat_in_qpos = np.concatenate(quat_in_qpos)
        self._quat_in_qpos = np.array([True if q in quat_in_qpos else False for q in self._qpos_ind])
        # calc mask for the root free joint velocities
        self._free_joint_qvel_ind = np.array(mj_jntname2qvelid(self._info_props["root_free_joint_xml_name"], model))
        self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)
        self._free_joint_qvel_mask[self._free_joint_qvel_ind] = True




    def init_state(self,
                   env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType):
        """
        Initialize the reward state.

        Args:
            env (Any): The environment instance.
            key (Any): Key for the reward state.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            NaturalRewardState: The initialized reward state.

        """
        return NaturalRewardState(
            last_qvel=data.qvel,
            last_action=backend.zeros(env.info.action_space.shape[0]),
            last_qfrc_actuator=data.qfrc_actuator,
            qfrc_actuator_history=backend.zeros((10, data.qfrc_actuator.shape[0]))
        )

    @check_traj_provided
    def __call__(self,
                state: Union[np.ndarray, jnp.ndarray],
                action: Union[np.ndarray, jnp.ndarray],
                next_state: Union[np.ndarray, jnp.ndarray],
                absorbing: bool,
                info: Dict[str, Any],
                env: Any,
                model: Union[MjModel, Model],
                data: Union[MjData, Data],
                carry: Any,
                backend: ModuleType) -> Tuple[float, Any]:
        """
        Computes a deep mimic tracking reward based on the deviation from the trajectory. The reward is computed as the
        negative exponential of the squared difference between the current state and the trajectory state. The reward
        is computed for the joint positions, joint velocities, relative site positions, relative site orientations, and
        relative site velocities.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.

        Raises:
            ValueError: If trajectory handler is not provided.

        """

        # get current reward state
        reward_state = carry.reward_state

        # # get trajectory data
        traj_data = env.th.traj.data

        # get all quantities from trajectory
        traj_data_single = traj_data.get(carry.traj_state.traj_no, carry.traj_state.subtraj_step_no, backend)
        qpos_traj, qvel_traj = traj_data_single.qpos[self._qpos_ind], traj_data_single.qvel[self._qvel_ind]
        qpos_quat_traj = qpos_traj[self._quat_in_qpos].reshape(-1, 4)
        if len(self._rel_site_ids) > 1:
            site_rpos_traj, site_rangles_traj, site_rvel_traj =\
                calculate_relative_site_quatities(traj_data_single, self._rel_site_ids,
                                                self._rel_body_ids, model.body_rootid, backend)
            
        # get all quantities from the current data
        qpos, qvel = data.qpos[self._qpos_ind], data.qvel[self._qvel_ind]
        qpos_quat = qpos[self._quat_in_qpos].reshape(-1, 4)
        if len(self._rel_site_ids) > 1:
            site_rpos, site_rangles, site_rvel = (
                calculate_relative_site_quatities(data, self._rel_site_ids, self._rel_body_ids,
                                                model.body_rootid, backend))
            
        # # calculate distances
        qpos_dist = backend.mean(backend.square(qpos[~self._quat_in_qpos] - qpos_traj[~self._quat_in_qpos]))
        qpos_dist += backend.mean(quaternion_angular_distance(qpos_quat, qpos_quat_traj, backend))
        qvel_dist = backend.mean(backend.square(qvel - qvel_traj))
        if len(self._rel_site_ids) > 1:
            rpos_dist = backend.mean(backend.square(site_rpos - site_rpos_traj))
            rangles_dist = backend.mean(backend.square(site_rangles - site_rangles_traj))
            rvel_rot_dist = backend.mean(backend.square(site_rvel[:,:3] - site_rvel_traj[:,:3]))
            rvel_lin_dist = backend.mean(backend.square(site_rvel[:,3:] - site_rvel_traj[:,3:]))



        # # calculate rewards
        qpos_reward = backend.exp(-self._qpos_w_exp*qpos_dist)
        qvel_reward = backend.exp(-self._qvel_w_exp*qvel_dist)
        if len(self._rel_site_ids) > 1:
            rpos_reward = backend.exp(-self._rpos_w_exp*rpos_dist)
            rangles_reward = backend.exp(-self._rquat_w_exp*rangles_dist)
            rvel_rot_reward = backend.exp(-self._rvel_w_exp*rvel_rot_dist)
            rvel_lin_reward = backend.exp(-self._rvel_w_exp*rvel_lin_dist)


        # Velocity reward
        target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._target_body)
        # body_vel = data.cvel[target_body_id]  # linear velocity
        # jax.debug.print('body_vel: {body_vel}', body_vel=body_vel)
        # body_vel_rot = data.cvel[target_body_id, :3]  # angular velocity
        body_vel_lin_x = data.cvel[target_body_id, 3]  # linear velocity

        vel_reward = backend.where(
            body_vel_lin_x < self._target_velocity,
            backend.exp(-self._vel_omega * (self._target_velocity - body_vel_lin_x)**2),
            backend.array(1.0),
        )


        # Velocity reward z direction
        target_body_id_z = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._target_body_z)
        body_vel_lin_z = data.cvel[target_body_id_z, 5]  # linear velocity
        vel_reward_z = backend.exp(-self._vel_omega_z * (self._target_velocity_z - body_vel_lin_z)**2)
        # vel_reward_z = backend.where(
        #     body_vel_lin_z == self._target_velocity_z,
        #     backend.exp(-self._vel_omega_z * (self._target_velocity_z - body_vel_lin_z)**2),
        #     backend.array(0),
        # )

        # jax.debug.print('body_vel_lin_z: {body_vel_lin_z}', body_vel_lin_z=body_vel_lin_z)

        # jax.debug.print('cvel pelvis: {vel_reward_z}', vel_reward_z=data.cvel[target_body_id_z,:])
        # jax.debug.print('vel_reward_z: {vel_reward_z}', vel_reward_z=vel_reward_z)

        # jax.lax.cond(body_vel_lin_x < self._target_velocity, 
        #              lambda _: backend.exp(-1.0 * (self._target_velocity - body_vel_lin_x)**2),
        #              lambda _: 1.0,
        #              operand=None)

        # jax.debug.print('vel_reward: {vel_reward}', vel_reward=vel_reward)

        # action rate reward
        if self._action_rate_coeff > 0.0:
            action_rate_norm = backend.sum(backend.square(action - reward_state.last_action))/env.info.action_space.shape[0]
            action_rate_reward = action_rate_norm
        else:
            action_rate_reward = 0.0

        # jax.debug.print('action_rate_reward: {action_rate_reward}', action_rate_reward=action_rate_reward)

        # Action rate reward with original action 
        # Other action penalties below use normalized action with sigmoid to muscle actuators
        muscle_skeleton_control_activation = SkeletonMuscleControlFunction(env)

        # call as an instance so the method receives the correct 'self' and capture updated carry
        _action, carry = muscle_skeleton_control_activation.generate_action(env, action, model, data, carry, backend)

        # calculate costs
        # out of bounds action reward
        if self._action_out_of_bounds_coeff > 0.0:
            """ Get the number of actuators whose activations are above the  threshold 
            Normalized by the total number of actuators
            """
            # fraction of actuators with |action| > threshold
            out_of_bound_count = backend.sum(
                backend.logical_or(_action > self._action_threshold, _action < -self._action_threshold)
            )
            action_out_of_bound_reward = out_of_bound_count / env.info.action_space.shape[0] #float(env.info.action_space.shape[0])

            # backend.sum(backend.where(action > self._action_threshold)[0].shape[0])/ env.info.action_space.shape[0]
        else:
            action_out_of_bound_reward = 0.0

        # jax.debug.print('action_out_of_bound_reward: {action_out_of_bound_reward}', action_out_of_bound_reward=action_out_of_bound_reward)

        
        # Action magnitude reward
        if self._action_coeff > 0.0:
            # split between muscle and motor actuators
            # action_muscle = _action[self.muscle_actuator_indices]
            # action_motor = _action[self.motor_actuator_indices]
            # # # cubic penalty on muscle action magnitude
            # action_reward_muscle = backend.sum(backend.power(backend.abs(action_muscle),3))
            # action_reward_motor = backend.sum(backend.power(backend.abs(action_motor),3))
            action_reward = backend.sum(backend.power(backend.abs(_action),3)) # cubic penalty on action magnitude
        else: 
            action_reward = 0.0
            # action_reward_muscle = 0.0
            # action_reward_motor = 0.0
        # jax.debug.print('action_muscle: {action_reward}', action_reward=action_muscle)
        # jax.debug.print('action_motor: {action_reward}', action_reward=action_motor)
        # jax.debug.print('action_reward_muscle: {action_muscle}', action_muscle=action_reward_muscle)
        # jax.debug.print('action_reward_motor: {action_motor}', action_motor=action_reward_motor)
        # jax.debug.print('action_reward: {action_reward}', action_reward=action_reward)

        if self._action_muscle_coeff > 0.0:
            # split between muscle and motor actuators
            action_muscle = _action[self.muscle_actuator_indices]
            action_reward_muscle = backend.sum(backend.power(backend.abs(action_muscle),3))
        else: 
            action_reward_muscle = 0.0

        # Max 81 

        # jax.debug.print('action_muscle_coeff: {action_muscle_coeff}', action_muscle_coeff=self._action_muscle_coeff)
        # jax.debug.print('action_muscle: {action_muscle}', action_muscle=action_muscle)
        # jax.debug.print('action_reward_muscle: {action_muscle}', action_muscle=action_reward_muscle)

        if self._action_motor_coeff > 0.0:
            # split between muscle and motor actuators
            action_motor = _action[self.motor_actuator_indices]
            action_reward_motor = backend.sum(backend.power(backend.abs(action_motor),3))
        else: 
            action_reward_motor = 0.0

        # jax.debug.print('self._torque_at_limit_coeff: {torque_at_limit_coeff}', torque_at_limit_coeff=self._torque_at_limit_coeff)

        # torque at limit reward
        if self._torque_at_limit_coeff > 0.0:

            # Build a joint-level mask from the qvel-level free-joint mask so we
            # index `model.jnt_range` (one row per joint) correctly. `model.jnt_range`
            # has shape (njnt, 2) while the `_free_joint_qvel_mask` has length `nv`.
            # free_qvel_inds = np.where(self._free_joint_qvel_mask)[0]
            free_jnt_mask = np.zeros(model.njnt, dtype=bool)
            for j in range(model.njnt):
                qvel_ids = np.array(mj_jntid2qvelid(j, model))
                if np.any(np.isin(qvel_ids, self._free_joint_qvel_ind)):
                    free_jnt_mask[j] = True

            # joint_limits: rows for non-free joints
            joint_limits = model.jnt_range[~free_jnt_mask]

            # For joint_positions and joint_torques we extract a representative
            # scalar per joint (the first qpos / first qvel-related actuator index)
            joint_positions = []
            joint_torques = []
            for j in range(model.njnt):
                if free_jnt_mask[j]:
                    continue
                qpos_ids = mj_jntid2qposid(j, model)
                # choose first qpos id as representative
                joint_positions.append(data.qpos[qpos_ids[0]])
                qvel_ids = mj_jntid2qvelid(j, model)
                # choose first qvel id to index actuator/torque arrays
                joint_torques.append(data.qfrc_actuator[qvel_ids[0]])

            joint_positions = backend.array(joint_positions)
            joint_torques = backend.array(joint_torques)

            torque_at_limit = backend.array(0.0)
            for i in range(joint_limits.shape[0]):
                lower_limit, upper_limit = joint_limits[i]
                joint_pos = joint_positions[i]
                joint_torque = joint_torques[i]

                # determine joint index corresponding to this non-free joint
                j = np.where(~free_jnt_mask)[0][i]

                # choose epsilon depending on joint type (sliding joints use tighter epsilon)
                if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE):
                    eps = self._joint_limit_threshold_slide
                else:
                    eps = self._joint_limit_threshold_hinge

                # check if joint is at limit (within small epsilon)
                at_lower_limit = backend.abs(joint_pos - lower_limit) < eps
                at_upper_limit = backend.abs(joint_pos - upper_limit) < eps


                # accumulate torque when joint is at limit and torque pushes against the limit
                torque_at_limit = torque_at_limit + backend.sum(backend.where(
                    backend.logical_and(at_lower_limit, joint_torque < 0),
                    backend.abs(joint_torque), 0.0))
                torque_at_limit = torque_at_limit + backend.sum(backend.where(
                    backend.logical_and(at_upper_limit, joint_torque > 0),
                    backend.abs(joint_torque), 0.0))

            torque_at_limit_reward = torque_at_limit
        else:
            torque_at_limit_reward = 0.0
        # jax.debug.print('torque_at_limit_reward: {torque_at_limit_reward}', torque_at_limit_reward=torque_at_limit_reward)



        if self._torque_at_limit_direction_coeff > 0.0:

            # Build a joint-level mask from the qvel-level free-joint mask so we
            # index `model.jnt_range` (one row per joint) correctly. `model.jnt_range`
            # has shape (njnt, 2) while the `_free_joint_qvel_mask` has length `nv`.
            free_jnt_mask = np.zeros(model.njnt, dtype=bool)
            for j in range(model.njnt):
                qvel_ids = np.array(mj_jntid2qvelid(j, model))
                if np.any(np.isin(qvel_ids, self._free_joint_qvel_ind)):
                    free_jnt_mask[j] = True

            # joint_limits: rows for non-free joints
            joint_limits = model.jnt_range[~free_jnt_mask]

            # For joint_positions and joint_torques we extract a representative
            # scalar per joint (the first qpos / first qvel-related actuator index)
            joint_positions = []
            joint_torques = []
            torque_change_list = []
            for j in range(model.njnt):
                if free_jnt_mask[j]:
                    continue
                qpos_ids = mj_jntid2qposid(j, model)
                # choose first qpos id as representative
                joint_positions.append(data.qpos[qpos_ids[0]])
                qvel_ids = mj_jntid2qvelid(j, model)
                qvel_id = qvel_ids[0]
                # choose first qvel id to index actuator/torque arrays
                joint_torques.append(data.qfrc_actuator[qvel_id])
                # compute smooth torque change: difference between oldest and newest in history
                oldest_torque = reward_state.qfrc_actuator_history[0, qvel_id]
                current_torque = data.qfrc_actuator[qvel_id]
                torque_change_list.append(current_torque - oldest_torque)

            joint_positions = backend.array(joint_positions)
            joint_torques = backend.array(joint_torques)
            torque_changes = backend.array(torque_change_list)

            torque_at_limit = backend.array(0.0)
            for i in range(joint_limits.shape[0]):
                lower_limit, upper_limit = joint_limits[i]
                joint_pos = joint_positions[i]
                joint_torque = joint_torques[i]
                torque_change = torque_changes[i]

                # determine joint index corresponding to this non-free joint
                j = np.where(~free_jnt_mask)[0][i]

                # choose epsilon depending on joint type (sliding joints use tighter epsilon)
                if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE):
                    eps = self._joint_limit_threshold_slide
                else:
                    eps = self._joint_limit_threshold_hinge

                # check if joint is at limit (within small epsilon)
                at_lower_limit = backend.abs(joint_pos - lower_limit) < eps
                at_upper_limit = backend.abs(joint_pos - upper_limit) < eps

                # jax.debug.print('joint_name: {joint_name}, at_lower_limit: {at_lower_limit}, joint_pos {joint_pos}, lower_limit {lower_limit}, margin= {margin}',
                #                 joint_name =mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
                #                 at_lower_limit=at_lower_limit,
                #                 joint_pos=backend.rad2deg(joint_pos),
                #                 lower_limit=backend.rad2deg(lower_limit),
                #                 margin = backend.rad2deg(eps))
                
                # jax.debug.print('joint_name: {joint_name}, at_upper_limit: {at_upper_limit}, joint_pos {joint_pos}, upper_limit {upper_limit}, margin= {margin}',
                #                 joint_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
                #                 at_upper_limit=at_upper_limit,
                #                 joint_pos=backend.rad2deg(joint_pos),
                #                 upper_limit=backend.rad2deg(upper_limit),
                #                 margin = backend.rad2deg(eps))

                # jax.debug.print('Joint {joint_name}: pos={position_deg}, lower_limit={lower_limit}, upper_limit={upper_limit}, torque={torque}, last_torque={last_torque}, torque_change={torque_change}',
                #                 joint_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
                #                 position_deg=backend.rad2deg(joint_pos),
                #                 lower_limit=backend.rad2deg(lower_limit),
                #                 upper_limit=backend.rad2deg(upper_limit),
                #                 torque=joint_torque,
                #                 last_torque=last_joint_torque,
                #                 torque_change=torque_change)

                # accumulate torque change when joint is at limit and torque direction worsens
                # at lower limit: penalize if torque is decreasing (becoming more negative)
                torque_at_limit = torque_at_limit + backend.where(
                    backend.logical_and(at_lower_limit, torque_change > 0),
                    backend.abs(joint_torque), 0.0)
                # at upper limit: penalize if torque is increasing (becoming more positive)
                torque_at_limit = torque_at_limit + backend.where(
                    backend.logical_and(at_upper_limit, torque_change < 0),
                    backend.abs(joint_torque), 0.0)
                
                # jax.debug.print('joint_name: {joint_name}, torque_change: {torque_change}, at_lower_limit: {at_lower_limit}, at_upper_limit: {at_upper_limit}',
                #                 joint_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
                #                 torque_change=torque_change,
                #                 at_lower_limit=at_lower_limit,
                #                 at_upper_limit=at_upper_limit)
            
            # # jax.debug.print the index and joint when the torque at limit direction reward is non-zero
            # condition = backend.logical_or(
            #     backend.where(backend.logical_and(at_upper_limit, torque_change < 0),
            #                   backend.abs(joint_torque), 0.0) > 0,
            #     backend.where(backend.logical_and(at_lower_limit, torque_change > 0),
            #                   backend.abs(joint_torque), 0.0) > 0
            # )

                # condition = backend.logical_or(
                #         backend.logical_and(at_upper_limit, torque_change < 0),
                #         backend.logical_and(at_lower_limit, torque_change > 0)
                #     )
                # jax.lax.cond(
                #         condition,
                #         lambda _: jax.debug.print('torque_at_limit - joint_name={joint_name}, lower_limit={lower_limit}, upper_limit={upper_limit}, torque={torque}, position_deg={position_deg}',
                #                                 joint_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
                #                                 lower_limit=backend.rad2deg(lower_limit),
                #                                 upper_limit=backend.rad2deg(upper_limit),
                #                                 torque=joint_torque,
                #                                 position_deg=backend.rad2deg(joint_pos)),
                #         lambda _: None,
                #         operand=None
                #     )
            torque_at_limit_direction_reward = torque_at_limit
        else:
            torque_at_limit_direction_reward = 0.0

        # jax.debug.print('torque_at_limit_direction_reward: {torque_at_limit_direction_reward}', torque_at_limit_direction_reward=torque_at_limit_direction_reward)


        # grf reward
        if self._grf_coeff > 0.0:
            foot_name = "toes"  # name of the foot box in the model
            calcn_name = "calcn"

            foot_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot_name + "_l")
            foot_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot_name + "_r")
            calcn_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, calcn_name + "_l")
            calcn_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, calcn_name + "_r")

            # use z-component (index 2) of contact force (first three entries are force x,y,z)
            grf_z_l = (backend.abs(data.cfrc_ext[foot_l_id, 5]) + backend.abs(data.cfrc_ext[calcn_l_id, 5])) / (backend.sum(model.body_mass) * 9.81)
            grf_z_r = (backend.abs(data.cfrc_ext[foot_r_id, 5]) + backend.abs(data.cfrc_ext[calcn_r_id, 5])) / (backend.sum(model.body_mass) * 9.81)

            # jax.debug.print('cfrc_ext: {grf_threshold}', grf_threshold=data.cfrc_ext)
            
            # jax.debug.print('grf_z_l: {grf_z_l}', grf_z_l=grf_z_l)
            # jax.debug.print('grf_z_r: {grf_z_r}', grf_z_r=grf_z_r)

            # jax.debug.print('grf_l: {grf_threshold}', grf_threshold=data.cfrc_ext[foot_l_id]+data.cfrc_ext[calcn_l_id])
            # jax.debug.print('grf_r: {grf_threshold}', grf_threshold=data.cfrc_ext[foot_r_id]+data.cfrc_ext[calcn_r_id])
            # reward if grf < threshold
            grf_reward = backend.maximum(0.0, grf_z_l-self._grf_threshold) + backend.maximum(0.0, grf_z_r-self._grf_threshold)
        else:
            grf_reward = 0.0

        # jax.debug.print('grf_reward: {grf_reward}', grf_reward=grf_reward)

        
        if self._lateral_range_coeff>0.0:
            # z_pos_reward_range = 0.3
            lateral_free_pos = data.qpos[1]
            # jax.debug.print('lateral_free_pos: {lateral_free_pos}', lateral_free_pos = lateral_free_pos)

            # give lateral_pos_reward if out of range so not between -range and +range
            lateral_pos_reward = backend.where(backend.logical_or(lateral_free_pos < -self._lateral_pos_reward_range, lateral_free_pos > self._lateral_pos_reward_range), self._lateral_range_coeff, 0.0)
            # jax.debug.print('lateral_pos_reward: {lateral_pos_reward}', lateral_pos_reward=lateral_pos_reward)

            # condition = jnp.logical_and(lateral_free_pos > -self._lateral_pos_reward_range, lateral_free_pos < self._lateral_pos_reward_range)
            # jax.debug.print('condition: {condition}', condition=condition)
            # lateral_pos_reward = jnp.where(condition, self._lateral_range_coeff, 0.0)
            # jax.debug.print('lateral_pos_reward: {lateral_pos_reward}', lateral_pos_reward=lateral_pos_reward)
        else: 
            lateral_pos_reward = 0.0
        
        # jax.debug.print('lateral_pos_reward: {lateral_pos_reward}', lateral_pos_reward=lateral_pos_reward)


        if self._joint_torque_vel_arm_coeff>0.0: 
            # torque_vel_arm_norm = backend.sum(backend.square(data.qfrc_actuator[self._arm_root_joint_qvel_mask]*data.qvel[self._arm_root_joint_qvel_mask]))
            torque_vel_arm_reward = backend.sum(backend.abs(data.qfrc_actuator[self._arm_joint_qvel_mask]*data.qvel[self._arm_joint_qvel_mask]))
            # torque_vel_arm_reward = self._joint_torque_vel_arm_coeff * -torque_vel_arm_norm
            torque_vel_nonarm_reward = backend.sum(backend.abs(data.qfrc_actuator[~self._arm_joint_qvel_mask]*data.qvel[~self._arm_joint_qvel_mask]))
        else: 
            torque_vel_arm_reward = 0.0
            torque_vel_nonarm_reward = 0.0

        # jax.debug.print('arm_joint_vel: {arm_joint_vel}', arm_joint_vel=backend.sum(backend.abs(data.qvel[self._arm_joint_qvel_mask])))
        # jax.debug.print('arm_joint_torque: {arm_joint_torque}', arm_joint_torque=backend.sum(backend.abs(data.qfrc_actuator[self._arm_joint_qvel_mask])))
        # jax.debug.print('torque_vel_arm_reward: {torque_vel_arm_reward}', torque_vel_arm_reward=torque_vel_arm_reward)
        # jax.debug.print('nonarm_joint_vel: {nonarm_joint_vel}', nonarm_joint_vel=backend.sum(backend.abs(data.qvel[~self._arm_joint_qvel_mask])))
        # jax.debug.print('nonarm_joint_torque: {nonarm_joint_torque}', nonarm_joint_torque=backend.sum(backend.abs(data.qfrc_actuator[~self._arm_joint_qvel_mask])))
        # jax.debug.print('torque_vel_nonarm_reward: {torque_vel_nonarm_reward}', torque_vel_nonarm_reward=torque_vel_nonarm_reward)



        total_penalties = self._action_out_of_bounds_coeff*action_out_of_bound_reward + \
                          self._action_muscle_coeff*action_reward_muscle + \
                          self._action_motor_coeff*action_reward_motor + \
                          self._action_coeff*action_reward + \
                          self._action_rate_coeff*action_rate_reward + \
                          self._torque_at_limit_coeff*torque_at_limit_reward + \
                          self._torque_at_limit_direction_coeff*torque_at_limit_direction_reward + \
                          self._grf_coeff*grf_reward + \
                          self._joint_torque_vel_arm_coeff*torque_vel_arm_reward + \
                          lateral_pos_reward #+ \
                          #self._lateral_range_coeff*lateral_pos_reward

        
        # jax.debug.print('action_out_of_bound_reward term: {term}', term=self._action_out_of_bounds_coeff*action_out_of_bound_reward)
        # jax.debug.print('action_reward term: {term}', term=self._action_coeff*action_reward)
        # jax.debug.print('action_rate_reward term: {term}', term=self._action_rate_coeff*action_rate_reward)
        # jax.debug.print('torque_at_limit_reward term with coeff: {term}', term=self._torque_at_limit_coeff*torque_at_limit_reward)
        # jax.debug.print('torque_at_limit_direction_coeff: {term}', term=self._torque_at_limit_direction_coeff*torque_at_limit_direction_reward)
        # jax.debug.print('grf_reward term: {term}', term=self._grf_coeff*grf_reward)
        # # jax.debug.print('lateral_pos_reward term: {term}', term=lateral_pos_reward)
        # jax.debug.print('action_reward_muscle: {term}', term=self._action_muscle_coeff*action_reward_muscle)
        # jax.debug.print('action_reward_motor: {term}', term=self._action_motor_coeff*action_reward_motor)
        # jax.debug.print('torque_vel_arm_reward term: {term}', term=self._joint_torque_vel_arm_coeff*torque_vel_arm_reward)

        # jax.debug.print('total_penalties: {total_penalties}', total_penalties=total_penalties)
        
        # conditional if penalties bigger than 0 then print 
        # jax.lax.cond(
        #     total_penalties > 0.0,
        #     lambda _: jax.debug.print('total_penalties > 0: {total_penalties}', total_penalties=total_penalties),
        #     lambda _: None,
        #     operand=None
        # )



        # calculate total reward from reference tracking rewards
        total_reward = (self._qpos_w_sum * qpos_reward + self._qvel_w_sum * qvel_reward)
        if len(self._rel_site_ids) > 1:
            total_reward = (total_reward
                        + self._rpos_w_sum * rpos_reward + self._rquat_w_sum * rangles_reward
                        + self._rvel_w_sum * rvel_rot_reward + self._rvel_w_sum * rvel_lin_reward)

        # jax.debug.print('total_reward before penalties: {total_reward}', total_reward=total_reward)

        total_reward += self._vel_coeff * vel_reward + \
                          self._vel_coeff_z * vel_reward_z 

        # jax.debug.print('vel_reward term: {term}', term=self._vel_coeff * vel_reward)
        # jax.debug.print('vel_reward_z term: {term}', term=self._vel_coeff_z * vel_reward_z)
        # jax.debug.print('total_reward before penalties: {total_reward}', total_reward=total_reward)

        total_reward = total_reward - total_penalties
        # total_reward = vel_reward - total_penalties
        # jax.debug.print('total_reward after penalties: {total_reward}', total_reward=total_reward)

        # clip to positive values
        total_reward = backend.maximum(total_reward, 0.0)

        # set nan values to 0
        total_reward = backend.nan_to_num(total_reward, nan=0.0)

        # jax.debug.print('total_reward final: {total_reward}', total_reward=total_reward)


        # update reward state with circular buffer for torque history
        qfrc_history = reward_state.qfrc_actuator_history
        qfrc_history = backend.concatenate([qfrc_history[1:], backend.expand_dims(data.qfrc_actuator, axis=0)], axis=0)
        reward_state = reward_state.replace(
            last_qvel=data.qvel,
            last_action=action,
            last_qfrc_actuator=data.qfrc_actuator,
            qfrc_actuator_history=qfrc_history
        )
        carry = carry.replace(reward_state=reward_state)

        return total_reward, carry

        