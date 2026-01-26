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
from loco_mujoco.core.utils import mj_jntname2qposid, mj_jntname2qvelid, mj_jntid2qposid, mj_jntid2qvelid
from loco_mujoco.core.utils.math import calculate_relative_site_quatities, quaternion_angular_distance
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast
from loco_mujoco.core.reward.utils import out_of_bounds_action_cost
from loco_mujoco.core.reward import MimicReward


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
class MimicRewardState:
    """
    State of MimicReward.
    """
    last_qvel: Union[np.ndarray, jnp.ndarray]
    last_action: Union[np.ndarray, jnp.ndarray]


class MimicRewardVel(MimicReward):
    """
    DeepMimic reward function that computes the reward based on the deviation from the trajectory. The reward is
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

        super().__init__(env, **kwargs)

        # reward coefficients
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
        self._action_out_of_bounds_coeff = kwargs.get("action_out_of_bounds_coeff", 0.01)
        self._joint_acc_coeff = kwargs.get("joint_acc_coeff", 0.0)
        self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 0.0)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 0.0)
        self._joint_torque_vel_coeff = kwargs.get("joint_torque_vel_coeff", 0.0)

        # get main body name of the environment
        self.main_body_name = self._info_props["upper_body_xml_name"]
        model = env._model
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
        quat_in_qpos = np.concatenate(quat_in_qpos)
        self._quat_in_qpos = np.array([True if q in quat_in_qpos else False for q in self._qpos_ind])

        # calc mask for the root free joint velocities
        self._free_joint_qvel_ind = np.array(mj_jntname2qvelid(self._info_props["root_free_joint_xml_name"], model))
        self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)
        self._free_joint_qvel_mask[self._free_joint_qvel_ind] = True

    def init_state(self, env: Any,
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
            MimicRewardState: The initialized reward state.

        """
        return MimicRewardState(last_qvel=data.qvel, last_action=backend.zeros(env.info.action_space.shape[0]))

    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType):
        """
        Reset the reward state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated data and carry.

        """
        reward_state = self.init_state(env, None, model, data, backend)
        carry = carry.replace(reward_state=reward_state)
        return data, carry

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

        # get trajectory data
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

        # calculate distances
        qpos_dist = backend.mean(backend.square(qpos[~self._quat_in_qpos] - qpos_traj[~self._quat_in_qpos]))
        qpos_dist += backend.mean(quaternion_angular_distance(qpos_quat, qpos_quat_traj, backend))
        qvel_dist = backend.mean(backend.square(qvel - qvel_traj))
        if len(self._rel_site_ids) > 1:
            rpos_dist = backend.mean(backend.square(site_rpos - site_rpos_traj))
            rangles_dist = backend.mean(backend.square(site_rangles - site_rangles_traj))
            rvel_rot_dist = backend.mean(backend.square(site_rvel[:,:3] - site_rvel_traj[:,:3]))
            rvel_lin_dist = backend.mean(backend.square(site_rvel[:,3:] - site_rvel_traj[:,3:]))

        # calculate rewards
        qpos_reward = backend.exp(-self._qpos_w_exp*qpos_dist)
        qvel_reward = backend.exp(-self._qvel_w_exp*qvel_dist)
        if len(self._rel_site_ids) > 1:
            rpos_reward = backend.exp(-self._rpos_w_exp*rpos_dist)
            rangles_reward = backend.exp(-self._rquat_w_exp*rangles_dist)
            rvel_rot_reward = backend.exp(-self._rvel_w_exp*rvel_rot_dist)
            rvel_lin_reward = backend.exp(-self._rvel_w_exp*rvel_lin_dist)

        # calculate costs
        # out of bounds action cost
        if self._action_out_of_bounds_coeff > 0.0:
            out_of_bound_reward = -out_of_bounds_action_cost(action, lower_bound=env.mdp_info.action_space.low,
                                                             upper_bound=env.mdp_info.action_space.high, backend=backend)
        else:
            out_of_bound_reward = 0.0
        # jax.debug.print("out_of_bound_reward: {out_of_bound_reward}", out_of_bound_reward=out_of_bound_reward)
        # joint acceleration reward
        if self._joint_acc_coeff > 0.0:
            last_joint_vel = reward_state.last_qvel[~self._free_joint_qvel_mask]
            joint_vel = data.qvel[~self._free_joint_qvel_mask]
            acceleration_norm = backend.sum(backend.square(joint_vel - last_joint_vel) / env.dt)
            acceleration_reward = self._joint_acc_coeff * -acceleration_norm
        else:
            acceleration_reward = 0.0


        # joint torque vel reward 
        if self._joint_torque_vel_coeff>0.0: 
            torque_vel_norm = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]*data.qvel[~self._free_joint_qvel_mask]))
            torque_vel_reward = self._joint_torque_vel_coeff* - torque_vel_norm
        else: 
            torque_vel_reward = 0.0

        # jax.debug.print('torque_vel_norm: {torque_vel_norm}', torque_vel_norm=torque_vel_norm)
        # jax.debug.print('torque_vel_reward after 1st scale: {torque_vel_reward}', torque_vel_reward=torque_vel_reward)
        # jax.debug.print('torque_vel_reward after 2nd scale: {torque_vel_reward}', torque_vel_reward=self._joint_torque_vel_coeff * torque_vel_reward)

        # joint torque reward
        if self._joint_torque_coeff > 0.0:
            torque_norm = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]))
            torque_reward = self._joint_torque_coeff * -torque_norm
            # jax.debug.print('torque_norm: {torque_norm}', torque_norm=torque_norm)  
        else:
            torque_reward = 0.0
        # jax.debug.print('Torque_reward: {torque_reward}', torque_reward=torque_reward)
        # action rate reward
        if self._action_rate_coeff > 0.0:
            action_rate_norm = backend.sum(backend.square(action - reward_state.last_action))
            action_rate_reward = self._action_rate_coeff * -action_rate_norm
        else:
            action_rate_reward = 0.0

        # total penality rewards
        total_penalities = (self._action_out_of_bounds_coeff * out_of_bound_reward
                            + self._joint_acc_coeff * acceleration_reward
                            + self._joint_torque_coeff * torque_reward
                            + self._action_rate_coeff * action_rate_reward
                            + self._joint_torque_vel_coeff * torque_vel_reward)
        # jax.debug.print('total_penalities before clip: {total_penalities}', total_penalities=total_penalities)
        
        total_penalities = backend.maximum(total_penalities, -1.0)
        # jax.debug.print('total_penalities with clip: {total_penalities}', total_penalities=total_penalities)

        # calculate total reward
        total_reward = (self._qpos_w_sum * qpos_reward + self._qvel_w_sum * qvel_reward)
        if len(self._rel_site_ids) > 1:
            total_reward = (total_reward
                        + self._rpos_w_sum * rpos_reward + self._rquat_w_sum * rangles_reward
                        + self._rvel_w_sum * rvel_rot_reward + self._rvel_w_sum * rvel_lin_reward)
        # jax.debug.print('total_reward: {total_reward}', total_reward=total_reward)

        total_reward = total_reward + total_penalities
        
        # clip to positive values
        total_reward = backend.maximum(total_reward, 0.0)

        # set nan values to 0
        total_reward = backend.nan_to_num(total_reward, nan=0.0)

        # update reward state
        reward_state = reward_state.replace(last_qvel=data.qvel, last_action=action)
        carry = carry.replace(reward_state=reward_state)

        return total_reward, carry





class MimicRewardVelArm(MimicReward):
    """
    !!!!!HARD CODED which joints to use!!!!!!!!!!!!
    DeepMimic reward function that computes the reward based on the deviation from the trajectory. The reward is
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

        super().__init__(env, **kwargs)

        # reward coefficients
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
        self._action_out_of_bounds_coeff = kwargs.get("action_out_of_bounds_coeff", 0.01)
        self._joint_acc_coeff = kwargs.get("joint_acc_coeff", 0.0)
        # self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 0.0)
        self._joint_torque_arm_coeff = kwargs.get("joint_torque_arm_coeff", 0.0)
        self._joint_torque_nonarm_coeff = kwargs.get("joint_torque_nonarm_coeff", 0.0)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 0.0)
        # self._joint_torque_vel_coeff = kwargs.get("joint_torque_vel_coeff", 0.0)
        self._joint_torque_vel_arm_coeff = kwargs.get("joint_torque_vel_arm_coeff", 0.0)
        self._joint_torque_vel_nonarm_coeff = kwargs.get("joint_torque_vel_nonarm_coeff", 0.0)
        self._action_coeff = kwargs.get("action_coeff", 0.0)
        self._lateral_range_coeff = kwargs.get("lateral_range_coeff", 0.0)
        self._lateral_pos_reward_range = kwargs.get("lateral_pos_reward_range",0.1)
        # joints_for_mimic = kwargs.get("joints_for_mimic", None) # For defining specific joints to mimic 


        # get main body name of the environment
        self.main_body_name = self._info_props["upper_body_xml_name"]
        model = env._model
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

        # Find arm joints 
        arm_joint_names_general = ["arm_flex", "arm_add", "arm_rot", "elbow_flex", "pro_sup", "wrist_flex", "wrist_dev"]

        self._arm_joint_names = [n + "_l" for n in arm_joint_names_general] + [n + "_r" for n in arm_joint_names_general]
        self._arm_joint_ind = [np.array(mj_jntname2qvelid(name, model)) for name in  self._arm_joint_names] #np.array(mj_jntname2qvelid(self._arm_joint_names, model))
        self._arm_joint_qvel_mask = np.zeros(model.nv,dtype=bool)
        self._arm_joint_qvel_mask[self._arm_joint_ind] = True 

        self._arm_root_joint_qvel_mask = np.zeros(model.nv,dtype=bool)
        self._arm_root_joint_qvel_mask[self._arm_joint_ind] = True 
        self._arm_root_joint_qvel_mask[self._free_joint_qvel_ind] = True 
        

    def init_state(self, env: Any,
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
            MimicRewardState: The initialized reward state.

        """
        return MimicRewardState(last_qvel=data.qvel, last_action=backend.zeros(env.info.action_space.shape[0]))

    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType):
        """
        Reset the reward state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated data and carry.

        """
        reward_state = self.init_state(env, None, model, data, backend)
        carry = carry.replace(reward_state=reward_state)
        return data, carry

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

        # get trajectory data
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
            
        # jax.debug.print("qpos: {qpos}", qpos=data.qpos[:5])
        # jax.debug.print("qpos_traj: {qpos_traj}", qpos_traj=traj_data_single.qpos[:5])
        # jax.debug.print("qvel: {qvel}", qvel=data.qvel[:5])
        # jax.debug.print("qvel_traj: {qvel_traj}", qvel_traj=traj_data_single.qvel[:5])

        # calculate distances
        qpos_dist = backend.mean(backend.square(qpos[~self._quat_in_qpos] - qpos_traj[~self._quat_in_qpos]))
        qpos_dist += backend.mean(quaternion_angular_distance(qpos_quat, qpos_quat_traj, backend))
        qvel_dist = backend.mean(backend.square(qvel - qvel_traj))
        if len(self._rel_site_ids) > 1:
            rpos_dist = backend.mean(backend.square(site_rpos - site_rpos_traj))
            rangles_dist = backend.mean(backend.square(site_rangles - site_rangles_traj))
            rvel_rot_dist = backend.mean(backend.square(site_rvel[:,:3] - site_rvel_traj[:,:3]))
            rvel_lin_dist = backend.mean(backend.square(site_rvel[:,3:] - site_rvel_traj[:,3:]))

        # calculate rewards
        qpos_reward = backend.exp(-self._qpos_w_exp*qpos_dist)
        qvel_reward = backend.exp(-self._qvel_w_exp*qvel_dist)
        if len(self._rel_site_ids) > 1:
            rpos_reward = backend.exp(-self._rpos_w_exp*rpos_dist)
            rangles_reward = backend.exp(-self._rquat_w_exp*rangles_dist)
            rvel_rot_reward = backend.exp(-self._rvel_w_exp*rvel_rot_dist)
            rvel_lin_reward = backend.exp(-self._rvel_w_exp*rvel_lin_dist)

        # calculate costs
        # out of bounds action cost
        if self._action_out_of_bounds_coeff > 0.0:
            out_of_bound_reward = -out_of_bounds_action_cost(action, lower_bound=env.mdp_info.action_space.low,
                                                             upper_bound=env.mdp_info.action_space.high, backend=backend)
        else:
            out_of_bound_reward = 0.0
        
        # joint acceleration reward
        if self._joint_acc_coeff > 0.0:
            last_joint_vel = reward_state.last_qvel[~self._free_joint_qvel_mask]
            joint_vel = data.qvel[~self._free_joint_qvel_mask]
            acceleration_norm = backend.sum(backend.square(joint_vel - last_joint_vel) / env.dt)
            acceleration_reward = self._joint_acc_coeff * -acceleration_norm
        else:
            acceleration_reward = 0.0
        # jax.debug.print("acceleration_reward: {acceleration_reward}", acceleration_reward=acceleration_reward)

        # joint torque reward arm
        if self._joint_torque_vel_arm_coeff>0.0: 
            # torque_vel_arm_norm = backend.sum(backend.square(data.qfrc_actuator[self._arm_root_joint_qvel_mask]*data.qvel[self._arm_root_joint_qvel_mask]))
            torque_vel_arm_reward = -1*(backend.sum(backend.abs(data.qfrc_actuator[self._arm_root_joint_qvel_mask]*data.qvel[self._arm_root_joint_qvel_mask])))
            # torque_vel_arm_reward = self._joint_torque_vel_arm_coeff * -torque_vel_arm_norm
        else: 
            torque_vel_arm_reward = 0.0
        # jax.debug.print('torque_vel_arm_reward: {torque_vel_norm}', torque_vel_norm=torque_vel_arm_reward)

        # joint_torque reward without arm 
        if self._joint_torque_vel_nonarm_coeff>0.0: 
            # torque_vel_nonarm_norm = backend.sum(backend.square(data.qfrc_actuator[~self._arm_root_joint_qvel_mask]*data.qvel[~self._arm_root_joint_qvel_mask]))
            torque_vel_nonarm_reward = -1*(backend.sum(backend.abs(data.qfrc_actuator[~self._arm_root_joint_qvel_mask]*data.qvel[~self._arm_root_joint_qvel_mask])))
            # torque_vel_nonarm_reward = self._joint_torque_vel_nonarm_coeff * -torque_vel_nonarm_norm
        else: 
            torque_vel_nonarm_reward = 0.0
        # jax.debug.print('torque_vel_nonarm_reward: {torque_vel_norm}', torque_vel_norm=torque_vel_nonarm_reward)


        # # joint torque vel reward 
        # if self._joint_torque_vel_coeff>0.0: 
        #     torque_vel_norm = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]*data.qvel[~self._free_joint_qvel_mask]))
        #     torque_vel_reward = self._joint_torque_vel_coeff* - torque_vel_norm
        # else: 
        #     torque_vel_reward = 0.0

        # jax.debug.print('torque_vel_norm: {torque_vel_norm}', torque_vel_norm=torque_vel_norm)
        # jax.debug.print('torque_vel_reward after 1st scale: {torque_vel_reward}', torque_vel_reward=torque_vel_reward)
        # jax.debug.print('torque_vel_reward after 2nd scale: {torque_vel_reward}', torque_vel_reward=self._joint_torque_vel_coeff * torque_vel_reward)

        # # joint torque reward
        # if self._joint_torque_coeff > 0.0:
        #     torque_norm = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]))
        #     torque_reward = self._joint_torque_coeff * -torque_norm
        #     # jax.debug.print('torque_norm: {torque_norm}', torque_norm=torque_norm)  
        # else:
        #     torque_reward = 0.0


        # joint torque reward arm
        if self._joint_torque_arm_coeff>0.0: 
            torque_arm_norm = backend.sum(backend.square(data.qfrc_actuator[self._arm_root_joint_qvel_mask]))
            torque_arm_reward = self._joint_torque_arm_coeff * -torque_arm_norm
        else: 
            torque_arm_reward = 0.0
        # jax.debug.print('torque_arm_reward: {torque_vel_norm}', torque_vel_norm=torque_arm_reward)

        # joint_torque reward without arm 
        if self._joint_torque_nonarm_coeff>0.0: 
            torque_nonarm_norm = backend.sum(backend.square(data.qfrc_actuator[~self._arm_root_joint_qvel_mask]))
            torque_nonarm_reward = self._joint_torque_nonarm_coeff * -torque_nonarm_norm
        else: 
            torque_nonarm_reward = 0.0
        # jax.debug.print('torque_nonarm_reward: {torque_vel_norm}', torque_vel_norm=torque_nonarm_reward)


        # jax.debug.print('Torque_reward: {torque_reward}', torque_reward=torque_reward)
        # action rate reward
        if self._action_rate_coeff > 0.0:
            action_rate_norm = backend.sum(backend.square(action - reward_state.last_action))
            action_rate_reward = self._action_rate_coeff * -action_rate_norm
        else:
            action_rate_reward = 0.0
        # jax.debug.print('action_rate_reward: {torque_vel_norm}', torque_vel_norm=self._action_rate_coeff * action_rate_reward)

        if self._action_coeff > 0.0:
            action_norm = backend.sum(backend.square(action))
            action_reward = -action_norm #self._action_coeff * -action_norm
        else: 
            action_reward = 0.0
        # jax.debug.print('action_reward: {torque_vel_norm}', torque_vel_norm=self._action_coeff *action_reward)
        

        # total penality rewards
        total_penalities = (self._action_out_of_bounds_coeff * out_of_bound_reward
                            + self._joint_acc_coeff * acceleration_reward
                            + self._joint_torque_arm_coeff * torque_arm_reward
                            + self._joint_torque_nonarm_coeff * torque_nonarm_reward
                            # + self._joint_torque_coeff * torque_reward
                            + self._action_rate_coeff * action_rate_reward
                            + self._action_coeff * action_reward
                            + self._joint_torque_vel_arm_coeff * torque_vel_arm_reward
                            + self._joint_torque_vel_nonarm_coeff * torque_vel_nonarm_reward)
                            # + self._joint_torque_vel_coeff * torque_vel_reward)
        # jax.debug.print('total_penalities before clip: {total_penalities}', total_penalities=total_penalities)
        
        # jax.debug.print('action out of bounds: {total_penalities}', total_penalities = self._action_out_of_bounds_coeff * out_of_bound_reward)
        # total_penalities = backend.maximum(total_penalities, -1.0)
        total_penalities = backend.maximum(total_penalities, -7.0)
        # jax.debug.print('total_penalities with clip: {total_penalities}', total_penalities=total_penalities)

        # calculate total reward
        total_reward = (self._qpos_w_sum * qpos_reward + self._qvel_w_sum * qvel_reward)
        if len(self._rel_site_ids) > 1:
            total_reward = (total_reward
                        + self._rpos_w_sum * rpos_reward + self._rquat_w_sum * rangles_reward
                        + self._rvel_w_sum * rvel_rot_reward + self._rvel_w_sum * rvel_lin_reward)
        # jax.debug.print('total_reward: {total_reward}', total_reward=total_reward)

        # if total_reward + total_penalities < 0.0:
        #     jax.debug.print("NEGATIVE REWARD OCCRED")


        if self._lateral_range_coeff>0.0:
            # z_pos_reward_range = 0.3
            lateral_free_pos = data.qpos[1]
            # jax.debug.print('lateral_free_pos: {lateral_free_pos}', lateral_free_pos = lateral_free_pos)

            # give lateral_pos_reward if out of range so not between -range and +range
            lateral_pos_reward = jnp.where(jnp.logical_or(lateral_free_pos < -self._lateral_pos_reward_range, lateral_free_pos > self._lateral_pos_reward_range), self._lateral_range_coeff, 0.0)
            # jax.debug.print('lateral_pos_reward: {lateral_pos_reward}', lateral_pos_reward=lateral_pos_reward)

            # condition = jnp.logical_and(lateral_free_pos > -self._lateral_pos_reward_range, lateral_free_pos < self._lateral_pos_reward_range)
            # jax.debug.print('condition: {condition}', condition=condition)
            # lateral_pos_reward = jnp.where(condition, self._lateral_range_coeff, 0.0)
            # jax.debug.print('lateral_pos_reward: {lateral_pos_reward}', lateral_pos_reward=lateral_pos_reward)
        else: 
            lateral_pos_reward = 0.0

        # jax.debug.print('z_pos_reward: {total_reward}', total_reward=lateral_pos_reward)

        total_reward = total_reward + total_penalities - lateral_pos_reward

        # jax.debug.print('total_reward - pen: {total_reward}', total_reward=total_reward)
# 
        # avoid Python boolean conversion of JAX tracers; use jax.lax.cond to run debug print only when predicate true
        # jax.lax.cond(total_reward < 0.0,
        #              lambda _: jax.debug.print("NEGATIVE REWARD OCCURED"),
        #              lambda _: None,
        #              None)

        
        # clip to positive values
        total_reward = backend.maximum(total_reward, 0.0)

        # set nan values to 0
        total_reward = backend.nan_to_num(total_reward, nan=0.0)

        # jax.debug.print('total_reward final: {total_reward}', total_reward=total_reward)


        # update reward state
        reward_state = reward_state.replace(last_qvel=data.qvel, last_action=action)
        carry = carry.replace(reward_state=reward_state)

        return total_reward, carry
    



            


