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
from loco_mujoco.core.utils import mj_jntname2qposid, mj_jntname2qvelid, mj_jntid2qposid, mj_jntid2qvelid, mj_check_collisions
from loco_mujoco.core.utils.math import calculate_relative_site_quatities, quaternion_angular_distance
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast
from loco_mujoco.core.reward.utils import out_of_bounds_action_cost
from loco_mujoco.core.reward.trajectory_based import TrajectoryBasedReward


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
class MimicRewardCrossState:
    """
    State of MimicRewardCross.
    """
    last_qvel: Union[np.ndarray, jnp.ndarray]
    last_action: Union[np.ndarray, jnp.ndarray]
    prev_foot_contact_left_flag: Union[np.ndarray, jax.Array] #jnp.array(False, dtype=bool)
    prev_foot_contact_right_flag: Union[np.ndarray, jax.Array] #jnp.array(False, dtype=bool)
    last_active_touchdown_side: Union[np.ndarray, jax.Array] #jnp.ndarray = struct.field(default_factory=lambda: jnp.array(0, dtype=jnp.int32)) # 0: None, 1: Left, 2: Right


class MimicRewardCross(TrajectoryBasedReward):
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
        model = env._model
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
        self._foot_cross_coeff = kwargs.get("foot_cross_coeff", 0.0)

        self._foot_names = self._info_props["foot_geom_names"]
        print('self._foot_names: ', self._foot_names)
        foot_name = "toes"  # name of the foot box in the model
        # calcn_name = "calcn"    
        self._left_toes_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY, f"{foot_name}_l")
        self._right_toes_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY, f"{foot_name}_r")
        # self._left_calcn_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY, f"{calcn_name}_l")
        # self._right_calcn_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY, f"{calcn_name}_r")

        self._floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in self._foot_names]

        self._left_foot_mask = jnp.array(['_l' in name for name in self._foot_names], dtype=bool)
        self._right_foot_mask = jnp.array(['_r' in name for name in self._foot_names], dtype=bool)



        # assert self._left_toes_id != -1, f"Left toes body '{self._left_toes_id}' not found."
        # assert self._right_toes_id != -1, f"Right toes body '{self._right_toes_id}' not found."
        # assert self._left_calcn_id != -1, f"Left calcn body '{self._left_calcn_id}' not found."
        # assert self._right_calcn_id != -1, f"Right calcn body '{self._right_calcn_id}' not found."


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
        return MimicRewardCrossState(last_qvel=data.qvel, last_action=backend.zeros(env.info.action_space.shape[0]),
                                prev_foot_contact_left_flag=backend.array(False),
                                prev_foot_contact_right_flag=backend.array(False),
                                last_active_touchdown_side= backend.array(0))

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


        if self._foot_cross_coeff > 0.0: 
            contacts_on_ground = backend.zeros(len(self._foot_ids))
            # jax.debug.print('amount of contacts: {ncon}', ncon=len(self._foot_ids))
            for i, id in enumerate(self._foot_ids):
                each_contacts_on_ground = mj_check_collisions(id, self._floor_id, data, backend)
                if backend == np: 
                    contacts_on_ground[i] = each_contacts_on_ground
                else: 
                    contacts_on_ground = contacts_on_ground.at[i].set(each_contacts_on_ground)

            # all_current_contact_right = []
            # all_current_contact_left = []
            # for i in range(len(self._foot_names)):
            #     if '_r' in self._foot_names[i]:
            #         all_current_contact_right.append(contacts_on_ground[i])
            #     elif '_l' in self._foot_names[i]:
            #         all_current_contact_left.append(contacts_on_ground[i])

            current_contact_left_elements = contacts_on_ground[self._left_foot_mask]
            current_contact_right_elements = contacts_on_ground[self._right_foot_mask]

            current_contact_left = backend.any(current_contact_left_elements)
            current_contact_right = backend.any(current_contact_right_elements)

            # current_contact_right = backend.any(backend.array(all_current_contact_right)) #backend.logical_or(contacts_on_ground[0], contacts_on_ground[1])
            # current_contact_left = backend.any(backend.array(all_current_contact_left)) #backend.logical_or(contacts_on_ground[2], contacts_on_ground[3])

            prev_contact_left = reward_state.prev_foot_contact_left_flag #backend.array(self.previous_foot_contact["left"])
            prev_contact_right = reward_state.prev_foot_contact_right_flag

            # jax.debug.print('contacts_on_ground: {contact_l}', contact_l= contacts_on_ground)
            # jax.debug.print('current_contact_right: {contact_l}', contact_l= current_contact_right)
            # jax.debug.print('current_contact_left: {contact_l}', contact_l= current_contact_left)
            # jax.debug.print('prev_contact_left: {contact_l}', contact_l= prev_contact_left)
            # jax.debug.print('prev_contact_right: {contact_l}', contact_l= prev_contact_right)

            # Get x-positions of the feet (using toes for foot position)
            left_foot_x_pos = data.xpos[self._left_toes_id, 0]
            right_foot_x_pos = data.xpos[self._right_toes_id, 0]
            # jax.debug.print('left_foot_x_pos: {contact_l}', contact_l= left_foot_x_pos)
            # jax.debug.print('right_foot_x_pos: {contact_r}', contact_r= right_foot_x_pos)
            # Determine new touch-down events using JAX-compatible logical operations
            left_new_touch_down = backend.logical_and(current_contact_left, backend.logical_not(prev_contact_left))
            right_new_touch_down = backend.logical_and(current_contact_right, backend.logical_not(prev_contact_right))
            
            # jax.debug.print('left_new_touch_down: {foot_crossing_reward}', foot_crossing_reward=left_new_touch_down)
            # jax.debug.print('right_new_touch_down: {foot_crossing_reward}', foot_crossing_reward=right_new_touch_down)


            touchdown_side_numeric = backend.where(
                left_new_touch_down,
                backend.array(1, dtype=backend.int32), # If left_new_touch_down is True
                backend.where( # Else (left_new_touch_down is False), check right_new_touch_down
                    right_new_touch_down,
                    backend.array(2, dtype=backend.int32), # If right_new_touch_down is True
                    backend.array(0, dtype=backend.int32)  # Else (neither left nor right new touch down)
                )
            )

            # jax.debug.print('touchdown_side_numeric: {touchdown_side_numeric}', touchdown_side_numeric=touchdown_side_numeric)


            last_active_touchdown_side = reward_state.last_active_touchdown_side

            # Determine the current touchdown side based on new events
            # This uses nested backend.where to implement the if-elif logic:
            # If left_new_touch_down, it's 1.
            # Else if right_new_touch_down, it's 2.
            # Else, it retains the previous last_active_touchdown_side (meaning no *new* touchdown occurred)
            current_active_touchdown_side = backend.where(
                left_new_touch_down,
                backend.array(1, dtype=backend.int32), # New Left touchdown
                backend.where(
                    right_new_touch_down,
                    backend.array(2, dtype=backend.int32), # New Right touchdown
                    last_active_touchdown_side # No new touchdown, so keep the last recorded one
                )
            )

            reward_for_left_case = right_foot_x_pos - left_foot_x_pos #* self._foot_cross_coeff
            reward_for_right_case = left_foot_x_pos - right_foot_x_pos #* self._foot_cross_coeff
            no_touchdown_reward_val = backend.array(0.0) # Ensure it's a JAX float array


            # jax.debug.print('reward_for_left_case: {reward_for_left_case}', reward_for_left_case=reward_for_left_case)
            # jax.debug.print('reward_for_right_case: {reward_for_left_case}', reward_for_left_case=reward_for_right_case)

            # foot_crossing_reward = backend.where(
            #     touchdown_side_numeric == 1, # Condition for 'left' touchdown (numeric 1)
            #     reward_for_left_case,
            #     backend.where( # Else (not 'left'), check for 'right' touchdown
            #         touchdown_side_numeric == 2, # Condition for 'right' touchdown (numeric 2)
            #         reward_for_right_case,
            #         no_touchdown_reward_val # Else (no active touchdown side)
            #     )
            # )

            foot_crossing_reward = backend.where(
                current_active_touchdown_side == 1, # If the active side is Left
                reward_for_left_case,
                backend.where( # Else, check if it's Right
                    current_active_touchdown_side == 2, # If the active side is Right
                    reward_for_right_case,
                    no_touchdown_reward_val # Else (current_active_touchdown_side is 0, meaning no active side)
                )
            )



            # if left_new_touch_down: 
            #     touchdown_side = 'left'
            # elif right_new_touch_down:
            #     touchdown_side = 'right'

            # if touchdown_side == 'left': 
            #     foot_crossing_reward = right_foot_x_pos - left_foot_x_pos
            # elif touchdown_side == 'right': 
            #     foot_crossing_reward = left_foot_x_pos - right_foot_x_pos



            # # Calculate reward terms conditionally
            # # Use backend.where (JAX's ternary operator) instead of Python `if`
            # reward_for_left_new_touch = backend.where(
            #     left_new_touch_down,
            #     (right_foot_x_pos - left_foot_x_pos),
            #     0.0
            # )
            # reward_for_right_new_touch = backend.where(
            #     right_new_touch_down,
            #     (left_foot_x_pos - right_foot_x_pos),
            #     0.0
            # )

            # jax.debug.print('reward_for_right_new_touch: {foot_crossing_reward}', foot_crossing_reward=reward_for_right_new_touch)
            # jax.debug.print('reward_for_left_new_touch: {foot_crossing_reward}', foot_crossing_reward=reward_for_left_new_touch)

            # # Sum the individual crossing rewards
            # foot_crossing_reward = reward_for_left_new_touch + reward_for_right_new_touch
            
            # jax.debug.print('foot_crossing_reward: {foot_crossing_reward}', foot_crossing_reward=foot_crossing_reward)

            
            foot_crossing_reward = backend.where(
                foot_crossing_reward<0,
                backend.maximum(-1.0, foot_crossing_reward),
                foot_crossing_reward
            )
            # jax.debug.print('foot_crossing_reward after min cut: {foot_crossing_reward}', foot_crossing_reward=foot_crossing_reward)
        else: 
            foot_crossing_reward = 0.0

        # total penality rewards
        total_penalities = (self._action_out_of_bounds_coeff * out_of_bound_reward
                            + self._joint_acc_coeff * acceleration_reward
                            + self._joint_torque_coeff * torque_reward
                            + self._action_rate_coeff * action_rate_reward)
                            # + self._foot_cross_coeff * foot_crossing_reward)
        # jax.debug.print('total_penalities: {total_reward}', total_reward = total_penalities)
        
        total_penalities = backend.maximum(total_penalities, -1.0)
        # jax.debug.print('total_penalities with lim: {total_reward}', total_reward = total_penalities)

        # jax.debug.print('total_penalities: {total_penalities}', total_penalities=total_penalities)

        # calculate total reward
        total_reward = (self._qpos_w_sum * qpos_reward + self._qvel_w_sum * qvel_reward)

        # jax.debug.print('total_reward pos: {total_reward}', total_reward = total_reward)
        if len(self._rel_site_ids) > 1:
            total_reward = (total_reward
                        + self._rpos_w_sum * rpos_reward + self._rquat_w_sum * rangles_reward
                        + self._rvel_w_sum * rvel_rot_reward + self._rvel_w_sum * rvel_lin_reward)
        # jax.debug.print('total_reward: {total_reward}', total_reward=total_reward)

        total_reward = total_reward + total_penalities + self._foot_cross_coeff * foot_crossing_reward

        # jax.debug.print('total_reward total: {total_reward}', total_reward = total_reward)

        
        # clip to positive values
        total_reward = backend.maximum(total_reward, 0.0)


        # set nan values to 0
        total_reward = backend.nan_to_num(total_reward, nan=0.0)

        # jax.debug.print('total_reward final: {total_reward}', total_reward = total_reward)
        

        # update reward state
        reward_state = reward_state.replace(last_qvel=data.qvel, last_action=action, 
                                            prev_foot_contact_left_flag=current_contact_left,
                                            prev_foot_contact_right_flag=current_contact_right,
                                            last_active_touchdown_side=current_active_touchdown_side)
        carry = carry.replace(reward_state=reward_state)

        return total_reward, carry
