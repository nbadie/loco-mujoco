from types import ModuleType
from typing import Any, Dict, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from jax._src.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R
import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.reward.base import Reward
from loco_mujoco.core.utils import mj_jntname2qposid, mj_jntname2qvelid, mj_jntid2qposid, mj_check_collisions
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast
from loco_mujoco.core.reward.utils import out_of_bounds_action_cost


@struct.dataclass
class TargetVelocityCross:
    """
    State of TargetVeloctityGoalRewardCross.
    """
    prev_foot_contact_left_flag: Union[np.ndarray, jax.Array] #jnp.array(False, dtype=bool)
    prev_foot_contact_right_flag: Union[np.ndarray, jax.Array] #jnp.array(False, dtype=bool)
    last_active_touchdown_side: Union[np.ndarray, jax.Array] 


class TargetVelocityGoalRewardCross(Reward):
    """
    Reward function that computes the reward based on the deviation from the goal velocity. The goal velocity is
    provided as an observation in the environment. The reward is computed as the negative exponential of the squared
    difference between the current velocity and the goal velocity. The reward is computed for the x, y, and yaw
    velocities of the root.
    Combine with energy efficiency and foot crossing reward

    """

    def __init__(self, env: Any, tracking_w_exp_xy=10.0, tracking_w_exp_yaw=10.0,
                 tracking_w_sum_xy=1.0, tracking_w_sum_yaw=1.0, joint_torque_coeff = 0.003, **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): The environment instance.
            tracking_w_exp_xy (float, optional): The exponential weight for xy-tracking reward.
            tracking_w_exp_yaw (float, optional): The exponential weight for yaw-tracking reward.
            **kwargs (Any): Additional keyword arguments.

        """

        super().__init__(env, **kwargs)

        self._free_jnt_name = self._info_props["root_free_joint_xml_name"]
        self._vel_idx = np.array(mj_jntname2qvelid(self._free_jnt_name, env._model))
        self._w_exp_xy = tracking_w_exp_xy
        self._w_exp_yaw = tracking_w_exp_yaw
        self._w_sum_xy = tracking_w_sum_xy
        self._w_sum_yaw = tracking_w_sum_yaw
        self._foot_cross_coeff = kwargs.get("foot_cross_coeff", 0.0)
        self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 0.0)
        self._action_out_of_bounds_coeff = kwargs.get("action_out_of_bounds_coeff", 0.01)

        foot_name = "toes"  # name of the foot box in the model
        calcn_name = "calcn"    
        self._left_toes_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY, f"{foot_name}_l")
        self._right_toes_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY, f"{foot_name}_r")
        self._left_calcn_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY, f"{calcn_name}_l")
        self._right_calcn_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY, f"{calcn_name}_r")

        assert self._left_toes_id != -1, f"Left toes body '{self._left_toes_id}' not found."
        assert self._right_toes_id != -1, f"Right toes body '{self._right_toes_id}' not found."
        assert self._left_calcn_id != -1, f"Left calcn body '{self._left_calcn_id}' not found."
        assert self._right_calcn_id != -1, f"Right calcn body '{self._right_calcn_id}' not found."

        # self.previous_foot_contact = {"left": False, "right": False}

        # find the goal velocity observation
        assert "GoalRandomRootVelocity" in env.obs_container or "GoalRootWalk" in env.obs_container, \
            f"GoalRandomRootVelocity is the required goal for the reward for{self.__class__.__name__}"

        super().__init__(env, **kwargs)


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
            LocomotionRewardState: The initialized reward state.

        """

        return TargetVelocityCross(prev_foot_contact_left_flag=backend.array(False),
                                     prev_foot_contact_right_flag=backend.array(False),
                                     last_active_touchdown_side= backend.array(0)) #, zeros(len(self._foot_ids)))


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
        Computes a tracking reward based on the deviation from the goal velocity.Tracking is done on the x, y, and yaw
        velocities of the root.

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
        """
        reward_state = carry.reward_state
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        if hasattr(carry.observation_states,"GoalRandomRootVelocity"):
            goal_state = getattr(carry.observation_states, "GoalRandomRootVelocity")
        elif hasattr(carry.observation_states,"GoalRootWalk"):
            goal_state = getattr(carry.observation_states, "GoalRootWalk")

        # get root orientation
        root_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self._free_jnt_name)

        assert root_jnt_id != -1, f"Joint {self._free_jnt_name} not found in the model."
        root_jnt_qpos_start_id = model.jnt_qposadr[root_jnt_id]
        root_qpos = backend.squeeze(data.qpos[root_jnt_qpos_start_id:root_jnt_qpos_start_id+7])
        root_quat = R.from_quat(quat_scalarfirst2scalarlast(root_qpos[3:7]))

        # get current local vel of root
        lin_vel_global = backend.squeeze(data.qvel[self._vel_idx])[:3]
        ang_vel_global = backend.squeeze(data.qvel[self._vel_idx])[3:]
        lin_vel_local = root_quat.as_matrix().T @ lin_vel_global
        vel_local = backend.concatenate([lin_vel_local[:2], backend.atleast_1d(ang_vel_global[2])]) # construct vel, x, y and yaw

        # calculate tracking reward
        goal_vel = backend.array([goal_state.goal_vel_x, goal_state.goal_vel_y, goal_state.goal_vel_yaw])
        tracking_reward_xy = backend.exp(-self._w_exp_xy * backend.mean(backend.square(vel_local[:2] - goal_vel[:2])))
        tracking_reward_yaw = backend.exp(-self._w_exp_yaw * backend.mean(backend.square(vel_local[2] - goal_vel[2])))
        total_tracking = self._w_sum_xy * tracking_reward_xy + self._w_sum_yaw * tracking_reward_yaw

        # Foot crossing 
        if self._foot_cross_coeff > 0.0: 
            contacts_on_ground = backend.zeros(len(self._foot_ids))
            jax.debug.print('amount of contacts: {ncon}', ncon=len(self._foot_ids))
            for i, id in enumerate(self._foot_ids):
                each_contacts_on_ground = mj_check_collisions(id, self._floor_id, data, backend)
                if backend == np: 
                    contacts_on_ground[i] = each_contacts_on_ground
                else: 
                    contacts_on_ground = contacts_on_ground.at[i].set(each_contacts_on_ground)

            current_contact_left_elements = contacts_on_ground[self._left_foot_mask]
            current_contact_right_elements = contacts_on_ground[self._right_foot_mask]

            current_contact_left = backend.any(current_contact_left_elements)
            current_contact_right = backend.any(current_contact_right_elements)

            prev_contact_left = reward_state.prev_foot_contact_left_flag #backend.array(self.previous_foot_contact["left"])
            prev_contact_right = reward_state.prev_foot_contact_right_flag

            jax.debug.print('contacts_on_ground: {contact_l}', contact_l= contacts_on_ground)
            jax.debug.print('current_contact_right: {contact_l}', contact_l= current_contact_right)
            jax.debug.print('current_contact_left: {contact_l}', contact_l= current_contact_left)
            jax.debug.print('prev_contact_left: {contact_l}', contact_l= prev_contact_left)
            jax.debug.print('prev_contact_right: {contact_l}', contact_l= prev_contact_right)

            # Get x-positions of the feet (using toes for foot position)
            left_foot_x_pos = data.xpos[self._left_toes_id, 0]
            right_foot_x_pos = data.xpos[self._right_toes_id, 0]
            jax.debug.print('left_foot_x_pos: {contact_l}', contact_l= left_foot_x_pos)
            jax.debug.print('right_foot_x_pos: {contact_r}', contact_r= right_foot_x_pos)
            # Determine new touch-down events using JAX-compatible logical operations
            left_new_touch_down = backend.logical_and(current_contact_left, backend.logical_not(prev_contact_left))
            right_new_touch_down = backend.logical_and(current_contact_right, backend.logical_not(prev_contact_right))
            
            jax.debug.print('left_new_touch_down: {foot_crossing_reward}', foot_crossing_reward=left_new_touch_down)
            jax.debug.print('right_new_touch_down: {foot_crossing_reward}', foot_crossing_reward=right_new_touch_down)


            touchdown_side_numeric = backend.where(
                left_new_touch_down,
                backend.array(1, dtype=backend.int32), # If left_new_touch_down is True
                backend.where( # Else (left_new_touch_down is False), check right_new_touch_down
                    right_new_touch_down,
                    backend.array(2, dtype=backend.int32), # If right_new_touch_down is True
                    backend.array(0, dtype=backend.int32)  # Else (neither left nor right new touch down)
                )
            )

            jax.debug.print('touchdown_side_numeric: {touchdown_side_numeric}', touchdown_side_numeric=touchdown_side_numeric)


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


            jax.debug.print('reward_for_left_case: {reward_for_left_case}', reward_for_left_case=reward_for_left_case)
            jax.debug.print('reward_for_right_case: {reward_for_left_case}', reward_for_left_case=reward_for_right_case)


            foot_crossing_reward = backend.where(
                current_active_touchdown_side == 1, # If the active side is Left
                reward_for_left_case,
                backend.where( # Else, check if it's Right
                    current_active_touchdown_side == 2, # If the active side is Right
                    reward_for_right_case,
                    no_touchdown_reward_val # Else (current_active_touchdown_side is 0, meaning no active side)
                )
            )

            jax.debug.print('foot_crossing_reward: {foot_crossing_reward}', foot_crossing_reward=foot_crossing_reward)

            
            foot_crossing_reward = backend.where(
                foot_crossing_reward<0,
                backend.maximum(-1.0, foot_crossing_reward),
                foot_crossing_reward
            )
            jax.debug.print('foot_crossing_reward after min cut: {foot_crossing_reward}', foot_crossing_reward=foot_crossing_reward)
        else: 
            foot_crossing_reward = 0.0

        # joint torque reward
        if self._joint_torque_coeff > 0.0:
            torque_norm = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]))
            torque_reward = self._joint_torque_coeff * -torque_norm
            # jax.debug.print('torque_norm: {torque_norm}', torque_norm=torque_norm)  
        else:
            torque_reward = 0.0


        # out of bounds action cost
        if self._action_out_of_bounds_coeff > 0.0:
            out_of_bound_reward = -out_of_bounds_action_cost(action, lower_bound=env.mdp_info.action_space.low,
                                                             upper_bound=env.mdp_info.action_space.high, backend=backend)
        else:
            out_of_bound_reward = 0.0

        # total penality rewards
        total_penalities = (self._action_out_of_bounds_coeff * out_of_bound_reward
                            + self._joint_torque_coeff * torque_reward)
        total_penalities = backend.maximum(total_penalities, -1.0)

        total_reward = total_tracking + foot_crossing_reward + total_penalities

        reward_state = reward_state.replace(prev_foot_contact_left_flag=current_contact_left,
            prev_foot_contact_right_flag=current_contact_right,
            last_active_touchdown_side=current_active_touchdown_side)

        carry = carry.replace(reward_state = reward_state)

        return total_reward, carry
