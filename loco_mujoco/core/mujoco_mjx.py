from typing import Any, Dict, Tuple
from types import ModuleType

import mujoco
from mujoco import mjx
from mujoco.mjx import Model, Data
from flax import struct
import numpy as np
import jax
import jax.numpy as jnp

from loco_mujoco.core.mujoco_base import Mujoco, AdditionalCarry
from loco_mujoco.core.visuals import MujocoViewer
from loco_mujoco.trajectory import TrajectoryData


@struct.dataclass
class MjxAdditionalCarry(AdditionalCarry):
    """
    Additional carry for the Mjx environment.

    """
    final_observation: jax.Array
    final_info: Dict[str, Any]


@struct.dataclass
class MjxState:
    """
    State of the Mjx environment.

    Args:
        data (Data): Mjx data structure.
        observation (jax.Array): Observation of the environment.
        reward (float): Reward of the environment.
        absorbing (bool): Whether the state is absorbing.
        done (bool): Whether the episode is done.
        additional_carry (Any): Additional carry information.
        info (Dict[str, Any]): Information dictionary.

    """
    data: Data
    observation: jax.Array
    reward: float
    absorbing: bool
    done: bool
    additional_carry: MjxAdditionalCarry
    info: Dict[str, Any] = struct.field(default_factory=dict)


class Mjx(Mujoco):
    """
    Base class for Mujoco environments using JAX.

    Args:
        n_envs (int): Number of environments to run in parallel.
        **kwargs: Additional arguments to pass to the Mujoco base class.

    """

    def __init__(self, **kwargs):

        # call base mujoco env
        super().__init__(**kwargs)

        # add information to mdp_info
        self._mdp_info.mjx_env = True

        # setup mjx model and data
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        self.sys = mjx.put_model(self._model)
        data = mjx.put_data(self._model, self._data)
        self._first_data = mjx.forward(self.sys, data)
        self.init_talus_pos = 0
        
        #if hasattr(self,'socket_ty_slack') and self.socket_ty_slack: 
        # Socket_ty hysterisis
        high_stiffness = 43500 #70000 #30000 #43500
        low_stiffness = 4350 #6000 #7000 #4350 #1000
        a = 0.038 #0.02 #0.038
        H = 0.025
        delta_shift =  0 #0.01
        self.xp = jnp.array([-a+delta_shift, 0+delta_shift, H+delta_shift, H+a+delta_shift])
        self.fp = jnp.array([-high_stiffness*(a+delta_shift), 0+delta_shift, low_stiffness*(H+delta_shift), low_stiffness*(H+delta_shift)+high_stiffness*(a+delta_shift)])
        # self.f_kp = jnp.interp(x,xp=[-a, 0, H, H+a], fp=[-high_stiffness*a, 0, low_stiffness*H, low_stiffness*H+high_stiffness*a], left="extrapolate", right="extrapolate") 
        #else: 
            #self.socket_ty_slack = False

    def mjx_reset(self, key: jax.random.PRNGKey) -> MjxState:
        """
        Resets the environment.

        Args:
            key (jax.random.PRNGKey): Random key for the reset.

        Returns:
            MjxState: The reset state of the environment.

        """

        key, subkey = jax.random.split(key)

        # reset data
        data = self._first_data

        carry = self._init_additional_carry(key, self._model, data, jnp)

        data, carry = self._mjx_reset_carry(self.sys, data, carry)

        # reset all stateful entities
        data, carry = self.obs_container.reset_state(self, self._model, data, carry, jnp)

        obs, carry = self._mjx_create_observation(self._model, data, carry)
        reward = 0.0
        absorbing = jnp.array(False, dtype=bool)
        done = jnp.array(False, dtype=bool)
        info = self._mjx_reset_info_dictionary(obs, data, subkey)

        return MjxState(data=data, observation=obs, reward=reward, absorbing=absorbing, done=done,
                        info=info, additional_carry=carry)

    def _mjx_reset_in_step(self, state: MjxState) -> MjxState:
        """
        Resets the environment if the episode is done. This function is called in the step function for asynchronous
        resetting of the environments.

        Args:
            state (MjxState): Current state of the environment.

        Returns:
            MjxState: The reset state of the environment.

        """

        carry = state.additional_carry

        # reset data
        data = self._first_data

        data, carry = self._mjx_reset_carry(self.sys, data, carry)

        # reset carry
        carry = carry.replace(cur_step_in_episode=1,
                              final_observation=state.observation,
                              last_action=jnp.zeros_like(carry.last_action),
                              final_info=state.info)

        # update all stateful entities
        data, carry = self.obs_container.reset_state(self, self._model, data, carry, jnp)

        # create new observation
        obs, carry = self._mjx_create_observation(self._model, data, carry)

        return state.replace(data=data, observation=obs, additional_carry=carry)

    def mjx_step(self, state: MjxState, action: jax.Array) -> MjxState:
        """

        Args:
            state (MjxState): Current state of the environment.
            action (jax.Array): Action to take in the environment.

        Returns:
            MjxState: The next state of the environment.

        """

        data = state.data
        cur_info = state.info
        carry = state.additional_carry
        carry = carry.replace(last_action=action)

        # reset dones
        state = state.replace(done=jnp.zeros_like(state.done, dtype=bool))

        # preprocess action
        processed_action, carry = self._mjx_preprocess_action(action, self._model, data, carry)

        # modify data and model *before* step if needed
        sys, data, carry = self._mjx_simulation_pre_step(self.sys, data, carry)

        def _adapt_qfrc_applied(_data):
            #Adapt qfrc_applied for socket_ty slack during swing phase 
            joint_id= mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'socket_ty'+self.prosthesis_side)
            # jax.debug.print("joint_id: {joint_id}", joint_id = joint_id)
            qpos_address = self._model.jnt_qposadr[joint_id]
            # jax.debug.print("qpos_address: {qpos_address}", qpos_address = qpos_address)
            pistoning_qpos_adr = qpos_address
            pistoning_dof_adr = self._model.jnt_dofadr[joint_id]
            # jax.debug.print("pistoning_dof_adr: {pistoning_dof_adr}", pistoning_dof_adr = pistoning_dof_adr)
            f_kp = jnp.interp(data.qpos[pistoning_qpos_adr],xp=self.xp, fp=self.fp, left="extrapolate", right="extrapolate")
            _data = _data.replace(qfrc_applied=_data.qfrc_applied.at[pistoning_dof_adr].set(f_kp))
        
            return _data

        def _no_adaptation(_data):
            return _data

        def _inner_loop(idx, _runner_state):

            _data, _carry = _runner_state

            ctrl_action, _carry = self._mjx_compute_action(processed_action, self._model, _data, _carry)

            # step in the environment using the action
            ctrl = _data.ctrl.at[jnp.array(self._action_indices)].set(ctrl_action)
            _data = _data.replace(ctrl=ctrl)

            _data = jax.lax.cond(self.socket_ty_slack, _adapt_qfrc_applied, _no_adaptation, _data)

            step_fn = lambda _, x: mjx.step(sys, x)
            _data = jax.lax.fori_loop(0, self._n_substeps, step_fn, _data)

            return _data, _carry

        # run inner loop
        data, carry = jax.lax.fori_loop(0, self._n_intermediate_steps, _inner_loop, (data, carry))

        # modify data *after* step if needed (does nothing by default)
        data, carry = self._mjx_simulation_post_step(self._model, data, carry)

        # create the observation
        cur_obs, carry = self._mjx_create_observation(sys, data, carry)

        # modify the observation and the data if needed (does nothing by default)
        cur_obs, data, cur_info, carry = self._mjx_step_finalize(cur_obs, self._model, data, cur_info, carry)

        # create info
        cur_info = self._mjx_update_info_dictionary(cur_info, cur_obs, data, carry)

        # check if the next obs is an absorbing state
        absorbing, carry = self._mjx_is_absorbing(cur_obs, cur_info, data, carry)

        # calculate the reward
        reward, carry = self._mjx_reward(state.observation, action, cur_obs, absorbing, cur_info, self._model, data, carry)

        # check if done
        done = self._mjx_is_done(cur_obs, absorbing, cur_info, data, carry)

        done = jnp.logical_or(done, jnp.any(jnp.isnan(cur_obs)))
        cur_obs = jnp.nan_to_num(cur_obs, nan=0.0)

        # create state
        carry = carry.replace(cur_step_in_episode=carry.cur_step_in_episode + 1)
        state = state.replace(data=data, observation=cur_obs, reward=reward,
                              absorbing=absorbing, done=done, info=cur_info, additional_carry=carry)

        # reset state if done
        state = jax.lax.cond(state.done, self._mjx_reset_in_step, lambda x: x, state)

        return state

    def _mjx_create_observation(self, model: Model,
                                data: Data,
                                carry: MjxAdditionalCarry) -> jax.Array:
        """
        Creates the observation for the environment.

        Args:
            model (Model): Mjx model.
            data (Data): Mjx data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            jax.Array: The observation of the environment.

        """
        return self._create_observation_compat(model, data, carry, jnp)

    def _mjx_reset_info_dictionary(self, obs: jnp.ndarray,
                                   data: Data,
                                   key: jax.random.PRNGKey) -> Dict:
        """
        Resets the info dictionary.

        Args:
            obs (jnp.ndarray): Observation of the environment.
            data (Data): Mjx data structure.
            key (jax.random.PRNGKey): Random key.

        Returns:
            Dict: The updated info dictionary.

        """
        return {}

    def _mjx_update_info_dictionary(self, info: Dict,
                                    obs: jnp.ndarray,
                                    data: Data,
                                    carry: MjxAdditionalCarry) -> Dict:
        """
        Updates the info dictionary.

        Args:
            obs (jnp.ndarray): Observation of the environment.
            data (Data): Mjx data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Dict: The updated info dictionary.

        """
        return info

    def _mjx_reward(self, obs: jnp.ndarray,
                    action: jnp.ndarray,
                    next_obs: jnp.ndarray,
                    absorbing: bool,
                    info: Dict,
                    model: Model,
                    data: Data,
                    carry: MjxAdditionalCarry) -> Tuple[float, MjxAdditionalCarry]:
        """
        Calls the reward function of the environment.

        Args:
            obs (jnp.ndarray): Observation of the environment.
            action (jnp.ndarray): Action taken in the environment.
            next_obs (jnp.ndarray): Next observation of the environment.
            absorbing (bool): Whether the next state is absorbing.
            info (Dict): Information dictionary.
            model (Model): Mjx model.
            data (Data): Mjx data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[float, MjxAdditionalCarry]: The reward and the updated carry.

        """
        reward, carry = self._reward_function(obs, action, next_obs, absorbing, info, self, model, data, carry, jnp)
        return reward, carry

    def _mjx_is_absorbing(self, obs: jnp.ndarray,
                          info: Dict,
                          data: Data,
                          carry: MjxAdditionalCarry) -> bool:
        """
        Determines if the current state is absorbing.

        Args:
            obs (jnp.ndarray): Current observation.
            info (Dict): Information dictionary.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            bool: True if the state is absorbing, False otherwise.
        """
        return self._terminal_state_handler.mjx_is_absorbing(self, obs, info, data, carry)

    def _mjx_is_done(self, obs: jnp.ndarray,
                     absorbing: bool,
                     info: Dict,
                     data: Data,
                     carry: MjxAdditionalCarry) -> bool:
        """
        Determines if the episode is done.

        Args:
            obs (jnp.ndarray): Current observation.
            absorbing (bool): Whether the next state is absorbing.
            info (Dict): Information dictionary.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            bool: True if the episode is done, False otherwise.
        """
        done = jnp.greater_equal(carry.cur_step_in_episode, self.info.horizon)
        done = jnp.logical_or(done, absorbing)
        return done

    def _mjx_simulation_pre_step(self, model: Model,
                                 data: Data,
                                 carry: MjxAdditionalCarry) -> Tuple[Model, Data, MjxAdditionalCarry]:
        """
        Applies pre-step modifications to the model, data, and carry.

        Args:
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[Model, Data, MjxAdditionalCarry]: Updated model, data, and carry.
        """
        model, data, carry = self._terrain.update(self, model, data, carry, jnp)
        model, data, carry = self._domain_randomizer.update(self, model, data, carry, jnp)
        return model, data, carry

    def _mjx_simulation_post_step(self, model: Model,
                                  data: Data,
                                  carry: MjxAdditionalCarry) -> Tuple[Data, MjxAdditionalCarry]:
        """
        Applies post-step modifications to the data and carry.

        Args:
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[Data, MjxAdditionalCarry]: Updated data and carry.
        """
        return data, carry

    def _mjx_preprocess_action(self, action: jnp.ndarray,
                               model: Model,
                               data: Data,
                               carry: MjxAdditionalCarry) -> Tuple[jnp.ndarray, MjxAdditionalCarry]:
        """
        Transforms the action before applying it to the environment.

        Args:
            action (jnp.ndarray): Action input.
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[jnp.ndarray, MjxAdditionalCarry]: Processed action and updated carry.
        """
        action, carry = self._domain_randomizer.update_action(self, action, model, data, carry, jnp)
        return action, carry

    def _mjx_compute_action(self, action: jnp.ndarray,
                            model: Model,
                            data: Data,
                            carry: MjxAdditionalCarry) -> Tuple[jnp.ndarray, MjxAdditionalCarry]:
        """
        Applies transformations to the action at intermediate steps.

        Args:
            action (jnp.ndarray): Action at the current step.
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[jnp.ndarray, MjxAdditionalCarry]: Computed action and updated carry.
        """
        action, carry = self._control_func.generate_action(self, action, model, data, carry, jnp)
        return action, carry

    def _mjx_reset_carry(self, model: Model,
                         data: Data,
                         carry: MjxAdditionalCarry) -> Tuple[Data, MjxAdditionalCarry]:
        """
        Resets the additional carry and allows modification to the Mujoco data.

        Args:
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[Data, MjxAdditionalCarry]: Updated data and carry.
        """
        data, carry = self._terminal_state_handler.reset(self, model, data, carry, jnp)
        data, carry = self._terrain.reset(self, model, data, carry, jnp)
        data, carry = self._init_state_handler.reset(self, model, data, carry, jnp)
        data, carry = self._domain_randomizer.reset(self, model, data, carry, jnp)
        data, carry = self._reward_function.reset(self, model, data, carry, jnp)
        return data, carry

    def _mjx_step_finalize(self, obs: jnp.ndarray,
                           model: Model,
                           data: Data,
                           info: Dict,
                           carry: MjxAdditionalCarry) -> Tuple[jnp.ndarray, Data, Dict, MjxAdditionalCarry]:
        """
        Allows information to be accessed at the end of a step.

        Args:
            obs (jnp.ndarray): Observation.
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            info (Dict): Information dictionary.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[jnp.ndarray, Data, Dict, MjxAdditionalCarry]: Updated observation, data, info, and carry.
        """
        obs, carry = self._domain_randomizer.update_observation(self, obs, model, data, carry, jnp)
        return obs, data, info, carry

    @staticmethod
    def mjx_set_sim_state_from_traj_data(data: Data,
                                         traj_data: TrajectoryData,
                                         carry: MjxAdditionalCarry) -> Data:
        """
        Sets the simulation state from the trajectory data.

        Args:
            data (Data): Current Mujoco data.
            traj_data (TrajectoryData): Data from the trajectory.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Data: Updated Mujoco data.
        """
        return data.replace(
            xpos=traj_data.xpos if traj_data.xpos.size > 0 else data.xpos,
            xquat=traj_data.xquat if traj_data.xquat.size > 0 else data.xquat,
            cvel=traj_data.cvel if traj_data.cvel.size > 0 else data.cvel,
            qpos=traj_data.qpos if traj_data.qpos.size > 0 else data.qpos,
            qvel=traj_data.qvel if traj_data.qvel.size > 0 else data.qvel)

    def _mjx_set_sim_state_from_obs(self, data: Data,
                                    obs: jnp.ndarray) -> Data:
        """
        Updates the simulation state from an observation.

        .. note:: This may not fully set the state of the simulation if the observation does not contain all the
                  necessary information.

        Args:
            data (Data): Current Mujoco data.
            obs (jnp.ndarray): Observation containing state information.

        Returns:
            Data: Updated Mujoco data.
        """
        data = data.replace(
            qpos=data.qpos.at[self._data_indices.free_joint_qpos].set(obs[self._obs_indices.free_joint_qpos]),
            qvel=data.qvel.at[self._data_indices.free_joint_qvel].set(obs[self._obs_indices.free_joint_qvel]))

        return data.replace(
            xpos=data.xpos.at[self._data_indices.body_xpos].set(obs[self._obs_indices.body_xpos].reshape(-1, 3)),
            xquat=data.xquat.at[self._data_indices.body_xquat].set(obs[self._obs_indices.body_xquat].reshape(-1, 4)),
            cvel=data.cvel.at[self._data_indices.body_cvel].set(obs[self._obs_indices.body_cvel].reshape(-1, 6)),
            qpos=data.qpos.at[self._data_indices.joint_qpos].set(obs[self._obs_indices.joint_qpos]),
            qvel=data.qvel.at[self._data_indices.joint_qvel].set(obs[self._obs_indices.joint_qvel]),
            site_xpos=data.site_xpos.at[self._data_indices.site_xpos].set(
                obs[self._obs_indices.site_xpos].reshape(-1, 3)),
            site_xmat=data.site_xmat.at[self._data_indices.site_xmat].set(
                obs[self._obs_indices.site_xmat].reshape(-1, 9)))

    def mjx_render(self, state,
                   record: bool = False) -> np.ndarray:
        """
        Renders all environments in parallel.

        Args:
            state: Current environment state.
            record (bool): Whether to record the rendering.

        Returns:
            np.ndarray: Rendered image.
        """
        if self._viewer is None:
            if "default_camera_mode" not in self._viewer_params.keys():
                self._viewer_params["default_camera_mode"] = "static"
            if 'ignore_modify_mjx_contact' in self._viewer_params.keys():
                del self._viewer_params['ignore_modify_mjx_contact']
            if 'add_sensors' in self._viewer_params.keys():
                del self._viewer_params['add_sensors']
            if 'socket_ty_slack' in self._viewer_params.keys():
                del self._viewer_params['socket_ty_slack']
            if 'add_pos_ori_to_observation' in self._viewer_params.keys():
                del self._viewer_params['add_pos_ori_to_observation']
            if 'limit_knee_extension' in self._viewer_params.keys():
                del self._viewer_params['limit_knee_extension']
            if 'knee_extension_limit' in self._viewer_params.keys():
                del self._viewer_params['knee_extension_limit']
            self._viewer = MujocoViewer(self._model, self.dt, record=record, **self._viewer_params)

        if self._terrain.is_dynamic:
            terrain_state = state.additional_carry.terrain_state
            assert hasattr(terrain_state, "height_field_raw"), "Terrain state does not have height_field_raw."
            assert self._terrain.hfield_id is not None, "Terrain hfield id is not set."
            hfield_data = np.array(terrain_state.height_field_raw)
            self._model.hfield_data = hfield_data[0]
            self._viewer.upload_hfield(self._model, hfield_id=self._terrain.hfield_id)

        return self._viewer.parallel_render(state, record)
    
    def mjx_render_domain_randomization(self, state,
                   record: bool = False) -> np.ndarray:
        """
        Renders all environments in parallel.

        Args:
            state: Current environment state.
            record (bool): Whether to record the rendering.

        Returns:
            np.ndarray: Rendered image.
        """
        model = self.update_mjM_post_domain_randomizer(state.additional_carry)
        if self._viewer is None:
            if "default_camera_mode" not in self._viewer_params.keys():
                self._viewer_params["default_camera_mode"] = "static"
            if 'ignore_modify_mjx_contact' in self._viewer_params.keys():
                del self._viewer_params['ignore_modify_mjx_contact']
            if 'add_sensors' in self._viewer_params.keys():
                del self._viewer_params['add_sensors']
            if 'socket_ty_slack' in self._viewer_params.keys():
                del self._viewer_params['socket_ty_slack']
            if 'add_pos_ori_to_observation' in self._viewer_params.keys():
                del self._viewer_params['add_pos_ori_to_observation']
            if 'limit_knee_extension' in self._viewer_params.keys():
                del self._viewer_params['limit_knee_extension']
            if 'knee_extension_limit' in self._viewer_params.keys():
                del self._viewer_params['knee_extension_limit']
            self._viewer = MujocoViewer(model, self.dt, record=record, **self._viewer_params)

        if self._terrain.is_dynamic:
            terrain_state = state.additional_carry.terrain_state
            assert hasattr(terrain_state, "height_field_raw"), "Terrain state does not have height_field_raw."
            assert self._terrain.hfield_id is not None, "Terrain hfield id is not set."
            hfield_data = np.array(terrain_state.height_field_raw)
            model.hfield_data = hfield_data[0]
            self._viewer.upload_hfield(model, hfield_id=self._terrain.hfield_id)

        return self._viewer.parallel_render(state, record)


    def mjx_render_trajectory(self, trajectory,
                              record: bool = False) -> None:
        """
        Renders a trajectory sequence.

        Args:
            trajectory: A sequence of environment states.
            record (bool): Whether to record the rendering.

        """
        assert len(trajectory) > 0, "Mjx render got provided with an empty trajectory."

        if self._viewer is None:
            self._viewer = MujocoViewer(self._model, self.dt, record=record, **self._viewer_params)

        n_envs = trajectory[0].data.qpos.shape[0]

        for i in range(n_envs):
            for state in trajectory:
                self._data.qpos, self._data.qvel = state.data.qpos[i, :], state.data.qvel[i, :]
                mujoco.mj_forward(self._model, self._data)
                self._viewer.render(self._data, record)

    def _init_additional_carry(self, key,
                               model: Model,
                               data: Data,
                               backend: ModuleType) -> MjxAdditionalCarry:
        """
        Initializes additional carry parameters.

        Args:
            key: Random key for initialization.
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            backend (ModuleType): Computational backend (either numpy or jax.numpy).

        Returns:
            MjxAdditionalCarry: Initialized carry object.
        """
        carry = super()._init_additional_carry(key, model, data, backend)
        return MjxAdditionalCarry(final_observation=backend.zeros(self.info.observation_space.shape),
                                  final_info={},
                                  **vars(carry))

    @property
    def n_envs(self) -> int:
        """Returns the number of environments."""
        return self._n_envs

    @property
    def mjx_env(self) -> bool:
        """Indicates whether this is an MJX environment."""
        return True
        

    def update_mjM_post_domain_randomizer(self, carry: MjxAdditionalCarry) -> None:
        """
        Updates the Mujoco model based on the MJX model after randomization.

        Args:
            mjxModel (Model): The Mujoco model to update.
        """
        domain_randomizer_state = carry.domain_randomizer_state

        model = self._model

        # dof_indices = list(self._domain_randomizer._dof_indices.values())
        # jnt_indices = list(self._domain_randomizer._joint_indices.values())
        # body_pos_indices = list(self._domain_randomizer._body_pos_indices.values())
        # body_quat_indices = list(self._domain_randomizer._body_quat_indices.values())   
        # print("Stiffness before update:", model.jnt_stiffness)
        if self._domain_randomizer.rand_conf["randomize_prosthesis_joint_stiffness"]:
            jnt_indices_map = self._domain_randomizer._joint_indices  # joint_name -> index
            sampled_stiffness_dict = domain_randomizer_state.prosthesis_joint_stiffness  # joint_name -> value

            # Make a copy of current stiffness values
            jnt_stiffness = model.jnt_stiffness.copy()

            for joint_name, stiffness_value in sampled_stiffness_dict.items():
                if joint_name not in jnt_indices_map:
                    raise KeyError(f"Joint '{joint_name}' not found in joint index mapping.")
                index = jnt_indices_map[joint_name]
                jnt_stiffness[index] = np.asarray(stiffness_value, dtype=np.float64).item()

            model.jnt_stiffness = jnt_stiffness
            # print('Stiffness updated:',  model.jnt_stiffness)

        # print("Damping before update:", model.dof_damping)
        if self._domain_randomizer.rand_conf["randomize_prosthesis_dof_damping"]:
            dof_indices_map = self._domain_randomizer._dof_indices  # dof_name -> index
            sampled_damping_dict = domain_randomizer_state.prosthesis_dof_damping  # dof_name -> value

            # Make a copy of current damping values
            dof_damping = model.dof_damping.copy()

            for dof_name, damping_value in sampled_damping_dict.items():
                if dof_name not in dof_indices_map:
                    raise KeyError(f"DOF '{dof_name}' not found in dof index mapping.")
                index = dof_indices_map[dof_name]
                dof_damping[index] = np.asarray(damping_value, dtype=np.float64).item()

            model.dof_damping = dof_damping
        #     dof_indices = list(self._domain_randomizer._dof_indices.values())
        # # if hasattr(domain_randomizer_state, "prosthesis_dof_damping"):
        #     model.dof_damping[dof_indices] = np.array(
        #         domain_randomizer_state.prosthesis_dof_damping, dtype=np.float64
        #     ).squeeze()
        #     # print('Damping updated:', model.dof_damping)

        # print("Position before update:", model.body_pos)
        # if self._domain_randomizer.rand_conf["randomize_prosthesis_body_position"]:
        #     body_pos_indices = list(self._domain_randomizer._body_pos_indices.values())
        # #if hasattr(domain_randomizer_state, "prosthesis_body_position"):
        #     model.body_pos[body_pos_indices] = np.array(
        #         domain_randomizer_state.prosthesis_body_position, dtype=np.float64
        #     ).squeeze()
        #     # print('Position updated:', model.body_pos)

        if self._domain_randomizer.rand_conf["randomize_prosthesis_body_position"]:
            # The sampled_position_dict holds {body_name: [x,y,z] array}
            sampled_position_dict = domain_randomizer_state.prosthesis_body_position 

            # Make a *mutable* copy of the current model's body positions.
            # This is crucial for NumPy, as model.body_pos might be read-only or we want to modify a copy.
            current_model_body_pos = model.body_pos.copy()

            #jax.debug.print("current_model_body_pos_shape: {shape}", shape=current_model_body_pos.shape)

            for body_name_str, position_value_array in sampled_position_dict.items():
                prefix = ""
                if self._domain_randomizer.rand_conf["prosthesis_side"] == "left_side":
                    prefix = "_l"
                elif self._domain_randomizer.rand_conf["prosthesis_side"] == "right_side":
                    prefix = "_r"

                full_mujoco_body_name = body_name_str + prefix

                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, full_mujoco_body_name)

                if body_id == -1:
                    print(f"Warning: MuJoCo body '{full_mujoco_body_name}' (from semantic name '{body_name_str}') not found in model. Skipping position update for this body.")
                    continue
                
                # Update the position for this specific body ID in our mutable copy

                # jax.debug.print("position_value_array_shape: {shape}", shape = position_value_array.shape)

                
                current_model_body_pos[body_id] = np.array(position_value_array[0], dtype=np.float64).squeeze()

            if 'socket_ty'+self._domain_randomizer.prosthesis_side_str in self._domain_randomizer._socket_joint_indices and self._domain_randomizer.rand_conf["randomize_prosthesis_socket_joint"]:
                talus_offset_y = domain_randomizer_state.prosthesis_socket_joint_value[f"socket_ty"+self._domain_randomizer.prosthesis_side_str]
                talus_offset_array = np.array([0,talus_offset_y[0], 0])
                # jax.debug.print("pos_y view: {pos_y}", pos_y = pos_y)
                # if not np.any(self.init_talus_pos):
                #     jax.debug.print("IN LOOOOOOPPPPP")
                #     self.init_talus_pos = model.body_pos[self._domain_randomizer._talus_idx].copy()
                # # jax.debug.print("talus_pos view: {talus_pos}", talus_pos = self.init_talus_pos)
                # new_talus_pos = self.init_talus_pos - np.array([0,pos_y[0],0])

                # current_model_body_pos = model.body_pos.copy()
                current_model_body_pos[self._domain_randomizer._talus_idx] -= np.array(talus_offset_array, dtype=np.float64).squeeze()

                # Assign the modified copy back to the model's body_pos
                model.body_pos = current_model_body_pos

            # Assign the modified copy back to the model's body_pos
            model.body_pos = current_model_body_pos

            # print('Position updated:', model.body_pos)

        # print("Orientation before update:", model.body_quat)
        if self._domain_randomizer.rand_conf["randomize_prosthesis_body_orientation"]:
            sampled_quat_dict = domain_randomizer_state.prosthesis_body_orientation
             
            current_model_body_quat = model.body_quat.copy()
            for body_name_str, quat_value_array in sampled_quat_dict.items():
                prefix = ""
                if self._domain_randomizer.rand_conf["prosthesis_side"] == "left_side":
                    prefix = "_l"
                elif self._domain_randomizer.rand_conf["prosthesis_side"] == "right_side":
                    prefix = "_r"

                full_mujoco_body_name = body_name_str + prefix

                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, full_mujoco_body_name)

                if body_id == -1:
                    print(f"Warning: MuJoCo body '{full_mujoco_body_name}' (from semantic name '{body_name_str}') not found in model. Skipping position update for this body.")
                    continue

                current_model_body_quat[body_id] = np.array(quat_value_array[0], dtype=np.float64).squeeze()

            # Assign the modified copy back to the model's body_pos
            model.body_quat = current_model_body_quat

        if self._domain_randomizer.rand_conf["randomize_prosthesis_socket_joint"] and not self._domain_randomizer.rand_conf["randomize_prosthesis_body_position"]:
            
            if 'socket_ty'+self._domain_randomizer.prosthesis_side_str in self._domain_randomizer._socket_joint_indices:
                
                pos_y = domain_randomizer_state.prosthesis_socket_joint_value[f"socket_ty"+self._domain_randomizer.prosthesis_side_str]
                # jax.debug.print("pos_y view: {pos_y}", pos_y = pos_y)
                if not np.any(self.init_talus_pos):
                    jax.debug.print("IN LOOOOOOPPPPP")
                    self.init_talus_pos = model.body_pos[self._domain_randomizer._talus_idx].copy()
                # jax.debug.print("talus_pos view: {talus_pos}", talus_pos = self.init_talus_pos)
                new_talus_pos = self.init_talus_pos - np.array([0,pos_y[0],0])

                current_model_body_pos = model.body_pos.copy()
                current_model_body_pos[self._domain_randomizer._talus_idx] = np.array(new_talus_pos, dtype=np.float64).squeeze()

                # Assign the modified copy back to the model's body_pos
                model.body_pos = current_model_body_pos
                # jax.debug.print('current_model_body_pos view: {current_model_body_pos}', current_model_body_pos=current_model_body_pos)
                
            
            # # The sampled_position_dict holds {body_name: [x,y,z] array}
            # sampled_position_dict = domain_randomizer_state.prosthesis_body_position 

            # # Make a *mutable* copy of the current model's body positions.
            # # This is crucial for NumPy, as model.body_pos might be read-only or we want to modify a copy.
            # current_model_body_pos = model.body_pos.copy()

            # #jax.debug.print("current_model_body_pos_shape: {shape}", shape=current_model_body_pos.shape)

            # for body_name_str, position_value_array in sampled_position_dict.items():
            #     prefix = ""
            #     if self._domain_randomizer.rand_conf["prosthesis_side"] == "left_side":
            #         prefix = "_l"
            #     elif self._domain_randomizer.rand_conf["prosthesis_side"] == "right_side":
            #         prefix = "_r"

            #     full_mujoco_body_name = body_name_str + prefix

            #     body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, full_mujoco_body_name)

            #     if body_id == -1:
            #         print(f"Warning: MuJoCo body '{full_mujoco_body_name}' (from semantic name '{body_name_str}') not found in model. Skipping position update for this body.")
            #         continue
                
            #     # Update the position for this specific body ID in our mutable copy

            #     # jax.debug.print("position_value_array_shape: {shape}", shape = position_value_array.shape)

                
            #     current_model_body_pos[body_id] = np.array(position_value_array[0], dtype=np.float64).squeeze()

            # # Assign the modified copy back to the model's body_pos
            # model.body_pos = current_model_body_pos
        
        
        
        
        #    body_quat_indices = list(self._domain_randomizer._body_quat_indices.values())   
        # #if hasattr(domain_randomizer_state, "prosthesis_body_orientation"):
        #     model.body_quat[body_quat_indices] = np.array(
        #         domain_randomizer_state.prosthesis_body_orientation, dtype=np.float64
        #     ).squeeze()
        #     # print('Orientation updated:', model.body_quat)

        return model 


    def mjx_step_test(self, state: MjxState, action: jax.Array) -> MjxState:
        """

        Args:
            state (MjxState): Current state of the environment.
            action (jax.Array): Action to take in the environment.

        Returns:
            MjxState: The next state of the environment.

        """

        data = state.data
        cur_info = state.info
        carry = state.additional_carry
        carry = carry.replace(last_action=action)

        # reset dones
        state = state.replace(done=jnp.zeros_like(state.done, dtype=bool))

        # preprocess action
        processed_action, carry = self._mjx_preprocess_action(action, self._model, data, carry)

        # modify data and model *before* step if needed
        sys, data, carry = self._mjx_simulation_pre_step(self.sys, data, carry)

        def _inner_loop(idx, _runner_state):

            _data, _carry = _runner_state

            ctrl_action, _carry = self._mjx_compute_action(processed_action, self._model, _data, _carry)

            # step in the environment using the action
            ctrl = _data.ctrl.at[jnp.array(self._action_indices)].set(ctrl_action)
            _data = _data.replace(ctrl=ctrl)
            step_fn = lambda _, x: mjx.step(sys, x)
            _data = jax.lax.fori_loop(0, self._n_substeps, step_fn, _data)

            return _data, _carry

        # run inner loop
        data, carry = jax.lax.fori_loop(0, self._n_intermediate_steps, _inner_loop, (data, carry))

        # modify data *after* step if needed (does nothing by default)
        data, carry = self._mjx_simulation_post_step(self._model, data, carry)

        # create the observation
        cur_obs, carry = self._mjx_create_observation(sys, data, carry)

        # modify the observation and the data if needed (does nothing by default)
        cur_obs, data, cur_info, carry = self._mjx_step_finalize(cur_obs, self._model, data, cur_info, carry)

        # create info
        cur_info = self._mjx_update_info_dictionary(cur_info, cur_obs, data, carry)

        # check if the next obs is an absorbing state
        absorbing, carry = self._mjx_is_absorbing(cur_obs, cur_info, data, carry)

        # calculate the reward
        reward, carry = self._mjx_reward(state.observation, action, cur_obs, absorbing, cur_info, self._model, data, carry)

        # check if done
        done = self._mjx_is_done(cur_obs, absorbing, cur_info, data, carry)

        done = jnp.logical_or(done, jnp.any(jnp.isnan(cur_obs)))
        cur_obs = jnp.nan_to_num(cur_obs, nan=0.0)

        # create state
        carry = carry.replace(cur_step_in_episode=carry.cur_step_in_episode + 1)
        state = state.replace(data=data, observation=cur_obs, reward=reward,
                              absorbing=absorbing, done=done, info=cur_info, additional_carry=carry)

        # def print_identity(x):
        #     jax.debug.print("Identity function called")
        #     return x

        # reset state if done
        # state = jax.lax.cond(state.done, self._mjx_reset_in_step, print_identity, state)
        state = jax.lax.cond(state.done, self._mjx_reset_in_step, lambda x: x, state)

        return state, sys


    def mjx_step_socket_ty(self, state: MjxState, action: jax.Array) -> MjxState:
        """

        Args:
            state (MjxState): Current state of the environment.
            action (jax.Array): Action to take in the environment.

        Returns:
            MjxState: The next state of the environment.

        """

        data = state.data
        cur_info = state.info
        carry = state.additional_carry
        carry = carry.replace(last_action=action)

        # reset dones
        state = state.replace(done=jnp.zeros_like(state.done, dtype=bool))

        # preprocess action
        processed_action, carry = self._mjx_preprocess_action(action, self._model, data, carry)

        # modify data and model *before* step if needed
        sys, data, carry = self._mjx_simulation_pre_step(self.sys, data, carry)

        def _inner_loop(idx, _runner_state):

            _data, _carry = _runner_state

            ctrl_action, _carry = self._mjx_compute_action(processed_action, self._model, _data, _carry)

            # step in the environment using the action
            ctrl = _data.ctrl.at[jnp.array(self._action_indices)].set(ctrl_action)
            _data = _data.replace(ctrl=ctrl)
            joint_id= mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'socket_ty'+self.prosthesis_side)
            qpos_address = self._model.jnt_qposadr[joint_id]
            pistoning_qpos_adr = qpos_address
            pistoning_dof_adr = self._model.jnt_dofadr[joint_id]
            f_kp = jnp.interp(data.qpos[pistoning_qpos_adr],xp=self.xp, fp=self.fp, left="extrapolate", right="extrapolate")
            _data = _data.replace(qfrc_applied=_data.qfrc_applied.at[pistoning_dof_adr].set(f_kp))
            step_fn = lambda _, x: mjx.step(sys, x)
            _data = jax.lax.fori_loop(0, self._n_substeps, step_fn, _data)

            return _data, _carry

        # run inner loop
        data, carry = jax.lax.fori_loop(0, self._n_intermediate_steps, _inner_loop, (data, carry))

        # modify data *after* step if needed (does nothing by default)
        data, carry = self._mjx_simulation_post_step(self._model, data, carry)

        # create the observation
        cur_obs, carry = self._mjx_create_observation(sys, data, carry)

        # modify the observation and the data if needed (does nothing by default)
        cur_obs, data, cur_info, carry = self._mjx_step_finalize(cur_obs, self._model, data, cur_info, carry)

        # create info
        cur_info = self._mjx_update_info_dictionary(cur_info, cur_obs, data, carry)

        # check if the next obs is an absorbing state
        absorbing, carry = self._mjx_is_absorbing(cur_obs, cur_info, data, carry)

        # calculate the reward
        reward, carry = self._mjx_reward(state.observation, action, cur_obs, absorbing, cur_info, self._model, data, carry)

        # check if done
        done = self._mjx_is_done(cur_obs, absorbing, cur_info, data, carry)

        done = jnp.logical_or(done, jnp.any(jnp.isnan(cur_obs)))
        cur_obs = jnp.nan_to_num(cur_obs, nan=0.0)

        # create state
        carry = carry.replace(cur_step_in_episode=carry.cur_step_in_episode + 1)
        state = state.replace(data=data, observation=cur_obs, reward=reward,
                              absorbing=absorbing, done=done, info=cur_info, additional_carry=carry)

        # reset state if done
        state = jax.lax.cond(state.done, self._mjx_reset_in_step, lambda x: x, state)

        return state