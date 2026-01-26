import ast
from omegaconf import open_dict
import warnings
from dataclasses import dataclass
from typing import Any
from omegaconf import DictConfig, OmegaConf, ListConfig

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import flax
from flax.serialization import to_state_dict
import optax
from pathlib import Path
from orbax import checkpoint as ocp
import pickle

from traitlets import This

from loco_mujoco.algorithms import (JaxRLAlgorithmBase, AgentConfBase, AgentStateBase, ActorCritic,
                                    Transition, TrainState, TrainStateBuffer, MetricHandlerTransition, PPOJax)
from loco_mujoco.core.wrappers import LogWrapper, NStepWrapper, LogEnvState, VecEnv, NormalizeVecReward, SummaryMetrics
from loco_mujoco.utils import MetricsHandler, ValidationSummary
from flax.training import orbax_utils

# from jax.experimental import io_callback as hcb
from jax.experimental import io_callback
import uuid

from jax import tree_util

@dataclass(frozen=True)
class PPOAgentConf(AgentConfBase):
    config: DictConfig
    network: ActorCritic
    tx: Any

    def serialize(self):
        """
        Serialize the agent configuration and network configuration.

        Returns:
            Serialized agent configuration as a dictionary.

        """
        conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        serialized_network = flax.serialization.to_state_dict(self.network)
        return {"config": conf_dict, "network": serialized_network}

    @classmethod
    def from_dict(cls, d):
        config = OmegaConf.create(d["config"])
        tx = PPOJax._get_optimizer(config)
        return cls(config=config,
                   network=flax.serialization.from_state_dict(ActorCritic, d["network"]),
                   tx=tx)


@struct.dataclass
class PPOAgentState(AgentStateBase):
    train_state: TrainState

    def serialize(self):
        serialized_train_state = flax.serialization.to_state_dict(self.train_state)
        return {"train_state": serialized_train_state}

    @classmethod
    def from_dict(cls, d, agent_conf):
        train_state = TrainState(apply_fn=agent_conf.network, tx=agent_conf.tx, **d["train_state"])
        return cls(train_state)


class SavePPOJax(PPOJax):

    _agent_conf = PPOAgentConf
    _agent_state = PPOAgentState

    @classmethod
    def save_agent_checkpoints(cls, path, agent_conf: AgentConfBase, agent_state: AgentStateBase, checkpoint_name=None):
        """
        Save the agent state to a file.
        
        Args:
            path: Base directory path
            agent_conf: Agent configuration
            agent_state: Agent state (can contain multiple seeds with vmap dimension)
            checkpoint_name: Optional custom checkpoint name
        """
        path = Path(path)
        
        if checkpoint_name is None:
            checkpoint_name = cls.__name__ + "_agent_saved"
        
        # Create checkpoints subdirectory
        checkpoint_dir = path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = checkpoint_dir / checkpoint_name
        # jax.debug.print("Saving agent checkpoint to: {save_path}", save_path=str(save_path))
        save_path = save_path.with_suffix(cls._saved_agent_suffix)
        # jax.debug.print("Saving agent checkpoint to: {save_path}", save_path=str(save_path))
        
        # Serialize both config and state
        serialized_data = {
            "agent_conf": agent_conf.serialize(),
            "agent_state": agent_state.serialize()
        }
        
        # Save to file
        with open(save_path, 'wb') as file:
            pickle.dump(serialized_data, file)
        
        print(f"Saved agent to: {save_path}")
        return save_path
    
    @classmethod
    def save_agent_with_name(cls, path, agent_conf: AgentConfBase, agent_state: AgentStateBase, checkpoint_name=None):
        """ Save the agent state to a file."""
        path = Path(path)
        path = path / (cls.__name__ + '_' + checkpoint_name + "_saved")
        path = path.with_suffix(cls._saved_agent_suffix)
        # serialize agent state
        serialized_state = cls.serialize(agent_conf, agent_state)
        # save agent state
        with open(path, 'wb') as file:
            pickle.dump(serialized_state, file)
        print(f"\nSaved agent to: {path}\n")
        return path

    @classmethod
    def init_agent_conf(cls, env, config):

        with (open_dict(config.experiment)):
            config.experiment.num_updates = (
                    config.experiment.total_timesteps // config.experiment.num_steps // config.experiment.num_envs)
            config.experiment.minibatch_size = (
                    config.experiment.num_envs * config.experiment.num_steps // config.experiment.num_minibatches)
            config.experiment.validation_interval = config.experiment.num_updates // config.experiment.validation.num
            config.experiment.validation.num = int(
                config.experiment.num_updates // config.experiment.validation_interval)
            config.experiment.checkpoint_interval = config.experiment.num_updates // config.experiment.checkpoint_num
            

        # INIT NETWORK
        hidden_layers = config.experiment.hidden_layers \
            if isinstance(config.experiment.hidden_layers, (list, ListConfig)) \
            else ast.literal_eval(config.experiment.hidden_layers)
        if hasattr(config.experiment, "actor_obs_group") and config.experiment.actor_obs_group is not None:
            actor_obs_ind = env.obs_container.get_obs_ind_by_group(config.experiment.actor_obs_group)
        else:
            actor_obs_ind = jnp.arange(env.mdp_info.observation_space.shape[0])
        if hasattr(config.experiment, "critic_obs_group") and config.experiment.critic_obs_group is not None:
            critic_obs_ind = env.obs_container.get_obs_ind_by_group(config.experiment.critic_obs_group)
        else:
            critic_obs_ind = jnp.arange(env.mdp_info.observation_space.shape[0])
        if hasattr(config.experiment, "len_obs_history") and config.experiment.len_obs_history > 1:
            obs_len = env.info.observation_space.shape[0]
            actor_obs_ind = jnp.concatenate([actor_obs_ind + i*obs_len
                                             for i in range(config.experiment.len_obs_history)])
            critic_obs_ind = jnp.concatenate([critic_obs_ind + i*obs_len
                                              for i in range(config.experiment.len_obs_history)])
        network = ActorCritic(
            env.info.action_space.shape[0],
            activation=config.experiment.activation,
            init_std=config.experiment.init_std,
            learnable_std=config.experiment.learnable_std,
            hidden_layer_dims=hidden_layers,
            actor_obs_ind=actor_obs_ind,
            critic_obs_ind=critic_obs_ind
        )

        # set up optimizers
        tx = cls._get_optimizer(config)

        return cls._agent_conf(config, network, tx)
    
    @classmethod
    def save_conf(cls, path, agent_conf):
        """ Save the agent state to a file."""
        path = Path(path)
        path = path / (cls.__name__ + "_agent_conf_saved")
        path = path.with_suffix(cls._saved_agent_suffix)
        # serialize agent state
        serialized_conf = agent_conf.serialize() #cls.serialize(agent_conf)
        # save agent state
        with open(path, 'wb') as file:
            pickle.dump(serialized_conf , file)
        print(f"\nSaved agent conf to: {path}\n")
        return path

    @classmethod
    def load_agent_conf(cls, path):
        """ Load the agent state from a file. """
        if isinstance(path, str):
            path = Path(path)
        if not path.is_file():
            raise ValueError(f'Not a file: {path}')
        if path.suffix != cls._saved_agent_suffix:
            raise ValueError(f'Not a {cls._saved_agent_suffix} file: {path}')
        print(f"Loading agent from {path}..." )
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return cls.agent_conf_from_dict(data)
    
    @classmethod
    def agent_conf_from_dict(cls, d):
        """ Load conf of an agent from a dictionary. """
        agent_conf = cls._agent_conf.from_dict(d) #["agent_conf"])
        # agent_state = cls._agent_state.from_dict(d["agent_state"], agent_conf)
        return agent_conf#, agent_state
    
    def load_checkpoint_with_device_fix(ckpt_path, train_state_template):
        """
        Load checkpoint with proper device handling and target tree
        This fixes the TFRT_CPU_0 device error and missing target tree warning
        """
        
        # Create proper restore arguments with target template
        template = {
            'params': train_state_template.params,
            'run_stats': train_state_template.run_stats, 
            'step': train_state_template.step,
            'opt_state': train_state_template.opt_state,
        }
        
        check_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=False)
        
        with ocp.CheckpointManager(ckpt_path, options=check_options, item_names=('agent_state',)) as mngr:
            latest_step = mngr.latest_step()
            print(f"Loading checkpoint from step: {latest_step}")
            
            # Use StandardRestore with template to avoid device mismatch
            restored = mngr.restore(
                latest_step,
                args=ocp.args.Composite(
                    agent_state=ocp.args.StandardRestore(template),
                )
            )
            
            # Transfer to current devices if needed
            loaded_state = restored['agent_state']
            
            # Ensure all arrays are on current devices
            loaded_state = jax.tree_util.tree_map( #jax.tree_map(
                lambda x: jax.device_put(x) if hasattr(x, 'device') else x,
                loaded_state
            )
            
            return loaded_state
        
    def load_checkpoint_with_device_fix_without_value(ckpt_path, train_state_template, network,env):
        """
        Load checkpoint with proper device handling and target tree
        This fixes the TFRT_CPU_0 device error and missing target tree warning
        """
        rng = jax.random.key(0) 
        rng, _rng1, _rng2 = jax.random.split(rng, 3)
        init_x = jnp.zeros(env.info.observation_space.shape)
        network_params = network.init(_rng1, init_x)
        
        # Create proper restore arguments with target template
        template = {
            'params': train_state_template.params,
            'run_stats': train_state_template.run_stats, 
            'step': train_state_template.step,
            'opt_state': train_state_template.opt_state,
        }
        
        check_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=False)
        
        with ocp.CheckpointManager(ckpt_path, options=check_options, item_names=('agent_state',)) as mngr:
            latest_step = mngr.latest_step()
            print(f"Loading checkpoint from step: {latest_step}")
            
            # Use StandardRestore with template to avoid device mismatch
            restored = mngr.restore(
                latest_step,
                args=ocp.args.Composite(
                    agent_state=ocp.args.StandardRestore(template),
                )
            )
            
            # Transfer to current devices if needed
            loaded_state = restored['agent_state']
            
            # Ensure all arrays are on current devices
            loaded_state = jax.tree_util.tree_map( #jax.tree_map(
                lambda x: jax.device_put(x) if hasattr(x, 'device') else x,
                loaded_state
            )

            loaded_state['params']['FullyConnectedNet_1'] = network_params['params']['FullyConnectedNet_1']
            
            return loaded_state
    
    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf: PPOAgentConf,
                  agent_state: PPOAgentState = None,
                  mh: MetricsHandler = None):

        # extract static agent info
        config, network, tx =\
            (agent_conf.config.experiment, agent_conf.network, agent_conf.tx)
        
        jax.debug.print('Training config updated num_updates: {num_updates}', num_updates=config.num_updates)

        env = cls._wrap_env(env, config)

        # extract current agent state
        if agent_state is not None:
            train_state = agent_state.train_state
        else:
            train_state = None

        if train_state is None:

            rng, _rng1, _rng2 = jax.random.split(rng, 3)
            init_x = jnp.zeros(env.info.observation_space.shape)
            network_params = network.init(_rng1, init_x)

        else:
            raise NotImplementedError("Loading of train state not implemented yet.")

        # init new train states from old params
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params["params"] if train_state is None else train_state.params,
            run_stats=network_params["run_stats"] if train_state is None else train_state.run_stats,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = env.reset(reset_rng)

        train_state_buffer = TrainStateBuffer.create(train_state, config.validation.num)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, train_state_buffer, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                y, updates = network.apply({'params': train_state.params,
                                                  'run_stats': train_state.run_stats},
                                                 last_obs, mutable=["run_stats"])
                pi, value = y
                train_state = train_state.replace(run_stats=updates['run_stats'])   # update stats
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)
                # jax.debug.print('obsv shape: {obs}', obs=obsv)
                # GET METRICS
                log_env_state = env_state.find(LogEnvState)
                logged_metrics = log_env_state.metrics

                transition = Transition(
                    done, absorbing, action, value, reward, log_prob, last_obs, info, env_state.additional_carry.traj_state,
                    logged_metrics
                )
                runner_state = (train_state, env_state, obsv, train_state_buffer, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, train_state_buffer, rng = runner_state
            y, _ = network.apply({'params': train_state.params,
                                              'run_stats': train_state.run_stats},
                                             last_obs, mutable=["run_stats"])
            pi, last_val = y

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, absorbing, value, reward, obs = (
                        transition.done,
                        transition.absorbing,
                        transition.value,
                        transition.reward,
                        transition.obs
                    )

                    delta = reward + config.gamma * next_value * (1 - absorbing) - value
                    gae = (
                        delta
                        + config.gamma * config.gae_lambda * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE ACTOR & CRITIC NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        y, _ = network.apply({'params': params, 'run_stats': train_state.run_stats},
                                             traj_batch.obs, mutable=["run_stats"])
                        pi, value = y
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE PPO ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config.clip_eps,
                                    1.0 + config.clip_eps,
                                )
                                * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config.vf_coef * value_loss
                            - config.ent_coef * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config.minibatch_size * config.num_minibatches
                assert (
                    batch_size == config.num_steps * config.num_envs
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config.num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
            rng = update_state[-1]

            counter = ((train_state.step + 1) // config.num_minibatches) // config.update_epochs

            logged_metrics = traj_batch.metrics

            metric = SummaryMetrics(
                mean_episode_return=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)) / jnp.sum(logged_metrics.done),
                mean_episode_length=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)) / jnp.sum(logged_metrics.done),
                max_timestep=jnp.max(logged_metrics.timestep * config.num_envs),
            )

            ckpt_path=f'{agent_conf.config.experiment.result_dir}/checkpoints'
            ckpt_path = Path(ckpt_path)
            ckpt_path.mkdir(parents=True, exist_ok=True)
            # jax.debug.print(f"Checkpoint path: {ckpt_path}")
            # current_step = jnp.array(metric.max_timestep, int)



            # def _save_checkpoint_callback(step, ckpt_path, agent_conf, train_state):
            #     """Callback function for saving checkpoints"""
            #     check_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
            #     path = ckpt_path / f"ckpt_{int(step)}"
            #     with ocp.CheckpointManager(path, options=check_options, item_names=('agent_conf', 'agent_state')) as mngr:
            #         mngr.save(
            #             int(step), 
            #             args=ocp.args.Composite(
            #                 agent_conf=ocp.args.StandardSave(agent_conf),
            #                 agent_state=ocp.args.StandardSave(train_state),
            #             )
            #         ) 

            # # Replace your jax.lax.cond call with:
            # def _conditional_save():
            #     jax.debug.callback(
            #         _save_checkpoint_callback,
            #         train_state.step + 1,
            #         ckpt_path,
            #         agent_conf,
            #         train_state
            #     )
            # jax.debug.print('counter: {counter}', counter=counter)
            # jax.debug.print('config.validation_interval: {interval}', interval=config.validation_interval)
            # jax.lax.cond(counter % config.validation_interval == 0,
            #             lambda _: _conditional_save(),
            #             lambda _: None,
            #             None)
            def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
                    orbax_checkpointer = ocp.PyTreeCheckpointer()
                    save_args = orbax_utils.save_args_from_target(params)
                    path = ckpt_path / f"{current_step}"
                    orbax_checkpointer.save(path, params, force=True, save_args=save_args)

            # def _save_checkpoint_callback(step, ckpt_path, train_state):
            #     """Callback function for saving checkpoints"""
            #     check_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
            #     path = ckpt_path / f"ckpt_{int(step)}"
                
            #     # # Corrected: Serialize the train_state using flax.serialization.to_state_dict
            #     # serialized_train_state_for_orbax = flax.serialization.to_state_dict(train_state)
                
            #     # with ocp.CheckpointManager(path, options=check_options, item_names=('agent_state')) as mngr:
            #     #     mngr.save(
            #     #         int(step), 
            #     #         args=ocp.args.Composite(
            #     #             agent_state=ocp.args.StandardSave(serialized_train_state_for_orbax),
            #     #         )
            #     #     )
            #     with ocp.CheckpointManager(path, options=check_options, item_names=('agent_state')) as mngr:
            #     # with ocp.CheckpointManager(path, options=check_options, item_names=('agent_conf', 'agent_state')) as mngr:
            #         mngr.save(
            #             int(step), 
            #             args=ocp.args.Composite(
            #                 # agent_conf=ocp.args.JsonSave(serialized_agent_conf),
            #                 agent_state=ocp.args.StandardSave(train_state), #agent_state),
            #             )
            #         )

            def _save_checkpoint_callback(step, ckpt_path, train_state): #, agent_conf):
                """Callback function for saving checkpoints"""
                check_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
                path = ckpt_path / f"ckpt_{int(step)}"
                
                # Extract only serializable parts from train_state
                serializable_train_state = {
                    'params': train_state.params,
                    'run_stats': train_state.run_stats,
                    'step': train_state.step,
                    'opt_state': train_state.opt_state,
                    # 'network': agent_conf.network,
                }

                # # move to host and convert to python containers
                # host_train_state = jax.device_get(flax.serialization.to_state_dict(train_state))

                # # build serializable payload
                # serializable_train_state = {
                #     'params': host_train_state['params'],
                #     'run_stats': host_train_state['run_stats'],
                #     'step': int(host_train_state.get('step', 0)),
                #     'opt_state': host_train_state['opt_state'],
                # }

                # serializable_train_conf = {
                #     'experiment': SavePPOJax._serialized_agent_conf['experiment'],
                #     'control_config': SavePPOJax._serialized_agent_conf.control_config,
                #     'randomization_config': SavePPOJax._serialized_agent_conf.randomization_config,
                # }

                # # serializable_train_network = {
                # #     'network': SavePPOJax._serialized_agent_conf['network'],
                # # }
                # serializable_train_conf = {
                #     'experiment': agent_conf.config.experiment,
                #     'control_config': agent_conf.config.control_config,
                #     'randomization_config': agent_conf.config.randomization_config,
                # }

                # serializable_train_network = {
                #     'network': agent_conf.network,
                # }


                with ocp.CheckpointManager(path, options=check_options, item_names=('agent_state',)) as mngr:
                    mngr.save(
                        int(step), #f"ckpt_{int(step)}", #int(step), 
                        args=ocp.args.Composite(
                            agent_state=ocp.args.StandardSave(serializable_train_state),
                            # agent_conf=ocp.args.JsonSave(serializable_train_conf), #SavePPOJax._serialized_agent_conf), #agent_conf),
                            # agent_conf=ocp.args.StandardSave(SavePPOJax._serialized_agent_conf),
                        )
                    )
                
            # def _save_checkpoint_callback(step, ckpt_path, train_state):
            #     """Callback function for saving checkpoints"""
            #     check_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
            #     path = ckpt_path / f"ckpt_{int(step)}"
                
            #     # Use the pre-serialized agent_conf from class attribute
            #     # serialized_agent_conf = SavePPOJax._serialized_agent_conf
                
            #     # Create serialized agent state
            #     serialized_agent_state = {
            #         'train_state': {
            #             'params': train_state.params,
            #             'run_stats': train_state.run_stats,
            #             'step': train_state.step,
            #             'opt_state': train_state.opt_state
            #         }
            #     }
                
            #     with ocp.CheckpointManager(path, options=check_options, item_names=('agent_state')) as mngr:
            #     # with ocp.CheckpointManager(path, options=check_options, item_names=('agent_conf', 'agent_state')) as mngr:
            #         mngr.save(
            #             int(step), 
            #             args=ocp.args.Composite(
            #                 # agent_conf=ocp.args.JsonSave(serialized_agent_conf),
            #                 agent_state=ocp.args.StandardSave(serialized_agent_state),
            #             )
            #         )

            #########################################################################
            # def _conditional_save():
            #     jax.debug.callback(
            #         _save_checkpoint_callback,
            #         total_env_steps,
            #         #train_state.step + 1,
            #         ckpt_path,
            #         train_state,  # Only pass train_state now
            #         # agent_conf
            #     )

            ######################################################################################
            # def _host_save(payload): #, transforms):
            #     # payload is a tuple: (step, ckpt_path_str, host_state, maybe_seed_id)
            #     step, host_state, seed_id = payload
            #     path = Path(str(ckpt_path)) / f"ckpt_{int(step)}"
            #     # if seed_id is provided, create seed-specific folder
            #     if seed_id is not None:
            #         path = path / step / f"seed_{seed_id}"
            #     path.mkdir(parents=True, exist_ok=True)
            #     opts = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
            #     with ocp.CheckpointManager(path, options=opts, item_names=('agent_state',)) as mngr:
            #         mngr.save(int(step), args=ocp.args.Composite(agent_state=ocp.args.StandardSave(host_state)))

            # # Build and call host save only when the conditional save is triggered.
            # def _conditional_save():
            #     total_env_steps = (train_state.step // (config.num_minibatches * config.update_epochs)) * (config.num_envs * config.num_steps)
            #     # device-side serializable dict (DeviceArrays will be materialized by id_tap)
            #     device_state = flax.serialization.to_state_dict(train_state)
            #     # Use seed_id if present on train_state, otherwise None
            #     seed_id = getattr(train_state, 'seed_id', None)
            #     # Prepare payload: (step, ckpt_path_str, device_state, seed_id)
            #     payload = (total_env_steps, device_state, seed_id)
            #     # Transfer payload to host and invoke _host_save there
            #     # hcb.id_tap(_host_save, payload)
            #     # io_callback(_host_save, None, payload)
            #     io_callback(_host_save, payload)
            #     leaves, treedef = tree_util.tree_flatten(train_state)
            #     # Collect leaf info for vmap/jit compatibility (no side effects)
            #     leaf_info = [
            #         {"index": i, "type": str(type(leaf)), "shape": getattr(leaf, "shape", None)}
            #         for i, leaf in enumerate(leaves)
            #     ]
            #     jax.debug.print("TrainState leaves: {leaf_info}", leaf_info=leaf_info)
            #     # Optionally, return or log leaf_info for debugging outside jit/vmap


            ##############################################################################
            # CLAUDE ERROR
            # def _host_save(payload):
            #     step, host_state, seed_id = payload
            #     path = Path(str(ckpt_path)) / f"ckpt_{int(step)}" / f"seed_{seed_id}"
            #     path.mkdir(parents=True, exist_ok=True)
            #     opts = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
            #     with ocp.CheckpointManager(path, options=opts, item_names=('agent_state',)) as mngr:
            #         mngr.save(int(step), args=ocp.args.Composite(
            #             agent_state=ocp.args.StandardSave(host_state)))

            # def _conditional_save():
            #     total_env_steps = (train_state.step // (config.num_minibatches * config.update_epochs)) * (config.num_envs * config.num_steps)
                
            #     # Get the current seed index from vmap
            #     # This requires passing seed_id through your runner_state
            #     seed_id = jax.lax.axis_index('batch')  # If using pmap/vmap with axis_name
                
            #     device_state = flax.serialization.to_state_dict(train_state)
            #     payload = (total_env_steps, device_state, seed_id)
            #     io_callback(_host_save, None, payload)

            ######################################################################################

            # # Save using jax.debug.callback (1 seed/file?)
            # def _conditional_save():
            #     total_env_steps = (train_state.step // (config.num_minibatches * config.update_epochs)) * (config.num_envs * config.num_steps)
                
            #     # Prepare the checkpoint path
            #     ckpt_save_path = ckpt_path / f"ckpt_{int(total_env_steps)}"
                
            #     # Use debug.callback to call save functions outside JIT
            #     jax.debug.callback(
            #         _checkpoint_callback,
            #         ckpt_save_path,
            #         agent_conf,
            #         train_state,
            #         total_env_steps
            #     )


            # def _checkpoint_callback(save_path, agent_conf, train_state, step):
            #     """
            #     Callback function that runs outside JIT context.
            #     This handles vmap dimension properly by saving each seed separately.
            #     """
            #     # Convert JAX arrays to numpy for inspection
            #     train_state_host = jax.device_get(train_state)
                
            #     # Check if we have multiple seeds (vmap dimension)
            #     first_param = jax.tree_util.tree_leaves(train_state_host.params)[0]
            #     has_vmap_dim = len(first_param.shape) > 1 and first_param.shape[0] == agent_conf.config.experiment.n_seeds
                
            #     if has_vmap_dim and agent_conf.config.experiment.n_seeds > 1:
            #         # Save each seed separately
            #         for seed_idx in range(agent_conf.config.experiment.n_seeds):
            #             # Extract single seed from vmapped state
            #             single_seed_state = jax.tree_util.tree_map(
            #                 lambda x: x[seed_idx] if (hasattr(x, 'shape') and len(x.shape) > 0) else x,
            #                 train_state_host
            #             )
                        
            #             # Reconstruct agent_state for this seed
            #             agent_state_single = SavePPOJax._agent_state(train_state=single_seed_state)
                        
            #             # Create seed-specific path
            #             seed_save_path = Path(str(save_path)) / f"seed_{seed_idx}"
            #             seed_save_path.mkdir(parents=True, exist_ok=True)
                        
            #             # Use existing save_agent function
            #             SavePPOJax.save_agent_checkpoints(
            #                 str(seed_save_path.parent.parent),  # Go back to base dir
            #                 agent_conf,
            #                 agent_state_single,
            #                 checkpoint_name=f"ckpt_{int(step)}_seed_{seed_idx}"
            #             )
            #     else:
            #         # Single seed case - save directly
            #         agent_state = SavePPOJax._agent_state(train_state=train_state_host)
            #         SavePPOJax.save_agent_checkpoints(
            #             str(save_path.parent),
            #             agent_conf,
            #             agent_state,
            #             checkpoint_name=f"ckpt_{int(step)}"
            #         )
                
            #     print(f"Checkpoint saved at step {int(step)}")


            ############################################

            # # jax.debug.callbak all seeds after each other and overwriting them XX

            # def _conditional_save():
            #     total_env_steps = (train_state.step // (config.num_minibatches * config.update_epochs)) * (config.num_envs * config.num_steps)
                
            #     # Use debug.callback to save the entire vmapped train_state
            #     jax.debug.callback(
            #         _checkpoint_callback_all_seeds,
            #         total_env_steps,
            #         agent_conf,
            #         train_state
            #     )

            # def _checkpoint_callback_all_seeds(step, agent_conf, train_state):
            #     """
            #     Callback function that saves all seeds together in one file.
            #     This matches the behavior of the final save_agent call.
            #     """
            #     # Move train_state to host (CPU)
            #     train_state_host = jax.device_get(train_state)
                
            #     # Create agent_state wrapper
            #     agent_state = SavePPOJax._agent_state(train_state=train_state_host)
                
            #     # Use the existing save_agent function with a custom checkpoint name
            #     checkpoint_name = f"ckpt_{int(step)}"
            #     save_path = SavePPOJax.save_agent_checkpoints(
            #         agent_conf.config.experiment.result_dir,
            #         agent_conf,
            #         agent_state,
            #         checkpoint_name=checkpoint_name
            #     )
                
            #     print(f"✓ Checkpoint saved at step {int(step)}: {save_path}")


            ################################################################################

            # def _conditional_save():
            #     total_env_steps = (train_state.step // (config.num_minibatches * config.update_epochs)) * (config.num_envs * config.num_steps)
                
            #     # Use jax.lax.cond to ensure callback runs only once per vmap batch
            #     # We check if we're at seed index 0, and only then save ALL seeds
            #     seed_idx = jax.lax.axis_index('batch') if config.n_seeds > 1 else 0
                
            #     def _save_all():
            #         jax.debug.callback(
            #             _checkpoint_callback_all_seeds,
            #             total_env_steps,
            #             agent_conf,
            #             train_state
            #         )
                
            #     # Only execute for the first seed to avoid multiple saves
            #     jax.lax.cond(
            #         seed_idx == 0,
            #         lambda _: _save_all(),
            #         lambda _: None,
            #         None
            #     )

            # def _checkpoint_callback_all_seeds(step, agent_conf, train_state):
            #     """Save checkpoint with all seeds in a single file"""
            #     train_state_host = jax.device_get(train_state)
            #     agent_state = SavePPOJax._agent_state(train_state=train_state_host)
                
            #     checkpoint_name = f"ckpt_{int(step)}"
            #     SavePPOJax.save_agent(
            #         agent_conf.config.experiment.result_dir,
            #         agent_conf,
            #         agent_state,
            #         checkpoint_name=checkpoint_name
            #     )

            # # Execute conditional save
            # jax.lax.cond(
            #     counter % config.checkpoint_interval == 0,
            #     lambda _: _conditional_save(),
            #     lambda _: None,
            #     None
            # )



            #############################################################################################
            # Option 2: Save each seed separately with unique names (Simpler, recommended)
            
            def _conditional_save():
                total_env_steps = (train_state.step // (config.num_minibatches * config.update_epochs)) * (config.num_envs * config.num_steps)
                
                # Get seed index if vmapped
                seed_idx = jax.lax.axis_index('batch') if config.n_seeds > 1 else 0
                
                jax.debug.callback(
                    _checkpoint_callback_single_seed,
                    total_env_steps,
                    agent_conf,
                    train_state,
                    seed_idx,
                    config.n_seeds
                )

            def _checkpoint_callback_single_seed(step, agent_conf, train_state, seed_idx, n_seeds):
                """Save checkpoint for a single seed"""
                train_state_host = jax.device_get(train_state)
                agent_state = SavePPOJax._agent_state(train_state=train_state_host)
                
                # Create unique checkpoint name per seed
                if n_seeds > 1:
                    checkpoint_name = f"ckpt_{int(step)}_seed_{int(seed_idx)}"
                else:
                    checkpoint_name = f"ckpt_{int(step)}"
                
                SavePPOJax.save_agent_checkpoints(
                    agent_conf.config.experiment.result_dir,
                    agent_conf,
                    agent_state,
                    checkpoint_name=checkpoint_name
                )

            # Execute conditional save
            jax.lax.cond(
                counter % config.checkpoint_interval == 0,
                lambda _: _conditional_save(),
                lambda _: None,
                None
            )

            #############################################################################################






            # ########## WRONGGG ############
            # # # def _conditional_save():
            # # #     jax.debug.callback(
            # # #         lambda current_step: _save_checkpoint(ckpt_path, int(current_step), agent_conf, train_state),
            # # #         train_state.step + 1
            # # #     )

            # # # # staticmethod
            # # def _save_checkpoint(ckpt_path: Path, current_step: int, agent_conf: PPOAgentConf, train_state: TrainState):
            # #     check_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
            # #     # params_to_save = (agent_conf, train_state)
            # #     # save_args = orbax_utils.save_args_from_target(params_to_save)
            # #     path = ckpt_path / f"ckpt_{current_step}"
            # #     with ocp.CheckpointManager(path, options=check_options, item_names=('agent_conf', 'agent_state')) as mngr:
            # #         mngr.save(
            # #             current_step, 
            # #             args=ocp.args.Composite(
            # #                 agent_conf=ocp.args.StandardSave(agent_conf),
            # #                 agent_state=ocp.args.StandardSave(train_state),
            # #             )
            # #         )
            # # total_env_steps = (train_state.step // (config.num_minibatches * config.update_epochs)) * (config.num_envs * config.num_steps)
            # # jax.debug.print('total_env_steps: {total_env_steps}', total_env_steps=total_env_steps)
            # jax.lax.cond(counter % config.checkpoint_interval==0, #config.validation_interval == 0, #total_env_steps % config.checkpoint_interval == 0,
            #              lambda _: _conditional_save(),
            #              lambda _: None,
            #              None)
            
            # # # current_step = jnp.array(metric.max_timestep, int) #jnp.max(logged_metrics.timestep * config.num_envs),int)
            # # # MODIFIED: Pass a lambda function to jax.lax.cond to defer the call to _save_checkpoint
            # # jax.lax.cond(counter % config.validation_interval == 0,
            # #                              lambda _: _save_checkpoint(ckpt_path, int(train_state.step + 1), agent_conf, train_state), # Use train_state directly
            # #                              lambda _: None, # No-op for the else branch
            # #                              None) # Dummy argument for the lambda

            def _evaluation_step():

                def _eval_env(runner_state, unused):
                    train_state, env_state, last_obs, train_state_buffer, rng = runner_state

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    y, updates = train_state.apply_fn({'params': train_state.params,
                                                       'run_stats': train_state.run_stats},
                                                      last_obs, mutable=["run_stats"])
                    pi, value = y
                    train_state = train_state.replace(run_stats=updates['run_stats'])  # update stats
                    action = pi.sample(seed=_rng)

                    # STEP ENV
                    obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                    # GET METRICS
                    log_env_state = env_state.find(LogEnvState)
                    logged_metrics = log_env_state.metrics

                    transition = MetricHandlerTransition(env_state, logged_metrics)

                    runner_state = (train_state, env_state, obsv, train_state_buffer, rng)
                    return runner_state, transition

                rng = runner_state[-1]
                reset_rng = jax.random.split(rng, config.validation.num_envs)
                obsv, env_state = env.reset(reset_rng)
                runner_state_eval = (train_state, env_state, obsv, train_state_buffer, rng)

                # do evaluation runs
                _, traj_batch_eval = jax.lax.scan(
                    _eval_env, runner_state_eval, None, config.validation.num_steps
                )

                env_states = traj_batch_eval.env_state

                validation_metrics = mh(env_states)

                return validation_metrics

            if mh is None:
                validation_metrics = ValidationSummary()
            else:
                validation_metrics = jax.lax.cond(counter % config.validation_interval == 0, _evaluation_step,
                                                   mh.get_zero_container)

            if config.debug:
                def callback(metrics):
                    return_values = metrics.returned_episode_returns[metrics.done]
                    timesteps = metrics.timestep[metrics.done] * config.num_envs

                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")

                jax.debug.callback(callback, env_state.metrics)

            # add train state to buffer if needed
            train_state_buffer = jax.lax.cond(counter % config.validation_interval == 0,
                                              lambda x, y: TrainStateBuffer.add(x, y),
                                              lambda x, y: x, train_state_buffer, train_state)

            runner_state = (train_state, env_state, last_obs, train_state_buffer, rng)
            return runner_state, (metric, validation_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, train_state_buffer, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config.num_updates
        )

        agent_state = cls._agent_state(train_state=runner_state[0])

        return {"agent_state": agent_state,
                "training_metrics": metrics[0],
                "validation_metrics": metrics[1]}


    @classmethod
    def play_policy(cls, env,
                    agent_conf: PPOAgentConf,
                    agent_state: PPOAgentState,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None):

        if use_mujoco and wrap_env:
            if hasattr(agent_conf.experiment, "len_obs_history"):
                assert agent_conf.experiment.len_obs_history == 1, "len_obs_history must be 1 for mujoco envs."
        if use_mujoco:
            assert n_envs == 1, "Only one mujoco env can be run at a time."

        def sample_actions(ts, obs, _rng):
            y, updates = agent_conf.network.apply({'params': ts.params,
                                                   'run_stats': ts.run_stats},
                                                  obs, mutable=["run_stats"])
            ts = ts.replace(run_stats=updates['run_stats'])  # update stats
            pi, _ = y
            a = pi.sample(seed=_rng)
            return a, ts

        config = agent_conf.config.experiment
        train_state = agent_state

        if deterministic:
            train_state.params["log_std"] = np.ones_like(train_state.params["log_std"]) * -np.inf

        if config.n_seeds > 1:
            assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
                                                  "train_state_seed for replay.")

            # take the seed queried for evaluation
            train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

        if not render and n_steps is None and not record:
            warnings.warn("No rendering, no record, no n_steps specified. This will run forever with no effect.")

        # create env
        if wrap_env and not use_mujoco:
            env = cls._wrap_env(env, config)

        if rng is None:
            rng = jax.random.key(0)

        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        plcy_call = jax.jit(sample_actions)

        # reset env
        if use_mujoco:
            obs = env.reset()
            env_state = None
        else:
            obs, env_state = env.reset(env_keys)

        if n_steps is None:
            n_steps = np.iinfo(np.int32).max

        for i in range(n_steps):

            # SAMPLE ACTION
            rng, _rng = jax.random.split(rng)
            action, train_state = plcy_call(train_state, obs, _rng)
            action = jnp.atleast_2d(action)

            # STEP ENV
            if use_mujoco:
                obs, reward, absorbing, done, info = env.step(action)
            else:
                obs, reward, absorbing, done, info, env_state = env.step(env_state, action)

            # RENDER
            if use_mujoco:
                env.render(record=True)
            else:
                env.mjx_render(env_state, record=record)

            # RESET MUJOCO ENV (MJX resets by itself)
            if use_mujoco:
                if done:
                    obs = env.reset()

        env.stop()


    @classmethod
    def _train_fn_continue(cls, rng, env,
                  agent_conf: PPOAgentConf,
                  agent_state: PPOAgentState = None,
                  mh: MetricsHandler = None):

        # extract static agent info
        config, network, tx =\
            (agent_conf.config.experiment, agent_conf.network, agent_conf.tx)

        env = cls._wrap_env(env, config)

        # extract current agent state
        if agent_state is not None:
            train_state = agent_state#.train_state
        else:
            train_state = None

      
        rng, _rng1, _rng2 = jax.random.split(rng, 3)
        init_x = jnp.zeros(env.info.observation_space.shape)
        network_params = network.init(_rng1, init_x)

        
        # init new train states from old params
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params["params"] if train_state is None else train_state.params,
            run_stats=network_params["run_stats"] if train_state is None else train_state.run_stats,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = env.reset(reset_rng)

        train_state_buffer = TrainStateBuffer.create(train_state, config.validation.num)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, train_state_buffer, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                y, updates = network.apply({'params': train_state.params,
                                                  'run_stats': train_state.run_stats},
                                                 last_obs, mutable=["run_stats"])
                pi, value = y
                train_state = train_state.replace(run_stats=updates['run_stats'])   # update stats
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                # GET METRICS
                log_env_state = env_state.find(LogEnvState)
                logged_metrics = log_env_state.metrics

                transition = Transition(
                    done, absorbing, action, value, reward, log_prob, last_obs, info, env_state.additional_carry.traj_state,
                    logged_metrics
                )
                runner_state = (train_state, env_state, obsv, train_state_buffer, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, train_state_buffer, rng = runner_state
            y, _ = network.apply({'params': train_state.params,
                                              'run_stats': train_state.run_stats},
                                             last_obs, mutable=["run_stats"])
            pi, last_val = y

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, absorbing, value, reward, obs = (
                        transition.done,
                        transition.absorbing,
                        transition.value,
                        transition.reward,
                        transition.obs
                    )

                    delta = reward + config.gamma * next_value * (1 - absorbing) - value
                    gae = (
                        delta
                        + config.gamma * config.gae_lambda * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE ACTOR & CRITIC NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        y, _ = network.apply({'params': params, 'run_stats': train_state.run_stats},
                                             traj_batch.obs, mutable=["run_stats"])
                        pi, value = y
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE PPO ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config.clip_eps,
                                    1.0 + config.clip_eps,
                                )
                                * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config.vf_coef * value_loss
                            - config.ent_coef * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config.minibatch_size * config.num_minibatches
                assert (
                    batch_size == config.num_steps * config.num_envs
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config.num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
            rng = update_state[-1]

            counter = ((train_state.step + 1) // config.num_minibatches) // config.update_epochs

            logged_metrics = traj_batch.metrics

            metric = SummaryMetrics(
                mean_episode_return=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)) / jnp.sum(logged_metrics.done),
                mean_episode_length=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)) / jnp.sum(logged_metrics.done),
                max_timestep=jnp.max(logged_metrics.timestep * config.num_envs),
            )

            ckpt_path=f'{agent_conf.config.experiment.result_dir}/checkpoints'
            ckpt_path = Path(ckpt_path)
            ckpt_path.mkdir(parents=True, exist_ok=True)
            # jax.debug.print(f"Checkpoint path: {ckpt_path}")
            current_step = jnp.array(metric.max_timestep, int)


            # def _save_checkpoint_callback(step, ckpt_path, train_state): #, agent_conf):
            #     """Callback function for saving checkpoints"""
            #     check_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
            #     path = ckpt_path / f"ckpt_{int(step)}"
                
            #     # Extract only serializable parts from train_state
            #     serializable_train_state = {
            #         'params': train_state.params,
            #         'run_stats': train_state.run_stats,
            #         'step': train_state.step,
            #         'opt_state': train_state.opt_state,
            #         # 'network': agent_conf.network,
            #     }


            #     with ocp.CheckpointManager(path, options=check_options, item_names=('agent_state',)) as mngr:
            #         mngr.save(
            #             int(step), #f"ckpt_{int(step)}", #int(step), 
            #             args=ocp.args.Composite(
            #                 agent_state=ocp.args.StandardSave(serializable_train_state),
            #                 # agent_conf=ocp.args.JsonSave(serializable_train_conf), #SavePPOJax._serialized_agent_conf), #agent_conf),
            #                 # agent_conf=ocp.args.StandardSave(SavePPOJax._serialized_agent_conf),
            #             )
            #         )
            

            # def _conditional_save():
            #     jax.debug.callback(
            #         _save_checkpoint_callback,
            #         total_env_steps,
            #         #train_state.step + 1,
            #         ckpt_path,
            #         train_state,  # Only pass train_state now
            #         # agent_conf
            #     )


            # total_env_steps = (train_state.step // (config.num_minibatches * config.update_epochs)) * (config.num_envs * config.num_steps)
            # # jax.debug.print('total_env_steps: {total_env_steps}', total_env_steps=total_env_steps)
            # jax.lax.cond(counter % config.checkpoint_interval==0, #config.validation_interval == 0, #total_env_steps % config.checkpoint_interval == 0,
            #              lambda _: _conditional_save(),
            #              lambda _: None,
            #              None)

            #############################################################################################
            # Option 2: Save each seed separately with unique names (Simpler, recommended)
            
            def _conditional_save():
                total_env_steps = (train_state.step // (config.num_minibatches * config.update_epochs)) * (config.num_envs * config.num_steps)
                
                # Get seed index if vmapped
                seed_idx = jax.lax.axis_index('batch') if config.n_seeds > 1 else 0
                
                jax.debug.callback(
                    _checkpoint_callback_single_seed,
                    total_env_steps,
                    agent_conf,
                    train_state,
                    seed_idx,
                    config.n_seeds
                )

            def _checkpoint_callback_single_seed(step, agent_conf, train_state, seed_idx, n_seeds):
                """Save checkpoint for a single seed"""
                train_state_host = jax.device_get(train_state)
                agent_state = SavePPOJax._agent_state(train_state=train_state_host)
                
                # Create unique checkpoint name per seed
                if n_seeds > 1:
                    checkpoint_name = f"ckpt_{int(step)}_seed_{int(seed_idx)}"
                else:
                    checkpoint_name = f"ckpt_{int(step)}"
                
                SavePPOJax.save_agent_checkpoints(
                    agent_conf.config.experiment.result_dir,
                    agent_conf,
                    agent_state,
                    checkpoint_name=checkpoint_name
                )

            # Execute conditional save
            jax.lax.cond(
                counter % config.checkpoint_interval == 0,
                lambda _: _conditional_save(),
                lambda _: None,
                None
            )

            #############################################################################################
            
           
            def _evaluation_step():

                def _eval_env(runner_state, unused):
                    train_state, env_state, last_obs, train_state_buffer, rng = runner_state

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    y, updates = train_state.apply_fn({'params': train_state.params,
                                                       'run_stats': train_state.run_stats},
                                                      last_obs, mutable=["run_stats"])
                    pi, value = y
                    train_state = train_state.replace(run_stats=updates['run_stats'])  # update stats
                    action = pi.sample(seed=_rng)

                    # STEP ENV
                    obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                    # GET METRICS
                    log_env_state = env_state.find(LogEnvState)
                    logged_metrics = log_env_state.metrics

                    transition = MetricHandlerTransition(env_state, logged_metrics)

                    runner_state = (train_state, env_state, obsv, train_state_buffer, rng)
                    return runner_state, transition

                rng = runner_state[-1]
                reset_rng = jax.random.split(rng, config.validation.num_envs)
                obsv, env_state = env.reset(reset_rng)
                runner_state_eval = (train_state, env_state, obsv, train_state_buffer, rng)

                # do evaluation runs
                _, traj_batch_eval = jax.lax.scan(
                    _eval_env, runner_state_eval, None, config.validation.num_steps
                )

                env_states = traj_batch_eval.env_state

                validation_metrics = mh(env_states)

                return validation_metrics

            if mh is None:
                validation_metrics = ValidationSummary()
            else:
                validation_metrics = jax.lax.cond(counter % config.validation_interval == 0, _evaluation_step,
                                                   mh.get_zero_container)

            if config.debug:
                def callback(metrics):
                    return_values = metrics.returned_episode_returns[metrics.done]
                    timesteps = metrics.timestep[metrics.done] * config.num_envs

                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")

                jax.debug.callback(callback, env_state.metrics)

            # add train state to buffer if needed
            train_state_buffer = jax.lax.cond(counter % config.validation_interval == 0,
                                              lambda x, y: TrainStateBuffer.add(x, y),
                                              lambda x, y: x, train_state_buffer, train_state)

            runner_state = (train_state, env_state, last_obs, train_state_buffer, rng)
            return runner_state, (metric, validation_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, train_state_buffer, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config.num_updates
        )

        agent_state = cls._agent_state(train_state=runner_state[0])

        return {"agent_state": agent_state,
                "training_metrics": metrics[0],
                "validation_metrics": metrics[1]}
