import pickle
from pathlib import Path
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 
# os.environ["JAX_PLATFORMS"] = "cpu"
import sys
import jax
# jax.config.update('jax_platform_name', 'cpu')
import wandb
import jax.numpy as jnp
import traceback
import time
import hydra 
from omegaconf import DictConfig, OmegaConf, open_dict
from dataclasses import fields

from loco_mujoco.utils.metrics import QuantityContainer
from loco_mujoco import TaskFactory, ImitationFactory
from loco_mujoco.algorithms import PPOJax, SavePPOJax #SavePPOJax
from loco_mujoco.utils import MetricsHandler

# Set MUJOCO_GL to egl
os.environ["MUJOCO_GL"] = "egl" 

@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    try: 
        result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        config.experiment.result_dir = result_dir

        # Setup wandb 
        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        run = wandb.init(project=config.wandb.project, 
                         name=f"{Path(result_dir).parent.name}_{Path(result_dir).name}_{config.wandb.run_name}", 
                         config=config_dict)

        # Get task factory and environment
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
        randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)
        if "prosthesis_side" in config.experiment.env_params:
            randomization_params["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]

        env = factory.make(domain_randomization_type=config.randomization_config["randomization_type"], 
                          domain_randomization_params=randomization_params,
                          **config.experiment.env_params, **config.experiment.task_factory.params)


        # Path to your two checkpoints 
        ckpt_path_1 = config.checkpoint_path_1
        ckpt_path_2 = config.checkpoint_path_2

        ##########################################
        # # 1 Load the agent 
        # _, state_1 = SavePPOJax.load_agent(ckpt_path_1)
        # _, state_2 = SavePPOJax.load_agent(ckpt_path_2)

        # # Extract and format train_states
        # ts0 = state_1.train_state
        # ts12 = state_2.train_state

        # # # Merge them: result will have leading dim of 3
        # # # jnp.atleast_2d ensures that a single seed (dim 0) becomes (1, dim)
        # # merged_train_state = jax.tree.map(
        # #     lambda s0, s12: jnp.concatenate([jnp.atleast_2d(s0), jnp.atleast_2d(s12)], axis=0),
        # #     ts0, ts12
        # # )
        # # 1. Convert objects to raw state dictionaries to avoid PyTree structure comparison errors
        # from flax.serialization import to_state_dict, from_state_dict
        # ts0_dict = to_state_dict(ts0)
        # ts12_dict = to_state_dict(ts12)

        # def merge_leaf(s0, s12):
        #     # Convert to arrays if they aren't already
        #     s0, s12 = jnp.array(s0), jnp.array(s12)
            
        #     # If they are already arrays with matching trailing dimensions, concatenate
        #     # We use atleast_1d to handle scalars correctly (making them shape (1,) and (2,))
        #     v0 = jnp.atleast_1d(s0)
        #     v12 = jnp.atleast_1d(s12)
            
        #     # Check if trailing dimensions match
        #     if v0.shape[1:] == v12.shape[1:]:
        #         return jnp.concatenate([v0, v12], axis=0)
        #     else:
        #         # This handles cases like 'step' count where shapes might be (1,) and (2,)
        #         # but nested differently. We force them into a flat 1D array of 3 seeds.
        #         return jnp.concatenate([v0.ravel(), v12.ravel()], axis=0)

        # merged_dict = jax.tree.map(merge_leaf, ts0_dict, ts12_dict)
        # merged_train_state = from_state_dict(ts0, merged_dict)

        #########################################

        # 1. Load the two files
        _, state_1 = SavePPOJax.load_agent(ckpt_path_1)  # Contains 1 seed
        _, state_2 = SavePPOJax.load_agent(ckpt_path_2)  # Contains 2 seeds

        from flax.serialization import to_state_dict, from_state_dict

        # 2. Convert to raw dictionaries
        dict_seed_0 = to_state_dict(state_1.train_state)
        dict_seeds_1_2 = to_state_dict(state_2.train_state)

        # 3. Slice the 2-seed dictionary into two individual dictionaries
        # We extract index 0 and index 1 from every leaf in the second file
        dict_seed_1 = jax.tree.map(lambda x: x[0], dict_seeds_1_2)
        dict_seed_2 = jax.tree.map(lambda x: x[1], dict_seeds_1_2)

        # 4. Define the stacking function
        def stack_three_seeds(s0, s1, s2):
            # Ensure they are arrays and add a batch dimension to each
            # This turns shape (weights) into (1, weights)
            nodes = [jnp.expand_dims(jnp.array(n), 0) for n in [s0, s1, s2]]
            return jnp.concatenate(nodes, axis=0)

        # 5. Merge all three individual seeds
        merged_dict = jax.tree.map(stack_three_seeds, dict_seed_0, dict_seed_1, dict_seed_2)

        # 6. Reconstruct the TrainState object
        merged_train_state = from_state_dict(state_1.train_state, merged_dict)

        # --- NEW STEP: Wrap it back into an AgentState ---
        # state_1 is an instance of PPOAgentState (or similar)
        # We use its constructor to wrap our merged TrainState
        runner_state = state_1.__class__(train_state=merged_train_state)

        # new_agent_conf = SavePPOJax.init_agent_conf(env, config)
            
        # agent_conf = new_agent_conf 


        print(f"Loading and merging checkpoints: {ckpt_path_1} and {ckpt_path_2}")


        # Initialize agent configuration
        agent_conf = PPOJax.init_agent_conf(env, config)
        save_path = SavePPOJax.save_conf(result_dir, agent_conf)
        run.config.update({"agent_conf_save_path": save_path})

        # Calculation for Chunking Logic
        # Total updates across the whole experiment
        total_updates = config.experiment.total_timesteps // (config.experiment.num_envs * config.experiment.num_steps)
        num_checkpoints = config.experiment.checkpoint_num
        updates_per_chunk = total_updates // num_checkpoints

        with open_dict(config.experiment):
            agent_conf.config.experiment.num_updates = updates_per_chunk
        
        # Build and JIT the training function
        mh = MetricsHandler(config, env) if config.experiment.validation.active else None
        
        # # We define a function that takes runner_state and runs for updates_per_chunk
        # def train_chunk_fn(rng): #, state=None):
        #     # Temporarily override num_updates for the JITed function
        #     with open_dict(config.experiment):
        #         config.experiment.num_updates = updates_per_chunk
        #     return SavePPOJax._train_fn(rng, env, agent_conf, mh=mh)
        # build training function
        # train_fn = PPOJax.build_train_fn(env, agent_conf, mh=mh) 

        # # jit and vmap training function
        # # train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)
        # train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

        
        # We define a function that takes runner_state and runs for updates_per_chunk
        def train_chunk_continue_fn(rng, state=None):
            # Temporarily override num_updates for the JITed function
            # with open_dict(config.experiment):
            #     config.experiment.num_updates = updates_per_chunk
            return PPOJax._train_fn_continue(rng, env, agent_conf, agent_state = state, mh=mh)

        # Apply vmap for multi-seed support
        if config.experiment.n_seeds > 1:
            train_chunk_continue_fn = jax.vmap(train_chunk_continue_fn) #, axis_name='batch')
        
        # jit_train_chunk = jax.jit(train_chunk_fn)
        jit_train_continue_chunk = jax.jit(train_chunk_continue_fn)

        # Initialize seeds
        skip_seed = 0 #1 if config.experiment.n_seeds > 1 else 0
        rngs = jnp.array([jax.random.PRNGKey(i) for i in range(skip_seed, config.experiment.n_seeds + 1)]) # skip_seed)])
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
        # out = train_fn(_rng)

        # runner_state = merged_train_state #runner_state = None
        global_step = 0


        # Training in chunks
        for chunk in range(num_checkpoints):
            print(f"\n--- Starting Chunk {chunk + 1}/{num_checkpoints} ---")
            t_chunk_start = time.time()
            
            # Execute JITed chunk
            out = jit_train_continue_chunk(_rng, state=runner_state)
            
            # Update states for next chunk
            runner_state = out["agent_state"] # Using agent_state as the state carrier
            training_metrics = out["training_metrics"]
            validation_metrics = out["validation_metrics"]
            
            # Update global step count
            chunk_steps = updates_per_chunk * config.experiment.num_envs * config.experiment.num_steps
            global_step += chunk_steps

            # --- SAVE CHECKPOINTS FOR ALL SEEDS ---
            for i in range(config.experiment.n_seeds):
                # Extract state for specific seed
                single_seed_state = jax.tree.map(lambda x: x[i], runner_state)
                checkpoint_name = f"ckpt_step_{int(global_step)}_seed_{i}"
                print(f"Saving checkpoint for seed {i} at step {global_step} in {checkpoint_name}...")
                SavePPOJax.save_agent_checkpoints(result_dir, agent_conf, single_seed_state, checkpoint_name=checkpoint_name)
            
            save_path = SavePPOJax.save_agent_with_name(result_dir, agent_conf, runner_state, checkpoint_name=f"ckpt_step_{int(global_step)}") #PPOJax.save_agent(result_dir, agent_conf, agent_state)
            # run.config.update({"agent_save_path": save_path})
            
            # --- LOGGING ---
            # Mean metrics across seeds for the current chunk
            m_train = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics)
            m_val = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics)


            if not config.experiment.debug:
                # calculate mean across seeds
                m_train = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics)
                m_val = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics)

                for i in range(len(m_train.mean_episode_return)):
                    actual_step = int(global_step - chunk_steps + m_train.max_timestep[i])
                    # run.log({"Mean Episode Return": m_train.mean_episode_return[i],
                    #          "Mean Episode Length": m_train.mean_episode_length[i]},
                    #         step=int(m_train.max_timestep[i]))
                    log_data = {
                            "Mean Episode Return": float(m_train.mean_episode_return[i]),
                            "Mean Episode Length": float(m_train.mean_episode_length[i])
                        }

                    if (i+1) % config.experiment.validation_interval == 0 and config.experiment.validation.active:
                        # run.log({"Validation Info/Mean Episode Return": m_val.mean_episode_return[i],
                        #          "Validation Info/Mean Episode Length": m_val.mean_episode_length[i]},
                        #         step=int(m_train.max_timestep[i]))
                        log_data.update({
                                "Validation Info/Mean Episode Return": float(m_val.mean_episode_return[i]),
                                "Validation Info/Mean Episode Length": float(m_val.mean_episode_length[i])
                            })

                        # log all measures
                        # metrics_to_log = {}
                        for field in fields(m_val):
                            attr = getattr(m_val, field.name)
                            if isinstance(attr, QuantityContainer):
                                # measure_name = field.name
                                for field_attr in fields(attr):
                                    attr_name = field_attr.name
                                    attr_value = getattr(attr, attr_name)
                                    if attr_value.size > 0:
                                        # metrics_to_log[f"Validation Measures/{measure_name}/{attr_name}"] = attr_value[i]
                                        log_data[f"Validation Measures/{field.name}/{attr_name}"] = float(attr_value[i])

                        # run.log(metrics_to_log, step=int(m_train.max_timestep[i]))
                        # metric for used for wandb sweep (optional)
                        # site_rpos = m_val.euclidean_distance.site_rpos[i]
                        # site_rrotvec = m_val.euclidean_distance.site_rpos[i]
                        # site_rvel = m_val.euclidean_distance.site_rpos[i]
                        site_rpos = m_val.euclidean_distance.site_rpos[i]
                        site_rrotvec = m_val.euclidean_distance.site_rrotvec[i] 
                        site_rvel = m_val.euclidean_distance.site_rvel[i]
                        # run.log({"Metric for Sweep": site_rpos + site_rrotvec + site_rvel},
                        #         step=int(m_train.max_timestep[i]))
                        log_data["Metric for Sweep"] = site_rpos + site_rrotvec + site_rvel

                    run.log(log_data, step=actual_step)


            # for i in range(len(m_train.mean_episode_return)):
            #     actual_step = int(global_step - chunk_steps + m_train.max_timestep[i])
            #     run.log({
            #         "Mean Episode Return": m_train.mean_episode_return[i],
            #         "Mean Episode Length": m_train.mean_episode_length[i]
            #     }, step=actual_step)

            # print(f"Chunk {chunk+1} finished. Time: {time.time() - t_chunk_start:.2f}s. Global Step: {global_step}")

        
        
        # if not config.experiment.debug:
        #     training_metrics = out["training_metrics"]
        #     validation_metrics = out["validation_metrics"]

        #     # calculate mean across seeds
        #     training_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics)
        #     validation_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics)

        #     for i in range(len(training_metrics.mean_episode_return)):
        #         run.log({"Mean Episode Return": training_metrics.mean_episode_return[i],
        #                  "Mean Episode Length": training_metrics.mean_episode_length[i]},
        #                 step=int(training_metrics.max_timestep[i]))

        #         if (i+1) % config.experiment.validation_interval == 0 and config.experiment.validation.active:
        #             run.log({"Validation Info/Mean Episode Return": validation_metrics.mean_episode_return[i],
        #                      "Validation Info/Mean Episode Length": validation_metrics.mean_episode_length[i]},
        #                     step=int(training_metrics.max_timestep[i]))

        #             # log all measures
        #             metrics_to_log = {}
        #             for field in fields(validation_metrics):
        #                 attr = getattr(validation_metrics, field.name)
        #                 if isinstance(attr, QuantityContainer):
        #                     measure_name = field.name
        #                     for field_attr in fields(attr):
        #                         attr_name = field_attr.name
        #                         attr_value = getattr(attr, attr_name)
        #                         if attr_value.size > 0:
        #                             metrics_to_log[f"Validation Measures/{measure_name}/{attr_name}"] = attr_value[i]

        #             run.log(metrics_to_log, step=int(training_metrics.max_timestep[i]))

        #             # metric for used for wandb sweep (optional)
        #             site_rpos = validation_metrics.euclidean_distance.site_rpos[i]
        #             site_rrotvec = validation_metrics.euclidean_distance.site_rpos[i]
        #             site_rvel = validation_metrics.euclidean_distance.site_rpos[i]
        #             run.log({"Metric for Sweep": site_rpos + site_rrotvec + site_rvel},
        #                     step=int(training_metrics.max_timestep[i]))

        run.config.update({"agent_save_path": save_path})
        # Final Video/Play policy
        PPOJax.play_policy(env, agent_conf, runner_state.train_state, deterministic=True, 
                               n_steps=200, n_envs=20, record=True, train_state_seed=0)
        run.log({"Agent Video": wandb.Video(env.video_file_path)})
        wandb.finish()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise

if __name__ == "__main__":
    experiment()