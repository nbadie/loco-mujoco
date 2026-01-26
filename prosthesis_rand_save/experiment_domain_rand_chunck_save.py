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
        train_fn = PPOJax.build_train_fn(env, agent_conf, mh=mh) 

        # jit and vmap training function
        # train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)
        train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

        
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
        skip_seed = 1 if config.experiment.n_seeds > 1 else 0
        rngs = jnp.array([jax.random.PRNGKey(i) for i in range(skip_seed, config.experiment.n_seeds + skip_seed)])
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
        # out = train_fn(_rng)

        runner_state = None
        global_step = 0

        # Training in chunks
        for chunk in range(num_checkpoints):
            print(f"\n--- Starting Chunk {chunk + 1}/{num_checkpoints} ---")
            t_chunk_start = time.time()
            
            # Execute JITed chunk
            if chunk == 0:
                out = train_fn(_rng) #, state=runner_state)
            else:
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
            run.config.update({"agent_save_path": save_path})
            
            # --- LOGGING ---
            # Mean metrics across seeds for the current chunk
            m_train = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics)
            m_val = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics)

            for i in range(len(m_train.mean_episode_return)):
                actual_step = global_step - chunk_steps + int(m_train.max_timestep[i])
                run.log({
                    "Mean Episode Return": m_train.mean_episode_return[i],
                    "Mean Episode Length": m_train.mean_episode_length[i]
                }, step=actual_step)

            print(f"Chunk {chunk+1} finished. Time: {time.time() - t_chunk_start:.2f}s. Global Step: {global_step}")

        
        if not config.experiment.debug:
            training_metrics = out["training_metrics"]
            validation_metrics = out["validation_metrics"]

            # calculate mean across seeds
            training_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics)
            validation_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics)

            for i in range(len(training_metrics.mean_episode_return)):
                run.log({"Mean Episode Return": training_metrics.mean_episode_return[i],
                         "Mean Episode Length": training_metrics.mean_episode_length[i]},
                        step=int(training_metrics.max_timestep[i]))

                if (i+1) % config.experiment.validation_interval == 0 and config.experiment.validation.active:
                    run.log({"Validation Info/Mean Episode Return": validation_metrics.mean_episode_return[i],
                             "Validation Info/Mean Episode Length": validation_metrics.mean_episode_length[i]},
                            step=int(training_metrics.max_timestep[i]))

                    # log all measures
                    metrics_to_log = {}
                    for field in fields(validation_metrics):
                        attr = getattr(validation_metrics, field.name)
                        if isinstance(attr, QuantityContainer):
                            measure_name = field.name
                            for field_attr in fields(attr):
                                attr_name = field_attr.name
                                attr_value = getattr(attr, attr_name)
                                if attr_value.size > 0:
                                    metrics_to_log[f"Validation Measures/{measure_name}/{attr_name}"] = attr_value[i]

                    run.log(metrics_to_log, step=int(training_metrics.max_timestep[i]))

                    # metric for used for wandb sweep (optional)
                    site_rpos = validation_metrics.euclidean_distance.site_rpos[i]
                    site_rrotvec = validation_metrics.euclidean_distance.site_rpos[i]
                    site_rvel = validation_metrics.euclidean_distance.site_rpos[i]
                    run.log({"Metric for Sweep": site_rpos + site_rrotvec + site_rvel},
                            step=int(training_metrics.max_timestep[i]))

        
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