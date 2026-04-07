import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 
# os.environ["JAX_PLATFORMS"] = "cpu"
import sys
import jax
# jax.config.update('jax_platform_name', 'cpu')
import wandb
import jax.numpy as jnp
import traceback

# Hydra:  key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line 
import hydra 
# from hydra.core.hydra_config import HydraConfig

from dataclasses import fields
from loco_mujoco.utils.metrics import QuantityContainer


from omegaconf import DictConfig, OmegaConf #OmegaConf is a YAML based hierarchical configuration system, with support for merging configurations from multiple sources

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import SavePPOJax
from loco_mujoco.utils import MetricsHandler
from loco_mujoco import ImitationFactory

import argparse

from flax.serialization import to_state_dict, from_state_dict



# Set MUJOCO_GL to egl
os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering, which is more compatible with headless environments


@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    try: 

        # can increase the speed by ~30% on some GPUs
        # os.environ['XLA_FLAGS'] = (
        #     '--xla_gpu_triton_gemm_any=True ')
        
        # Accessing the current sweep number
        result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir


        # Extract date and time from the result directory path for wandb run name
        result_dir_parts = result_dir.split("/")
        result_dir_date, result_dir_time = result_dir_parts[-2], result_dir_parts[-1]
        formatted_result_dir = f"{result_dir_date}_{result_dir_time}_"


        # setup wandb 
        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        run = wandb.init(project=config.wandb.project, name= formatted_result_dir + config.wandb.run_name, config=config_dict)


        # get task factory
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

        randomization_type = config.randomization_config["randomization_type"]
        # Convert to plain dict to allow adding new keys
        randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)
        
        # add prosthesis side to randomization params if it exists in config.experiment.env_params
        if "prosthesis_side" in config.experiment.env_params:
            randomization_params["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]

        # create env
        env = factory.make(domain_randomization_type=randomization_type, domain_randomization_params=randomization_params,
            **config.experiment.env_params, **config.experiment.task_factory.params)
        
        # Path to your two checkpoints 
        ckpt_path_1 = config.checkpoint_path_1
        ckpt_path_2 = config.checkpoint_path_2

        # 1. Load the two files
        _, state_1 = SavePPOJax.load_agent(ckpt_path_1)  # Contains 1 seed
        _, state_2 = SavePPOJax.load_agent(ckpt_path_2)  # Contains 2 seeds

        # 2. Convert to raw dictionaries
        dict_seed_0 = to_state_dict(state_1.train_state)
        dict_seed_1 = to_state_dict(state_2.train_state)


        # 4. Define the stacking function
        def stack_two_seeds(s0, s1):
            # Ensure they are arrays and add a batch dimension to each
            # This turns shape (weights) into (1, weights)
            nodes = [jnp.expand_dims(jnp.array(n), 0) for n in [s0, s1]]
            return jnp.concatenate(nodes, axis=0)

        # 5. Merge all three individual seeds
        merged_dict = jax.tree.map(stack_two_seeds, dict_seed_0, dict_seed_1)

        # 6. Reconstruct the TrainState object
        merged_train_state = from_state_dict(state_1.train_state, merged_dict)

        # --- NEW STEP: Wrap it back into an AgentState ---
        # state_1 is an instance of PPOAgentState (or similar)
        # We use its constructor to wrap our merged TrainState
        runner_state = state_1.__class__(train_state=merged_train_state)

        print(f"Loading and merging checkpoints: {ckpt_path_1} and {ckpt_path_2}")
        
        # Initialize agent configuration
        agent_conf = SavePPOJax.init_agent_conf(env, config)

        # # Check if a checkpoint path is provided and exists in the Hydra config
        # if config.checkpoint_path is not None:# and os.path.exists(config.checkpoint_path):
        #     print(f"Loading agent state from: {config.checkpoint_path}")
        #     # Load the agent configuration and state
        #     agent_conf, agent_state = SavePPOJax.load_agent(config.checkpoint_path)
        #     print("Agent state loaded successfully.")
        #     new_agent_conf = SavePPOJax.init_agent_conf(env, config)
            
        #     agent_conf = new_agent_conf 
        # else:
        #     print("No checkpoint path provided or file not found. Initializing new agent.")
        #     # If no checkpoint, initialize a new agent configuration
        #     agent_conf = SavePPOJax.init_agent_conf(env, config)

        save_path = SavePPOJax.save_conf(result_dir, agent_conf)
        run.config.update({"agent_conf_save_path": save_path})
        
        config.experiment.result_dir = result_dir
       
        # setup metric handler (optional)
        mh = MetricsHandler(config, env) if config.experiment.validation.active else None

        # build training function
        # train_state_seed = 1
        # train_state = jax.tree.map(lambda x: x[train_state_seed], agent_state.train_state)
        # train_fn = SavePPOJax.build_train_fn_continue(env, agent_conf, agent_state=train_state, mh=mh) #PPOJax.build_train_fn(env, agent_conf, mh=mh)
       

        # 2. Jit and vmap training function
        if config.experiment.n_seeds > 1:
            # 1. Build the base training function (singular)
            # This function expects a SINGLE seed's state
            train_fn_base = SavePPOJax.build_train_fn_continue_multiseeds(env, agent_conf, agent_state=runner_state, mh=mh)

            # We map over axis 0 for both the RNG (arg 0) and the State (arg 1)
            train_fn = jax.jit(jax.vmap(train_fn_base, axis_name='batch', in_axes=(0, 0))) #, in_axes=(0, 0)))
            skip_seed = 4 #1
            rngs_list = [jax.random.PRNGKey(i) for i in range(skip_seed, config.experiment.n_seeds + 1)]
            _rngs = jnp.stack(rngs_list[1:]) # This gives (2, 2) which matches your (2, 510) state
            
            # 4. Run training passing BOTH rngs and the train_state
            print('Starting Training with multiple inputs and multiple seeds...')
            out = train_fn(_rngs, runner_state.train_state) #agent_state.train_state)


            # # ####################WRONG?????????????? 
            # train_fn_base = SavePPOJax.build_train_fn_continue(env, agent_conf, agent_state=runner_state, mh=mh)

            # # We map over axis 0 for both the RNG (arg 0) and the State (arg 1)
            # train_fn = jax.jit(jax.vmap(train_fn_base, axis_name='batch')) #, in_axes=(0, 0)))
            
            # 3. Prepare RNGs (Ensure you have 1 key for each seed in the checkpoint)
            # Since your checkpoint has 2 seeds, we need 2 keys.
            # Your logic: if n_seeds=3 and skip=1, rngs=[1,2,3]. _rng=rngs[1:]=[2,3] (Size 2)
            # skip_seed = 0 #1
            # rngs_list = [jax.random.PRNGKey(i) for i in range(skip_seed, config.experiment.n_seeds + skip_seed)]
            # _rngs = jnp.stack(rngs_list[1:]) # This gives (2, 2) which matches your (2, 510) state
            
            # # 4. Run training passing BOTH rngs and the train_state
            # print('Starting Training with multiple inputs and multiple seeds...')
            # out = train_fn(_rngs) #, runner_state) #agent_state.train_state)
            #########################################??????????????
        # else:
        #     train_fn = SavePPOJax.build_train_fn_continue(env, agent_conf, agent_state=agent_state.train_state, mh=mh)
        #     # Single seed case
        #     train_fn = jax.jit(jax.vmap(train_fn, axis_name='batch')) if config.experiment.n_seeds > 1 else jax.jit(train_fn)
        #     rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds+1)]
        #     # Slice the state to get only the first seed if only 1 seed is requested
        #     rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
        #     out = train_fn(_rng)


        # save agent state
        agent_state = out["agent_state"]
        save_path = SavePPOJax.save_agent(result_dir, agent_conf, agent_state) #PPOJax.save_agent(result_dir, agent_conf, agent_state)
        run.config.update({"agent_save_path": save_path})

        print(f"CHECKPOINT_PATH:{save_path}")

        import time
        t_start = time.time()
        # get the metrics and log them
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
                    # if site_rpos is jnp.empty(0) then calculate metric sweep differently: vel_x + vel_z + torque_at_limit else keep it as it is 
                    # site_rpos = validation_metrics.euclidean_distance.site_rpos[i]
                    # site_rrotvec = validation_metrics.euclidean_distance.site_rrotvec[i]
                    # site_rvel = validation_metrics.euclidean_distance.site_rvel[i]

                    if jnp.all(validation_metrics.euclidean_distance.site_rpos == 0):
                        metric_value = validation_metrics.euclidean_distance.vel_x[i] + validation_metrics.euclidean_distance.vel_z[i] + validation_metrics.euclidean_distance.torque_at_limit[i]
                    else:
                        site_rpos = validation_metrics.euclidean_distance.site_rpos[i]
                        site_rrotvec = validation_metrics.euclidean_distance.site_rrotvec[i]
                        site_rvel = validation_metrics.euclidean_distance.site_rvel[i]
                        metric_value = site_rpos + site_rrotvec + site_rvel

                    run.log({"Metric for Sweep": metric_value},
                            step=int(training_metrics.max_timestep[i]))

        print(f"Time taken to log metrics: {time.time() - t_start}s")

        # run the environment with the trained agent to record video
        SavePPOJax.play_policy(env, agent_conf, agent_state.train_state, deterministic=True, n_steps=200, n_envs=20, record=True,
                           train_state_seed=0)
        # PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=200, n_envs=20, record=True,
        #                    train_state_seed=0)
        video_file = env.video_file_path
        run.log({"Agent Video": wandb.Video(video_file)})

        wandb.finish()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
