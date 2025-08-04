import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 
# os.environ["JAX_PLATFORMS"] = "cpu"
import orbax.checkpoint as ocp
import jax
# jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from flax.training.train_state import TrainState # Make sure this import is correct
from loco_mujoco.algorithms.common.dataclasses import TrainState #as YourActualTrainStateClass # Or your actual class
import argparse
import hydra 
import traceback
import sys

from omegaconf import DictConfig, OmegaConf #OmegaConf is a YAML based hierarchical configuration system, with support for merging configurations from multiple sources



from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import SavePPOJax
from loco_mujoco.utils import MetricsHandler

from omegaconf import OmegaConf
import wandb

from dataclasses import fields
from loco_mujoco.utils.metrics import QuantityContainer


# SOLUTION for your specific error:
def load_checkpoint_with_device_fix(ckpt_path, train_state_template):
    """
    Load checkpoint with proper device handling and target tree
    This fixes the TFRT_CPU_0 device error and missing target tree warning
    """
    import jax
    
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



os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering, which is more compatible with headless environments



@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    try: 
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
        

        config.experiment.result_dir = result_dir
        if config.checkpoint_path is not None:# and os.path.exists(config.checkpoint_path):
            print(f"Loading agent state from: {config.checkpoint_path}")
            # Load the agent configuration and state
            # agent_conf, agent_state = SavePPOJax.load_agent(config.checkpoint_path)
            # print("Agent state loaded successfully.")
            new_agent_conf = SavePPOJax.init_agent_conf(env, config)
            agent_conf = new_agent_conf
            

            checkpoint_path = config.checkpoint_path
            # extract static agent info
            config, network, tx =\
                (agent_conf.config.experiment, agent_conf.network, agent_conf.tx)

            rng = jax.random.PRNGKey(seed=0)
            rng, _rng1, _rng2 = jax.random.split(rng, 3)
            init_x = jnp.zeros(env.info.observation_space.shape)
            network_params = network.init(_rng1, init_x)
            train_state=None

            # init new train states from old params
            train_state_template = TrainState.create(
                apply_fn=network.apply,
                params=network_params["params"] if train_state is None else train_state.params,
                run_stats=network_params["run_stats"] if train_state is None else train_state.run_stats,
                tx=tx,
            )

            # To continue training from your saved checkpoint:
            loaded_state = SavePPOJax.load_checkpoint_with_device_fix(checkpoint_path, train_state_template) #load_checkpoint_callback(ckpt_path)

            # Extract the components
            params = loaded_state['params']
            run_stats = loaded_state['run_stats'] 
            current_step = loaded_state['step']  # This should be 3073
            opt_state = loaded_state['opt_state']


            train_state= TrainState.create(
                apply_fn=network.apply,
                params=params,
                run_stats=run_stats,
                tx=tx,
            )
        else:
            print("No checkpoint path provided or file not found. Initializing new agent.")
            # If no checkpoint, initialize a new agent configuration
            agent_conf = SavePPOJax.init_agent_conf(env, config)

        save_path = SavePPOJax.save_conf(result_dir, agent_conf)
        run.config.update({"agent_conf_save_path": save_path})
        # setup metric handler (optional)
        mh = MetricsHandler(agent_conf.config, env) if agent_conf.config.experiment.validation.active else None

        # build training function
        train_fn = SavePPOJax.build_train_fn_continue(env, agent_conf, agent_state=train_state, mh=mh) #PPOJax.build_train_fn(env, agent_conf, mh=mh)


        # jit and vmap training function
        train_fn = jax.jit(jax.vmap(train_fn)) if agent_conf.config.experiment.n_seeds > 1 else jax.jit(train_fn)

        # get rng keys and run training
        # skip_seed = 1
        # rngs = [jax.random.PRNGKey(i) for i in range(skip_seed,config.experiment.n_seeds+1)]  # create rngs from seed
        rngs = [jax.random.PRNGKey(i) for i in range(agent_conf.config.experiment.n_seeds+1)]  # create rngs from seed
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
        out = train_fn(_rng) #, agent_state=agent_state)


        # save agent state
        agent_state = out["agent_state"]
        save_path = SavePPOJax.save_agent(result_dir, agent_conf, agent_state) #PPOJax.save_agent(result_dir, agent_conf, agent_state)
        run.config.update({"agent_save_path": save_path})

        import time
        t_start = time.time()
        # get the metrics and log them
        if not agent_conf.config.experiment.debug:
            training_metrics = out["training_metrics"]
            validation_metrics = out["validation_metrics"]

            # calculate mean across seeds
            training_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics)
            validation_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics)

            for i in range(len(training_metrics.mean_episode_return)):
                run.log({"Mean Episode Return": training_metrics.mean_episode_return[i],
                         "Mean Episode Length": training_metrics.mean_episode_length[i]},
                        step=int(training_metrics.max_timestep[i]))

                if (i+1) % agent_conf.config.experiment.validation_interval == 0 and agent_conf.config.experiment.validation.active:
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












# # Set up argument parser
# parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
# parser.add_argument('--state_path', type=str, required=True, help='Path to the agent state file')
# parser.add_argument('--conf_path', type=str, required=True, help='Path to the agent conf file')
# parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
# args = parser.parse_args()

# # Use the path from command line arguments
# path = args.conf_path
# agent_conf, agent_state = SavePPOJax.load_agent(path)
# config = agent_conf.config
# # config.experiment.env_params.reward_params.joint_torque_coeff=0.003 #5 #2 #0.001 #0.01
# # config.experiment.env_params.reward_params.sites_for_mimic = "upper_body_mimic"
# # get task factory
# factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# # create env
# OmegaConf.set_struct(config, False)  # Allow modifications
# config.experiment.env_params["headless"] = False
# config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic

# randomization_type = config.randomization_config["randomization_type"]
# # Convert to plain dict to allow adding new keys
# randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)

# # add prosthesis side to randomization params if it exists in config.experiment.env_params
# if "prosthesis_side" in config.experiment.env_params:
#     randomization_params["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]


# env = factory.make(domain_randomization_type=randomization_type, domain_randomization_params=randomization_params,
#                    **config.experiment.env_params, **config.experiment.task_factory.params)










# # extract static agent info
# config, network, tx =\
#     (agent_conf.config.experiment, agent_conf.network, agent_conf.tx)

# rng = jax.random.PRNGKey(seed=0)
# rng, _rng1, _rng2 = jax.random.split(rng, 3)
# init_x = jnp.zeros(env.info.observation_space.shape)
# network_params = network.init(_rng1, init_x)
# train_state=None
# # init new train states from old params
# train_state_template = TrainState.create(
#     apply_fn=network.apply,
#     params=network_params["params"] if train_state is None else train_state.params,
#     run_stats=network_params["run_stats"] if train_state is None else train_state.run_stats,
#     tx=tx,
# )

# # To continue training from your saved checkpoint:
# ckpt_path = args.state_path
# #'/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-07-24/19-20-30/checkpoints/ckpt_3073/'
# #"ckpt_3073"  # Base path to your checkpoint
# loaded_state = load_checkpoint_with_device_fix(ckpt_path, train_state_template) #load_checkpoint_callback(ckpt_path)

# # Extract the components
# params = loaded_state['params']
# run_stats = loaded_state['run_stats'] 
# current_step = loaded_state['step']  # This should be 3073
# opt_state = loaded_state['opt_state']


# # # Reconstruct your train_state (adapt this to your TrainState structure)
# # train_state = TrainState(
# #     params=params,
# #     run_stats=run_stats,
# #     step=current_step,
# #     opt_state=opt_state
# # )

# train_state= TrainState.create(
#     apply_fn=network.apply,
#     params=params,
#     run_stats=run_stats,
#     tx=tx,
# )
# # # Determine which evaluation environment to run
# # if args.use_mujoco:
# #     # run eval mujoco
# #     SavePPOJax.play_policy_mujoco(env, agent_conf, train_state, deterministic=False, n_steps=1000, record=True,
# #                               train_state_seed=0)
# # else:
# #     # run eval mjx
# #     SavePPOJax.play_policy(env, agent_conf, train_state, deterministic=False, n_steps=1000, n_envs=1, record=True,
# #                        train_state_seed=0)

# # # Accessing the current sweep number
# result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir


# # Extract date and time from the result directory path for wandb run name
# result_dir_parts = result_dir.split("/")
# result_dir_date, result_dir_time = result_dir_parts[-2], result_dir_parts[-1]
# formatted_result_dir = f"{result_dir_date}_{result_dir_time}_"

# agent_conf.experiment.result_dir=result_dir


# # setup wandb 
# wandb.login()
# config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
# run = wandb.init(project=config.wandb.project, name= formatted_result_dir + config.wandb.run_name, config=config_dict)


# # setup metric handler (optional)
# mh = MetricsHandler(config, env) if config.experiment.validation.active else None

# # build training function
# train_fn = SavePPOJax.build_train_fn_continue(env, agent_conf, agent_state=agent_state, mh=mh)


# # jit and vmap training function
# train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

# # get rng keys and run training
# # skip_seed = 1
# # rngs = [jax.random.PRNGKey(i) for i in range(skip_seed,config.experiment.n_seeds+1)]  # create rngs from seed
# rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds+1)]  # create rngs from seed
# rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
# out = train_fn(_rng)


# # save agent state
# agent_state = out["agent_state"]
# save_path = SavePPOJax.save_agent(config.experiment.result_dir, agent_conf, agent_state) #PPOJax.save_agent(result_dir, agent_conf, agent_state)
# run.config.update({"agent_save_path": save_path})

# import time
# t_start = time.time()
# # get the metrics and log them
# if not config.experiment.debug:
#     training_metrics = out["training_metrics"]
#     validation_metrics = out["validation_metrics"]

#     # calculate mean across seeds
#     training_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics)
#     validation_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics)

#     for i in range(len(training_metrics.mean_episode_return)):
#         run.log({"Mean Episode Return": training_metrics.mean_episode_return[i],
#                     "Mean Episode Length": training_metrics.mean_episode_length[i]},
#                 step=int(training_metrics.max_timestep[i]))

#         if (i+1) % config.experiment.validation_interval == 0 and config.experiment.validation.active:
#             run.log({"Validation Info/Mean Episode Return": validation_metrics.mean_episode_return[i],
#                         "Validation Info/Mean Episode Length": validation_metrics.mean_episode_length[i]},
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

# print(f"Time taken to log metrics: {time.time() - t_start}s")

# # # run the environment with the trained agent to record video
# # SavePPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=200, n_envs=20, record=True,
# #                    train_state_seed=0)
# # # PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=200, n_envs=20, record=True,
# # #                    train_state_seed=0)
# # video_file = env.video_file_path
# # run.log({"Agent Video": wandb.Video(video_file)})

# # wandb.finish()

