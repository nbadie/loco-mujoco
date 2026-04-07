"""
alignment_eval uses a trained policy to systematically test and evaluate different alignments 
"""
import os 

# Uncomment the following lines to force JAX to use CPU instead of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
# Uncomment the following line to force JAX to use CPU instead of GPU
jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
import argparse
import pickle
from datetime import datetime
from omegaconf import OmegaConf
import numpy as np

from loco_mujoco.core.wrappers import VecEnv
from loco_mujoco import TaskFactory


from loco_mujoco.algorithms import PPOJax
from loco_mujoco.evaluation.eval_metrics import ProsthesisEvalMetrics



os.environ["MUJOCO_GL"] = "egl"  # Use EGL for headless rendering

# Setup folder for saving results
subfolder_name = 'parallel_eval'


def _infer_leading_batch_size(pytree):
    sizes = []
    for leaf in jax.tree_util.tree_leaves(pytree):
        if hasattr(leaf, "shape") and len(leaf.shape) > 0:
            sizes.append(int(leaf.shape[0]))
    if not sizes:
        return None
    unique_sizes = set(sizes)
    if len(unique_sizes) == 1:
        return sizes[0]
    return None


def _get_train_state_batch_size(train_state):
    params_batch = _infer_leading_batch_size(train_state.params)
    if params_batch is not None:
        return params_batch
    return _infer_leading_batch_size(train_state.run_stats)


def _prepare_train_state_for_eval(train_state, n_seeds):
    batch_size = _get_train_state_batch_size(train_state)
    if batch_size is None:
        return train_state, False

    if batch_size == n_seeds:
        return train_state, True

    if batch_size == 1 and n_seeds > 1:
        expanded_train_state = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, n_seeds, axis=0)
            if hasattr(x, "shape") and len(x.shape) > 0 and x.shape[0] == 1
            else x,
            train_state,
        )
        return expanded_train_state, True

    if n_seeds == 1 and batch_size > 1:
        squeezed_train_state = jax.tree_util.tree_map(
            lambda x: x[0]
            if hasattr(x, "shape") and len(x.shape) > 0 and x.shape[0] == batch_size
            else x,
            train_state,
        )
        return squeezed_train_state, False

    raise ValueError(
        f"Incompatible train_state batch size ({batch_size}) for n_seeds={n_seeds}."
    )


def _set_all_axes_zero_for_body(rand_conf, param_name, body_name):
    range_key = f"{param_name}_range"
    for axis in rand_conf[range_key][body_name]:
        rand_conf[range_key][body_name][axis] = [0.0, 0.0]


def _generate_output_path(save_dir, param_info, seed_idx):
    if param_info is None:
        tag = "base"
    else:
        value = param_info['val']
        if 'orientation' in param_info['name']:
            value = np.rad2deg(value)
            unit = 'deg'
        elif 'position' in param_info['name']:
            value = value * 1000
            unit = 'mm'
        else:
            unit = ''

        tag = (
            f"eval_{param_info['n_steps']}steps_"
            f"{param_info['name']}_{param_info['body']}_"
            f"{param_info['direction']}_{int(np.round(value))}{unit}"
        )

    out_folder = os.path.join(save_dir, tag)
    #os.makedirs(out_folder, exist_ok=True)
    return os.path.join(f"{out_folder}_seed{seed_idx}.pkl")

def main():
    # Setup argument parser
    # example usage:
    # python examples/parameters_perturbations/parameter_eval.py --path .pkl --param randomize_prosthesis_body_position --range -0.01 0.01 --direction x --n_steps 2000
    parser = argparse.ArgumentParser(description="Evaluate a trained policy under different alignments.")
    parser.add_argument("--path", type=str, required=True, help="Path to the trained agent pkl file")
    parser.add_argument("--param", type=str, required=True,  help="Specific parameter to randomize")
    parser.add_argument("--direction", type=str, required=False, help="Direction to randomize (e.g., x, y, z for position/orientation)")
    parser.add_argument("--range", type=float, nargs=2, required=True, help="Range for the parameter (min max)")
    parser.add_argument("--n_steps", type=int, default=2000, help="Number of steps to run for each evaluation")
    args = parser.parse_args()


    # Load agent 
    path = args.path 
    agent_conf, agent_state = PPOJax.load_agent(path)
    config = agent_conf.config
    n_seeds = config.experiment.n_seeds 

    def sample_actions_uncompiled(ts, obs, _rng):
        y, updated_state = agent_conf.network.apply(
            {'params': ts.params, 'run_stats': ts.run_stats},
            obs,
            mutable=['run_stats']
        )
        pi, _ = y
        action = pi.sample(seed=_rng)
        ts = ts.replace(run_stats=updated_state['run_stats'])
        return action, ts

    parameter_to_randomize = args.param.replace("randomize_", "")
    list_parameter_to_randomize = [parameter_to_randomize]
    rand_param_min, rand_param_max = args.range
    parameter_direction = args.direction
    n_steps = args.n_steps

    # Randomization options
    #randomization_param_names = ['randomize_prosthesis_dof_damping','randomize_prosthesis_joint_stiffness','randomize_prosthesis_body_position','randomize_prosthesis_body_orientation']
    # Build randomization_param_eval based on command-line arguments
    randomization_param_eval = {
        "randomize_prosthesis_dof_damping": False,
        "prosthesis_dof_damping_range": {'ankle_angle': [800, 1000]},
        "randomize_prosthesis_joint_stiffness": False,
        "prosthesis_joint_stiffness_range": {'ankle_angle': [1, 5]},
        "randomize_prosthesis_body_position": False, #True,
        "prosthesis_body_position_range": {'pylon_socket': {'x': [0.0,0.0], 'z': [0,0]}, 'talus': {'x': [0.0,0.0], 'z': [0,0]}},
        "randomize_prosthesis_body_orientation": True,
        "prosthesis_body_orientation_range": {'pylon_socket': {'x': [0.0,0.0], 'z':[0,0]}, 'talus': {'x': [0.0,0.0],'z':[0,0]}},
    }
    
    # Override with command-line parameters
    randomize_key = f"randomize_{parameter_to_randomize}"
    range_key = f"{parameter_to_randomize}_range"
    
    print(randomize_key)

    if randomize_key in randomization_param_eval:
        randomization_param_eval[randomize_key] = True
        
        # Update range from command-line arguments
        if range_key in randomization_param_eval:
            if "position" in parameter_to_randomize or "orientation" in parameter_to_randomize:
                body_name = list(randomization_param_eval[range_key].keys())[0]
                if args.direction:
                    randomization_param_eval[range_key][body_name] = {parameter_direction: [rand_param_min, rand_param_max]}
            else:
                joint_name = list(randomization_param_eval[range_key].keys())[0]
                randomization_param_eval[range_key] = {joint_name: [rand_param_min, rand_param_max]}
    
    randomization_increments = {
        "prosthesis_dof_damping": 100, 
        "prosthesis_joint_stiffness": 1, 
        "prosthesis_body_position": 0.005, 
        "prosthesis_body_orientation": 0.005, 
    }

    OmegaConf.set_struct(config, False)
    config.experiment.env_params["headless"] = True #False
    config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic
    config.experiment.env_params["add_sensors"] = True

    # config.experiment.env_params= {"prosthesis_side":"right", #"bilateral", #
    #     "prosthesis_type":"transtibial", #"None", #"dynamic_ankle",
    #     "prosthesis_subtype":"SACH", #"rigid_socket",
    #     "remove_joint_names" : ['subtalar_angle'],
    #     # use_box_feet=True,
    #     "multi_contact_geom_type":"2boxes",
    #     "amputated_tibia_length" : 0.2, 
    #     "tibia_socket_overlap" : 0.2, 
    #     "adapt_joint_range" : {'ankle_angle_l': [-20,20], 'ankle_angle_r': [-10,10], 'hip_flexion_r': [-30,35],'hip_flexion_l': [-30,35], 'knee_angle_r': [-120,5], 'knee_angle_l': [-120,5]},
    #     "contact_geom_solref":  [-900,-300], 
    #     "joint_stiffness" : {'ankle_angle': 900},
    #     "reattach_muscles" : {'med_gas': [0,0,0]},
    #     "add_pos_ori_to_observation": True}



    randomization_type = config.randomization_config["randomization_type"]
    #randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

    print('randomization_params', randomization_param_eval)

    env = factory.make(
        **config.experiment.env_params,
        **config.experiment.task_factory.params,
        domain_randomization_type=randomization_type,
        domain_randomization_params=randomization_param_eval
    )

    env.th.to_jax()
    env = VecEnv(env)

    prosthesis_eval_metrics = ProsthesisEvalMetrics(env)

    def _collect_sensor_forces(all_sensor_force, env_data):
        sensor_force, sensor_force_names = prosthesis_eval_metrics.get_sensor_data(env_data)
        for sensor_name, force in sensor_force.items():
            if sensor_name not in all_sensor_force:
                all_sensor_force[sensor_name] = []
            all_sensor_force[sensor_name].append(force)
        return sensor_force_names
    

    def _collect_grf(env_data):
        foot_name = "toes" #+ prosthesis_side_suffix
        calcn_name = "calcn"
        # GRF collection
        grf_foot_l = prosthesis_eval_metrics.get_grf(env_data, f"{foot_name}_l")
        grf_foot_r = prosthesis_eval_metrics.get_grf(env_data, f"{foot_name}_r")
        grf_calcn_l = prosthesis_eval_metrics.get_grf(env_data, f"{calcn_name}_l")
        grf_calcn_r = prosthesis_eval_metrics.get_grf(env_data, f"{calcn_name}_r")

        grf_l = grf_foot_l + grf_calcn_l
        grf_r = grf_foot_r + grf_calcn_r
        return grf_l, grf_r
    

    # Always use batched reset/step; this supports both n_seeds == 1 and n_seeds > 1.
    jit_step  = jax.jit(jax.vmap(env.mjx_step)) 
    rng = jax.random.key(0)
    keys = jax.random.split(rng, n_seeds + 1)
    rng, env_keys = keys[0], keys[1:]
    # rng_seeds = jnp.array([jax.random.PRNGKey(i) for i in range(1, n_seeds + 1)])
    # prepared_train_state, train_state_is_batched = _prepare_train_state_for_eval(
    #     agent_state.train_state,
    #     n_seeds,
    # )
    prepared_train_state = agent_state.train_state
    sample_actions= jax.jit(sample_actions_uncompiled)

    def run_evaluation_parallel(
            n_steps: int, 
            n_seeds: int,
            env_states: jnp.ndarray,
            agent_state, 
            rng: jnp.ndarray,
            save_dir: str,
            param_info= None
    ) -> None:
        """
        Run evaluation in parallel across multiple seeds.
        
        Args:
            n_steps: Number of simulation steps per seed
            n_seeds: Number of parallel seeds to evaluate (batch size)
            env_states: Batched environment states
            agent_state: Agent training state
            rng: Random number generator keys
            save_dir: Directory to save results
            param_info: Parameter configuration info (optional)
        """
        all_sensor_force = {}
        all_grf_l = []
        all_grf_r = []

        current_train_state = prepared_train_state
        for t in range(n_steps):
            rng, _rng = jax.random.split(rng)

            # Get actions for all seeds
            obs = env_states.observation
            actions, current_train_state = sample_actions(current_train_state, obs, _rng)
            actions = jnp.atleast_2d(actions)
            # Step physics for all seeds
            env_states = jit_step(env_states, actions)

            if (t % 500) == 0:
                print(f"  Step {t}/{n_steps} completed.")


            sensor_force_names = _collect_sensor_forces(all_sensor_force, env_states.data)
            grf_l, grf_r = _collect_grf(env_states.data)
            all_grf_l.append(grf_l)
            all_grf_r.append(grf_r)

        # Save results for each checkpoint
        for seed_idx in range(n_seeds):
            # checkpoint_file = checkpoint_files[batch_idx]
            print(f"Processing {seed_idx+1}")
            all_relevant_data = {
                "all_grf_l": prosthesis_eval_metrics.extract_checkpoint_data(all_grf_l, seed_idx),
                "all_grf_r": prosthesis_eval_metrics.extract_checkpoint_data(all_grf_r, seed_idx),
                "all_sensor_force": prosthesis_eval_metrics.extract_checkpoint_data(all_sensor_force, seed_idx),
                "sensor_force_names": sensor_force_names,
            }

            #tag = f"eval_{n_steps}steps_{param_info['name']}_{param_info['body']}_{param_info['direction']}_{int(np.round(name_param_value,0))}{name_units}" if param_info else "base"
            #filename = f"seed{seed_idx}.pkl"
            # out_path = os.path.join(save_dir, filename)
            out_path = _generate_output_path(save_dir, param_info, seed_idx)

            with open(out_path, "wb") as f:
                pickle.dump(all_relevant_data, f)
            print(f"Saved Seed {seed_idx} -> {out_path}")


    dt_init = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(path), f'{dt_init}_{subfolder_name}')
    os.makedirs(save_dir, exist_ok=True)


    def process_stiffness_damping_param(param_name, randomization_params_eval, randomization_increments):
        """Process stiffness/damping parameters."""
        joint_name = list(randomization_params_eval[f"{param_name}_range"].keys())[0]
        min_val, max_val = randomization_params_eval[f"{param_name}_range"][joint_name]
        increment = randomization_increments[param_name]
        
        return {
            "joint_name": joint_name,
            "min_val": min_val,
            "max_val": max_val,
            "increment": increment,
            "is_range_type": False,
        }


    def process_position_orientation_param(param_name, randomization_params_eval, randomization_increments):
        """Process position/orientation parameters."""
        body_range = randomization_params_eval[f"{param_name}_range"]
        body_name = list(body_range.keys())[0]
        directions = list(body_range[body_name].keys())
        increment = randomization_increments[param_name]
        
        return {
            "body_name": body_name,
            "directions": directions,
            "body_range": body_range,
            "increment": increment,
            "is_range_type": True,
        }


    def run_and_save_evaluation(param_name, current_value, axis_or_joint, body_or_joint_name, 
                                n_steps, n_seeds, env_states, agent_state, rng, save_dir):
        """Unified function to run evaluation with consistent parameters."""
        param_info = {
            "name": param_name,
            "val": current_value,
            "direction": axis_or_joint,
            "body": body_or_joint_name,
            "n_steps": n_steps
        }

        run_evaluation_parallel(
            n_steps=n_steps,
            n_seeds=n_seeds,
            env_states=env_states,
            agent_state=agent_state,
            rng=rng,
            save_dir=save_dir,
            param_info=param_info
        )


    def reset_and_eval(param_name, current_value, axis_or_joint, body_or_joint_name):
        jit_reset = jax.jit(jax.vmap(env.mjx_reset))
        env_states = jit_reset(env_keys)
        run_and_save_evaluation(
            param_name=param_name,
            current_value=current_value,
            axis_or_joint=axis_or_joint,
            body_or_joint_name=body_or_joint_name,
            n_steps=n_steps,
            n_seeds=n_seeds,
            env_states=env_states,
            agent_state=agent_state,
            rng=rng,
            save_dir=save_dir,
        )


    # Iterate over enabled randomization parameters
    for param_name in list_parameter_to_randomize: #randomization_param_names:
        randomize_param_name = f"randomize_{param_name}"
        if randomize_param_name not in randomization_param_eval or not randomization_param_eval.get(randomize_param_name, False):
            continue
        
        if "damping" in param_name or "stiffness" in param_name:
            param_config = process_stiffness_damping_param(param_name, randomization_param_eval, randomization_increments)
            min_val, max_val = param_config["min_val"], param_config["max_val"]
            increment = param_config["increment"]
            
            current_value = min_val
            while current_value <= max_val + 1e-9:
                env._domain_randomizer.rand_conf[f"{param_name}_range"][param_config["joint_name"]] = [
                    current_value,
                    current_value,
                ]
                print('Current value:', param_config["joint_name"], current_value)
                print('env._domain_randomizer.rand_conf:', env._domain_randomizer.rand_conf)
                reset_and_eval(param_name, current_value, param_config["joint_name"], param_config["joint_name"])
                current_value += increment
        
        elif "position" in param_name or "orientation" in param_name:
            param_config = process_position_orientation_param(param_name, randomization_param_eval, randomization_increments)
            body_name = param_config["body_name"]
            increment = param_config["increment"]
            
            for direction in param_config["directions"]:
                min_val, max_val = param_config["body_range"][body_name][direction]
                current_value = min_val
                
                while current_value <= max_val + 1e-9:
                    _set_all_axes_zero_for_body(env._domain_randomizer.rand_conf, param_name, body_name)
                    env._domain_randomizer.rand_conf[f"{param_name}_range"][body_name][direction] = [current_value, current_value]
                    print('Current value:', body_name, direction, current_value)
                    print('env._domain_randomizer.rand_conf:', env._domain_randomizer.rand_conf)
                    reset_and_eval(param_name, current_value, direction, body_name)
                    current_value += increment



if __name__ == "__main__":
    main()














    
















