import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"

import jax 
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import lax
import pickle
import numpy as np
import argparse
import sys
from loco_mujoco.core.wrappers import VecEnv
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from omegaconf import OmegaConf
sys.path.append(os.path.join(os.path.dirname(__file__), "/home/nadinebadie/loco-mujoco/prosthesis_test"))
from loco_mujoco.evaluation.evaluation_prosthesis import ProsthesisMetricsHandler
from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction
import mujoco
from datetime import datetime
import timeit 

# Parameter randomization initialization
randomization_params_names = ["prosthesis_dof_damping", "prosthesis_joint_stiffness", "prosthesis_body_position", "prosthesis_body_orientation"]
randomization_params_eval = {
    "prosthesis_side": "left_side",
    "randomize_prosthesis_dof_damping": False,
    "prosthesis_dof_damping_range": {'ankle_angle': [2, 10]},
    "randomize_prosthesis_joint_stiffness": False,
    "prosthesis_joint_stiffness_range": {'ankle_angle': [50, 100]},
    "randomize_prosthesis_body_position": True,
    "prosthesis_body_position_range": {'pylon_socket': {'x': [0.2, 0.25]}},
    "randomize_prosthesis_body_orientation": False,
    "prosthesis_body_orientation_range": {'pylon_socket': {'x': [-0.1, 0.1]}},
}
randomization_increments = {
    "prosthesis_joint_stiffness": 10,
    "prosthesis_dof_damping": 5,
    "prosthesis_body_position": 0.05,
    "prosthesis_body_orientation": 0.1
}

os.environ["MUJOCO_GL"] = "egl"

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

# Load agent
path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config

randomization_type = config.randomization_config["randomization_type"]
randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)

# Update values in randomization_params based on randomization_params_eval
for key, value in randomization_params_eval.items():
    if key in randomization_params:
        randomization_params[key] = value
    else:
        randomization_params[key] = value

# Add prosthesis side to randomization params if it exists in config.experiment.env_params
if "prosthesis_side" in config.experiment.env_params:
    randomization_params["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]

# Get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# Create env
OmegaConf.set_struct(config, False)
config.experiment.env_params["headless"] = True
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"
config.experiment.env_params["add_sensors"] = True
env = factory.make(domain_randomization_type=randomization_type, domain_randomization_params=randomization_params,
                   **config.experiment.env_params, **config.experiment.task_factory.params)
env.th.to_jax()
env = VecEnv(env)
model = env.get_model()

prosthesis_metrics_handler = ProsthesisMetricsHandler(env)
muscle_skeleton_control_activation = SkeletonMuscleControlFunction(env)

n_steps = 200
n_envs = 1
rng = jax.random.key(0)
train_state_seed = 0

keys = jax.random.split(rng, n_envs + 1)
rng, env_keys = keys[0], keys[1:]

def sample_actions_uncompiled(ts, obs, _rng):
    y, updates = agent_conf.network.apply({'params': ts.params,
                                           'run_stats': ts.run_stats},
                                           obs, mutable=["run_stats"])
    ts = ts.replace(run_stats=updates['run_stats'])
    pi, _ = y
    a = pi.sample(seed=_rng)
    return a, ts

sample_actions = jax.jit(sample_actions_uncompiled)

if config.experiment.n_seeds > 1:
    assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
                                            "train_state_seed for replay.")
    train_state = jax.tree.map(lambda x: x[train_state_seed], agent_state.train_state)
else: 
    train_state = agent_state.train_state

# JIT compile step and reset functions once
jit_step = jax.jit(jax.vmap(env.mjx_step_test))
jit_reset = jax.jit(jax.vmap(env.mjx_reset))

def create_parameter_configs():
    """Create all parameter configurations as separate lists to avoid JAX array issues"""
    configs = []
    
    for param_idx, param_name in enumerate(randomization_params_names):
        if not randomization_params_eval[f"randomize_{param_name}"]:
            continue
            
        if "stiffness" in param_name or "damping" in param_name:
            joint_name = list(randomization_params_eval[f"{param_name}_range"].keys())[0]
            min_val, max_val = randomization_params_eval[f"{param_name}_range"][joint_name]
            increment = randomization_increments[param_name]
            
            current_value = min_val
            while current_value <= max_val:
                config = {
                    'param_idx': param_idx,
                    'param_name': param_name,
                    'param_value': float(current_value),
                    'joint_name': joint_name,
                    'body_name': None,
                    'axis': None
                }
                configs.append(config)
                current_value += increment
                
        elif "position" in param_name or "orientation" in param_name:
            body_range = randomization_params_eval[f"{param_name}_range"]
            body_name = list(body_range.keys())[0]
            directions = list(body_range[body_name].keys())
            
            for axis in directions:
                min_val = body_range[body_name][axis][0]
                max_val = body_range[body_name][axis][1]
                inc = randomization_increments[param_name]
                
                current_value = min_val
                while current_value <= max_val + 1e-6:
                    config = {
                        'param_idx': param_idx,
                        'param_name': param_name,
                        'param_value': float(current_value),
                        'joint_name': None,
                        'body_name': body_name,
                        'axis': axis
                    }
                    configs.append(config)
                    current_value += inc
    
    return configs

def update_env_randomization_config(param_config):
    """Update environment randomization parameters based on config"""
    param_name = param_config['param_name']
    param_value = param_config['param_value']
    joint_name = param_config['joint_name']
    body_name = param_config['body_name']
    axis = param_config['axis']
    
    # Reset all randomization flags
    for name in randomization_params_names:
        env._domain_randomizer.rand_conf[f"randomize_{name}"] = False
    
    # Enable only the current parameter
    env._domain_randomizer.rand_conf[f"randomize_{param_name}"] = True
    
    if "stiffness" in param_name or "damping" in param_name:
        env._domain_randomizer.rand_conf[f"{param_name}_range"][joint_name] = [param_value, param_value]
    elif "position" in param_name or "orientation" in param_name:
        # Reset all axes to zero first
        for other_axis in env._domain_randomizer.rand_conf[f"{param_name}_range"][body_name].keys():
            env._domain_randomizer.rand_conf[f"{param_name}_range"][body_name][other_axis] = [0, 0]
        # Set current axis
        env._domain_randomizer.rand_conf[f"{param_name}_range"][body_name][axis] = [param_value, param_value]

@jax.jit
def evaluation_step_jit(carry, step_i):
    """JIT-compiled evaluation step - only the computational part"""
    env_state, train_state, rng = carry
    
    obs = env_state.observation
    rng, _rng = jax.random.split(rng)
    action, train_state = sample_actions(train_state, obs, _rng)
    action = jnp.atleast_2d(action)
    
    # Apply muscle activation
    for i in range(model.nu):
        if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE:
            action = action.at[..., i].set(muscle_skeleton_control_activation.adapted_sigmoid(action[..., i]))
    
    env_state, sys = jit_step(env_state, action)
    
    return (env_state, train_state, rng), (env_state, action, step_i)

def extract_metrics_from_step_data(step_data_list):
    """Extract all metrics from step data outside of JAX compilation"""
    all_metrics = {
        'grf_l': [],
        'grf_r': [],
        'contact_left': [],
        'contact_right': [],
        'joint_angles': [],
        'joint_velocities': [],
        'joint_forces_constraint': [],
        'joint_forces_smooth': [],
        'joint_torques': [],
        'joint_energy_exp': [],
        'sensor_forces': [],
        'sensor_force_names': [],
        'actions': []
    }
    
    for env_state, action, step_i in step_data_list:
        # Contact detection
        contact_left, contact_right = prosthesis_metrics_handler.get_contact_steps_batched(env_state.data, step_i)
        
        # GRF calculations
        grf_foot_l = prosthesis_metrics_handler.get_grf(env_state.data, "toes_l")
        grf_foot_r = prosthesis_metrics_handler.get_grf(env_state.data, "toes_r")
        grf_calcn_l = prosthesis_metrics_handler.get_grf(env_state.data, "calcn_l")
        grf_calcn_r = prosthesis_metrics_handler.get_grf(env_state.data, "calcn_r")
        
        grf_l = grf_foot_l + grf_calcn_l
        grf_r = grf_foot_r + grf_calcn_r
        
        # Joint data
        joint_angles = prosthesis_metrics_handler.get_joint_angles(env_state.data)
        joint_velocities = prosthesis_metrics_handler.get_joint_vels(env_state.data)
        joint_forces_constraint, joint_forces_smooth = prosthesis_metrics_handler.get_joint_frces(env_state.data)
        joint_torques = prosthesis_metrics_handler.get_joint_trques(env_state.data)
        joint_energy_exp = prosthesis_metrics_handler.calc_joint_energy_exp(joint_torques, joint_velocities)
        
        # Sensor data
        sensor_force, sensor_force_names = prosthesis_metrics_handler.get_sensor_data_batched(env_state.data)
        
        # Store all metrics
        all_metrics['grf_l'].append(grf_l)
        all_metrics['grf_r'].append(grf_r)
        all_metrics['contact_left'].append(contact_left)
        all_metrics['contact_right'].append(contact_right)
        all_metrics['joint_angles'].append(joint_angles)
        all_metrics['joint_velocities'].append(joint_velocities)
        all_metrics['joint_forces_constraint'].append(joint_forces_constraint)
        all_metrics['joint_forces_smooth'].append(joint_forces_smooth)
        all_metrics['joint_torques'].append(joint_torques)
        all_metrics['joint_energy_exp'].append(joint_energy_exp)
        all_metrics['sensor_forces'].append(sensor_force)
        all_metrics['sensor_force_names'].append(sensor_force_names)
        all_metrics['actions'].append(action)
    
    # Convert lists to arrays where appropriate
    for key in ['grf_l', 'grf_r', 'joint_angles', 'joint_velocities', 
                'joint_forces_constraint', 'joint_forces_smooth', 'joint_torques', 
                'joint_energy_exp', 'sensor_forces', 'actions']:
        all_metrics[key] = jnp.array(all_metrics[key])
    
    return all_metrics

def run_single_configuration_evaluation(param_config, env_state, train_state, rng):
    """Run evaluation for a single parameter configuration"""
    
    # Run the JIT-compiled simulation loop
    final_carry, step_data_collected = lax.scan(
        evaluation_step_jit,
        (env_state, train_state, rng),
        jnp.arange(n_steps)
    )
    
    env_state, train_state, rng = final_carry
    
    # Extract all metrics outside of JAX compilation
    all_metrics = extract_metrics_from_step_data(step_data_collected)
    
    # Create comprehensive results
    results = {
        'param_config': param_config,
        'metrics': all_metrics,
        'n_steps': n_steps,
        'final_env_state': env_state
    }
    
    return results

def save_individual_result(result, base_path):
    """Save individual result with all parameters"""
    param_config = result['param_config']
    param_name = param_config['param_name']
    param_value = param_config['param_value']
    
    # Create more descriptive filename
    if param_config['joint_name']:
        identifier = f"{param_name}_{param_config['joint_name']}_{param_value:.3f}"
    elif param_config['body_name'] and param_config['axis']:
        identifier = f"{param_name}_{param_config['body_name']}_{param_config['axis']}_{param_value:.3f}"
    else:
        identifier = f"{param_name}_{param_value:.3f}"
    
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dt_str}_evaluation_{identifier}.pkl"
    output_path = os.path.join(os.path.dirname(base_path), filename)
    
    with open(output_path, "wb") as f:
        pickle.dump(result, f)
    
    print(f"Saved complete result for {param_name}={param_value} to {output_path}")
    return output_path

def run_improved_evaluation():
    """Main evaluation function with improved structure"""
    # Create all parameter configurations
    all_configs = create_parameter_configs()
    print(f"Created {len(all_configs)} parameter configurations")
    
    all_results = []
    
    for i, param_config in enumerate(all_configs):
        print(f"Running configuration {i+1}/{len(all_configs)}: {param_config['param_name']}={param_config['param_value']}")
        
        # Update environment configuration (this happens outside JAX)
        update_env_randomization_config(param_config)
        
        # Reset environment with new configuration
        env_state = jit_reset(env_keys)
        
        # Run evaluation for this configuration
        result = run_single_configuration_evaluation(param_config, env_state, train_state, rng)
        
        # Save complete result with all parameters
        save_individual_result(result, path)
        
        # Keep summary for final report
        summary_result = {
            'param_config': param_config,
            'mean_grf_l': jnp.mean(result['metrics']['grf_l']),
            'mean_grf_r': jnp.mean(result['metrics']['grf_r']),
            'n_steps': result['n_steps'],
            'saved_file': f"evaluation_{param_config['param_name']}_{param_config['param_value']:.3f}.pkl"
        }
        all_results.append(summary_result)
    
    return all_results

# Main execution
if __name__ == "__main__":
    time_all = []
    time_all.append(timeit.default_timer())
    
    print("Starting improved evaluation with full parameter saving...")
    
    # Run improved evaluation
    results = run_improved_evaluation()
    
    # Save summary results
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(os.path.dirname(path), f"{dt_str}_evaluation_summary_{n_steps}steps.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump(results, f)
    
    time_all.append(timeit.default_timer())
    total_time = time_all[-1] - time_all[0]
    
    print(f"Total evaluation time: {total_time:.2f} seconds")
    print(f"Average time per configuration: {total_time/len(results):.2f} seconds")
    print(f"Saved summary results to {summary_path}")
    print(f"Individual result files saved with complete parameter data")
    
    env.stop()