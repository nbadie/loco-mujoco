import pickle
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
import jax
jax.config.update('jax_platform_name', 'cpu')
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



# Set MUJOCO_GL to egl
os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering, which is more compatible with headless environments




@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    checkpoint_path = config.checkpoint_path

    
    print(f"Loading agent from: {checkpoint_path}")
    try: 

        # can increase the speed by ~30% on some GPUs
        # os.environ['XLA_FLAGS'] = (
        #     '--xla_gpu_triton_gemm_any=True ')
        print('TOTAL TIME STPES: ', config.experiment.total_timesteps)
        
        # Accessing the current sweep number
        result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir


        # Extract date and time from the result directory path for wandb run name
        result_dir_parts = result_dir.split("/")
        result_dir_date, result_dir_time = result_dir_parts[-2], result_dir_parts[-1]
        formatted_result_dir = f"{result_dir_date}_{result_dir_time}_"

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise



if __name__ == "__main__":
    experiment()


