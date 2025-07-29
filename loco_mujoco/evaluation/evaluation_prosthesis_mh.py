from loco_mujoco.utils.metrics import MetricsHandler
from omegaconf import OmegaConf, DictConfig

class ProsthesisHandler(MetricsHandler):
    def __init__(self, config: DictConfig, env):
        self._config = config.experiment

        if env.th is not None:
            self._traj_data = env.th.traj.data
        else:
            self._traj_data = None
