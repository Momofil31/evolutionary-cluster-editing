from typing import Any, Dict

from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)

def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"logger"`: The logger.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    logger = object_dict["logger"]

    if not logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["input_file"] = cfg.get("input_file")
    hparams["data"] = cfg["data"]
    hparams["evolution"] = cfg.get("evolution")
    hparams["mutation"] = cfg.get("mutation")
    hparams["crossover"] = cfg.get("crossover")
    hparams["selection"] = cfg.get("selection")
    hparams["local_search"] = cfg.get("local_search")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["seed"] = cfg.get("seed")


    logger.log_hyperparams(hparams)