from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.profilers import Profiler
from omegaconf import DictConfig

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def instantiate_profiler(profiler_cfg: DictConfig) -> List[Profiler]:
    """Instantiates profiler from config.

    :param profiler_cfg: A DictConfig object containing profiler configurations.
    :return: An instantiated profiler.
    """
    profiler: List[Profiler] = []

    if not profiler_cfg:
        log.warning("No profiler configs found! Skipping...")
        return profiler

    if not isinstance(profiler_cfg, DictConfig):
        raise TypeError("Profiler config must be a DictConfig!")

    for _, pr_conf in profiler_cfg.items():
        if isinstance(pr_conf, DictConfig) and "_target_" in pr_conf:
            log.info(f"Instantiating profiler <{pr_conf._target_}>")
            profiler.append(hydra.utils.instantiate(pr_conf))

    return profiler
