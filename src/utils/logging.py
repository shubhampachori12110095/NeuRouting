from abc import ABC, abstractmethod
from numbers import Number
from typing import Dict, List, Union
from torch import nn

import wandb


class Logger(ABC):
    @abstractmethod
    def log(self, info: Dict[str, Number], phase: str):
        pass


class WandBLogger(Logger):
    def __init__(self, model: List[nn.Module],
                 run_name: str = None, project: str = "NeuRouting", username: str = "mazzio97"):
        wandb.init(name=run_name, project=project, entity=username)
        wandb.watch(*model)

    def log(self, info: Dict[str, Number], phase: str):
        keys = []
        for k in info.keys():
            keys.append(f"{phase}/{k}")
        info = dict(zip(keys, list(info.values())))
        wandb.log(info)


class ConsoleLogger(Logger):
    def log(self, info: Dict[str, Number], phase: str):
        print(f"[{phase.upper()}]")
        for k, v in info.items():
            print(f"{k}: {v}")


class MultipleLogger(Logger):
    def __init__(self, loggers: List[Logger]):
        self.loggers = set() if loggers is None else set(loggers)

    def add(self, logger: Logger):
        self.loggers.add(logger)

    def remove(self, logger: Logger):
        self.loggers.remove(logger)

    def log(self, info: Dict[str, Number], phase: str):
        for logger in self.loggers:
            logger.log(info, phase)
