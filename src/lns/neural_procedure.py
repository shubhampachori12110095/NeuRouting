import ntpath
import os
import time
from abc import abstractmethod
from math import ceil
from typing import Union, Optional, List

import numpy as np
import torch
from torch import nn

from environments.lns_env import LNSEnvironment
from environments.batch_lns_env import BatchLNSEnvironment
from instances import VRPInstance, VRPSolution
from lns import LNSProcedure, RepairProcedure, DestroyProcedure, LNSOperator
from utils.logging import Logger


class NeuralProcedure(LNSProcedure):
    def __init__(self, model: nn.Module, device: str = "cpu", logger: Optional[Logger] = None):
        self.model = model.to(device)
        self.device = device
        self.logger = logger
        self.val_env = None

    def train(self,
              opposite_procedure: Union[DestroyProcedure, RepairProcedure],
              train_instances: List[VRPInstance],
              batch_size: int,
              n_epochs: int,
              ckpt_path: str,
              log_interval: int = 1,
              val_instances: Optional[List[VRPInstance]] = None,
              val_interval: Optional[int] = None,
              val_neighborhood: Optional[int] = None,
              val_steps: Optional[int] = None):
        run_name = ntpath.basename(ckpt_path)
        run_name = run_name[:run_name.rfind('.')]
        if self.logger is not None:
            self.logger.new_run(run_name=run_name)

        incumbent_cost = np.inf
        train_size = len(train_instances)
        n_batches = ceil(float(train_size) / batch_size)

        start_time = time.time()
        print(f"Starting {run_name} training...")
        self._init_train()
        for epoch in range(n_epochs):
            for batch_idx in range(n_batches):
                begin = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, train_size)
                self.model.train()

                self._train_step(opposite_procedure, train_instances[begin:end])

                if self.logger is not None and (batch_idx + 1) % log_interval == 0:
                    self.logger.log(self._train_info(epoch, batch_idx, log_interval), phase="train")

                if val_instances is not None:
                    n_customers = val_instances[0].n_customers
                    val_steps = val_steps if val_steps is not None else n_customers
                    val_neighborhood = val_neighborhood if val_neighborhood is not None else n_customers
                    val_interval = val_interval if val_interval is not None else n_batches

                if val_instances is not None and (batch_idx + 1) % val_interval == 0 or batch_idx + 1 == n_batches:
                    self.model.eval()
                    start_eval_time = time.time()
                    solutions = self.validate(opposite_procedure, val_instances, val_neighborhood, val_steps)
                    runtime = time.time() - start_eval_time
                    mean_cost = np.mean([sol.cost() for sol in solutions])
                    if self.logger is not None:
                        self.logger.log({"epoch": epoch + 1,
                                         "batch_idx": batch_idx + 1,
                                         "mean_cost": mean_cost,
                                         "runtime": runtime}, phase="val")
                    if mean_cost < incumbent_cost:
                        incumbent_cost = mean_cost
                        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                        torch.save(self._ckpt_info(epoch, batch_idx), ckpt_path)
        self.val_env = None
        print(f"Training completed successfully in {time.time() - start_time} seconds.")

    def validate(self,
                 opposite_procedure: Union[DestroyProcedure, RepairProcedure],
                 instances: List[VRPInstance],
                 neighborhood_size: int,
                 steps: int):
        if self.val_env is None:
            if isinstance(opposite_procedure, RepairProcedure) and isinstance(self, DestroyProcedure):
                base_env = LNSEnvironment(operators=[LNSOperator(self, opposite_procedure)],
                                          neighborhood_size=neighborhood_size)
            elif isinstance(opposite_procedure, DestroyProcedure) and isinstance(self, RepairProcedure):
                base_env = LNSEnvironment(operators=[LNSOperator(opposite_procedure, self)],
                                          neighborhood_size=neighborhood_size)
            else:
                base_env = None
            assert base_env is not None, f"{opposite_procedure} and {self} should be two opposite LNS procedures."
            self.val_env = BatchLNSEnvironment(base_env)
        return self.val_env.solve(instances, max_steps=steps)

    def load_model(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, self.device)
        self.model.load_state_dict(ckpt["parameters"])
        self.model.eval()

    @abstractmethod
    def _init_train(self):
        pass

    @abstractmethod
    def _train_step(self, opposite_procedure, train_batch):
        pass

    @abstractmethod
    def _train_info(self, epoch, batch_idx, log_interval) -> dict:
        pass

    @abstractmethod
    def _ckpt_info(self, epoch, batch_idx) -> dict:
        pass
