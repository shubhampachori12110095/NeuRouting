from abc import ABC, abstractmethod
from typing import List

from instances import VRPSolution
from lns.destroy.destroy_operators import DestroyProcedure


class RepairProcedure(ABC):

    @abstractmethod
    def __call__(self, partial_solution: VRPSolution):
        pass

    def multiple(self, partial_solutions: List[VRPSolution]):
        for sol in partial_solutions:
            self(sol)


class NeuralRepairProcedure(RepairProcedure):

    @abstractmethod
    def train(self, destroy_procedure: DestroyProcedure, n_samples: int, val_split: float, batch_size: int):
        pass
