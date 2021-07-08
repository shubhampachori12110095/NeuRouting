from abc import ABC, abstractmethod
from typing import List

from instances import VRPSolution


class RepairProcedure(ABC):
    @abstractmethod
    def __call__(self, partial_solution: VRPSolution) -> VRPSolution:
        pass

    def multiple(self, partial_solutions: List[VRPSolution]) -> List[VRPSolution]:
        return [self(sol) for sol in partial_solutions]


class NeuralRepairProcedure(RepairProcedure):
    @abstractmethod
    def train(self, destroy_procedure, n_samples, val_split, batch_size):
        pass
