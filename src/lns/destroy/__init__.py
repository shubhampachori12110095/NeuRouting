from abc import ABC, abstractmethod
from typing import List

from instances import VRPSolution


class DestroyProcedure(ABC):
    @abstractmethod
    def __call__(self, solution: VRPSolution) -> VRPSolution:
        pass

    def multiple(self, solutions: List[VRPSolution]) -> List[VRPSolution]:
        return [self(sol) for sol in solutions]


class NeuralDestroyProcedure(DestroyProcedure):
    @abstractmethod
    def train(self, repair_procedure, n_samples, val_split, batch_size):
        pass
