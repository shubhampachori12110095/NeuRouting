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
    def train(self, instances, repair_procedure, val_split, batch_size, epochs):
        pass
