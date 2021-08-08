from abc import ABC, abstractmethod
from typing import List

from instances import VRPSolution


class LNSProcedure(ABC):
    @abstractmethod
    def __call__(self, solution: VRPSolution):
        pass

    def multiple(self, solutions: List[VRPSolution]):
        return [self(sol) for sol in solutions]


class DestroyProcedure(LNSProcedure):
    @abstractmethod
    def __call__(self, solution: VRPSolution):
        pass


class RepairProcedure(LNSProcedure):
    @abstractmethod
    def __call__(self, solution: VRPSolution):
        pass


class LNSOperator:
    def __init__(self, destroy_procedure: DestroyProcedure, repair_procedure: RepairProcedure):
        self.destroy = destroy_procedure
        self.repair = repair_procedure


from .neural_procedure import NeuralProcedure
