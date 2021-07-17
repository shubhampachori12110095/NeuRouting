from abc import ABC, abstractmethod
from typing import List

from instances import VRPSolution


class RepairProcedure(ABC):
    @abstractmethod
    def __call__(self, partial_solution: VRPSolution):
        pass

    def multiple(self, partial_solutions: List[VRPSolution]):
        return [self(sol) for sol in partial_solutions]
