from abc import ABC, abstractmethod
from typing import List

from instances import VRPSolution


class DestroyProcedure(ABC):
    @abstractmethod
    def __call__(self, solution: VRPSolution):
        pass

    def multiple(self, solutions: List[VRPSolution]):
        return [self(sol) for sol in solutions]
