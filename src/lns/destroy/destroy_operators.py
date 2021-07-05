from abc import ABC, abstractmethod
from typing import List

from instances import VRPSolution


class DestroyProcedure(ABC):

    def multiple(self, solutions: List[VRPSolution]):
        for sol in solutions:
            self(sol)

    @abstractmethod
    def __call__(self, solution: VRPSolution):
        pass
