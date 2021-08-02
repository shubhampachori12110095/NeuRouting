from abc import ABC, abstractmethod
from typing import List, Union, Optional

from matplotlib import pyplot as plt

from instances import VRPInstance, VRPSolution

INF = 1e20  # Ensure compatibility with SCIP optimizer


class VRPSolver(ABC):
    def __init__(self):
        self.instance = None
        self.solution = None

    @abstractmethod
    def reset(self, instance: Union[VRPInstance, List[VRPInstance]]):
        pass

    @abstractmethod
    def solve(self,
              instance: Union[VRPInstance, List[VRPInstance]],
              time_limit: Optional[int],
              max_steps: Optional[int]) -> Union[VRPSolution, List[VRPSolution]]:
        pass

    def render(self, ax=None):
        if type(self.instance) is VRPInstance:
            self.instance.plot(solution=self.solution, ax=ax)
            plt.show()


class VRPEnvironment(VRPSolver):
    def __init__(self):
        super().__init__()
        self.current_cost = 0
        self.n_steps = 0
        self.max_steps = INF
        self.time_limit = INF

    @abstractmethod
    def step(self):
        pass
