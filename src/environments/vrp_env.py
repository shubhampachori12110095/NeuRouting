from abc import ABC, abstractmethod
from typing import List, Union, Optional

from matplotlib import pyplot as plt

from instances import VRPInstance, VRPSolution

INF = 1e20  # Ensure compatibility with SCIP optimizer maximum time limit


class VRPSolver(ABC):
    def __init__(self, name: str):
        self.name = name
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
            self.instance.plot(solution=self.solution, ax=ax, title=self.name)
            plt.show()


class VRPEnvironment(VRPSolver):
    def __init__(self, name):
        super().__init__(name)
        self.n_steps = 0
        self.improvements = 0
        self.max_steps = INF
        self.time_limit = INF

    def reset(self, instance):
        self.instance = instance
        self.n_steps = 0
        self.improvements = 0
        self.max_steps = INF
        self.time_limit = INF

    @abstractmethod
    def step(self):
        pass
