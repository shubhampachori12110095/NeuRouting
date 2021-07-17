import time
from abc import ABC, abstractmethod
from typing import List

from matplotlib import pyplot as plt

from instances import VRPInstance, VRPSolution


class VRPEnvironment(ABC):
    def __init__(self):
        self.instance = None
        self.solution = None

    @abstractmethod
    def reset(self, instance: VRPInstance):
        pass

    @abstractmethod
    def step(self):
        pass

    def render(self, ax=None):
        self.instance.plot(solution=self.solution, ax=ax)
        plt.show()

    def solve(self, instance: VRPInstance, time_limit: int = 60) -> VRPSolution:
        start_time = time.time()
        self.reset(instance)
        while time.time() - start_time < time_limit:
            self.step()
        self.solution.verify()
        return self.solution


class BatchVRPEnvironment(ABC):
    def __init__(self):
        self.instances = None
        self.solutions = None

    @abstractmethod
    def reset(self, instances: List[VRPInstance]):
        pass

    @abstractmethod
    def step(self):
        pass

    def solve(self, instances: List[VRPInstance], time_limit: int = 60) -> List[VRPSolution]:
        start_time = time.time()
        self.reset(instances)
        while time.time() - start_time < time_limit:
            self.step()
        for sol in self.solutions:
            sol.verify()
        return self.solutions
