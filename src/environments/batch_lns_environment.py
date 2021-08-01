import time
from copy import deepcopy
from typing import List

import numpy as np

from environments import LNSEnvironment, VRPSolver
from instances import VRPInstance, VRPSolution


class BatchLNSEnvironment(VRPSolver):
    def __init__(self, base_env: LNSEnvironment):
        super().__init__()
        self._base_env = base_env
        self.envs = None

    def reset(self, instance: List[VRPInstance]):
        self.envs = [deepcopy(self._base_env) for _ in range(len(instance))]
        self.instance = instance
        self.solution = []
        for env, inst in zip(self.envs, self.instance):
            env.reset(inst)
            self.solution.append(env.solution)

    def solve(self, instance: List[VRPInstance], max_steps=None, time_limit=3600) -> List[VRPSolution]:
        start_time = time.time()
        self.reset(instance)
        while any([env.n_steps < max_steps for env in self.envs]) and time.time() - start_time < time_limit:
            # Create a envs of copies of the same solution that can be repaired in parallel
            for i, env in enumerate(self.envs):
                env.neighborhood = [deepcopy(env.solution) for _ in range(env.neighborhood_size)]
                criteria = env.step()
                if env.acceptance_criteria(criteria):
                    best_idx = np.argmin(env.neighborhood_costs)
                    env.solution = env.neighborhood[best_idx]
                    self.solution[i] = env.solution
                    self.solution[i].verify()
        return self.solution
