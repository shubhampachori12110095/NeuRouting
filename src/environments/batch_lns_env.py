import time
from copy import deepcopy
from math import ceil
from typing import List

import torch

from environments import VRPEnvironment
from environments.lns_env import LargeNeighborhoodSearch
from instances import VRPInstance, VRPSolution, VRPNeuralSolution
from nlns import LNSOperator
from nlns.initial import nearest_neighbor_solution


class BatchLNSEnvironment(LargeNeighborhoodSearch, VRPEnvironment):

    def __init__(self, batch_size: int,
                 operators: List[LNSOperator],
                 initial=nearest_neighbor_solution):
        VRPEnvironment.__init__(self, "Batch LNS")
        LargeNeighborhoodSearch.__init__(self, operators, initial, False)
        self.batch_size = batch_size
        self.instance = None
        self.solution = None
        self.costs = None
        self.n_steps = 0

    def reset(self, instances: List[VRPInstance], **args):
        super(BatchLNSEnvironment, self).reset(instances)
        self.solution = [self.initial(inst) for inst in self.instance]
        if any([callable(getattr(op.repair, "_actor_model_forward", None)) for op in self.operators]):
            self.solution = [VRPNeuralSolution.from_solution(sol) for sol in self.solution]
        self.costs = [sol.cost() for sol in self.solution]

    def step(self):
        backup_copies = [deepcopy(sol) for sol in self.solution]
        n_solutions = len(self.solution)

        destroy_procedure, repair_procedure, idx = self.select_operator_pair()

        n_batches = ceil(float(n_solutions) / self.batch_size)
        for i in range(n_batches):
            with torch.no_grad():
                begin = i * self.batch_size
                end = min((i + 1) * self.batch_size, n_solutions)
                destroy_procedure.multiple(self.solution[begin:end])
                repair_procedure.multiple(self.solution[begin:end])

        for i in range(n_solutions):
            cost = self.solution[i].cost()
            # Only "accept" improving solutions
            if self.costs[i] < cost:
                self.solution[i] = backup_copies[i]
            else:
                self.costs[i] = cost

    def solve(self, instance: List[VRPInstance], time_limit=None, max_steps=None) -> List[VRPSolution]:
        self.reset(instance)
        self.max_steps = max_steps if max_steps is not None else self.max_steps
        self.time_limit = time_limit if time_limit is not None else self.time_limit
        start_time = time.time()
        while self.n_steps < self.max_steps and time.time() - start_time < self.time_limit:
            self.step()
            self.n_steps += 1
        return self.solution
