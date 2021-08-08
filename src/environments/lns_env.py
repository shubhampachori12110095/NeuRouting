import time
from copy import deepcopy
import numpy as np
import torch
from typing import List, Tuple

from environments import VRPEnvironment
from instances import VRPInstance, VRPSolution
from lns import DestroyProcedure, RepairProcedure, LNSOperator
from lns.initial import nearest_neighbor_solution

EMA_ALPHA = 0.2  # Exponential Moving Average Alpha


class LargeNeighborhoodSearch:
    def __init__(self,
                 operators: List[LNSOperator],
                 initial=nearest_neighbor_solution,
                 adaptive=False):
        self.initial = initial
        self.operators = operators
        self.n_operators = len(operators)
        self.adaptive = adaptive
        self.performances = [np.inf] * self.n_operators if adaptive else None

    def select_operator_pair(self) -> Tuple[DestroyProcedure, RepairProcedure, int]:
        if self.adaptive:
            idx = np.argmax(self.performances)
        else:
            idx = np.random.randint(0, self.n_operators)
        return self.operators[idx].destroy, self.operators[idx].repair, idx


class LNSEnvironment(LargeNeighborhoodSearch, VRPEnvironment):
    def __init__(self,
                 operators: List[LNSOperator],
                 neighborhood_size: int,
                 initial=nearest_neighbor_solution,
                 adaptive=False,
                 name="LNS"):
        LargeNeighborhoodSearch.__init__(self, operators, initial, adaptive)
        VRPEnvironment.__init__(self, name)
        self.neighborhood_size = neighborhood_size
        self.neighborhood = None
        self.neighborhood_costs = None

    def reset(self, instance: VRPInstance):
        super(LNSEnvironment, self).reset(instance)
        self.solution = self.initial(instance)

    def step(self) -> dict:
        current_cost = self.solution.cost()

        destroy_operator, repair_operator, idx = self.select_operator_pair()

        iter_start_time = time.time()
        with torch.no_grad():
            destroy_operator.multiple(self.neighborhood)
            repair_operator.multiple(self.neighborhood)
        lns_iter_duration = time.time() - iter_start_time

        self.neighborhood_costs = [sol.cost() for sol in self.neighborhood]
        new_cost = min(self.neighborhood_costs)

        # If adaptive search is used, update performance scores
        if self.adaptive:
            delta = (current_cost - new_cost) / lns_iter_duration
            if self.performances[idx] == np.inf:
                self.performances[idx] = delta
            self.performances[idx] = self.performances[idx] * (1 - EMA_ALPHA) + delta * EMA_ALPHA

        self.n_steps += 1

        return {"cost": new_cost}

    def solve(self, instance: VRPInstance, max_steps=None, time_limit=None) -> VRPSolution:
        self.max_steps = max_steps if max_steps is not None else self.max_steps
        self.time_limit = time_limit if time_limit is not None else self.time_limit
        start_time = time.time()
        self.reset(instance)
        while self.n_steps < max_steps and time.time() - start_time < time_limit:
            # Create a envs of copies of the same solution that can be repaired in parallel
            self.neighborhood = [deepcopy(self.solution) for _ in range(self.neighborhood_size)]
            criteria = self.step()
            if self.acceptance_criteria(criteria):
                best_idx = np.argmin(self.neighborhood_costs)
                self.solution = self.neighborhood[best_idx]
                self.solution.verify()
                # self.render()
        return self.solution

    def acceptance_criteria(self, criteria: dict) -> bool:
        # Accept a solution if the acceptance criteria is fulfilled
        return criteria["cost"] < self.solution.cost()

    def __deepcopy__(self, memo):
        return LNSEnvironment(self.operators, self.neighborhood_size, self.initial, self.adaptive, self.name)
