import math
import time
from copy import deepcopy
from math import ceil
from typing import List

import numpy as np
import torch

from lns.environments import VRPEnvironment, BatchVRPEnvironment
from instances import VRPInstance, VRPSolution
from lns import LargeNeighborhoodSearch, LNSOperatorPair
from lns.initial import nearest_neighbor_solution


class LNSEnvironment(LargeNeighborhoodSearch, VRPEnvironment):
    def __init__(self, operator_pairs: List[LNSOperatorPair], initial=nearest_neighbor_solution, adaptive=False):
        super().__init__(operator_pairs, initial, adaptive)
        self.cost = None

    def reset(self, instance: VRPInstance):
        self.instance = instance
        self.solution = self.initial(instance)
        self.cost = self.solution.cost()

    def step(self) -> float:
        prev_cost = self.cost
        copy = deepcopy(self.solution)
        destroy_operator, repair_operator, idx = self.select_operator_pair()
        iter_start_time = time.time()
        with torch.no_grad():
            destroy_operator(self.solution)
            repair_operator(self.solution)
        lns_iter_duration = time.time() - iter_start_time

        new_cost = self.solution.cost()
        # If adaptive search is used, update performance scores
        if self.adaptive:
            delta = (prev_cost - new_cost) / lns_iter_duration
            if self.performances[idx] == np.inf:
                self.performances[idx] = delta
            self.performances[idx] = self.performances[idx] * (1 - self.EMA_ALPHA) + delta * self.EMA_ALPHA

        self.render()
        if new_cost < prev_cost:
            self.cost = new_cost
        else:
            self.solution = copy
        return prev_cost - new_cost


class BatchLNSEnvironment(LargeNeighborhoodSearch, BatchVRPEnvironment):

    def __init__(self, batch_size: int, operator_pairs: List[LNSOperatorPair],
                 initial=nearest_neighbor_solution, adaptive=False):
        super().__init__(operator_pairs, initial, adaptive)
        self.batch_size = batch_size
        self.n_steps = 0
        self.costs = None

    def reset(self, instances: List[VRPInstance]):
        self.instances = instances
        self.solutions = [self.initial(inst) for inst in self.instances]
        self.costs = [sol.cost() for sol in self.solutions]
        self.n_steps = 0

    def step(self) -> float:
        prev_costs = np.mean(self.costs)
        copies = [deepcopy(sol) for sol in self.solutions]
        n_solutions = len(self.solutions)

        destroy_procedure, repair_procedure, idx = self.select_operator_pair()

        iter_start_time = time.time()
        n_batches = ceil(float(n_solutions) / self.batch_size)
        for i in range(n_batches):
            with torch.no_grad():
                begin = i * self.batch_size
                end = min((i + 1) * self.batch_size, n_solutions)
                destroy_procedure.multiple(self.solutions[begin:end])
                repair_procedure.multiple(self.solutions[begin:end])
        lns_iter_duration = time.time() - iter_start_time

        self.n_steps += 1

        for i in range(n_solutions):
            cost = self.solutions[i].cost()
            # Only "accept" improving solutions
            if self.costs[i] < cost:
                self.solutions[i] = copies[i]
            else:
                self.costs[i] = cost

        # If adaptive search is used, update performance scores
        if self.adaptive:
            delta = (prev_costs - np.mean(self.costs)) / lns_iter_duration
            if self.performances[idx] == np.inf:
                self.performances[idx] = delta
            self.performances[idx] = self.performances[idx] * (1 - self.EMA_ALPHA) + delta * self.EMA_ALPHA

        return np.mean(prev_costs - self.costs)


class SimAnnealingLNSEnvironment(LargeNeighborhoodSearch, VRPEnvironment):
    def __init__(self, operator_pairs: List[LNSOperatorPair], neighborhood_size,
                 initial=nearest_neighbor_solution, n_reheating=5, reset_percentage=0.8):
        super().__init__(operator_pairs, initial)
        self.neighborhood_size = neighborhood_size
        self.n_reheating = n_reheating
        self.reset_percentage = reset_percentage
        self.copies = None
        self.costs = None

    def reset(self, instance: VRPInstance):
        self.instance = instance
        self.solution = self.initial(instance)

    def step(self):
        # Set a certain percentage of the data/solutions in the envs to the last accepted solution
        for i in range(int(self.reset_percentage * self.neighborhood_size)):
            self.copies[i] = deepcopy(self.solution)

        destroy_operator, repair_operator, _ = self.select_operator_pair()

        with torch.no_grad():
            destroy_operator.multiple(self.copies)
            repair_operator.multiple(self.copies)

        self.costs = [sol.cost() for sol in self.copies]

    def solve(self, instance: VRPInstance, time_limit: int = 60) -> VRPSolution:
        start_time = time.time()
        self.reset(instance)
        while time.time() - start_time < time_limit:
            current_cost = self.solution.cost()

            reheating_time = time.time()
            # Create a envs of copies of the same solution that can be repaired in parallel
            self.copies = [deepcopy(self.solution) for _ in range(self.neighborhood_size)]

            reheat = True
            t_max, t_factor = 0, 0
            # Repeat until the time limit of one reheating iteration is reached
            while time.time() - reheating_time < time_limit / self.n_reheating:
                self.step()
                # Calculate the T_max and T_factor values for simulated annealing in the first iteration
                if reheat:
                    q75, q25 = np.percentile(self.costs, [75, 25])
                    t_min = 10
                    t_max = q75 - q25 + t_min
                    t_factor = -math.log(t_max / t_min)
                    reheat = False

                min_cost = min(self.costs)

                # Calculate simulated annealing temperature
                temp = t_max * math.exp(t_factor * (time.time() - reheating_time) / (time_limit / self.n_reheating))

                # Accept a solution if the acceptance criteria is fulfilled
                if min_cost <= current_cost or np.random.rand() < math.exp(-(min(self.costs) - current_cost) / temp):
                    min_idx = np.argmin(self.costs)
                    self.solution = self.copies[min_idx]
                    current_cost = min_cost
                    self.solution.verify()
                    self.render()

        return self.solution
