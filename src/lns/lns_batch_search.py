import time
from copy import deepcopy
from math import ceil
from typing import List

import numpy as np
import torch

from instances import VRPSolution
from lns.destroy.destroy_operators import DestroyProcedure
from lns.repair.repair_operators import RepairProcedure, NeuralRepairProcedure


class LNSOperatorPair:

    def __init__(self, destroy_operator: DestroyProcedure, repair_operator: RepairProcedure):
        self.destroy = destroy_operator
        self.repair = repair_operator


EMA_ALPHA = 0.2


def lns_batch_search(solutions: List[VRPSolution],
                     operator_pairs: List[LNSOperatorPair],
                     batch_size,
                     max_iterations,
                     timelimit,
                     adaptive_search=True):

    costs = [sol.cost() for sol in solutions]  # Costs for each instance
    performance_ema = [np.inf] * len(operator_pairs)  # Exponential moving average of improvements in last iterations

    print("\t> Starting evaluation...")
    start_time = time.time()
    for it in range(max_iterations):

        if time.time() - start_time > timelimit:
            break

        mean_cost_before_iteration = np.mean(costs)

        solution_copies = [deepcopy(sol) for sol in solutions]

        # Select an LNS operator pair (destroy + repair operator)
        if adaptive_search:
            operator_pair_idx = np.argmax(performance_ema)  # select operator pair with the best EMA
        else:
            operator_pair_idx = np.random.randint(0, len(operator_pairs))  # select operator pair at random

        destroy_procedure = operator_pairs[operator_pair_idx].destroy
        repair_procedure = operator_pairs[operator_pair_idx].repair

        start_time_destroy = time.time()

        # Destroy instances
        destroy_procedure.multiple(solutions)

        # Repair instances
        n_batches = ceil(float(len(solutions)) / batch_size)
        for i in range(n_batches):
            if repair_procedure is NeuralRepairProcedure:
                with torch.no_grad():
                    repair_procedure.multiple(solutions[i * batch_size: min((i + 1) * batch_size, len(solutions))])
            else:
                repair_procedure.multiple(solutions[i * batch_size: min((i + 1) * batch_size, len(solutions))])

        lns_iter_duration = time.time() - start_time_destroy

        print(f"> Iteration {it} completed in {lns_iter_duration} seconds...")

        for i in range(len(solutions)):
            cost = solutions[i].cost()
            # Only "accept" improving solutions
            if costs[i] < cost:
                solutions[i] = solution_copies[i]
            else:
                costs[i] = cost

        # If adaptive search is used, update performance scores
        if adaptive_search:
            delta = (mean_cost_before_iteration - np.mean(costs)) / lns_iter_duration
            if performance_ema[operator_pair_idx] == np.inf:
                performance_ema[operator_pair_idx] = delta
            performance_ema[operator_pair_idx] = performance_ema[operator_pair_idx] * (1 - EMA_ALPHA) + delta * EMA_ALPHA

    # Verify solutions
    for sol in solutions:
        sol.verify()

    return costs
