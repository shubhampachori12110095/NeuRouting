from copy import deepcopy
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from instances import VRPSolution
from lns.destroy.destroy_operators import DestroyProcedure
from lns.initial.nearest_neighbor import nearest_neighbor_solution
from utils.io import read_vrp


class DestroyRandom(DestroyProcedure):
    """Random destroy. Select customers that should be removed at random and remove them from tours."""

    def __init__(self, percentage: float):
        assert 0 <= percentage <= 1
        self.percentage = percentage

    def __call__(self, solution: VRPSolution):
        n = solution.instance.n_customers
        random_customers = np.random.choice(range(1, n + 1), int(n * self.percentage), replace=False)
        solution.destroy_nodes(to_remove=random_customers)


class DestroyPointBased(DestroyProcedure):
    """Point based destroy.
    Select customers that should be removed based on their distance to a random point and remove them from tours."""

    def __init__(self, percentage: float, point: Tuple[float, float] = None):
        assert 0 <= percentage <= 1
        self.percentage = percentage
        self.point = lambda: np.random.rand(1, 2) if point is None else point

    def __call__(self, solution: VRPSolution):
        n = solution.instance.n_customers
        customers = np.array(solution.instance.customers)
        dist = np.sum((customers - self.point()) ** 2, axis=1)
        closest_customers = np.argsort(dist)[:int(n * self.percentage)] + 1
        solution.destroy_nodes(to_remove=closest_customers)


class DestroyTourBased(DestroyProcedure):
    """Tour based destroy. Remove all tours closest to a randomly selected point from a solution."""

    def __init__(self, percentage: float, point: Tuple[float, float] = None):
        assert 0 <= percentage <= 1
        self.percentage = percentage
        if point is None:
            self.point = lambda _: np.random.rand(1, 2)
        else:
            self.point = point

    def __call__(self, solution: VRPSolution):
        # Make a dictionary that maps customers to tours
        customer_to_tour = {}
        for i, tour in enumerate(solution.routes):
            for e in tour[1:-1]:
                if e in customer_to_tour:
                    customer_to_tour[e].append(i + 1)
                else:
                    customer_to_tour[e] = [i + 1]
        tours_to_remove = []
        n = solution.instance.n_customers
        n_to_remove = int(n * self.percentage)  # Number of customer that should be removed
        n_removed = 0
        customers = np.array(solution.instance.customers)
        dist = np.sum((customers - self.point) ** 2, axis=1)
        closest_customers = np.argsort(dist) + 1
        # Iterate over customers starting with the customer closest to the random point.
        for customer_idx in closest_customers:
            # Iterate over the tours of the customer
            for i in customer_to_tour[customer_idx]:
                # and if the tour is not yet marked for removal
                if i not in tours_to_remove:
                    # mark it for removal
                    tours_to_remove.append(i)
                    n_removed += len(solution.routes[i - 1]) - 2
            # Stop once enough tours are marked for removal
            if n_removed >= n_to_remove and len(tours_to_remove) >= 1:
                break
        to_remove = set()
        for tour in tours_to_remove:
            for c in solution.routes[tour - 1][1:-1]:
                to_remove.add(c)
        solution.destroy_nodes(to_remove=list(to_remove))


if __name__ == "__main__":
    inst = read_vrp("../../../res/A-n32-k5.vrp", grid_dim=100)
    nn_sol = nearest_neighbor_solution(inst)
    partial_sols = [deepcopy(nn_sol) for _ in range(10)]
    destroy_random = DestroyPointBased(0.2)
    destroy_random.multiple(partial_sols)
    for sol in partial_sols:
        inst.plot(sol)
        plt.show()
