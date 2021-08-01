import numpy as np

from instances import VRPSolution
from lns import DestroyProcedure


class DestroyRandom(DestroyProcedure):
    """Random destroy. Select customers that should be removed at random and remove them from tours."""

    def __init__(self, percentage: float):
        assert 0 <= percentage <= 1
        self.percentage = percentage

    def __call__(self, solution: VRPSolution):
        n = solution.instance.n_customers
        random_customers = np.random.choice(range(1, n + 1), int(n * self.percentage), replace=False)
        solution.destroy_nodes(to_remove=random_customers)
