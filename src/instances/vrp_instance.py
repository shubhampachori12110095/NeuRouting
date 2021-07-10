from typing import List

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix

from utils.visualize import plot_vrp


class VRPInstance:
    def __init__(self, depot: tuple, customers: List[tuple], demands: List[int], capacity: int):
        assert len(customers) == len(demands)
        self.depot = depot
        self.customers = customers
        self.n_customers = len(customers)
        self.demands = demands
        self.capacity = capacity
        self.distance_matrix = distance_matrix([depot] + customers, [depot] + customers)

    def plot(self, solution=None, ax=None):
        if ax is None:
            ax = plt.gca()
        plot_vrp(ax, self, solution)

    def to_networkx(self):
        coords = [self.depot] + self.customers
        graph = nx.from_numpy_matrix(self.distance_matrix)
        pos = dict(zip(range(self.n_customers + 1), coords))
        return graph, pos


def generate_instance(n_customers: int = 50,
                      capacity: int = 40,
                      distribution: str = 'uniform') -> VRPInstance:
    assert distribution in ['uniform', 'uchoa']
    if distribution == 'uniform':
        depot = tuple(np.random.random_sample((2,)))
        customers = list(map(lambda loc: (loc[0], loc[1]),
                             [tuple(np.random.random_sample((2,))) for _ in range(n_customers)]))
        demands = [np.random.choice(capacity // 4, replace=False) + 1 for _ in range(n_customers)]
        return VRPInstance(depot, customers, demands, capacity)
    if distribution == 'uchoa':
        # TODO
        pass


def generate_multiple_instances(n_instances: int,
                                n_customers: int = 50,
                                capacity: int = 40,
                                distribution: str = 'uniform') -> List[VRPInstance]:
    return [generate_instance(n_customers, capacity, distribution) for _ in range(n_instances)]


if __name__ == "__main__":
    for inst in generate_multiple_instances(5):
        inst.plot()
        plt.show()
