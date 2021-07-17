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

    def adjacency_matrix(self, num_neighbors: int):
        """
        0: node-node
        1: node-node knn
        2: node self-loop
        3: node-depot
        4: node-depot knn
        5: depot self-loop
        """
        num_nodes = self.n_customers
        if num_neighbors == -1:
            adj = np.ones((num_nodes + 1, num_nodes + 1))  # Graph is fully connected
        else:
            adj = np.zeros((num_nodes + 1, num_nodes + 1))
            # Determine k-nearest neighbors for each node
            knns = np.argpartition(self.distance_matrix, kth=num_neighbors, axis=-1)[:, num_neighbors::-1]
            # Make connections
            for idx in range(num_nodes):
                adj[idx][knns[idx]] = 1
        np.fill_diagonal(adj, 2)  # Special token for self-connections
        # Special token for depot connection (3 or 4 depending on whether it is knn or not)
        adj[:, 0] += 3
        adj[0, :] += 3
        # Depot self-connection
        adj[0, 0] = 5
        return adj

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
