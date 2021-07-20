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


def generate_instance(n_customers: int = 100,
                      distribution: str = 'nazari') -> VRPInstance:
    if distribution == 'nazari':
        acceptable = [10, 20, 30, 50, 100]
        assert n_customers in acceptable, f"{n_customers} should be one of {acceptable} for {distribution}"
        capacity = {10: 20, 20: 30, 50: 40, 100: 50}
        depot = tuple(np.random.random_sample((2,)))
        customers = list(map(lambda loc: (loc[0], loc[1]),
                             [tuple(np.random.random_sample((2,))) for _ in range(n_customers)]))
        demands = [np.random.randint(1, 10) + 1 for _ in range(n_customers)]
        return VRPInstance(depot, customers, demands, capacity[n_customers])
    else:
        Exception(f"{distribution} is unknown.")


def generate_multiple_instances(n_instances: int,
                                n_customers: int = 100,
                                distribution: str = 'nazari',
                                seed=42) -> List[VRPInstance]:
    np.random.seed(seed)
    if distribution == 'nazari':
        acceptable = [10, 20, 30, 50, 100]
        assert n_customers in acceptable, f"{n_customers} should be one of {acceptable} for {distribution}"
        capacity_map = {10: 20, 20: 30, 50: 40, 100: 50}
        capacity = capacity_map[n_customers]
        return [VRPInstance(tuple(depot), list(customers), list(demands), capacity) for depot, customers, demands
                in zip(list(np.random.uniform(size=(n_instances, 2))),
                       list(np.random.uniform(size=(n_instances, n_customers, 2))),
                       list(np.random.randint(1, 10, size=(n_instances, n_customers))))]
    else:
        Exception(f"{distribution} is unknown.")
