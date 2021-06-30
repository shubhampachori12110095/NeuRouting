from typing import List

import numpy as np

from instances.vrp_instance import VRPInstance


class Route(List[int]):
    def __init__(self, nodes: List[int], instance: VRPInstance):
        super().__init__(nodes)
        self.distance_matrix = instance.distance_matrix
        self.demands = instance.demands

    def is_complete(self) -> bool:
        return len(self) > 1 and (self[0] == self[-1] == 0)

    def is_incomplete(self) -> bool:
        return len(self) > 1 and (self[0] != 0 or self[-1] != 0)

    def total_distance(self) -> float:
        return np.sum([self.distance_matrix[from_idx, to_idx] for from_idx, to_idx in zip(self[:-1], self[1:])])

    def distance_till_customer(self, customer_idx: int) -> float:
        assert customer_idx in self, f"Customer {customer_idx} not in this route"
        distance = 0
        for from_idx, to_idx in zip(self[:-1], self[1:]):
            distance += self.distance_matrix[from_idx, to_idx]
            if to_idx == customer_idx:
                break
        return distance

    def total_demand(self) -> int:
        return np.sum(self.demands[idx - 1] for idx in self[1:-1])

    def demand_till_customer(self, customer_idx: int) -> int:
        assert customer_idx in self, f"Customer {customer_idx} not in this route"
        demand = 0
        for idx in self[1:-1]:
            demand += self.demands[idx - 1]
            if idx == customer_idx:
                break
        return demand


class VRPSolution:
    def __init__(self, instance: VRPInstance, routes: List[Route] = None):
        self.instance = instance
        self.routes = routes if routes is not None else []

    @classmethod
    def from_edges(cls, instance: VRPInstance, edges: List[tuple] = None):
        routes = []
        for edge in edges:
            if edge[0] == 0:
                successor = edge[1]
                path = [0, successor]
                while successor != 0:
                    for e in edges:
                        if e[0] == successor:
                            successor = e[1]
                            path.append(successor)
                path = Route(path, instance)
                routes.append(path)
        return cls(instance=instance, routes=routes)

    def as_edges(self) -> List[tuple]:
        return [(from_id, to_id) for route in self.complete_routes() + self.incomplete_routes()
                for from_id, to_id in zip(route[:-1], route[1:])]

    def missing_customers(self) -> List[int]:
        return list(set(range(self.instance.n_customers + 1)) - set([from_id for from_id, _ in self.as_edges()]))

    def verify(self) -> bool:
        # Each tour does not exceed the vehicle capacity
        demands = [0] + self.instance.demands
        loads = [np.sum([demands[node] for node in tour]) for tour in self.routes]
        for i, load in enumerate(loads):
            assert load <= self.instance.capacity, \
                f"Tour {i} with a total of {load} exceeds the maximum capacity of {self.instance.capacity}."
        # Each tour starts and ends at the depot
        for i, route in enumerate(self.routes):
            assert route[0] == route[-1] == 0, \
                f"Route {i} is incomplete because it starts at {route[0]} and ends at {route[-1]}."
        # All customers have been visited
        missing = self.missing_customers()
        assert len(missing) == 0, \
            f"Customers {missing} are not present in any tour."
        return True

    def cost(self) -> float:
        return np.sum([self.instance.distance_matrix[from_id, to_id] for from_id, to_id in self.as_edges()])

    def complete_routes(self) -> List[Route]:
        return [route for route in self.routes if route.is_complete()]

    def incomplete_routes(self) -> List[Route]:
        return [route for route in self.routes if route.is_incomplete()]

    def get_customer_route(self, customer_idx: int) -> Route:
        for route in self.routes:
            if customer_idx in route:
                return route
