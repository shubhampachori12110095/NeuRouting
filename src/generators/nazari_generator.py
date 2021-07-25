import numpy as np

from instances import VRPInstance


def generate_nazari_instances(n_instances: int, n_customers: int):
    acceptable = [10, 20, 50, 100]
    assert n_customers in acceptable, f"{n_customers} should be one of {acceptable} for Nazari distribution"
    capacity_map = {10: 20, 20: 30, 50: 40, 100: 50}
    capacity = capacity_map[n_customers]
    return [VRPInstance(tuple(depot), list(customers), list(demands), capacity) for depot, customers, demands
            in zip(list(np.random.uniform(size=(n_instances, 2))),
                   list(np.random.uniform(size=(n_instances, n_customers, 2))),
                   list(np.random.randint(1, 10, size=(n_instances, n_customers))))]