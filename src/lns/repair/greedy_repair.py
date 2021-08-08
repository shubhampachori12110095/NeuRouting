import numpy as np

from environments.lns_env import LNSEnvironment
from generators import generate_instance
from instances import VRPSolution
from lns import RepairProcedure, LNSOperator
from lns.destroy import DestroyRandom
from lns.initial.nearest_neighbor import closest_locations
from lns.repair import SCIPRepair


class GreedyRepair(RepairProcedure):
    def __call__(self, solution: VRPSolution):
        instance = solution.instance
        i = 0
        while len(solution.missing_customers()) > 0:
            missing = np.array(solution.missing_customers())

            mask = np.array([False] * (instance.n_customers + 1), dtype=bool)
            mask[missing] = True
            cust = np.random.choice(missing)
            cust_route = solution.get_customer_route(cust)
            self_head = cust_route[0] == cust
            mask[cust_route[0]] = False
            mask[cust_route[-1]] = False
            mask[0] = True
            nearest = closest_locations(instance, cust, mask=mask)

            for neigh in nearest:
                if neigh == 0:
                    cust_route.append_route([neigh], self_head, True)
                    break
                neigh_route = solution.get_customer_route(neigh)
                neigh_head = neigh_route[0] == neigh
                if cust_route.total_demand() + neigh_route.total_demand() <= instance.capacity:
                    cust_route.append_route(neigh_route, self_head, neigh_head)
                    solution.routes.remove(neigh_route)
                    break


if __name__ == "__main__":
    inst = generate_instance(n_customers=50, seed=12)
    env = LNSEnvironment([LNSOperator(DestroyRandom(0.05), GreedyRepair())], neighborhood_size=50)
    env.solve(inst, 50, 60)
