from typing import List

from matplotlib import pyplot as plt

from instances import VRPSolution, Route
from lns.initial.nearest_neighbor import nearest_neighbor_solution
from utils.io import read_vrp


def destroy_solution(solution: VRPSolution, to_remove: List[int]):
    incomplete_routes = []
    complete_routes = []
    removed = []
    for route in solution.routes:
        last_split_idx = 0
        for i in range(1, len(route) - 1):
            if route[i] in to_remove:
                # Create two new tours:
                # The first consisting of the tour from the depot or from the last removed customer to the
                # customer that should be removed
                if i > last_split_idx and i > 1:
                    new_tour_pre = route[last_split_idx:i]
                    # complete_routes.append(new_tour_pre)
                    incomplete_routes.append(new_tour_pre)
                # The second consisting of only the customer to be removed
                customer_idx = route[i]
                # make sure the customer has not already been extracted from a different tour
                if customer_idx not in removed:
                    new_tour = [customer_idx]
                    # complete_routes.append(new_tour)
                    incomplete_routes.append(new_tour)
                    removed.append(customer_idx)
                last_split_idx = i + 1
        if last_split_idx > 0:
            # Create another new tour consisting of the remaining part of the original tour
            if last_split_idx < len(route) - 1:
                new_tour_post = route[last_split_idx:]
                # complete_routes.append(new_tour_post)
                incomplete_routes.append(new_tour_post)
        else:  # add unchanged tour
            complete_routes.append(route)
    complete_routes = [Route(cr, solution.instance) for cr in complete_routes]
    incomplete_routes = [Route(ir, solution.instance) for ir in incomplete_routes]
    return VRPSolution(instance=solution.instance, routes=complete_routes + incomplete_routes)


if __name__ == "__main__":
    inst = read_vrp("../../../res/A-n32-k5.vrp")
    inst.plot()
    plt.show()
    nn_sol = nearest_neighbor_solution(inst)
    inst.plot(nn_sol)
    plt.show()
    partial_sol = destroy_solution(nn_sol, [5, 12, 27])
    inst.plot(partial_sol)
    plt.show()
