from matplotlib import pyplot as plt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from instances import VRPSolution, VRPInstance
from instances.vrp_solution import Route
from utils.io import read_vrp, GRID_DIM


class OrToolsSolver:
    def __init__(self, instance: VRPInstance):
        self.instance = instance
        n_vehicles = int(sum(instance.demands) // instance.capacity) + 1
        self.data = {
            'distance_matrix': instance.distance_matrix * GRID_DIM,
            'demands': [0] + instance.demands,
            'num_vehicles': n_vehicles,
            'vehicle_capacities': [instance.capacity] * n_vehicles,
            'depot': 0
        }
        self.manager = pywrapcp.RoutingIndexManager(instance.n_customers + 1, n_vehicles, self.data['depot'])
        self.routing = pywrapcp.RoutingModel(self.manager)
        transit_callback_index = self.routing.RegisterTransitCallback(self._distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        demand_callback_index = self.routing.RegisterUnaryTransitCallback(self._demand_callback)
        self.routing.AddDimensionWithVehicleCapacity(evaluator_index=demand_callback_index,
                                                     slack_max=0,  # null capacity slack
                                                     vehicle_capacities=self.data['vehicle_capacities'],
                                                     fix_start_cumul_to_zero=True,  # start cumulative to zero
                                                     name='Capacity')

    def _distance_callback(self, from_index: int, to_index: int) -> float:
        """Returns the distance between the two nodes."""
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.data['distance_matrix'][from_node][to_node]

    def _demand_callback(self, from_index: int) -> int:
        """Returns the demand of the node."""
        from_node = self.manager.IndexToNode(from_index)
        return self.data['demands'][from_node]

    def _process_solution(self, solution) -> VRPSolution:
        routes = []
        for vehicle_id in range(self.data['num_vehicles']):
            index = self.routing.Start(vehicle_id)
            path = []
            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                path.append(node_index)
                index = solution.Value(self.routing.NextVar(index))
            path.append(self.manager.IndexToNode(index))
            # Check if a vehicle remained in the depot
            if len(path) > 2:
                routes.append(path)
        routes = [Route(r, self.instance) for r in routes]
        return VRPSolution(self.instance, routes)

    def solve(self, time_limit: int = 60) -> VRPSolution:
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.FromSeconds(time_limit)

        solution = self.routing.SolveWithParameters(search_parameters)
        return self._process_solution(solution)


if __name__ == "__main__":
    inst = read_vrp("../../res/A-n32-k5.vrp", grid_dim=100)
    orsolver = OrToolsSolver(inst)
    sol = orsolver.solve(10)
    inst.plot(sol)
    plt.show()
