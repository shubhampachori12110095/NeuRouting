import time
from typing import List, Dict

import numpy as np
from matplotlib import pyplot as plt

from environments import VRPSolver, VRPEnvironment
from instances import VRPInstance


class Stats(Dict[VRPSolver, Dict[VRPInstance, dict]]):
    def __init__(self, solvers: List[VRPSolver]):
        super().__init__([(solver, {}) for solver in solvers])

    def mean_cost(self) -> Dict[str, float]:
        return {solver.name: np.mean([info["cost"] for info in self[solver].values()]) for solver in self.keys()}


class Evaluator:
    def __init__(self, solvers: List[VRPSolver], render=False):
        self.solvers = solvers
        self.render = render

    def compare(self, instances: List[VRPInstance], max_steps: int, time_limit: int) -> Stats:
        stats = Stats(self.solvers)
        if self.render:
            fig, axes = plt.subplots(len(instances), len(self.solvers),
                                     sharex=True, sharey=True, squeeze=False,
                                     figsize=(3*len(self.solvers), 2*len(instances)))
        for i, inst in enumerate(instances):
            for j, solver in enumerate(self.solvers):
                stats[solver][inst] = {}
                inst_stats = stats[solver][inst]
                start_time = time.time()
                sol = solver.solve(inst, max_steps=max_steps, time_limit=time_limit)
                solve_time = time.time() - start_time
                sol.verify()
                if self.render:
                    axes[i, j].set_aspect("equal")
                    inst.plot(sol, axes[i, j])
                inst_stats["solution"] = sol
                inst_stats["n_vehicles"] = len(sol.routes)
                inst_stats["cost"] = sol.cost()
                inst_stats["time"] = solve_time
                if isinstance(solver, VRPEnvironment):
                    inst_stats["steps"] = solver.n_steps
                    inst_stats["improvements"] = solver.improvements
                print(f"Instance {i} solved by {solver.name} with cost {inst_stats['cost']}")
                solver.render()
        if self.render:
            plt.show()
        return stats
