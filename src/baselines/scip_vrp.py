import re
from typing import List

from matplotlib import pyplot as plt
from pyscipopt import Model

from instances import VRPInstance, VRPSolution
from instances.vrp_model_scip import VRPModelSCIP
from utils.io import read_vrp


class SCIPSolver:
    def __init__(self, instance: VRPInstance, lns_only=False):
        self.instance = instance
        self.model = VRPModelSCIP(instance, lns_only)
        self.model.hideOutput()

    def solve(self, time_limit: int = 60) -> VRPSolution:
        self.model.setParam("limits/time", time_limit)
        self.model.optimize()
        return VRPSolution.from_edges(instance=self.instance, edges=get_solution_edges(self.model))

    def solve_subproblem(self, partial_solution: VRPSolution) -> VRPSolution:
        assert partial_solution.instance is self.instance, \
            "The specified solution does not correspond to the required problem instance."
        sub_mip = Model(sourceModel=self.model, origcopy=True)
        varname2var = {v.name: v for v in sub_mip.getVars() if 'x' in v.name}
        for x, y in partial_solution.as_edges():
            sub_mip.fixVar(varname2var[f'x({x}, {y})'], 1)
        sub_mip.setParam("limits/time", 1e20)  # no time limit
        sub_mip.optimize()
        return VRPSolution.from_edges(self.instance, get_solution_edges(sub_mip))


def get_solution_edges(model: Model) -> List[tuple]:
    assignment = {var.name: model.getVal(var) for var in model.getVars() if 'x' in var.name}
    return [tuple([int(n) for n in re.findall(r'\d+', name)])
            for name, val in assignment.items() if val > 0.99]


if __name__ == "__main__":
    inst = read_vrp("../../res/A-n32-k5.vrp")
    scipsolver = SCIPSolver(inst, lns_only=True)
    sol = scipsolver.solve(20)
    inst.plot(sol)
    plt.show()
