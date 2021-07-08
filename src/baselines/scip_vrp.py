import re

from matplotlib import pyplot as plt

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
        assignment = {var.name: self.model.getVal(var) for var in self.model.getVars() if 'x' in var.name}
        edges = [tuple([int(n) for n in re.findall(r'\d+', name)])
                for name, val in assignment.items() if val > 0.99]
        return VRPSolution.from_edges(instance=self.instance, edges=edges)


if __name__ == "__main__":
    inst = read_vrp("../../res/A-n32-k5.vrp", grid_dim=100)
    scipsolver = SCIPSolver(inst, lns_only=True)
    sol = scipsolver.solve(20)
    inst.plot(sol)
    plt.show()
