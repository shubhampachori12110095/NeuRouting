import re
from typing import List

from pyscipopt import Model

from instances.vrp_solution import VRPSolution
from instances.vrp_model_scip import VRPModelSCIP
from lns.repair import RepairProcedure


class SCIPRepair(RepairProcedure):

    @staticmethod
    def _get_solution_edges(model: Model) -> List[tuple]:
        assignment = {var.name: model.getVal(var) for var in model.getVars() if 'x' in var.name}
        return [tuple([int(n) for n in re.findall(r'\d+', name)])
                for name, val in assignment.items() if val > 0.99]

    def __call__(self, partial_solution: VRPSolution):
        model = VRPModelSCIP(partial_solution.instance, lns_only=True)
        model.hideOutput()
        sub_mip = Model(sourceModel=model, origcopy=True)
        varname2var = {v.name: v for v in sub_mip.getVars() if 'x' in v.name}
        for x, y in partial_solution.as_edges():
            sub_mip.fixVar(varname2var[f'x({x}, {y})'], 1)
        sub_mip.setParam("limits/time", 1e20)  # no time limit
        sub_mip.optimize()
        new_sol = VRPSolution.from_edges(partial_solution.instance, self._get_solution_edges(sub_mip))
        partial_solution.routes = new_sol.routes
