import os
import re
from multiprocessing import Pool
from typing import List

from pyscipopt import Model

from instances.vrp_solution import VRPSolution
from instances.vrp_model_scip import VRPModelSCIP
from nlns import RepairProcedure


class SCIPRepair(RepairProcedure):

    def __init__(self, time_limit: int = 1e20):
        self.time_limit = time_limit

    @staticmethod
    def get_solution_edges(model: Model) -> List[tuple]:
        assignment = {var.name: model.getVal(var) for var in model.getVars() if 'x' in var.name}
        return [tuple([int(n) for n in re.findall(r'\d+', name)])
                for name, val in assignment.items() if val > 0.99]

    def multiple(self, partial_solutions: List[VRPSolution]):
        with Pool(os.cpu_count()) as pool:
            results = pool.map(self, partial_solutions)
            pool.close()
            pool.join()
        for sol, res in zip(partial_solutions, results):
            sol.routes = res.routes

    def __call__(self, partial_solution: VRPSolution):
        model = VRPModelSCIP(partial_solution.instance, lns_only=True)
        model.hideOutput()
        sub_mip = Model(sourceModel=model, origcopy=True)
        varname2var = {v.name: v for v in sub_mip.getVars() if 'x' in v.name}
        for x, y in partial_solution.as_edges():
            sub_mip.fixVar(varname2var[f'x({x}, {y})'], 1)
        sub_mip.setParam("limits/time", self.time_limit)
        sub_mip.optimize()
        new_sol = VRPSolution.from_edges(partial_solution.instance, self.get_solution_edges(sub_mip))
        partial_solution.routes = new_sol.routes
        return new_sol
