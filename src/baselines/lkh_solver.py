import os
import tempfile
from subprocess import check_output

import matplotlib.pyplot as plt

from instances import generate_instance, VRPSolution, Route
from utils.io import write_vrp, read_solution


class LKHSolver:
    def __init__(self, instance, executable):
        self.instance = instance
        self.executable = executable

    @staticmethod
    def write_lkh_par(filename, parameters):
        default_parameters = {  # Use none to include as flag instead of kv
            "SPECIAL": None,
            "RUNS": 10,
            "TRACE_LEVEL": 1,
            "SEED": 0
        }
        with open(filename, 'w') as f:
            for k, v in {**default_parameters, **parameters}.items():
                if v is None:
                    f.write("{}\n".format(k))
                else:
                    f.write("{} = {}\n".format(k, v))

    def solve(self, max_trials=None):
        if max_trials is None:
            max_trials = self.instance.n_customers
        with tempfile.TemporaryDirectory() as tempdir:
            problem_filename = os.path.join(tempdir, "problem.vrp")
            output_filename = os.path.join(tempdir, "output.tour")
            param_filename = os.path.join(tempdir, "params.par")

            write_vrp(self.instance, problem_filename)
            params = {"PROBLEM_FILE": problem_filename,
                      "OUTPUT_TOUR_FILE": output_filename,
                      "MAX_TRIALS": max_trials}
            self.write_lkh_par(param_filename, params)
            check_output([self.executable, param_filename])
            tours = read_solution(output_filename, self.instance.n_customers)
            tours = [Route(t, self.instance) for t in tours]
            return VRPSolution(self.instance, tours)


if __name__ == "__main__":
    inst = generate_instance(50)
    lkhsolver = LKHSolver(inst, "../../executables/LKH")
    sol = lkhsolver.solve()
    inst.plot(sol)
    plt.show()
