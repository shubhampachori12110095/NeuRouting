from ecole.environment import Configuring
from matplotlib import pyplot as plt

from environments.ecole_env import EcoleEnvironment, VRPInfo
from utils.io import read_vrp


class SCIPSolver(EcoleEnvironment):
    def __init__(self, lns_only=False):
        super().__init__(base_env=Configuring(observation_function=None,
                                              information_function=VRPInfo()),
                         name="SCIP")
        self.model = None
        self.lns_only = lns_only

    def step(self):
        return self.env.step({})


if __name__ == "__main__":
    inst = read_vrp("../../res/A-n32-k5.vrp", grid_dim=100)
    scipsolver = SCIPSolver(lns_only=True)
    sol = scipsolver.solve(inst, time_limit=20)
    inst.plot(sol)
    plt.show()
