import re
import time
from abc import abstractmethod

import ecole
import numpy as np
import torch
from ecole.environment import Branching
from ecole.observation import NodeBipartite
from matplotlib import pyplot as plt

from environments import VRPEnvironment
from instances import VRPInstance, VRPSolution, VRPModelSCIP
from models.gcn import GCNModel
from utils.io import read_vrp


class VRPInfo:
    def before_reset(self, model: ecole.scip.Model) -> None:
        pass

    def extract(self, model: ecole.scip.Model, done: bool):
        scip_model = model.as_pyscipopt()
        edges_vars = [var for var in scip_model.getVars() if scip_model.getVal(var) > 0.99 and 'x' in str(var)]
        return [tuple([int(n) for n in re.findall(r'\d+', str(edge))]) for edge in edges_vars]


class EcoleEnvironment(VRPEnvironment):
    def __init__(self, base_env, name="Ecole"):
        super(EcoleEnvironment, self).__init__(name)
        self.scip_model = None
        self.env = base_env

        self.obs = None
        self.action_set = None
        self.reward = None

    def reset(self, instance: VRPInstance):
        super(EcoleEnvironment, self).reset(instance)
        self.scip_model = ecole.scip.Model.from_pyscipopt(VRPModelSCIP(instance))
        return self.env.reset(self.scip_model)

    @abstractmethod
    def step(self):
        pass

    def solve(self, instance: VRPInstance, time_limit=None, max_steps=None) -> VRPSolution:
        self.obs, self.action_set, self.reward, done, edges = self.reset(instance)
        # self.solution = VRPSolution.from_edges(self.instance, edges)

        self.max_steps = max_steps if max_steps is not None else self.max_steps
        self.time_limit = time_limit if time_limit is not None else self.time_limit
        start_time = time.time()
        while not done and self.n_steps < self.max_steps and time.time() - start_time < self.time_limit:
            self.obs, self.action_set, self.reward, done, edges = self.step()
            # self.solution = VRPSolution.from_edges(self.instance, edges)
            self.n_steps += 1
        return self.solution


class GCNEcoleEnvironment(EcoleEnvironment):
    def __init__(self, model: GCNModel, device: str = "cpu"):
        super().__init__(base_env=Branching(observation_function=NodeBipartite(),
                                            information_function=VRPInfo()),
                         name="GCN Ecole")
        self.model = model.to(device)
        self.device = device

    def step(self):
        with torch.no_grad():
            observation = (torch.from_numpy(self.obs.row_features.astype(np.float32)).to(self.device),
                           torch.from_numpy(self.obs.edge_features.indices.astype(np.int64)).to(self.device),
                           torch.from_numpy(self.obs.edge_features.values.astype(np.float32)).view(-1, 1).to(self.device),
                           torch.from_numpy(self.obs.column_features.astype(np.float32)).to(self.device))
            logits = self.model(*observation)
            action = self.action_set[logits[self.action_set.astype(np.int64)].argmax()]
            return self.env.step(action)


if __name__ == "__main__":
    inst = read_vrp("../../res/A-n32-k5.vrp", grid_dim=100)
    scipsolver = GCNEcoleEnvironment(GCNModel())
    sol = scipsolver.solve(inst, time_limit=10, max_steps=100)
    inst.plot(sol)
    plt.show()
