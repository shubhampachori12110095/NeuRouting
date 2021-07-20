from heapq import nsmallest
from typing import List

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from instances import VRPSolution, VRPInstance
from lns import RepairProcedure
from lns.neural.neural_procedure import NeuralProcedure
from lns.destroy import DestroyProcedure


class ResidualGatedGCNDestroy(NeuralProcedure, DestroyProcedure):

    def __init__(self, model: nn.Module, percentage: float, device: str = "cpu"):
        super(ResidualGatedGCNDestroy, self).__init__()
        self.model = nn.DataParallel(model)
        self.percentage = percentage
        self.device = device
        self.current_instances = None
        self.edges_features = None

    def multiple(self, solutions: List[VRPSolution]):
        self.edges_features = self._compute_features([sol.instance for sol in solutions])
        for sol, feats in zip(solutions, self.edges_features):
            edges = sol.as_edges()
            map_edges_prob = {e: feats[e] for e in edges}
            n_remove = int(sol.instance.n_customers * self.percentage)
            to_remove = nsmallest(n_remove, map_edges_prob, key=map_edges_prob.get)
            sol.destroy_edges(to_remove)
            # to_remove = [node for edge in to_remove for node in edge]
            # sol.destroy_nodes(to_remove)

    def _compute_features(self, instances, num_neighbors=-1) -> np.ndarray:
        if instances is None or instances == self.current_instances:
            return self.edges_features  # Features have already been computed

        edges = np.stack([inst.adjacency_matrix(num_neighbors) for inst in instances])
        self.edges = Variable(torch.LongTensor(edges), requires_grad=False)
        edges_values = np.stack([inst.distance_matrix for inst in instances])
        self.edges_values = Variable(torch.FloatTensor(edges_values), requires_grad=False)
        nodes = np.zeros(instances[0].n_customers + 1, dtype=int)
        nodes[0] = 1
        self.nodes = Variable(torch.LongTensor(np.stack([nodes for _ in instances])), requires_grad=False)
        nodes_coord = np.stack([np.array([inst.depot] + inst.customers) for inst in instances])
        nodes_demands = np.stack([np.array(inst.demands, dtype=float) / inst.capacity for inst in instances])
        nodes_values = np.concatenate((nodes_coord, np.pad(nodes_demands, ((0, 0), (1, 0)))[:, :, None]), -1)
        self.nodes_values = Variable(torch.FloatTensor(nodes_values), requires_grad=False)
        with torch.no_grad():
            edges_preds, _ = self.model.forward(self.edges, self.edges_values, self.nodes, self.nodes_values)
        prob_preds = torch.log_softmax(edges_preds, -1)[:, :, :, -1]
        return np.exp(prob_preds)

    def __call__(self, solution: VRPSolution):
        self.multiple([solution])

    def train(self, train_instances: List[VRPInstance], val_instances: List[VRPInstance],
              opposite_procedure: RepairProcedure, path: str, batch_size: int, epochs: int):
        pass

    def load_weights(self, path: str):
        weights = torch.load(path, self.device)
        self.model.load_state_dict(weights['model_state_dict'])
        self.model.eval()
