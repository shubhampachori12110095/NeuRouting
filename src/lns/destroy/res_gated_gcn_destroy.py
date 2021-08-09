from heapq import nsmallest
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
from sklearn.utils import compute_class_weight

from baselines import LKHSolver
from instances import VRPSolution
from lns import DestroyProcedure
from lns.neural import NeuralProcedure
from models import ResidualGatedGCNModel


class ResidualGatedGCNDestroy(NeuralProcedure, DestroyProcedure):

    def __init__(self, model: ResidualGatedGCNModel, percentage, num_neighbors=-1, device="cpu", logger=None):
        super(ResidualGatedGCNDestroy, self).__init__(model, device, logger)
        assert 0 <= percentage <= 1
        self.percentage = percentage
        self.num_neighbors = num_neighbors
        self.current_instances = None
        self.edges_probs = None

    def multiple(self, solutions: List[VRPSolution]):
        edges_preds, _ = self.model.forward(*self.features([sol.instance for sol in solutions]))
        prob_preds = torch.log_softmax(edges_preds, -1)[:, :, :, -1]
        self.edges_probs = np.exp(prob_preds)
        for sol, feats in zip(solutions, self.edges_probs):
            sol.verify()
            edges = sol.as_edges()
            map_edges_prob = {e: feats[e] for e in edges}
            n_remove = int(sol.instance.n_customers * self.percentage)
            to_remove = nsmallest(n_remove, map_edges_prob, key=map_edges_prob.get)
            sol.destroy_edges(to_remove)
            # to_remove = [node for edge in to_remove for node in edge]
            # sol.destroy_nodes(to_remove)

    def __call__(self, solution: VRPSolution):
        self.multiple([solution])

    def _init_train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.lkh = LKHSolver("../../executables/LKH3")
        self.n_samples = 0
        self.running_loss = 0.0
        self.running_preds_mean_cost = 0.0
        self.running_lkh_mean_cost = 0.0

    def _train_step(self, opposite_procedure, train_batch):
        batch_size = len(train_batch)
        batch_edges_target = []
        lkh_costs = []
        for instance in train_batch:
            sol = self.lkh.solve(instance)
            batch_edges_target.append(sol.adjacency_matrix())
            lkh_costs.append(sol.cost())
        edges_target = np.stack(batch_edges_target, axis=0)
        edges_target = Variable(torch.LongTensor(edges_target), requires_grad=False)
        edges, edges_values, nodes, nodes_values = self.features(train_batch)
        edge_labels = edges_target.cpu().numpy().flatten()
        edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
        edges_preds, loss = self.model.forward(edges, edges_values, nodes, nodes_values, edges_target, edge_cw)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        preds_mean_cost = self._mean_tour_len_edges(edges_values, edges_preds)
        lkh_mean_cost = np.mean(lkh_costs)
        self.n_samples += batch_size
        self.running_loss += batch_size * loss.data.item()
        self.running_preds_mean_cost += batch_size * preds_mean_cost
        self.running_lkh_mean_cost += batch_size * lkh_mean_cost

    def _train_info(self, epoch, batch_idx, log_interval) -> dict:
        return {"epoch": epoch + 1,
                "batch_idx": batch_idx + 1,
                "loss": self.running_loss / self.n_samples,
                "preds_mean_cost": self.running_preds_mean_cost / self.n_samples,
                "lkh_mean_cost": self.running_lkh_mean_cost / self.n_samples}

    def _ckpt_info(self, epoch, batch_idx) -> dict:
        return {"epoch": epoch + 1,
                "batch_idx": batch_idx + 1,
                "parameters": self.model.state_dict(),
                "optim": self.optimizer.state_dict()}

    def features(self, instances):
        if instances is None or instances == self.current_instances:
            return self.edges_probs  # Probabilities have already been computed

        edges = np.stack([inst.adjacency_matrix(self.num_neighbors) for inst in instances])
        edges = Variable(torch.LongTensor(edges), requires_grad=False)
        edges_values = np.stack([inst.distance_matrix for inst in instances])
        edges_values = Variable(torch.FloatTensor(edges_values), requires_grad=False)
        nodes = np.zeros(instances[0].n_customers + 1, dtype=int)
        nodes[0] = 1
        nodes = Variable(torch.LongTensor(np.stack([nodes for _ in instances])), requires_grad=False)
        nodes_coord = np.stack([np.array([inst.depot] + inst.customers) for inst in instances])
        nodes_demands = np.stack([np.array(inst.demands, dtype=float) / inst.capacity for inst in instances])
        nodes_values = np.concatenate((nodes_coord, np.pad(nodes_demands, ((0, 0), (1, 0)))[:, :, None]), -1)
        nodes_values = Variable(torch.FloatTensor(nodes_values), requires_grad=False)
        return edges, edges_values, nodes, nodes_values

    @staticmethod
    def _mean_tour_len_edges(edges_values, edges_preds):
        y = F.softmax(edges_preds, dim=-1)  # B x V x V x voc_edges
        y = y.argmax(dim=3)  # B x V x V
        # Divide by 2 because edges_values is symmetric
        tour_lens = (y.float() * edges_values.float()).sum(dim=1).sum(dim=1) / 2
        mean_tour_len = tour_lens.sum().to(dtype=torch.float).item() / tour_lens.numel()
        return mean_tour_len
