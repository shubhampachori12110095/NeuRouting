from copy import deepcopy
from typing import Tuple, List

import numpy as np
import torch
from torch import optim
from torch_geometric.data import DataLoader

from instances import VRPSolution
from lns.initial import nearest_neighbor_solution
from lns import DestroyProcedure
from lns.neural import NeuralProcedure
from utils.buffer import Buffer
from utils.running_mean_std import RunningMeanStd
from models import EgateModel


class EgateDestroy(NeuralProcedure, DestroyProcedure):
    def __init__(self, model: EgateModel, percentage: float, device="cpu", logger=None):
        super().__init__(model, device, logger)
        assert 0 <= percentage <= 1
        self.percentage = percentage

    def multiple(self, solutions: List[VRPSolution]):
        n_nodes = solutions[0].instance.n_customers + 1
        n_remove = int(n_nodes * self.percentage)
        nodes, edges = zip(*[self.features(sol) for sol in solutions])
        data_loader = Buffer.create_data(nodes, edges)
        for batch in data_loader:
            batch = batch.to(self.device)
            actions, log_p, values, entropy = self.model(batch, n_remove)
            to_remove = actions.squeeze().tolist()
        assert len(to_remove) == len(solutions), "Each solution must have a corresponding list of nodes to remove."
        for sol, remove in zip(solutions, to_remove):
            sol.destroy_nodes(remove)

    def __call__(self, solution: VRPSolution):
        self.multiple([solution])

    def _init_train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.model.train()
        self.n_rollout = 2
        self.rollout_steps = 5
        self.alpha = 1.0
        self.loss_vs = []
        self.loss_ps = []
        self.losses = []
        self.entropies = []

    def _train_step(self, opposite_procedure, train_batch):
        batch_solutions = [nearest_neighbor_solution(inst) for inst in train_batch]
        batch_size = len(train_batch)
        # Rollout phase
        all_datas = []
        for i in range(self.n_rollout):
            print(f"> Rollout {i + 1}:")
            is_last = (i == self.n_rollout - 1)
            rollout_solutions = [deepcopy(sol) for sol in batch_solutions]
            datas, states = self._rollout(rollout_solutions, opposite_procedure, self.rollout_steps, is_last)
            all_datas.extend(datas)
            print(f"{len(all_datas)} samples currently present.")

        # Training phase
        data_loader = DataLoader(all_datas, batch_size=batch_size, shuffle=True)

        for batch in data_loader:
            batch = batch.to(self.device)
            batch_size = batch.num_graphs
            actions = batch.action.reshape((batch_size, -1))
            log_p, v, entropy = self.model.evaluate(batch, actions)
            self.entropies.append(entropy.mean().item())

            target_vs = batch.v.squeeze(-1)
            old_log_p = batch.log_prob.squeeze(-1)
            adv = batch.adv.squeeze(-1)

            loss_v = ((v - target_vs) ** 2).mean()

            ratio = torch.exp(log_p - old_log_p)
            obj = ratio * adv
            obj_clipped = ratio.clamp(1.0 - 0.2,
                                      1.0 + 0.2) * adv
            loss_p = -torch.min(obj, obj_clipped).mean()
            loss = loss_p + self.alpha * loss_v

            self.losses.append(loss.item())
            self.loss_vs.append(loss_v.item())
            self.loss_ps.append(loss_p.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _train_info(self, epoch, batch_idx, log_interval) -> dict:
        loss_v = np.mean(self.loss_vs[-log_interval:])
        loss_p = np.mean(self.loss_ps[-log_interval:])
        loss = np.mean(self.losses[-log_interval:])
        entropy = np.mean(self.entropies[-log_interval:])
        return {"batch_idx": batch_idx + 1,
                "loss_v": loss_v,
                "loss_p": loss_p,
                "loss": loss,
                "entropy": entropy}

    def _ckpt_info(self, epoch, batch_idx) -> dict:
        return {"epoch": epoch + 1,
                "batch_idx": batch_idx + 1,
                "parameters": self.model.state_dict(),
                "optim": self.optimizer.state_dict()}

    def _rollout(self, solutions, repair_procedure, n_steps, is_last):
        n_nodes = solutions[0].instance.n_customers
        n_remove = int(n_nodes * self.percentage)
        all_nodes, all_edges = zip(*[self.features(sol) for sol in solutions])
        buffer = Buffer()
        reward_norm = RunningMeanStd()
        with torch.no_grad():
            self.model.eval()
            _sum = 0
            _entropy = []
            for i in range(n_steps):
                data_loader = buffer.create_data(all_nodes, all_edges, batch_size=len(solutions))
                data = list(data_loader)[0].to(self.device)
                actions, log_p, values, entropy = self.model(data, n_remove)
                for sol, to_remove in zip(solutions, actions):
                    prev_cost = sol.cost()
                    sol.destroy_nodes(to_remove)

                repair_procedure.multiple(solutions)

                new_all_nodes, new_all_edges, rewards = [], [], []
                for sol in solutions:
                    sol.verify()
                    new_cost = sol.cost()
                    nodes, edges = self.features(sol)
                    new_all_nodes.append(nodes)
                    new_all_edges.append(edges)
                    rewards.append(prev_cost - new_cost)

                rewards = np.array(rewards)
                _sum = _sum + rewards
                rewards = reward_norm(rewards)
                _entropy.append(entropy.mean().cpu().numpy())

                buffer.obs(all_nodes, all_edges, actions.cpu().numpy(), rewards, log_p.cpu().numpy(), values.cpu().numpy())
                all_nodes, all_edges = new_all_nodes, new_all_edges
                print(f"\t* Step {i + 1} completed.")

            if not is_last:
                data_loader = buffer.create_data(all_nodes, all_edges)
                data = list(data_loader)[0].to(self.device)
                _, _, values, _ = self.model(data, n_remove)
                values = values.cpu().numpy()
            else:
                values = 0

            datas = buffer.gen_datas(values, _lambda=0.99)
            return datas, (all_nodes, all_edges)

    @staticmethod
    def features(solution: VRPSolution) -> Tuple[np.ndarray, np.ndarray]:
        inst = solution.instance
        n = inst.n_customers + 1
        nodes = np.zeros((n, 5))
        for i in range(1, n):
            sol_route = solution.get_customer_route(i)
            nodes[i] = [float(inst.demands[i - 1]) / inst.capacity,
                        float(sol_route.demand_till_customer(i)) / inst.capacity,
                        float(sol_route.total_demand()) / inst.capacity,
                        sol_route.distance_till_customer(i),
                        sol_route.total_distance()]
        edges = np.zeros((n, n))
        for i, j in solution.as_edges():
            edges[i][j] = 1
        edges = np.stack([inst.distance_matrix, edges], axis=-1)
        edges = edges.reshape(-1, 2)
        return nodes, edges
