import os
import time
from copy import deepcopy
from math import ceil
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

from environments import BatchLNSEnvironment
from instances import VRPSolution, VRPInstance
from lns import LNSOperatorPair, nearest_neighbor_solution
from lns.destroy import DestroyProcedure
from lns.repair import RepairProcedure
from lns.neural import NeuralProcedure
from lns.utils import Buffer, RunningMeanStd
from models import EgateModel
from utils.logging import Logger


class EgateDestroy(DestroyProcedure, NeuralProcedure):

    def __init__(self, model: EgateModel, percentage: float, device="cpu"):
        assert 0 <= percentage <= 1
        self.percentage = percentage
        self.model = model
        self.device = device

    def __call__(self, solution: VRPSolution):
        self.multiple([solution])

    def multiple(self, solutions: List[VRPSolution]):
        n_nodes = solutions[0].instance.n_customers + 1
        n_remove = int(n_nodes * self.percentage)
        edge_index = torch.LongTensor([[i, j] for i in range(n_nodes) for j in range(n_nodes)]).T
        states = [self.features(sol) for sol in solutions]
        dataset = []
        for nodes, edges in states:
            data = Data(x=torch.from_numpy(nodes).float(),
                        edge_index=edge_index,
                        edge_attr=torch.from_numpy(edges).float())
            dataset.append(data)
        data_loader = DataLoader(dataset)
        for i, batch in enumerate(data_loader):
            actions, log_p, values, entropy = self.model(batch, n_remove)
            to_remove = actions.squeeze().tolist()
            solutions[i].destroy_nodes(to_remove)

    def train(self,
              opposite_procedure: RepairProcedure,
              train_instances: List[VRPInstance],
              batch_size: int,
              n_rollout: int,
              val_instances: List[VRPInstance],
              val_steps: int,
              val_interval: int,
              checkpoint_path: str,
              n_epochs: int = 1,
              logger: Optional[Logger] = None,
              log_interval: Optional[int] = None):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        neural_op_pair = LNSOperatorPair(self, opposite_procedure)
        val_env = BatchLNSEnvironment(operator_pairs=[neural_op_pair], batch_size=batch_size)

        incumbent_cost = np.inf

        train_size = len(train_instances)
        n_batches = ceil(float(train_size) / batch_size)
        for epoch in range(n_epochs):
            for batch_idx in range(n_batches):
                print(f"Batch {batch_idx + 1}:")
                begin = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, train_size)
                init_solutions = [nearest_neighbor_solution(inst) for inst in train_instances[begin:end]]

                # Rollout phase
                rollout_steps = ceil(float(val_steps) / n_rollout)
                print(rollout_steps)
                all_datas = []
                for i in range(n_rollout):
                    is_last = (i == n_rollout - 1)
                    batch_solutions = [deepcopy(sol) for sol in init_solutions]
                    datas, states = self._rollout(batch_solutions, opposite_procedure, rollout_steps, is_last)
                    all_datas.extend(datas)
                    print(f"> Rollout {i + 1} completed successfully: {len(all_datas)} samples.")

                # Training phase
                data_loader = DataLoader(all_datas, batch_size=batch_size, shuffle=True)
                loss_v, loss_p, loss, entropy = self._train_step(data_loader, optimizer)
                print("Training step performed!")

                if logger is not None and (batch_idx + 1) % log_interval == 0:
                    logger.log({"batch_idx": batch_idx + 1,
                                "loss_v": loss_v,
                                "loss_p": loss_p,
                                "loss": loss,
                                "entropy": entropy}, phase="train")

                if (batch_idx + 1) % val_interval == 0 or batch_idx == n_batches - 1:
                    self.model.eval()
                    start_eval_time = time.time()
                    val_env.solve(val_instances, max_steps=val_steps, time_limit=3600)
                    runtime = time.time() - start_eval_time
                    self.model.train()
                    mean_cost = np.mean([sol.cost() for sol in val_env.solutions])

                    if logger is not None:
                        logger.log({"batch_idx": batch_idx + 1,
                                    "mean_cost": mean_cost,
                                    "runtime": runtime}, phase="val")

                    if mean_cost < incumbent_cost:
                        incumbent_cost = mean_cost
                        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                        model_data = {"batch_idx": batch_idx + 1,
                                      "weights": self.model.state_dict(),
                                      "optim": optimizer.state_dict()}
                        torch.save(model_data, checkpoint_path)

    def load_weights(self, checkpoint_path: str):
        model_data = torch.load(checkpoint_path, self.device)
        self.model.load_state_dict(model_data["weights"])
        self.model.eval()

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
                data = buffer.create_data(all_nodes, all_edges).to(self.device)
                actions, log_p, values, entropy = self.model(data, n_remove)
                new_all_nodes, new_all_edges, rewards = [], [], []
                for sol, to_remove in zip(solutions, actions):
                    prev_cost = sol.cost()
                    sol.destroy_nodes(to_remove)

                repair_procedure.multiple(solutions)

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

            if not is_last:
                data = buffer.create_data(all_nodes, all_edges).to(self.device)
                actions, log_p, values, entropy = self.model(data, n_remove)
                values = values.cpu().numpy()
            else:
                values = 0

            datas = buffer.gen_datas(values, _lambda=0.99)
            return datas, (all_nodes, all_edges)

    def _train_step(self, data_loader, opt, alpha=1.0):
        self.model.train()

        loss_vs = []
        loss_ps = []
        losses = []
        entropies = []

        for i, batch in enumerate(data_loader):
            batch = batch.to(self.device)
            batch_size = batch.num_graphs
            actions = batch.action.reshape((batch_size, -1))
            log_p, v, entropy = self.model.evaluate(batch, actions)
            entropies.append(entropy.mean().item())

            target_vs = batch.v.squeeze(-1)
            old_log_p = batch.log_prob.squeeze(-1)
            adv = batch.adv.squeeze(-1)

            loss_v = ((v - target_vs) ** 2).mean()

            ratio = torch.exp(log_p - old_log_p)
            obj = ratio * adv
            obj_clipped = ratio.clamp(1.0 - 0.2,
                                      1.0 + 0.2) * adv
            loss_p = -torch.min(obj, obj_clipped).mean()
            loss = loss_p + alpha * loss_v

            losses.append(loss.item())
            loss_vs.append(loss_v.item())
            loss_ps.append(loss_p.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

        return np.mean(loss_vs), np.mean(loss_ps), np.mean(losses), np.mean(entropies)

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
