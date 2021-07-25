from typing import Tuple, List

import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

from lns.environments import BatchLNSEnvironment
from instances import VRPSolution, VRPInstance
from lns import LNSOperatorPair
from lns.destroy import DestroyProcedure, DestroyRandom
from lns.repair import RepairProcedure, SCIPRepair
from lns.neural import NeuralProcedure
from lns.utils import Buffer, RunningMeanStd
from models import EgateModel


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
        states = [self._features(sol) for sol in solutions]
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

    @staticmethod
    def _features(solution: VRPSolution) -> Tuple[np.ndarray, np.ndarray]:
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

    def train(self, train_instances: List[VRPInstance], val_instances: List[VRPInstance],
              opposite_procedure: RepairProcedure, path: str, batch_size: int, epochs: int):

        train_instances = np.array(train_instances)
        opt = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        random_op_pair = LNSOperatorPair(DestroyRandom(self.percentage), SCIPRepair())
        train_env = BatchLNSEnvironment(operator_pairs=[random_op_pair], batch_size=batch_size)

        for epoch in range(epochs):
            train_env.reset(train_instances[np.random.choice(len(train_instances), size=batch_size, replace=False)])
            pre_steps = 2
            for i in range(pre_steps):
                train_env.step()
            for sol in train_env.solutions:
                sol.verify()
            print(f"Random initialization completed.")

            # Rollout phase
            n_rollout, steps_rollout = 10, 10
            all_datas = []
            for i in range(n_rollout):
                is_last = (i == n_rollout - 1)
                datas, states = self._rollout(train_env.solutions, opposite_procedure, n_steps=steps_rollout, is_last=is_last)
                print(f"Rollout {i} completed.")
                all_datas.extend(datas)
            print("Rollout completed successfully.")

            # Training phase
            train_steps = 4
            dl = DataLoader(all_datas, batch_size=batch_size, shuffle=True)
            for step in range(train_steps):
                loss_v, loss_p, loss, entropy = self._train_once(dl, opt)
                print("Epoch:", epoch, "step:", step, "loss_v:", loss_v, "loss_p:", loss_p, "loss:",
                      loss, "entropy:", entropy)

        torch.save(self.model.state_dict(), path)

    def _rollout(self, solutions, repair_procedure, n_steps, is_last):
        n_nodes = solutions[0].instance.n_customers
        n_remove = int(n_nodes * self.percentage)
        all_nodes, all_edges = zip(*[self._features(sol) for sol in solutions])
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
                    repair_procedure(sol)
                    sol.verify()
                    new_cost = sol.cost()
                    nodes, edges = self._features(sol)
                    new_all_nodes.append(nodes)
                    new_all_edges.append(edges)
                    rewards.append(prev_cost - new_cost)
                print(f"> Step {i} completed successfully.")

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

            dl = buffer.gen_datas(values, _lambda=0.99)
            return dl, (all_nodes, all_edges)

    def _train_once(self, data_loader, opt, alpha=1.0):
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

    def load_weights(self, path: str):
        weights = torch.load(path, self.device)
        self.model.load_state_dict(weights)
        self.model.eval()
