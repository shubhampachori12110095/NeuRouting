import time
from copy import deepcopy
from math import ceil
from typing import List

import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from torch import optim

from environments.lns_environment import BatchLNSEnvironment
from instances import generate_multiple_instances, VRPSolution
from lns import LNSOperatorPair
from lns.destroy import DestroyProcedure
from lns.initial import nearest_neighbor_solution
from lns.repair.utils import VRPNeuralSolution
from lns.repair import NeuralRepairProcedure


class ActorCriticRepair(NeuralRepairProcedure):

    def __init__(self, actor: nn.Module, critic: nn.Module = None, device: str = 'cpu'):
        self.actor = actor.to(device)
        self.critic = critic.to(device) if critic is not None else critic
        self.device = device

    def __call__(self, partial_solution: VRPSolution) -> VRPSolution:
        with torch.no_grad():
            return self.multiple([partial_solution])[0]

    def _actor_model_forward(self,
                             incomplete_solutions: List[VRPNeuralSolution],
                             static_input,
                             dynamic_input,
                             vehicle_capacity: int):
        batch_size = static_input.shape[0]
        tour_idx, tour_logp = [], []

        solutions_repaired = np.zeros(batch_size)

        origin_idx = np.zeros(batch_size, dtype=int)

        while not solutions_repaired.all():
            # if origin_idx == 0 select the next tour-end that serves as the origin at random
            for i, solution in enumerate(incomplete_solutions):
                if origin_idx[i] == 0 and not solutions_repaired[i]:
                    origin_idx[i] = np.random.choice(solution.incomplete_nn_idx, 1).item()

            mask = self._get_mask(origin_idx, dynamic_input, incomplete_solutions, vehicle_capacity).to(self.device).float()

            # Rescale customer demand based on vehicle capacity
            dynamic_input_float = dynamic_input.float()
            dynamic_input_float[:, :, 0] = dynamic_input_float[:, :, 0] / float(vehicle_capacity)

            origin_static_input = static_input[torch.arange(batch_size), origin_idx]
            origin_dynamic_input = dynamic_input_float[torch.arange(batch_size), origin_idx]

            # Forward pass:
            # Returns a probability distribution over the point (tour end or depot) that origin should be connected to.
            probs = self.actor.forward(static_input, dynamic_input, origin_static_input, origin_dynamic_input)
            probs = F.softmax(probs + mask.log(), dim=1)  # Set prob of masked tour ends to zero

            if self.actor.training:
                m = torch.distributions.Categorical(probs)

                ptr = m.sample()
                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)  # Greedy selection
                logp = prob.log()

            # Perform action  for all data sequentially
            nn_input_updates = []
            ptr_np = ptr.cpu().numpy()
            for i, solution in enumerate(incomplete_solutions):
                idx_from = origin_idx[i].item()
                idx_to = ptr_np[i]
                if idx_from == 0 and idx_to == 0:  # No need to update in this case
                    continue

                nn_input_update, cur_nn_input_idx = solution.connect(idx_from, idx_to)  # Connect origin to select point

                for s in nn_input_update:
                    s.insert(0, i)
                    nn_input_updates.append(s)

                # Update origin
                if len(solution.incomplete_nn_idx) == 0:
                    solutions_repaired[i] = 1
                    origin_idx[i] = 0  # If instance is repaired set origin to 0
                else:
                    origin_idx[i] = cur_nn_input_idx  # Otherwise, set to tour end of the connect tour

            # Update network input
            nn_input_update = np.array(nn_input_updates)
            nn_input_update = torch.from_numpy(nn_input_update).to(self.device).long()
            dynamic_input[nn_input_update[:, 0], nn_input_update[:, 1]] = nn_input_update[:, 2:]

            logp = logp * (1. - torch.from_numpy(solutions_repaired).float().to(self.device))
            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))

        tour_idx = torch.cat(tour_idx, dim=1)
        tour_logp = torch.cat(tour_logp, dim=1)
        return tour_idx, tour_logp

    def _critic_model_forward(self, static_input, dynamic_input, vehicle_capacity: int):
        dynamic_input_float = dynamic_input.float()

        dynamic_input_float[:, :, 0] = dynamic_input_float[:, :, 0] / float(vehicle_capacity)

        return self.critic.forward(static_input, dynamic_input_float).view(-1)

    def train(self, destroy_procedure: DestroyProcedure, n_samples: int, val_split: float, batch_size: int):
        print("Generating training data...")
        train_size = round(n_samples * (1 - val_split))
        val_size = n_samples - train_size
        # Create training and validation set. The initial solutions are created greedily
        training_set = [nearest_neighbor_solution(inst)
                        for inst in generate_multiple_instances(n_instances=train_size, n_customers=30)]
        print(f"{len(training_set)} samples generated.")
        print("Generating validation data...")
        validation_set = [inst for inst in generate_multiple_instances(n_instances=val_size, n_customers=30)]
        print(f"{len(validation_set)} samples generated.")
        actor_optim = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.actor.train()
        critic_optim = optim.Adam(self.critic.parameters(), lr=5e-4)
        self.critic.train()

        losses_actor, rewards, diversity_values, losses_critic = [], [], [], []
        incumbent_costs = np.inf

        eval_env = BatchLNSEnvironment(batch_size, [LNSOperatorPair(destroy_procedure, self)])

        start_time = time.time()

        print("Starting training...")
        n_batches = ceil(float(train_size) / batch_size)
        for batch_idx in range(n_batches):
            begin = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, train_size)
            print(f"From {begin} to {end}")
            tr_solutions = [deepcopy(sol) for sol in training_set[begin:end]]
            print(f"Batch {batch_idx}: {len(tr_solutions)} samples")

            destroy_procedure.multiple(tr_solutions)
            costs_destroyed = [solution.cost() for solution in tr_solutions]
            _, tour_logp, critic_est = self._forward(tr_solutions)
            costs_repaired = [solution.cost() for solution in tr_solutions]

            # Reward/Advantage computation
            reward = np.array(costs_repaired) - np.array(costs_destroyed)
            reward = torch.from_numpy(reward).float().to(self.device)
            advantage = reward - critic_est

            # Actor loss computation and backpropagation
            max_grad_norm = 2.
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
            actor_optim.step()

            # Critic loss computation and backpropagation
            critic_loss = torch.mean(advantage ** 2)
            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
            critic_optim.step()

            rewards.append(torch.mean(reward.detach()).item())
            losses_actor.append(torch.mean(actor_loss.detach()).item())
            losses_critic.append(torch.mean(critic_loss.detach()).item())

            # Replace the solution of the training set instances with the new created solutions
            for i in range(end - begin):
                training_set[batch_idx * batch_size + i] = tr_solutions[i]

            log_interval = 1
            eval_interval = 4

            # Log performance
            if (batch_idx + 1) % log_interval == 0:
                mean_loss = np.mean(losses_actor[-log_interval:])
                mean_critic_loss = np.mean(losses_critic[-log_interval:])
                mean_reward = np.mean(rewards[-log_interval:])
                print(f'Batch {batch_idx + 1}/{n_batches}, repair costs (reward): {mean_reward:2.3f}, '
                      f'actor loss: {mean_loss:2.6f}, critic_loss: {mean_critic_loss:2.6f}')

            # Evaluate and save model every bunch of batches
            if (batch_idx + 1) % eval_interval == 0 or batch_idx == n_batches - 1:
                val_instances = [deepcopy(instance) for instance in validation_set]
                self.actor.eval()
                solutions = eval_env.solve(val_instances, time_limit=10)
                self.actor.train()
                mean_costs = np.mean([sol.cost() for sol in solutions])

                if mean_costs < incumbent_costs:
                    incumbent_costs = mean_costs
                    incumbent_model_path = "../../pretrained/model.pt"
                    model_data = {
                        'model_name': "VrpActorModel",
                        'parameters': self.actor.state_dict()
                    }
                    torch.save(model_data, incumbent_model_path)

                runtime = (time.time() - start_time)
                print(f"Validation (Batch {batch_idx}) Costs: {mean_costs:.3f} ({incumbent_costs:.3f}) Runtime: {runtime}")

    def _forward(self, partial_solutions: List[VRPSolution]):
        neural_solutions = [VRPNeuralSolution(solution) for solution in partial_solutions]
        emb_size = max([solution.min_nn_repr_size() for solution in neural_solutions])  # Max. input points of envs
        batch_size = len(neural_solutions)

        # Create envs input
        static_input = np.zeros((batch_size, emb_size, 2))
        dynamic_input = np.zeros((batch_size, emb_size, 2), dtype='int')
        for i, solution in enumerate(neural_solutions):
            static_nn_input, dynamic_nn_input = solution.network_representation(emb_size)
            static_input[i] = static_nn_input
            dynamic_input[i] = dynamic_nn_input

        static_input = torch.from_numpy(static_input).to(self.device).float()
        dynamic_input = torch.from_numpy(dynamic_input).to(self.device).long()
        # Assumes that the vehicle capacity is identical for all the incomplete solutions
        capacity = partial_solutions[0].instance.capacity

        cost_estimate = None
        if self.critic is not None:
            cost_estimate = self._critic_model_forward(static_input, dynamic_input, capacity)

        tour_idx, tour_logp = self._actor_model_forward(neural_solutions, static_input, dynamic_input, capacity)
        return tour_idx, tour_logp, cost_estimate

    def multiple(self, partial_solutions: List[VRPSolution]) -> List[VRPSolution]:
        self._forward(partial_solutions)
        return partial_solutions

    @staticmethod
    def _get_mask(origin_nn_input_idx, dynamic_input, solutions: List[VRPNeuralSolution], capacity: int):
        """ Returns a mask for the current nn_input"""
        batch_size = origin_nn_input_idx.shape[0]

        # Start with all used input positions
        mask = (dynamic_input[:, :, 1] != 0).cpu().long().numpy()

        for i in range(batch_size):
            idx_from = origin_nn_input_idx[i]
            origin_tour = solutions[i].map_network_idx_to_route[idx_from][0]
            origin_pos = solutions[i].map_network_idx_to_route[idx_from][1]

            # Find the start of the tour in the nn input
            # e.g. for the tour [2, 3] two entries in nn input exists
            if origin_pos == 0:
                idx_same_tour = origin_tour[-1][2]
            else:
                idx_same_tour = origin_tour[0][2]

            mask[i, idx_same_tour] = 0

            # Do not allow origin location = destination location
            mask[i, idx_from] = 0

        mask = torch.from_numpy(mask)

        origin_demands = dynamic_input[torch.arange(batch_size), origin_nn_input_idx, 0]
        combined_demands = origin_demands.unsqueeze(1).expand(batch_size, dynamic_input.shape[1]) + dynamic_input[:, :, 0]
        mask[combined_demands > capacity] = 0

        mask[:, 0] = 1  # Always allow to go to the depot

        return mask

    def load_actor_weights(self, path: str):
        model = torch.load(path, self.device)
        self.actor.load_state_dict(model['parameters'])
        self.actor.eval()
