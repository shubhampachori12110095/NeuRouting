# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import operator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from experimental.neurewriter.data_utils import np_to_tensor
from experimental.neurewriter.modules import SeqLSTM, MLPModel
from experimental.neurewriter.base_model import BaseModel

eps = 1e-3
log_eps = np.log(eps)


class RegionPickerModel(BaseModel):
    def __init__(self, args):
        super(RegionPickerModel, self).__init__(args)
        self.input_encoder = SeqLSTM(args)
        self.value_estimator = MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 4, self.MLP_hidden_size, 1,
                                        self.cuda_flag, self.dropout_rate)

    def forward(self, dm_list, eval_flag=False):
        torch.set_grad_enabled(not eval_flag)

        batch_size = len(dm_list)
        dm_list = self.input_encoder.calc_embedding(dm_list, eval_flag)

        node_idxes = []
        node_states = []
        depot_states = []
        for dm_idx in range(batch_size):
            dm = dm_list[dm_idx]
            for i in range(1, len(dm.vehicle_state) - 1):
                node_idxes.append((dm_idx, i))
                node_states.append(dm.encoder_outputs[i].unsqueeze(0))
                depot_states.append(dm.encoder_outputs[0].clone().unsqueeze(0))

        pred_rewards = []
        for st in range(0, len(node_idxes), self.batch_size):
            cur_node_states = node_states[st: st + self.batch_size]
            cur_node_states = torch.cat(cur_node_states, 0)
            cur_depot_states = depot_states[st: st + self.batch_size]
            cur_depot_states = torch.cat(cur_depot_states, 0)
            cur_pred_rewards = self.value_estimator(torch.cat([cur_node_states, cur_depot_states], dim=1))
            pred_rewards.append(cur_pred_rewards)
        pred_rewards = torch.cat(pred_rewards, 0)

        candidate_rewrite_pos = [[] for _ in range(batch_size)]
        for idx, (dm_idx, node_idx) in enumerate(node_idxes):
            candidate_rewrite_pos[dm_idx].append((pred_rewards[idx].data[0], pred_rewards[idx], node_idx))

        return dm_list, candidate_rewrite_pos


class RulePickerModel(BaseModel):
    def __init__(self, args):
        super(RulePickerModel, self).__init__(args)
        self.embedding_size = args.embedding_size
        self.attention_size = args.attention_size
        self.sqrt_attention_size = int(np.sqrt(self.attention_size))
        self.reward_thres = -0.01

        self.policy_embedding = MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 6 + self.embedding_size * 2,
                                         self.MLP_hidden_size, self.attention_size, self.cuda_flag,
                                         self.dropout_rate)
        self.policy = MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 4, self.MLP_hidden_size,
                               self.attention_size, self.cuda_flag, self.dropout_rate)

    def rewrite(self, dm, trace_rec, candidate_rewrite_pos, eval_flag, max_search_pos, reward_thres=None):
        candidate_rewrite_pos.sort(reverse=True, key=operator.itemgetter(0))
        if not eval_flag:
            sample_exp_reward_tensor = []
            for idx, (cur_pred_reward, cur_pred_reward_tensor, rewrite_pos) in enumerate(candidate_rewrite_pos):
                sample_exp_reward_tensor.append(cur_pred_reward_tensor)
            sample_exp_reward_tensor = torch.cat(sample_exp_reward_tensor, 0)
            sample_exp_reward_tensor = torch.exp(sample_exp_reward_tensor * 10)

        candidate_dm = []
        candidate_rewrite_rec = []

        if not eval_flag:
            sample_rewrite_pos_dist = Categorical(sample_exp_reward_tensor)
            sample_rewrite_pos = sample_rewrite_pos_dist.sample(sample_shape=[len(candidate_rewrite_pos)])
            # sample_rewrite_pos = torch.multinomial(sample_exp_reward_tensor, len(candidate_rewrite_pos))
            sample_rewrite_pos = sample_rewrite_pos.data.cpu().numpy()
            indexes = np.unique(sample_rewrite_pos, return_index=True)[1]
            sample_rewrite_pos = [sample_rewrite_pos[i] for i in sorted(indexes)]
            sample_rewrite_pos = sample_rewrite_pos[:self.num_sample_rewrite_pos]
            # sample_exp_reward = [sample_exp_reward[i] for i in sample_rewrite_pos]
            sample_rewrite_pos = [candidate_rewrite_pos[i] for i in sample_rewrite_pos]
        else:
            sample_rewrite_pos = candidate_rewrite_pos.copy()

        for idx, (pred_reward, cur_pred_reward_tensor, rewrite_pos) in enumerate(sample_rewrite_pos):
            if len(candidate_dm) > 0 and idx >= max_search_pos:
                break
            if reward_thres is not None and pred_reward < reward_thres:
                if eval_flag:
                    break
                elif np.random.random() > self.cont_prob:
                    continue
            candidate_neighbor_idxes = dm.get_neighbor_idxes(rewrite_pos)
            cur_node_idx = dm.vehicle_state[rewrite_pos][0]
            cur_node = dm.get_node(cur_node_idx)
            pre_node_idx = dm.vehicle_state[rewrite_pos - 1][0]
            pre_node = dm.get_node(pre_node_idx)
            pre_capacity = dm.vehicle_state[rewrite_pos - 1][1]
            depot = dm.get_node(0)
            depot_state = dm.encoder_outputs[0].unsqueeze(0)
            cur_state = dm.encoder_outputs[rewrite_pos].unsqueeze(0)
            cur_states_0 = []
            cur_states_1 = []
            cur_states_2 = []
            new_embeddings_0 = []
            new_embeddings_1 = []
            for i in candidate_neighbor_idxes:
                neighbor_idx = dm.vehicle_state[i][0]
                neighbor_node = dm.get_node(neighbor_idx)
                cur_states_0.append(depot_state.clone())
                cur_states_1.append(cur_state.clone())
                cur_states_2.append(dm.encoder_outputs[i].unsqueeze(0))
                if pre_capacity >= neighbor_node.demand:
                    new_embedding = [neighbor_node.x, neighbor_node.y, neighbor_node.demand * 1.0 / dm.capacity,
                                     pre_node.x, pre_node.y, neighbor_node.demand * 1.0 / pre_capacity,
                                     dm.get_dis(pre_node, neighbor_node)]
                else:
                    new_embedding = [neighbor_node.x, neighbor_node.y, neighbor_node.demand * 1.0 / dm.capacity,
                                     pre_node.x, pre_node.y, neighbor_node.demand * 1.0 / dm.capacity,
                                     dm.get_dis(pre_node, depot) + dm.get_dis(depot, neighbor_node)]
                new_embeddings_0.append(new_embedding[:])
                if pre_capacity >= neighbor_node.demand:
                    new_embedding = [(neighbor_node.x - depot.x) * (pre_node.x - depot.x),
                                     (neighbor_node.y - depot.y) * (pre_node.y - depot.y),
                                     (neighbor_node.demand - cur_node.demand) * 1.0 / pre_capacity, pre_node.px,
                                     pre_node.py,
                                     (neighbor_node.demand - cur_node.demand) * 1.0 / dm.capacity,
                                     dm.get_dis(pre_node, depot) + dm.get_dis(depot, neighbor_node)]
                else:
                    new_embedding = [(neighbor_node.x - depot.x) * (pre_node.x - depot.x),
                                     (neighbor_node.y - depot.y) * (pre_node.y - depot.y),
                                     (neighbor_node.demand - cur_node.demand) * 1.0 / dm.capacity, pre_node.px,
                                     pre_node.py,
                                     (neighbor_node.demand - cur_node.demand) * 1.0 / dm.capacity,
                                     dm.get_dis(pre_node, depot) + dm.get_dis(depot, neighbor_node)]
                new_embeddings_1.append(new_embedding[:])
            cur_states_0 = torch.cat(cur_states_0, 0)
            cur_states_1 = torch.cat(cur_states_1, 0)
            cur_states_2 = torch.cat(cur_states_2, 0)
            new_embeddings_0 = np_to_tensor(new_embeddings_0, 'float', self.cuda_flag)
            new_embeddings_1 = np_to_tensor(new_embeddings_1, 'float', self.cuda_flag)
            policy_inputs = torch.cat([cur_states_0, cur_states_1, cur_states_2, new_embeddings_0, new_embeddings_1], 1)
            ctx_embeddings = self.policy_embedding(policy_inputs)
            cur_state_key = self.policy(torch.cat([cur_state, depot_state], dim=1))
            ac_logits = torch.matmul(cur_state_key, torch.transpose(ctx_embeddings, 0, 1)) / self.sqrt_attention_size
            ac_logprobs = nn.LogSoftmax(dim=1)(ac_logits)
            ac_probs = nn.Softmax(dim=1)(ac_logits)
            ac_logprobs = ac_logprobs.squeeze(0)
            ac_probs = ac_probs.squeeze(0)
            if eval_flag:
                _, candidate_acs = torch.sort(ac_logprobs, descending=True)
                candidate_acs = candidate_acs.data.cpu().numpy()
            else:
                candidate_acs_dist = Categorical(ac_probs)
                candidate_acs = candidate_acs_dist.sample(sample_shape=[ac_probs.size()[0]])
                # candidate_acs = torch.multinomial(ac_probs, ac_probs.size()[0])
                candidate_acs = candidate_acs.data.cpu().numpy()
                indexes = np.unique(candidate_acs, return_index=True)[1]
                candidate_acs = [candidate_acs[i] for i in sorted(indexes)]

            for i in candidate_acs:
                neighbor_idx = candidate_neighbor_idxes[i]
                new_dm = self.move(dm, rewrite_pos, neighbor_idx)
                if new_dm.tot_dis[-1] in trace_rec:
                    continue
                candidate_dm.append(new_dm)
                candidate_rewrite_rec.append(
                    (ac_logprobs, pred_reward, cur_pred_reward_tensor, rewrite_pos, i, new_dm.tot_dis[-1]))
                if len(candidate_dm) >= max_search_pos:
                    break

        return candidate_dm, candidate_rewrite_rec

    def forward(self, dm_list, trace_rec, candidate_rewrite_pos, eval_flag=False):
        torch.set_grad_enabled(not eval_flag)
        candidate_dm = []
        candidate_rewrite_rec = []
        for i in range(len(dm_list)):
            cur_candidate_dm, cur_candidate_rewrite_rec = self.rewrite(dm_list[i], trace_rec[i],
                                                                       candidate_rewrite_pos[i], eval_flag,
                                                                       max_search_pos=1, reward_thres=self.reward_thres)
            candidate_dm.append(cur_candidate_dm)
            candidate_rewrite_rec.append(cur_candidate_rewrite_rec)
        return candidate_dm, candidate_rewrite_rec

    @staticmethod
    def move(dm, cur_route_idx, neighbor_route_idx):
        min_update_idx = min(cur_route_idx, neighbor_route_idx)
        res = dm.clone()
        old_vehicle_state = res.vehicle_state[:]
        old_vehicle_state[cur_route_idx], old_vehicle_state[neighbor_route_idx] = old_vehicle_state[neighbor_route_idx], \
                                                                                  old_vehicle_state[cur_route_idx]
        if old_vehicle_state[neighbor_route_idx][0] == 0:
            del old_vehicle_state[neighbor_route_idx]
        res.vehicle_state = res.vehicle_state[:min_update_idx]
        res.route = res.route[:min_update_idx]
        res.tot_dis = res.tot_dis[:min_update_idx]
        cur_node_idx, cur_capacity = res.vehicle_state[-1]
        for t in range(min_update_idx, len(old_vehicle_state)):
            new_node_idx, new_capacity = old_vehicle_state[t]
            new_node = res.get_node(new_node_idx)
            if new_node_idx != 0 and cur_capacity < new_node.demand:
                res.add_route_node(0)
            res.add_route_node(new_node_idx)
            cur_capacity = res.vehicle_state[-1][1]
        return res


class NeuRewriterModel(BaseModel):
    """
    Model architecture for vehicle routing.
    """

    def __init__(self, args):
        super(NeuRewriterModel, self).__init__(args)

        self.region_picker = RegionPickerModel(args)
        self.rule_picker = RulePickerModel(args)

        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        elif args.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        else:
            raise ValueError('optimizer undefined: ', args.optimizer)

    def forward(self, batch_data, eval_flag=False):
        torch.set_grad_enabled(not eval_flag)
        dm_list = []
        batch_size = len(batch_data)
        for dm in batch_data:
            dm_list.append(dm)

        active = True
        reduce_steps = 0
        dm_rec = [[] for _ in range(batch_size)]
        trace_rec = [{} for _ in range(batch_size)]
        rewrite_rec = [[] for _ in range(batch_size)]

        for idx in range(batch_size):
            dm_rec[idx].append(dm_list[idx])
            trace_rec[idx][dm_list[idx].tot_dis[-1]] = 0

        while active and (self.max_reduce_steps is None or reduce_steps < self.max_reduce_steps):
            reduce_steps += 1
            dm, candidate_rewrite_pos = self.region_picker(dm_list, eval_flag)
            candidate_dm, candidate_rewrite_rec = self.rule_picker(dm, trace_rec, candidate_rewrite_pos, eval_flag)
            active = False
            for dm_idx in range(batch_size):
                cur_candidate_dm = candidate_dm[dm_idx]
                cur_candidate_rewrite_rec = candidate_rewrite_rec[dm_idx]
                if len(cur_candidate_dm) > 0:
                    active = True
                    cur_dm = cur_candidate_dm[0]
                    cur_rewrite_rec = cur_candidate_rewrite_rec[0]
                    dm_list[dm_idx] = cur_dm
                    rewrite_rec[dm_idx].append(cur_rewrite_rec)
                    trace_rec[dm_idx][cur_dm.tot_dis[-1]] = 0
            if not active:
                break
            updated_dm = self.region_picker.input_encoder.calc_embedding(dm_list, eval_flag)
            for i in range(batch_size):
                if updated_dm[i].tot_dis[-1] != dm_rec[i][-1].tot_dis[-1]:
                    dm_rec[i].append(updated_dm[i])

        total_policy_loss = np_to_tensor(np.zeros(1), 'float', self.cuda_flag)
        total_value_loss = np_to_tensor(np.zeros(1), 'float', self.cuda_flag)

        pred_value_rec = []
        value_target_rec = []
        total_reward = 0
        for dm_idx, cur_dm_rec in enumerate(dm_rec):
            pred_dis = []
            for dm in cur_dm_rec:
                pred_dis.append(dm.tot_dis[-1])
            best_reward = pred_dis[0]

            for idx, (ac_logprob, pred_reward, cur_pred_reward_tensor, rewrite_pos, applied_op, new_dis) \
                    in enumerate(rewrite_rec[dm_idx]):
                cur_reward = pred_dis[idx] - pred_dis[idx + 1]
                best_reward = min(best_reward, pred_dis[idx + 1])

                if self.gamma > 0.0:
                    decay_coef = 1.0
                    num_rollout_steps = len(cur_dm_rec) - idx - 1
                    for i in range(idx + 1, idx + 1 + num_rollout_steps):
                        cur_reward = max(decay_coef * (pred_dis[idx] - pred_dis[i]), cur_reward)
                        decay_coef *= self.gamma

                cur_reward_tensor = np_to_tensor(np.array([cur_reward], dtype=np.float32), 'float',
                                                 self.cuda_flag, volatile_flag=True)
                if ac_logprob.data.cpu().numpy()[0] > log_eps or cur_reward - pred_reward > 0:
                    ac_mask = np.zeros(ac_logprob.size()[0])
                    ac_mask[applied_op] = cur_reward - pred_reward
                    ac_mask = np_to_tensor(ac_mask, 'float', self.cuda_flag, eval_flag)
                    total_policy_loss -= ac_logprob[applied_op] * ac_mask[applied_op]
                pred_value_rec.append(cur_pred_reward_tensor)
                value_target_rec.append(cur_reward_tensor)

            total_reward += best_reward

        if len(pred_value_rec) > 0:
            pred_value_rec = torch.cat(pred_value_rec, 0)
            value_target_rec = torch.cat(value_target_rec, 0)
            pred_value_rec = pred_value_rec.unsqueeze(1)
            value_target_rec = value_target_rec.unsqueeze(1)
            total_value_loss = F.smooth_l1_loss(pred_value_rec, value_target_rec, reduction='sum')
        total_policy_loss /= batch_size
        total_value_loss /= batch_size
        total_loss = total_policy_loss * self.value_loss_coef + total_value_loss
        total_reward = total_reward * 1.0 / batch_size

        return total_loss, total_reward, dm_rec
