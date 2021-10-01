# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import torch
import torch.multiprocessing as mp

from neurewriter.data_utils import process_batch


class Supervisor(object):
    """
    The base class to manage the high-level model execution processes.
    The concrete classes for different applications are derived from it.
    """

    def __init__(self, model, args):
        self.processes = args.processes
        self.model = model
        self.dropout_rate = args.dropout_rate
        self.global_step = 0
        self.batch_size = args.batch_size

    def load_pretrained(self, load_model):
        print("Read model parameters from %s." % load_model)
        checkpoint = torch.load(load_model)
        self.model.load_state_dict(checkpoint)

    def save_model(self, ckpt_path, ckpt_name):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({
            "parameters": self.model.state_dict(),
            "region_picker": self.model.region_picker.state_dict(),
            "rule_picker": self.model.rule_picker.state_dict()
        }, ckpt_path + ckpt_name)


class VRPSupervisor(Supervisor):
    """
    Management class for vehicle routing.
    """

    def __init__(self, model, args):
        super(VRPSupervisor, self).__init__(model, args)

    def train(self, batch_data):
        self.model.dropout_rate = self.dropout_rate
        self.model.optimizer.zero_grad()
        avg_loss, avg_reward, dm_rec = self.model(batch_data)
        self.global_step += 1
        if type(avg_loss) != float:
            avg_loss.backward()
            self.model.train()
        return avg_loss.item(), avg_reward

    def batch_eval(self, eval_data):
        cum_loss = 0
        cum_reward = 0
        cum_cost = 0
        data_size = len(eval_data)

        for batch_idx in range(0, data_size, self.batch_size):
            batch_data = process_batch(eval_data, self.batch_size, batch_idx)
            cur_avg_loss, cur_avg_reward, dm_rec = self.model(batch_data, eval_flag=True)
            cum_cost += sum([dm[-1].to_solution().cost() for dm in dm_rec])
            cum_loss += cur_avg_loss.item() * len(batch_data)
            cum_reward += cur_avg_reward * len(batch_data)
        return cum_loss, cum_reward, cum_cost

    def eval(self, data):
        data_size = len(data)
        if self.processes == 1:
            cum_loss, cum_reward, cum_cost = self.batch_eval(data)
        else:
            cum_loss = 0
            cum_reward = 0
            cum_cost = 0
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass
            pool = mp.Pool(processes=self.processes)
            res = []
            batch_per_process = data_size // self.processes
            if data_size % batch_per_process > 0:
                batch_per_process += 1
            for st in range(0, data_size, batch_per_process):
                res += [pool.apply_async(self.batch_eval, (data[st: st + batch_per_process]))]
            for i in range(len(res)):
                cur_cum_loss, cur_cum_reward, cur_cum_cost = res[i].get()
                cum_loss += cur_cum_loss
                cum_reward += cur_cum_reward
                cum_cost += cur_cum_cost

        return cum_loss / data_size, cum_reward / data_size, cum_cost / data_size
