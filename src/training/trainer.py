import os

import random
from typing import Optional

# from environments.gcn_ecole_env import GCNEcoleEnvironment
from generators import generate_multiple_instances
from nlns.destroy import EgateDestroy, ResidualGatedGCNDestroy, DestroyRandom, DestroyPointBased, DestroyTourBased
from nlns.repair import RLAgentRepair, SCIPRepair
from nlns.repair.greedy_repair import GreedyRepair
from models import EgateModel, ResidualGatedGCNModel, VRPActorModel, VRPCriticModel
from models.bipartite_gcn import BipartiteGCNModel
from utils.logging import Logger


class Trainer:
    def __init__(self,
                 n_customers: int,
                 n_train_instances: int,
                 n_val_instances: int,
                 distribution: str = "nazari",
                 device: str = "cpu",
                 ckpt_path: str = "./pretrained/",
                 logger: Optional[Logger] = None):

        self.neural_models = {
            "bipartite_gcn": BipartiteGCNModel(),
            "egate": EgateModel(),
            "res_gated_gcn": ResidualGatedGCNModel(),
            "rl_agent": VRPActorModel()
        }
        self.neural_envs = {
            # "bipartite_gcn": GCNEcoleEnvironment,
        }
        self.neural_procedures = {
            "egate": EgateDestroy,
            "res_gated_gcn": ResidualGatedGCNDestroy,
            "rl_agent": RLAgentRepair,
        }
        self.destroy_procedures = {
            "random": DestroyRandom,
            "point": DestroyPointBased,
            "tour": DestroyTourBased,
            "egate": EgateDestroy,
            "res_gated_gcn": ResidualGatedGCNDestroy,
        }
        self.repair_procedures = {
            "greedy": GreedyRepair,
            "scip": SCIPRepair,
            "rl_agent": RLAgentRepair
        }

        self.n_customers = n_customers
        self.train_instances = generate_multiple_instances(n_instances=n_train_instances,
                                                           n_customers=n_customers,
                                                           distribution=distribution,
                                                           seed=42)
        self.val_instances = generate_multiple_instances(n_instances=n_val_instances,
                                                         n_customers=n_customers,
                                                         distribution=distribution,
                                                         seed=73)
        self.device = device
        self.ckpt_path = ckpt_path
        self.logger = logger

    def train_env(self, model_name: str, batch_size: int, epochs: int):
        neural_env, model = self._select_neural_env(model_name, self.neural_envs)
        # logger = ConsoleLogger()
        neural_env = neural_env(model, self.device, self.logger)
        ckpt_file = self.ckpt_path + f"env_{model_name}_n{self.n_customers}.pt"
        return neural_env.train(train_instances=self.train_instances,
                                val_instances=self.val_instances,
                                n_epochs=epochs,
                                batch_size=batch_size,
                                ckpt_path=ckpt_file)

    def train_procedure(self, model_name: str, opposite_name: str, epochs: int, batch_size: int,
                        destroy_percentage: float, neighborhood_size: int):
        neural_proc, model = self._select_neural_env(model_name, self.neural_procedures)

        if model_name in self.destroy_procedures.keys():
            neural_proc = neural_proc(model, destroy_percentage, self.device, self.logger)
            destroy_name = model_name
            repair_name = opposite_name
        else:
            neural_proc = neural_proc(model, VRPCriticModel(), self.device, self.logger)
            destroy_name = opposite_name
            repair_name = model_name

        if opposite_name in self.neural_procedures.keys():
            opposite = self.neural_procedures[opposite_name](self.neural_models[opposite_name], device=self.device)
            model_ckpts = [self.ckpt_path + ckpt_file for ckpt_file in os.listdir(self.ckpt_path)
                           if opposite_name in ckpt_file and str(self.n_customers) in ckpt_file]
            opposite.load_model(ckpt_path=random.choice(model_ckpts))
        elif opposite_name == destroy_name:
            opposite = self.destroy_procedures[opposite_name](destroy_percentage)
        else:
            opposite = self.repair_procedures[opposite_name]()

        ckpt_file = f"destroy_{destroy_name}_{destroy_percentage}_repair_{repair_name}_n{self.n_customers}.pt"

        return neural_proc.train(train_instances=self.train_instances,
                                 val_instances=self.val_instances,
                                 opposite_procedure=opposite,
                                 n_epochs=epochs,
                                 batch_size=batch_size,
                                 val_neighborhood=neighborhood_size,
                                 ckpt_path=self.ckpt_path + ckpt_file)

    def _select_neural_env(self, model_name: str, envs: dict):
        neural_envs_name = list(envs.keys())
        assert model_name in neural_envs_name, \
            f"Unknown neural environment {model_name}, select one between {neural_envs_name}."
        model = self.neural_models[model_name]
        neural_env = envs[model_name]
        return neural_env, model
