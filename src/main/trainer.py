from typing import Optional

from generators import generate_multiple_instances
from nlns.factory import *
from models import VRPCriticModel
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
        neural_env, model = self._select_neural_env(model_name, neural_envs)
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
        neural_proc, model = self._select_neural_env(model_name, neural_procedures)

        if model_name in destroy_procedures.keys():
            neural_proc = neural_proc(model, destroy_percentage, device=self.device, logger=self.logger)
        else:
            neural_proc = neural_proc(model, VRPCriticModel(), device=self.device, logger=self.logger)

        if opposite_name in neural_procedures.keys():
            proc, model, ckpt = get_neural_procedure(opposite_name, model_name, self.ckpt_path)
            opposite = proc(model, device=self.device) if opposite_name in repair_procedures.keys() else \
                proc(model, destroy_percentage, device=self.device)
            if ckpt is not None:
                print(f"Loading {ckpt} destroy checkpoint...")
                opposite.load_model(ckpt)
        elif opposite_name in destroy_procedures.keys():
            opposite = destroy_procedures[opposite_name](destroy_percentage)
        else:
            opposite = repair_procedures[opposite_name]()

        ckpt_file = f"model_{model_name}_opposite_{opposite_name}_n{self.n_customers}_p{destroy_percentage}.pt"

        return neural_proc.train(train_instances=self.train_instances,
                                 val_instances=self.val_instances,
                                 opposite_procedure=opposite,
                                 n_epochs=epochs,
                                 batch_size=batch_size,
                                 val_neighborhood=neighborhood_size,
                                 ckpt_path=self.ckpt_path + ckpt_file)

    @staticmethod
    def _select_neural_env(model_name: str, envs: dict):
        neural_envs_name = list(envs.keys())
        assert model_name in neural_envs_name, \
            f"Unknown neural environment {model_name}, select one between {neural_envs_name}."
        model = neural_models[model_name]
        neural_env = envs[model_name]
        return neural_env, model
