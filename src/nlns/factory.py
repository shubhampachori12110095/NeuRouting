# from environments.gcn_ecole_env import GCNEcoleEnvironment
import os
import random

from environments.lns_env import LNSEnvironment
from models import EgateModel, ResidualGatedGCNModel, VRPActorModel
from models.bipartite_gcn import BipartiteGCNModel
from nlns import LNSOperator
from nlns.destroy import EgateDestroy, ResidualGatedGCNDestroy, DestroyRandom, DestroyPointBased, DestroyTourBased
from nlns.initial import nearest_neighbor_solution
from nlns.neural import NeuralProcedure
from nlns.repair import RLAgentRepair, SCIPRepair
from nlns.repair.greedy_repair import GreedyRepair


destroy_procedures = {
    "random": DestroyRandom,
    "point": DestroyPointBased,
    "tour": DestroyTourBased,
    "egate": EgateDestroy,
    "resgatedgcn": ResidualGatedGCNDestroy,
}

repair_procedures = {
    "greedy": GreedyRepair,
    "scip": SCIPRepair,
    "rlagent": RLAgentRepair
}

neural_procedures = {
    "egate": EgateDestroy,
    "resgatedgcn": ResidualGatedGCNDestroy,
    "rlagent": RLAgentRepair,
}

neural_envs = {
    # "bipartite_gcn": GCNEcoleEnvironment,
}

neural_models = {
    "bipartite_gcn": BipartiteGCNModel(),
    "egate": EgateModel(),
    "resgatedgcn": ResidualGatedGCNModel(),
    "rlagent": VRPActorModel()
}


def nlns_builder(destroy_name: str,
                 repair_name: str,
                 neighborhood_size: int,
                 destroy_percentage=0.15,
                 initial=nearest_neighbor_solution,
                 name="NLNS",
                 device="cpu",
                 ckpt_path="./pretrained/") -> LNSEnvironment:

    if destroy_name in neural_procedures.keys():
        proc, model, ckpt = get_neural_procedure(destroy_name, repair_name, ckpt_path)
        destroy = proc(model, destroy_percentage, device=device)
        if ckpt is not None:
            print(f"Loading {ckpt} destroy checkpoint...")
            destroy.load_model(ckpt)
    else:
        destroy = destroy_procedures[destroy_name](destroy_percentage)

    if repair_name in neural_procedures.keys():
        proc, model, ckpt = get_neural_procedure(repair_name, destroy_name, ckpt_path)
        repair = proc(model, device=device)
        if ckpt is not None:
            print(f"Loading {ckpt} repair checkpoint...")
            repair.load_model(ckpt)
    else:
        repair = repair_procedures[repair_name]()

    return LNSEnvironment([LNSOperator(destroy, repair)], neighborhood_size, initial, name=name)


def get_neural_procedure(neural_name: str, opposite_name: str, ckpt_path: str):
    neural_proc = neural_procedures[neural_name]
    neural_model = neural_models[neural_name]
    model_ckpts = [ckpt_path + ckpt_file for ckpt_file in os.listdir(ckpt_path)
                   if "model_" + neural_name in ckpt_file]
    ckpt_file = None
    exists = False
    for ckpt in model_ckpts:
        if "opposite_" + opposite_name in ckpt:
            ckpt_file = ckpt
            break
    if not exists and len(model_ckpts) > 0:
        ckpt_file = random.choice(model_ckpts)
    return neural_proc, neural_model, ckpt_file
