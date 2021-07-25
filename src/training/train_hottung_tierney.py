import sys

import torch.cuda
import wandb

sys.path.append("src")

from generators import generate_multiple_instances
from lns.destroy import DestroyPointBased
from lns.neural import ActorCriticRepair
from models import VRPActorModel, VRPCriticModel

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for training.")
    actor = VRPActorModel(hidden_size=128, device=device)
    critic = VRPCriticModel(hidden_size=128)
    destroy_procedure = DestroyPointBased(percentage=0.15)
    repair_procedure = ActorCriticRepair(actor, critic, device=device)

    wandb.init(project='NeuRouting', entity='mazzio97')
    config = wandb.config
    wandb.watch(actor, critic)

    train_instances = generate_multiple_instances(n_instances=2000, n_customers=20, seed=42)
    val_instances = generate_multiple_instances(n_instances=200, n_customers=20, seed=4321)
    repair_procedure.train(train_instances=train_instances,
                           val_instances=val_instances,
                           opposite_procedure=destroy_procedure,
                           path="pretrained/nlns_repair.pt",
                           batch_size=32)
