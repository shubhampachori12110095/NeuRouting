import sys
import torch

sys.path.append("src")

from lns.destroy import DestroyPointBased
from lns.neural import ActorCriticRepair
from generators import generate_multiple_instances
from models import VRPActorModel, VRPCriticModel
from utils.logging import MultipleLogger, ConsoleLogger, WandBLogger

if __name__ == "__main__":
    # Select the best available device depending on the machine
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for training.")

    # Define the parameters needed for the training
    n_train_instances = 10000
    n_val_instances = 100
    n_customers = 50
    destroy_percentage = 0.15
    batch_size = 64
    n_epochs = 1
    log_interval = 5
    val_interval = 20
    ckpt_file = f"n_{n_customers}_destroy_point_{destroy_percentage}_repair_neural.pt"

    train_instances = generate_multiple_instances(n_instances=n_train_instances, n_customers=n_customers, seed=42)
    val_instances = generate_multiple_instances(n_instances=n_val_instances, n_customers=n_customers, seed=73)

    hidden_size = 128
    model = VRPActorModel(hidden_size=hidden_size, device=device)

    logger = MultipleLogger(loggers=[ConsoleLogger(),
                                     WandBLogger(model=model)])

    destroy_procedure = DestroyPointBased(percentage=destroy_percentage)
    repair_procedure = ActorCriticRepair(model, VRPCriticModel(hidden_size=hidden_size), device=device, logger=logger)

    repair_procedure.train(opposite_procedure=destroy_procedure,
                           train_instances=train_instances,
                           val_instances=val_instances,
                           batch_size=batch_size,
                           n_epochs=n_epochs,
                           ckpt_path=f'./pretrained/{ckpt_file}',
                           log_interval=log_interval,
                           val_interval=val_interval,
                           val_steps=n_customers)
