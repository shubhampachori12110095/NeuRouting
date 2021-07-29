import sys
import time
from math import floor

import torch.cuda

from utils.logging import ConsoleLogger, MultipleLogger, WandBLogger

sys.path.append("src")

from generators import generate_multiple_instances
from lns.destroy import DestroyPointBased
from lns.neural import ActorCriticRepair
from models import VRPActorModel, VRPCriticModel

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
    log_interval = floor(float(n_train_instances // batch_size) / 10)
    val_interval = log_interval * 5
    val_steps = 20
    run_name = f"n_{n_customers}_destroy_point_{destroy_percentage}_repair_neural.pt"

    # Create the actor & critic models for the repair operator
    hidden_size = 128
    actor = VRPActorModel(hidden_size=hidden_size, device=device)
    critic = VRPCriticModel(hidden_size=hidden_size)

    # Create the actual LNS operators
    destroy_procedure = DestroyPointBased(percentage=0.15)
    repair_procedure = ActorCriticRepair(actor, critic, device=device)

    # Generate the instances to be used for training and validation
    train_instances = generate_multiple_instances(n_instances=n_train_instances, n_customers=n_customers, seed=42)
    val_instances = generate_multiple_instances(n_instances=n_val_instances, n_customers=n_customers, seed=73)

    logger = MultipleLogger(loggers=[ConsoleLogger(),
                                     WandBLogger(model=[actor, critic], run_name=run_name)])

    # Run the training of the repair operator
    start_time = time.time()
    repair_procedure.train(opposite_procedure=destroy_procedure,
                           train_instances=train_instances,
                           val_instances=val_instances,
                           batch_size=batch_size,
                           val_steps=val_steps,
                           val_interval=val_interval,
                           logger=logger,
                           log_interval=log_interval,
                           checkpoint_path=f"./pretrained/{run_name}")
    print(f"Training completed successfully in {time.time() - start_time} seconds.")
