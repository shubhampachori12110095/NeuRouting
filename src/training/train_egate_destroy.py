import time
from math import floor

import torch

from generators import generate_multiple_instances
from lns.neural import EgateDestroy
from lns.repair import SCIPRepair
from models.egate import EgateModel
from utils.logging import MultipleLogger, ConsoleLogger, WandBLogger

if __name__ == "__main__":
    # Select the best available device depending on the machine
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for training.")

    model = EgateModel(5, 64, 2, 16)

    # Define the parameters needed for the training
    n_train_instances = 10000
    n_val_instances = 100
    n_customers = 50
    destroy_percentage = 0.15
    batch_size = 64
    log_interval = floor(float(n_train_instances // batch_size) / 10)
    print(log_interval)
    val_interval = log_interval * 5
    val_steps = 20
    n_rollout = 4
    run_name = f"n_{n_customers}_destroy_egate_{destroy_percentage}_repair_scip.pt"

    destroy_procedure = EgateDestroy(model, percentage=destroy_percentage, device=device)
    repair_procedure = SCIPRepair()

    train_instances = generate_multiple_instances(n_instances=n_train_instances, n_customers=n_customers, seed=42)
    val_instances = generate_multiple_instances(n_instances=n_val_instances, n_customers=n_customers, seed=73)

    logger = MultipleLogger(loggers=[ConsoleLogger(),
                                     WandBLogger(model=[model], run_name=run_name)])

    start_time = time.time()
    destroy_procedure.train(opposite_procedure=repair_procedure,
                            train_instances=train_instances,
                            val_instances=val_instances,
                            batch_size=batch_size,
                            val_steps=val_steps,
                            val_interval=val_interval,
                            n_rollout=n_rollout,
                            logger=logger,
                            log_interval=log_interval,
                            checkpoint_path=f"./pretrained/{run_name}")
    print(f"Training completed successfully in {time.time() - start_time} seconds.")
