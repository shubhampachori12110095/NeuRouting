import sys
import torch

sys.path.append("src")

from generators import generate_multiple_instances
from lns.neural import EgateDestroy
from lns.repair import SCIPRepair
from models.egate import EgateModel
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
    log_interval = 4
    val_interval = 16
    run_name = f"n_{n_customers}_destroy_egate_{destroy_percentage}_repair_scip.pt"

    train_instances = generate_multiple_instances(n_instances=n_train_instances, n_customers=n_customers, seed=42)
    val_instances = generate_multiple_instances(n_instances=n_val_instances, n_customers=n_customers, seed=73)

    hidden_size = 128
    model = EgateModel(5, 64, 2, 16)

    logger = MultipleLogger(loggers=[ConsoleLogger(),
                                     WandBLogger(model=model)])

    destroy_procedure = EgateDestroy(model, percentage=destroy_percentage, device=device, logger=ConsoleLogger())
    repair_procedure = SCIPRepair()

    destroy_procedure.train(opposite_procedure=repair_procedure,
                            train_instances=train_instances,
                            val_instances=val_instances,
                            batch_size=batch_size,
                            n_epochs=n_epochs,
                            ckpt_path=f'./pretrained/{run_name}',
                            log_interval=log_interval,
                            val_interval=val_interval,
                            val_steps=n_customers)
