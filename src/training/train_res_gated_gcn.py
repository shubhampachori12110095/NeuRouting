import sys
import torch

from lns.repair.greedy_repair import GreedyRepair
from models import ResidualGatedGCNModel

sys.path.append("src")

from generators import generate_multiple_instances
from lns.destroy import ResidualGatedGCNDestroy
from utils.logging import MultipleLogger, ConsoleLogger, WandBLogger

if __name__ == "__main__":
    # Select the best available device depending on the machine
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for training.")

    # Define the parameters needed for the training
    n_train_instances = 10
    n_val_instances = 4
    n_customers = 50
    destroy_percentage = 0.15
    batch_size = 4
    n_epochs = 1
    log_interval = 1
    val_interval = 1
    ckpt_file = f"n_{n_customers}_destroy_resgatedgcn_{destroy_percentage}_repair_greedy.pt"

    train_instances = generate_multiple_instances(n_instances=n_train_instances, n_customers=n_customers, seed=42)
    val_instances = generate_multiple_instances(n_instances=n_val_instances, n_customers=n_customers, seed=73)

    model = ResidualGatedGCNModel()

    logger = MultipleLogger(loggers=[ConsoleLogger(),
                                     WandBLogger(model=model)])

    destroy_procedure = ResidualGatedGCNDestroy(model, percentage=destroy_percentage, device=device, logger=ConsoleLogger())
    repair_procedure = GreedyRepair()

    destroy_procedure.train(opposite_procedure=repair_procedure,
                            train_instances=train_instances,
                            val_instances=val_instances,
                            batch_size=batch_size,
                            n_epochs=n_epochs,
                            ckpt_path=f'./pretrained/{ckpt_file}',
                            log_interval=log_interval,
                            val_interval=val_interval,
                            val_steps=n_customers // 5,
                            val_neighborhood=n_customers // 5)
