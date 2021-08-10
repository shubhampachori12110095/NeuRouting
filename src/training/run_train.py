import argparse

import torch

from training.trainer import Trainer
from utils.logging import MultipleLogger, ConsoleLogger, WandBLogger

parser = argparse.ArgumentParser(description='Train Neural VRP')
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-o', '--opposite', type=str, required=False)
parser.add_argument('-n', '--n_customers', type=int, required=True)
parser.add_argument('-p', '--destroy_percentage', type=float, default=0.15)
parser.add_argument('-ts', '--train_samples', type=int, default=10000)
parser.add_argument('-vs', '--val_samples', type=int, default=100)
parser.add_argument('-dist', '--distribution', type=str, default="nazari")
parser.add_argument('-bs', '--batch_size', type=int, default=64)
parser.add_argument('-e', '--epochs', type=int, default=10)
parser.add_argument('-ns', '--neighborhood_size', type=int, default=64)

args = parser.parse_args()

if __name__ == "__main__":
    print(args)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for training.")

    logger = MultipleLogger(loggers=[ConsoleLogger(), WandBLogger()])

    trainer = Trainer(n_customers=args.n_customers,
                      n_train_instances=args.train_samples,
                      n_val_instances=args.val_samples,
                      distribution=args.distribution,
                      device=device,
                      logger=logger)

    train_params = {"model_name": args.model,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size}

    if args.opposite is None:
        train = trainer.train_env
    else:
        train_params = {**train_params,
                        "opposite_name": args.opposite,
                        "destroy_percentage": args.destroy_percentage,
                        "neighborhood_size": args.neighborhood_size}
        train = trainer.train_procedure

    train(**train_params)
