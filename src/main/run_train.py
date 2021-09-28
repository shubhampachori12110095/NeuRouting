import argparse
import sys
import torch

sys.path.append("src")

from main.trainer import Trainer
from utils.logging import MultipleLogger, ConsoleLogger, WandBLogger

parser = argparse.ArgumentParser(description='Train Neural VRP')
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-o', '--opposite', type=str, required=False)
parser.add_argument('-n', '--n_customers', type=int, required=True)
parser.add_argument('-p', '--destroy_percentage', type=float, required=True)
parser.add_argument('-ts', '--train_samples', type=int, default=100000)
parser.add_argument('-vs', '--val_samples', type=int, default=100)
parser.add_argument('-dist', '--distribution', type=str, default="nazari")
parser.add_argument('-bs', '--batch_size', type=int, default=256)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-vi', '--val_interval', type=int, required=False)
parser.add_argument('-log', '--log_interval', type=int, required=False)

args = parser.parse_args()

if __name__ == "__main__":
    print(args)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for training.")

    # logger = MultipleLogger(loggers=[ConsoleLogger(), WandBLogger()])
    logger = ConsoleLogger()

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
        train = trainer.train_environment
    else:
        train_params["opposite_name"] = args.opposite
        train_params["destroy_percentage"] = args.destroy_percentage

        if args.log_interval is not None:
            train_params["log_interval"] = args.log_interval

        if args.val_interval is not None:
            train_params["val_interval"] = args.val_interval

        train = trainer.train_procedure

    train(**train_params)

