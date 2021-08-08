from pathlib import Path

import torch

from environments.gcn_ecole_env import GCNEcoleEnvironment
from generators import generate_multiple_instances
from generators.ecole_branching_samples import generate_branching_samples
from utils.bipartite_graph_data import GraphDataset
from models.gcn import GCNModel
from utils.logging import ConsoleLogger


if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} for training.")

    n_train_instances = 10
    n_val_instances = 4
    n_customers = 20
    batch_size = 8
    n_epochs = 5
    n_samples_instance = 10
    ckpt_file = f"n_{n_customers}_ecole_gcn_branching.pt"

    model = GCNModel()

    train_instances = generate_multiple_instances(n_instances=n_train_instances, n_customers=n_customers, seed=42)
    val_instances = generate_multiple_instances(n_instances=n_val_instances, n_customers=n_customers, seed=73)

    train_files = generate_branching_samples(train_instances, n_samples_instance, folder="train")
    val_files = generate_branching_samples(val_instances, n_samples_instance, folder="val")
    # train_files = [str(path) for path in Path('train/').glob('instance_*.pkl')]
    # val_files = [str(path) for path in Path('val/').glob('instance_*.pkl')]

    train_data = GraphDataset(train_files, n_samples_instance)
    val_data = GraphDataset(val_files, n_samples_instance)

    env = GCNEcoleEnvironment(model, device, logger=ConsoleLogger())
    env.train(train_data=train_data,
              val_data=val_data,
              batch_size=batch_size,
              n_epochs=n_epochs,
              ckpt_path=f"pretrained/{ckpt_file}")
