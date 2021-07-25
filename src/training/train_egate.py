import torch

from generators import generate_multiple_instances
from lns.neural import EgateDestroy
from lns.repair import SCIPRepair
from models.egate import EgateModel

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    destroy_procedure = EgateDestroy(EgateModel(5, 64, 2, 16), percentage=0.15)
    repair_procedure = SCIPRepair()
    train_instances = generate_multiple_instances(n_instances=10, n_customers=20, seed=42)
    val_instances = generate_multiple_instances(n_instances=2, n_customers=20, seed=4321)
    destroy_procedure.train(train_instances=train_instances,
                            val_instances=val_instances,
                            opposite_procedure=repair_procedure,
                            path="pretrained/egate_destroy.pt",
                            batch_size=4,
                            epochs=2)
