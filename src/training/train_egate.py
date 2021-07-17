from instances import generate_multiple_instances
from lns.neural import EgateDestroy
from lns.repair import SCIPRepair
from models.egate import EgateModel

if __name__ == "__main__":
    device = "cpu"
    destroy_procedure = EgateDestroy(EgateModel(5, 64, 2, 16), 0.1)
    repair_procedure = SCIPRepair()
    instances = generate_multiple_instances(n_instances=10, n_customers=30)
    destroy_procedure.train(opposite_procedure=repair_procedure,
                            instances=instances,
                            val_split=0.05,
                            batch_size=16,
                            epochs=2)
