from instances import generate_multiple_instances
from lns.destroy.traditional.destroy_point import DestroyPointBased
from lns.repair.neural import ActorCriticRepair
from models import VRPActorModel, VRPCriticModel

if __name__ == "__main__":
    device = "cpu"
    actor = VRPActorModel(hidden_size=128, device="cpu")
    critic = VRPCriticModel(hidden_size=128)
    destroy_procedure = DestroyPointBased(0.1)
    repair_procedure = ActorCriticRepair(actor, critic, device=device)

    instances = generate_multiple_instances(n_instances=1000, n_customers=30)
    repair_procedure.train(destroy_procedure=destroy_procedure,
                           instances=instances,
                           val_split=0.05,
                           batch_size=32)
