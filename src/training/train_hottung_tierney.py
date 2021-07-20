from instances import generate_multiple_instances
from lns.destroy import DestroyPointBased
from lns.neural import ActorCriticRepair
from models import VRPActorModel, VRPCriticModel

if __name__ == "__main__":
    device = "cpu"
    actor = VRPActorModel(hidden_size=128, device=device)
    critic = VRPCriticModel(hidden_size=128)
    destroy_procedure = DestroyPointBased(0.1)
    repair_procedure = ActorCriticRepair(actor, critic, device=device)

    train_instances = generate_multiple_instances(n_instances=100000, n_customers=100, seed=42)
    val_instances = generate_multiple_instances(n_instances=10000, n_customers=100, seed=4321)
    repair_procedure.train(train_instances=train_instances,
                           val_instances=val_instances,
                           opposite_procedure=destroy_procedure,
                           path="../../pretrained/nlns_hottung_tierney_actor.pt",
                           batch_size=64)
