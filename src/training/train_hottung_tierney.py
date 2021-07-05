from lns.destroy import DestroyRandom
from lns.repair.neural_repair_procedures import ActorCriticRepair
from models import VrpActorModel, VrpCriticModel

if __name__ == "__main__":
    device = "cpu"
    actor = VrpActorModel(hidden_size=128, device="cpu")
    critic = VrpCriticModel(hidden_size=128)
    repair_procedure = ActorCriticRepair(actor, critic, device=device)
    repair_procedure.train(destroy_procedure=DestroyRandom(0.1),
                           n_samples=320,
                           val_split=0.1,
                           batch_size=16)
