from abc import ABC, abstractmethod
from typing import List, Union

from instances import VRPInstance
from lns import DestroyProcedure, RepairProcedure


class NeuralProcedure(ABC):
    @abstractmethod
    def train(self,
              train_instances: List[VRPInstance],
              val_instances: List[VRPInstance],
              opposite_procedure: Union[DestroyProcedure, RepairProcedure],
              path: str,
              batch_size: int,
              epochs: int):
        pass

    @abstractmethod
    def load_weights(self, path: str):
        pass
