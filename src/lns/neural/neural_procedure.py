from abc import ABC, abstractmethod
from typing import List

from instances import VRPInstance


class NeuralProcedure(ABC):
    @abstractmethod
    def train(self, instances: List[VRPInstance], opposite_procedure, val_split: float, batch_size: int, epochs: int):
        pass

    @abstractmethod
    def load_weights(self, path: str):
        pass
