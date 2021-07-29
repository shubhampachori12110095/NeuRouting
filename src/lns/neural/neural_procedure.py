from abc import ABC, abstractmethod
from typing import List, Union, Optional

from instances import VRPInstance
from lns import DestroyProcedure, RepairProcedure
from utils.logging import Logger


class NeuralProcedure(ABC):
    @abstractmethod
    def train(self,
              opposite_procedure: Union[DestroyProcedure, RepairProcedure],
              train_instances: List[VRPInstance],
              batch_size: int,
              n_epochs: int,
              val_instances: List[VRPInstance],
              val_steps: int,
              val_interval: int,
              checkpoint_path: str,
              logger: Optional[Logger],
              log_interval: Optional[int]):
        pass

    @abstractmethod
    def load_weights(self, checkpoint_path: str):
        pass
