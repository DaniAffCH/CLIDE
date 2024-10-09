from torch.nn.modules import Module
from clide.abstractModel.model import Model
from abc import ABC
import time
from typing import Optional, List

class StudentModel(Model, ABC):
    def __init__(self, model: Module, name: str, hookLayers: Optional[List[str]] = None) -> None:
        super().__init__(model, name, hookLayers)
        self.trainingStepExecuted()

    def trainingStepExecuted(self) -> None:
        self._updateTimestamp = int(round(time.time()))