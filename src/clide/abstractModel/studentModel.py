from torch.nn.modules import Module
from clide.abstractModel.model import Model
from abc import ABC
import time

class StudentModel(Model, ABC):
    def __init__(self, model: Module, name: str) -> None:
        super().__init__(model, name)
        self.trainingStepExecuted()

    def trainingStepExecuted(self) -> None:
        self._updateTimestamp = int(round(time.time()))