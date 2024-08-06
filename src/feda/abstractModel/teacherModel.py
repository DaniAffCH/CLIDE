from torch.nn.modules import Module
from feda.abstractModel.model import Model
from abc import ABC

class TeacherModel(Model, ABC):
    def __init__(self, model: Module, name: str, isVLM: bool) -> None:
        super().__init__(model, name)
        self._isVLM = isVLM
        self._prompt = ""

    @property
    def prompt(self):
        return self._prompt
    
    @property
    def isVLM(self):
        return self._isVLM
    
    @prompt.setter
    def prompt(self, p: str):
        if not self._isVLM:
            raise AssertionError("It's possible to set a prompt only for VLM.")
        
        self._prompt = p