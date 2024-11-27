from torch.nn.modules import Module
from clide.abstractModel.model import Model
from clide.managers.hookManager import HookType
from abc import ABC
from typing import Optional, List

class TeacherModel(Model, ABC):
    def __init__(self, model: Module, name: str, isVLM: bool, hookLayers: Optional[List[str]] = None) -> None:
        super().__init__(model, name, hookLayers)
        self._isVLM = isVLM
        self._prompt = ""

    def bindMasksForImportance(self, masks, scale, probability):
        self._hookManager._bindMasks(masks, scale, probability)

    def unbindMasksForImportance(self):
        self._hookManager._unbindMasks()

    def registerHooksForImportance(self, hookList: List[str]):
        self.registerHooks(hookList, HookType.RandomPruning)

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