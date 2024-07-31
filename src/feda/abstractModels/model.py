from abc import ABC
from typing import List, Dict
import torch
from torch import nn
from feda.managers.hookManager import HookManager

class Model(ABC):

    def __init__(self, model: nn.Module, name: str) -> None:
        super().__init__()
        self.name = name
        self._model = model
        # Size expressed in bytes
        self._size = self._computeModelSize()
        self._hookManager = HookManager(model)

    def _computeModelSize(self) -> int:
        size_model = 0
        for param in self._model.parameters():
            if param.data.is_floating_point():
                size_model += param.numel() * torch.finfo(param.data.dtype).bits
            else:
                size_model += param.numel() * torch.iinfo(param.data.dtype).bits
        return size_model // 8  # Convert bits to bytes

    @property
    def model(self) -> nn.Module:
        return self._model
    
    @property
    def size(self) -> int:
        return self._size

    @model.setter
    def model(self, model: nn.Module) -> None:
        self._model = model

    def _forwardHooks(self) -> Dict[str, torch.Tensor]:
        return self._hookManager.getActivationOutputs()
    
    def registerHooks(self, hookList: List[str]) -> None:
        self._hookManager.registerHooks(hookList)

    def forward(self, *inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = self._model(*inputs)
        output_hooks = self._forwardHooks()
        output_hooks["output"] = output
        return output_hooks
