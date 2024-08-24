from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch
from torch import nn
from feda.managers.hookManager import HookManager
from transformers.image_utils import ImageInput
from feda.tasks.tasks import TaskType, KeyMapping, TaskResult

class Model(ABC):

    def __init__(self, model: nn.Module, name: str) -> None:
        super().__init__()
        self.name = name
        self._model = model
        # Size expressed in bytes
        self._size = self._computeModelSize()
        self._hookManager = HookManager(model)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._taskType = TaskType.DETECTION # TODO: this should be related to a config
        

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
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def taskType(self) -> TaskType:
        return self._taskType

    @model.setter
    def model(self, model: nn.Module) -> None:
        self._model = model

    def _forwardHooks(self) -> Dict[str, Any]:
        return self._hookManager.getActivationOutputs()
    
    def registerHooks(self, hookList: List[str]) -> None:
        self._hookManager.registerHooks(hookList)

    @abstractmethod
    def _getKeyMapping(self) -> KeyMapping:
        pass

    @abstractmethod
    def _preprocess(self, inputs: ImageInput) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _postprocess(self, outputs: torch.Tensor) -> TaskResult:
        pass

    def _inference(self, processed_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._model(processed_inputs)

    def forward(self, inputs: ImageInput) -> Dict[str, Any]:
        processed_inputs = self._preprocess(inputs)
        output = self._inference(processed_inputs)
        output_hooks = self._forwardHooks()
        output_hooks["output"] = self._postprocess(output)
        return output_hooks
