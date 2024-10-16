from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional    
import torch
from torch import nn
from clide.managers.hookManager import HookManager
from transformers.image_utils import ImageInput
from clide.adapters.tasks import TaskType, KeyMapping, TaskResult
from collections import OrderedDict

class Model(ABC):

    def __init__(self, model: nn.Module, name: str, hookLayers: Optional[List[str]] = None) -> None:
        super().__init__()
        self.name = name
        self._model = model
        # Size expressed in bytes
        self._size = self._computeModelSize()
        self._hookManager = HookManager(model)
        self._hookLayers = hookLayers
        assert "output" not in hookLayers
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

    def getOutputChannels(self, layer_name: str) -> int:
        layer = dict(self._model.named_modules()).get(layer_name, None)

        if layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")

        if list(layer.children()):
            first_valid_layer = None

            # Iterate through the submodules of the specified layer
            for _, submodule in layer.named_modules():
                if isinstance(submodule, (nn.Conv2d, nn.Linear)):
                    first_valid_layer = submodule
                    break
        else:
            first_valid_layer = layer if isinstance(layer, (nn.Conv2d, nn.Linear)) else None

        if first_valid_layer is None:
            raise TypeError(f"No supported layer found in the specified layer '{layer_name}'.")

        # Return output channels or features of the first valid layer
        if isinstance(first_valid_layer, nn.Conv2d):
            return first_valid_layer.out_channels
        elif isinstance(first_valid_layer, nn.Linear):
            return first_valid_layer.out_features
        else:
            raise TypeError(f"Layer '{layer_name}' is not supported for output channel extraction.")

    @property
    def model(self) -> nn.Module:
        return self._model
    
    @property
    def hookLayers(self) -> List[str]:
        return self._hookLayers
    
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

    def getHooks(self, dropOutput=True) -> OrderedDict:
        out = self._hookManager.getActivationOutputs()
        if dropOutput:
            out.pop("output", None)
        return out
    
    def registerHooks(self, hookList: List[str]) -> None:
        self._hookManager.registerHooks(hookList)

    @abstractmethod
    def _getKeyMapping(self) -> KeyMapping:
        pass

    @abstractmethod
    def _getClassNameMapping(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def _preprocess(self, inputs: ImageInput) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _postprocess(self, outputs: Any) -> TaskResult:
        pass

    def _inference(self, processed_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._model(processed_inputs)

    def forward(self, inputs: ImageInput) -> Dict[str, Any]:
        processed_inputs = self._preprocess(inputs)
        output = self._inference(processed_inputs)
        output_hooks = self.getHooks()
        output_hooks["output"] = self._postprocess(output)
        return output_hooks
