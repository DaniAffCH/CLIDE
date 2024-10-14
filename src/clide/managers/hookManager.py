import torch
from torch import nn
from typing import List, Dict
import collections

class HookManager:

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._activationOutputMap = {}
        self._layerHookNames = []
        self._ringbuff = None

    def activationHook(self, model, input, output) -> None:
            name = self._ringbuff[0]
            self._activationOutputMap[name] = output
            self._ringbuff.rotate(-1)

    def registerHooks(self, hookList: List[str]) -> None:
        if self._layerHookNames:
            raise AssertionError("Hooks have already been registered. The 'registerHooks' method cannot be called multiple times.")

        available_layers = dict(self._model.named_modules())
        invalid_hooks = [hook for hook in hookList if hook not in available_layers]
        if invalid_hooks:
            raise ValueError(f"Invalid hook names detected: {', '.join(invalid_hooks)}. All hooks must be valid layer names in the model.")
        
        self._layerHookNames = hookList
        self._ringbuff = collections.deque(hookList)

        for hook in hookList:
            layer = available_layers[hook]
            layer.register_forward_hook(self.activationHook) #type: ignore

    def getActivationOutput(self, layer_name: str) -> torch.Tensor:
        if layer_name not in self._activationOutputMap:
            raise KeyError(f"Layer name '{layer_name}' not found in activation output map.")
        
        return self._activationOutputMap[layer_name]
    
    def getActivationOutputs(self) -> Dict[str, torch.Tensor]:        
        return self._activationOutputMap