import torch
from torch import nn
from typing import List, Dict
from collections import OrderedDict

class HookManager:
    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._activationOutputMap: OrderedDict = OrderedDict()
        self._layerHookNames: List[str] = []

    def activationHook(self, layer_name: str, module, input, output) -> None:
        """The hook function to store activations for a specific layer."""
        self._activationOutputMap[layer_name] = output

    def registerHooks(self, hookList: List[str]) -> None:
        """Registers forward hooks for the specified layers."""
        if self._layerHookNames:
            raise AssertionError("Hooks have already been registered. The 'registerHooks' method cannot be called multiple times.")
        
        # Validate hooks
        available_layers = dict(self._model.named_modules())
        invalid_hooks = [hook for hook in hookList if hook not in available_layers]
        if invalid_hooks:
            raise ValueError(f"Invalid hook names detected: {', '.join(invalid_hooks)}. All hooks must be valid layer names in the model.")

        # Update internal state
        self._layerHookNames = hookList
        
        # Register hooks
        for layer_name in hookList:
            layer = available_layers[layer_name]
            layer.register_forward_hook(self._create_hook(layer_name))

    def _create_hook(self, layer_name: str):
        """Creates and registers a hook function by passing layer name as an argument."""
        return HookWrapper(self, layer_name)  # Passes manager and layer name to the wrapper

    def getActivationOutput(self, layer_name: str) -> torch.Tensor:
        """Retrieves the activation output for a specific layer."""
        if layer_name not in self._activationOutputMap:
            raise KeyError(f"Layer name '{layer_name}' not found in activation output map.")
        return self._activationOutputMap[layer_name]

    def getActivationOutputs(self) -> OrderedDict:
        """Returns all activation outputs."""
        return self._activationOutputMap


class HookWrapper:
    def __init__(self, manager: HookManager, layer_name: str):
        """Initializes the hook wrapper, which is serializable by pickle."""
        self.manager = manager
        self.layer_name = layer_name

    def __call__(self, module, input, output):
        """Handles hook calls by referencing the manager and storing the output."""
        self.manager.activationHook(self.layer_name, module, input, output)
