from io import UnsupportedOperation
import torch
from torch.utils.hooks import RemovableHandle
from torch import nn
from typing import List, Dict, Tuple
from collections import OrderedDict
from enum import Enum
from clide.distiller.importanceDistiller import ImportanceDistiller
from copy import copy

class HookType(Enum):
    Standard = 0
    RandomPruning = 1

class HookManager:
    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._activationOutputMap: OrderedDict = OrderedDict()
        self._layerHookNames: List[str] = []
        self._handles: Dict[str, RemovableHandle] = {}

    def activationHook(self, layer_name: str, module, input, output) -> None:
        """The hook function to store activations for a specific layer."""
        self._activationOutputMap[layer_name] = output

    def _bindMasks(self, masks: torch.Tensor, scale: Tuple[int, int], probability: float):
        self.masks = masks
        self.scale = scale 
        self.probability = probability

    def randomPruningHook(self, module, input, output):
        assert self.masks is not None
        self.masks.resize_(output.shape)

        generatedMasks = ImportanceDistiller._generate_mask_batch(output.shape[1:], self.scale, self.probability, output.shape[0])

        self.masks.copy_(generatedMasks)
        maskedOutput = output * generatedMasks

        return maskedOutput

    def _unbindMasks(self):
        self.masks = None
        self.scale = None
        self.probability = None

    def registerHooks(self, hookList: List[str], type: HookType) -> None:
        """Registers forward hooks for the specified layers."""
        if self._layerHookNames:
            raise AssertionError("Hooks have already been registered. The 'registerHooks' method cannot be called multiple times without removing the previous hooks first.")
        
        # Validate hooks
        available_layers = dict(self._model.named_modules())
        invalid_hooks = [hook for hook in hookList if hook not in available_layers]
        if invalid_hooks:
            raise ValueError(f"Invalid hook names detected: {', '.join(invalid_hooks)}. All hooks must be valid layer names in the model.")

        # Update internal state
        self._layerHookNames = copy(hookList)
        # Register hooks
        for layer_name in hookList:
            layer = available_layers[layer_name]
            self._handles[layer_name] = layer.register_forward_hook(self._create_hook(type, layer_name))
    
    def removeHooks(self):        
        for handle in self._handles.values():
            handle.remove()

        self._handles.clear()
        self._layerHookNames.clear()
        self._activationOutputMap.clear()
        self._unbindMasks()

    def _create_hook(self, type: HookType, layer_name: str):
        """Creates and registers a hook function by passing layer name as an argument."""
        return HookWrapper(self, type, layer_name) 

    def getActivationOutput(self, layer_name: str) -> torch.Tensor:
        """Retrieves the activation output for a specific layer."""
        if layer_name not in self._activationOutputMap:
            raise KeyError(f"Layer name '{layer_name}' not found in activation output map.")
        return self._activationOutputMap[layer_name]

    def getActivationOutputs(self) -> OrderedDict:
        """Returns all activation outputs."""
        return self._activationOutputMap


class HookWrapper:
    def __init__(self, manager: HookManager, type: HookType, layer_name):
        """Initializes the hook wrapper, which is serializable by pickle."""
        self.manager = manager
        self.layer_name = layer_name
        self.type = type

    def __call__(self, module, input, output):
        """Handles hook calls by referencing the manager and storing the output."""
        if self.type == HookType.Standard:
            return self.manager.activationHook(self.layer_name, module, input, output)
        elif self.type == HookType.RandomPruning:
            return self.manager.randomPruningHook(module, input, output)
        else:
            raise UnsupportedOperation(f"Unsupported type {type.__name__}")
    
