from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Type
from abc import ABC, abstractmethod

class TaskType(Enum):
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"

@dataclass
class KeyMapping(ABC):
    @abstractmethod
    def map_keys(self, values: Dict) -> Dict:
        pass

@dataclass
class DetectionKeyMapping(KeyMapping):
    bounding_boxes: str
    classes: str

    def map_keys(self, values: Dict) -> Dict:
        return {
            "bounding_boxes": values[self.bounding_boxes],
            "classes": values[self.classes],
        }

@dataclass
class TaskResult(ABC):
    pass

@dataclass
class DetectionResult(TaskResult):
    bounding_boxes: List[Tuple[int, int, int, int]]
    classes: List[str]

    def __post_init__(self):
        if len(self.bounding_boxes) != len(self.classes):
            raise ValueError("The number of bounding boxes must match the number of classes.")

class TaskFactory:
    _result_registry: Dict[TaskType, Type[TaskResult]] = {
        TaskType.DETECTION: DetectionResult
    }

    _mapping_registry: Dict[TaskType, Type[KeyMapping]] = {
        TaskType.DETECTION: DetectionKeyMapping
    }

    @staticmethod
    def create_result(task_type: TaskType, key_mapping: KeyMapping, values: Dict) -> TaskResult:
        if task_type not in TaskFactory._result_registry:
            raise ValueError(f"Unsupported task type: {task_type}")

        mapped_values = key_mapping.map_keys(values)
        result_class = TaskFactory._result_registry[task_type]
        return result_class(**mapped_values)

    @staticmethod
    def create_key_mapping(task_type: TaskType, *args, **kwargs) -> KeyMapping:
        if task_type not in TaskFactory._mapping_registry:
            raise ValueError(f"Unsupported task type: {task_type}")

        mapping_class = TaskFactory._mapping_registry[task_type]
        return mapping_class(*args, **kwargs)
