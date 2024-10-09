from enum import Enum
from clide.abstractModel.studentModel import StudentModel
from ultralytics import YOLO
from overrides import override
from transformers.image_utils import ImageInput
from clide.adapters.tasks import KeyMapping, TaskType, TaskResult, TaskFactory
from clide.adapters.classes import ClassAdapter
import torch
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class YoloModelType(Enum):
    YOLOV8N = "yolov8n.pt"
    YOLOV8S = "yolov8s.pt"

class Yolo(StudentModel):
    def __init__(self, modelType: str, hookLayers: Optional[List[str]] = None) -> None:
        modelType = YoloModelType[modelType].value
        name = modelType.split(".")[0]
        
        model = YOLO(modelType, "detect")

        super().__init__(model, name, hookLayers)

    @override
    def _getKeyMapping(self) -> KeyMapping:
        mapping_dict = {
            TaskType.DETECTION: TaskFactory.create_key_mapping(TaskType.DETECTION, bounding_boxes="alo", classes="miao")
        }

        return mapping_dict[self.taskType]

    @override
    def _getClassNameMapping(self) -> Dict[str, str]:
        return {
            "person": "person",
            "car": "car",
            "bicycle": "bike"
        }
    
    @override
    def _postprocess(self, outputs: Any) -> TaskResult:
        mapping = self._getClassNameMapping()

        dict_out = {
            "cls": [
                    ClassAdapter.adaptClassesToId([o.names[cl] 
                                                   for cl in o.boxes.cls.tolist()], mapping) 
                    for o in outputs
                    ],
            "bbox": [o.boxes.xywhn.tolist() for o in outputs]
        }

        logger.info(dict_out["cls"])

        return TaskFactory.create_result(self.taskType, self._getKeyMapping(), dict_out)
    
    @override
    def _preprocess(self, inputs: ImageInput) -> Dict[str, torch.Tensor]:
        images = [torch.tensor(input) for input in inputs]
        return {"input": torch.stack(images)}
    
    @override
    def _inference(self, processed_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        results = self.model(processed_inputs["input"])
        return results
