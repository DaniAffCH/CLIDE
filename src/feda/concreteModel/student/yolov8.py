from enum import Enum
from feda.abstractModel.studentModel import StudentModel
from ultralytics import YOLO
from overrides import override
from transformers.image_utils import ImageInput
from feda.tasks.tasks import KeyMapping, TaskType, TaskResult, TaskFactory
import torch
from typing import Dict

class YoloV8ModelType(Enum):
    YOLOV8N = "yolov8n.pt"
    YOLOV8S = "yolov8s.pt"

class YoloV8(StudentModel):
    def __init__(self, modelType: str) -> None:
        modelType = YoloV8ModelType[modelType].value
        name = modelType.split(".")[0]

        model = YOLO(modelType, "detect")

        super().__init__(model, name)

    @override
    def _getKeyMapping(self) -> KeyMapping:
        mapping_dict = {
            TaskType.DETECTION: TaskFactory.create_key_mapping(TaskType.DETECTION, bounding_boxes="alo", classes="miao")
        }

        return mapping_dict[self.taskType]

    @override
    def _preprocess(self, inputs: ImageInput) -> Dict[str, torch.Tensor]:
        images = [torch.tensor(input) for input in inputs]
        return {"input": torch.stack(images)}
    
    @override
    def _postprocess(self, outputs: torch.Tensor) -> TaskResult:
        return TaskFactory.create_result(self.taskType, self._getKeyMapping(), outputs)
    
    @override
    def _inference(self, processed_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        results = self.model(processed_inputs["input"])
        return results
