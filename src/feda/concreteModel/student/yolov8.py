from enum import Enum
from feda.abstractModel.studentModel import StudentModel
from ultralytics import YOLO
from overrides import override
from transformers.image_utils import ImageInput
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
    def _preprocess(self, *inputs: ImageInput) -> Dict[str, torch.Tensor]:
        images = [torch.tensor(input) for input in inputs]
        return {"input": torch.stack(images)}
    
    @override
    def _inference(self, processed_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        results = self.model(processed_inputs["input"])
        return results
