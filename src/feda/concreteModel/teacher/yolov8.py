from enum import Enum
from feda.abstractModel.teacherModel import TeacherModel
from ultralytics import YOLO
from overrides import override
from transformers.image_utils import ImageInput
import torch
from typing import Dict

class YoloV8ModelType(Enum):
    YOLOV8M = "yolov8m.pt"
    YOLOV8L = "yolov8l.pt"
    YOLOV8X = "yolov8x.pt"

class YoloV8(TeacherModel):
    def __init__(self, modelType: str) -> None:
        isVLM = False
        modelType = YoloV8ModelType[modelType].value
        name = modelType.split(".")[0]

        model = YOLO(modelType, "detect")

        super().__init__(model, name, isVLM)

    @override
    def _preprocess(self, *inputs: ImageInput) -> Dict[str, torch.Tensor]:
        images = [torch.tensor(input) for input in inputs]
        return {"input": torch.stack(images)}
    
    @override
    def _inference(self, processed_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        results = self.model(processed_inputs["input"])
        return results
