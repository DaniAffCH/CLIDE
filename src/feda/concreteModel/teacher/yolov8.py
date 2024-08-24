from enum import Enum
from feda.abstractModel.teacherModel import TeacherModel
from feda.tasks.tasks import KeyMapping, TaskType, TaskResult, TaskFactory
from torchvision import transforms
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
    def _getKeyMapping(self) -> KeyMapping:
        mapping_dict = {
            TaskType.DETECTION: TaskFactory.create_key_mapping(TaskType.DETECTION, bounding_boxes="alo", classes="miao")
        }

        return mapping_dict[self.taskType]

    @override
    def _preprocess(self, inputs: ImageInput) -> Dict[str, torch.Tensor]:
        if inputs.ndim == 3:
            print("UNSQ")
            inputs = inputs.unsqueeze(0)

        inputs = inputs.permute(0, 3, 2, 1)

        # This should be fine just for SONY cameras as they resize the input tensor internally 
        transform = transforms.Resize(size=(640, 640))

        inputs = transform(inputs)

        if torch.dtype != torch.float32:
            inputs = inputs.to(dtype=torch.float32) / 255.

        return {"input": inputs}
    
    @override
    def _postprocess(self, outputs: torch.Tensor) -> TaskResult:
        print(outputs)
        return TaskFactory.create_result(self.taskType, self._getKeyMapping(), outputs)
    
    @override
    def _inference(self, processed_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        results = self.model(processed_inputs["input"])
        return results
