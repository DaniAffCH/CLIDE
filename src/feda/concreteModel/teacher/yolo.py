from enum import Enum
from feda.abstractModel.teacherModel import TeacherModel
from feda.adapters.tasks import KeyMapping, TaskType, TaskResult, TaskFactory
from feda.adapters.classes import ClassAdapter
from torchvision import transforms
from ultralytics import YOLO
from overrides import override
from transformers.image_utils import ImageInput
import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)
class YoloModelType(Enum):
    YOLOV8M = "yolov8m.pt"
    YOLOV8L = "yolov8l.pt"
    YOLOV8X = "yolov8x.pt"
    YOLOV10M = "yolov10m.pt"
    YOLOV10L = "yolov10l.pt"
    YOLOV10X = "yolov10x.pt"


class Yolo(TeacherModel):
    def __init__(self, modelType: str) -> None:
        isVLM = False
        modelType = YoloModelType[modelType].value
        name = modelType.split(".")[0]

        model = YOLO(modelType, "detect")
        model.model.args["verbose"] = False

        super().__init__(model, name, isVLM)

    @override
    def _getKeyMapping(self) -> KeyMapping:
        mapping_dict = {
            TaskType.DETECTION: TaskFactory.create_key_mapping(TaskType.DETECTION, bounding_boxes="bbox", classes="cls")
        }

        return mapping_dict[self.taskType]

    @override
    def _preprocess(self, inputs: ImageInput) -> Dict[str, torch.Tensor]:
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)

        inputs = inputs.permute(0, 3, 1, 2)

        transform = transforms.Resize(size=(640, 640))

        inputs = transform(inputs)

        if inputs.dtype != torch.float32 or inputs.max() > 1.:
            inputs = inputs.to(dtype=torch.float32) / 255.

        return {"input": inputs}
    
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

        # TODO: It doesn't work for batched elements for now :)
        o = outputs[0]
        adapted_cls, adaptation_mask = ClassAdapter.adaptClassesToId([o.names[cl] 
                                                for cl in o.boxes.cls.tolist()], mapping) 
        
        adapted_bbox = [e for e, m in zip(o.boxes.xywhn.tolist(), adaptation_mask) if m]

        dict_out = {"cls": adapted_cls, "bbox": adapted_bbox}

        return TaskFactory.create_result(self.taskType, self._getKeyMapping(), dict_out)
    
    @override
    def _inference(self, processed_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        results = self.model(processed_inputs["input"], verbose=False)
        return results
