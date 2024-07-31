from torch.nn.modules import Module
from feda.abstractModels.teacherModel import TeacherModel
from enum import Enum
from typing import Dict
from transformers import AutoProcessor, AutoModelForCausalLM 
from transformers.image_utils import ImageInput
from overrides import override
import torch

class FlorenceModelType(Enum):
    FLORENCE_2_BASE = "microsoft/Florence-2-base"
    FLORENCE_2_LARGE = "microsoft/Florence-2-large"
    FLORENCE_2_BASE_FINETUNED = "microsoft/Florence-2-base-ft"
    FLORENCE_2_LARGE_FINETUNED = "microsoft/Florence-2-large-ft"


class Florence2(TeacherModel):
    def __init__(self, modelType: FlorenceModelType) -> None:
        isVLM = True
        name = modelType.value.split("/")[-1]

        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._processor = AutoProcessor.from_pretrained(modelType.value, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(modelType.value, torch_dtype=self.dtype, trust_remote_code=True)

        super().__init__(model, name, isVLM)

    @override
    def forward(self, inputs: ImageInput) -> Dict[str, torch.Tensor]:
        processed_inputs = self._processor(text=self.prompt, images=inputs, return_tensors="pt")
        print(type(processed_inputs))
        # TODO: won't work 
        return super().forward(processed_inputs)