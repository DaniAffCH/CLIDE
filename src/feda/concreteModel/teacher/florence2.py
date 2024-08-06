from torch.nn.modules import Module
from feda.abstractModel.teacherModel import TeacherModel
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
    def __init__(self, modelType: FlorenceModelType, maxNewTokens: int = 1024, numBeams = 3) -> None:
        isVLM = True
        name = modelType.value.split("/")[-1]

        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._processor = AutoProcessor.from_pretrained(modelType.value, trust_remote_code=True)
        self.maxNewTokens = maxNewTokens
        self.numBeams = numBeams

        model = AutoModelForCausalLM.from_pretrained(modelType.value, torch_dtype=self.dtype, trust_remote_code=True)

        super().__init__(model, name, isVLM)

    @override
    def _preprocess(self, *inputs: ImageInput) -> torch.Tensor:
        return self._processor(text=self.prompt, images=inputs, return_tensors="pt").to(self._device, self.dtype)
    
    def _inference(self, processed_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        generated_ids = self._model.generate(
            input_ids=processed_inputs["input_ids"],
            pixel_values=processed_inputs["pixel_values"],
            max_new_tokens=self.maxNewTokens,
            num_beams=self.numBeams,
            do_sample=False
        )

        return generated_ids