from feda.abstractModel.teacherModel import TeacherModel
from enum import Enum
from typing import Dict, Any
from transformers import AutoProcessor, AutoModelForCausalLM 
from feda.adapters.tasks import KeyMapping, TaskType, TaskResult, TaskFactory
from transformers.image_utils import ImageInput
from feda.adapters.classes import ClassAdapter
from overrides import override
import logging
import torch

logger = logging.getLogger(__name__)
class FlorenceModelType(Enum):
    FLORENCE_2_BASE = "microsoft/Florence-2-base"
    FLORENCE_2_LARGE = "microsoft/Florence-2-large"
    FLORENCE_2_BASE_FINETUNED = "microsoft/Florence-2-base-ft"
    FLORENCE_2_LARGE_FINETUNED = "microsoft/Florence-2-large-ft"


class Florence2(TeacherModel):
    def __init__(self, modelType: str, maxNewTokens: int = 1024, numBeams = 3) -> None:
        isVLM = True
        modelType = FlorenceModelType[modelType].value
        name = modelType.split("/")[-1]

        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._processor = AutoProcessor.from_pretrained(modelType, trust_remote_code=True)
        self.maxNewTokens = maxNewTokens
        self.numBeams = numBeams

        model = AutoModelForCausalLM.from_pretrained(modelType, torch_dtype=self.dtype, trust_remote_code=True)
        super().__init__(model, name, isVLM)

        self._prompt = "<OD>" # TODO: this should be dependend on the type of task

    @override
    def _getKeyMapping(self) -> KeyMapping:
        mapping_dict = {
            TaskType.DETECTION: TaskFactory.create_key_mapping(TaskType.DETECTION, bounding_boxes="bboxes", classes="labels")
        }

        return mapping_dict[self.taskType]
    
    @override
    def _preprocess(self, inputs: ImageInput) -> Dict[str, torch.Tensor]:
        return self._processor(text=self.prompt, images=inputs, return_tensors="pt").to(self._device, self.dtype)
    
    @override
    def _getClassNameMapping(self) -> Dict[str, str]:
        return {
            "car": "car",
            "bicycle": "bike",
            "person": "person"
        }
    
    @override
    def _postprocess(self, outputs: Any) -> TaskResult:
        # TODO: make it batchable, currently just 1 elem
        generatedText = self._processor.batch_decode(outputs, skip_special_tokens=False)[0]

        # normalized output
        postProcessed = self._processor.post_process_generation(generatedText, task=self.prompt, image_size=(1,1))
        postProcessed = postProcessed[self.prompt]

        mapping = self._getClassNameMapping()

        # TODO: this won't work for batched inputs I guess
        postProcessed["labels"], adaptation_mask = ClassAdapter.adaptClassesToId(postProcessed["labels"], mapping)
        postProcessed["bboxes"] = [e for e, m in zip(postProcessed["bboxes"], adaptation_mask) if m] 

        return TaskFactory.create_result(self.taskType, self._getKeyMapping(), postProcessed)
    
    def _inference(self, processed_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        generated_ids = self._model.generate(
            input_ids=processed_inputs["input_ids"],
            pixel_values=processed_inputs["pixel_values"],
            max_new_tokens=self.maxNewTokens,
            num_beams=self.numBeams,
            do_sample=False
        )

        return generated_ids