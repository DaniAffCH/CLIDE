from feda.adapters.yolov8MCT import yolov8_pytorch, PostProcessWrapper
from model_compression_toolkit.core import BitWidthConfig
from model_compression_toolkit.core.common.network_editors import NodeNameScopeFilter
from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from feda.abstractModel.studentModel import StudentModel
from typing import Iterator, List 
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import model_compression_toolkit as mct
import os 

class MCTQuantizer():
    def __init__(self, batchSize: int):
        self.batchSize = batchSize
        self.score_threshold = 0.001
        self.iou_threshold = 0.7
        self.max_detections = 300 
        self._onnxFileName = "qmodel.onnx"

    @property
    def modelFileName(self): 
        return self._onnxFileName

    def ultralytics2MCT(self, model: StudentModel) -> nn.Module:
        mct_model, _ = yolov8_pytorch("yolov8n.yaml")
        modelSD = model.model.state_dict()
        mct_model.load_state_dict(modelSD, strict=False)
        mct_model.eval()
        return mct_model

    def get_representative_dataset(self, n_iter: int, dataLoader: DataLoader):
        """
        This function creates a representative dataset generator. The generator yields numpy
            arrays of batches of shape: [Batch, C, H, W].
        Args:
            n_iter: number of iterations for MCT to calibrate on
        Returns:
            A representative dataset generator
        """       
        def representative_dataset() -> Iterator[List]:
            ds_iter = iter(dataLoader)
            for _ in range(n_iter):
                yield [next(ds_iter)["img"] / 255]

        return representative_dataset

    def quantize(self, model: StudentModel, outputPath: str, dataset: Dataset) -> str:
        numIter = len(dataset) // self.batchSize
        assert numIter > 0, f"Dataset contains too few samples for quantization ({len(dataset)} samples)"
        dataLoader = DataLoader(dataset, self.batchSize)


        mctModel = self.ultralytics2MCT(model)
        representative_dataset = self.get_representative_dataset(numIter, dataLoader)

        tpc = mct.get_target_platform_capabilities(fw_name="pytorch",
                                                target_platform_name='imx500',
                                                target_platform_version='v3')
        
        # TODO: hardcoded for now, find a customizable way
        manual_bit_cfg = BitWidthConfig()
        manual_bit_cfg.set_manual_activation_bit_width(
            [NodeNameScopeFilter('mul'),
            NodeNameScopeFilter('sub'),
            NodeNameScopeFilter('sub_1'),
            NodeNameScopeFilter('add_6'),
            NodeNameScopeFilter('add_7'),
            NodeNameScopeFilter('stack')], 16)
        
        config = mct.core.CoreConfig(mixed_precision_config=mct.core.MixedPrecisionQuantizationConfig(num_of_images=10),
                                     quantization_config=mct.core.QuantizationConfig(concat_threshold_update=True),
                                     bit_width_config=manual_bit_cfg)

        # Define target Resource Utilization for mixed precision weights quantization (76% of 'standard' 8bits quantization).
        # We measure the number of parameters to be 3146176 and calculate the target memory (in Bytes).
        resource_utilization = mct.core.ResourceUtilization(weights_memory=3146176 * 0.76)

        quant_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=mctModel,
                                                            representative_data_gen=representative_dataset,
                                                            target_resource_utilization=resource_utilization,
                                                            core_config=config,
                                                            target_platform_capabilities=tpc)
        device = get_working_device()

        quant_model_pp = PostProcessWrapper(model=quant_model,
                                            score_threshold=self.score_threshold,
                                            iou_threshold=self.iou_threshold,
                                            max_detections=self.max_detections).to(device=device)

        modelPath = os.path.join(outputPath, self.modelFileName)

        mct.exporter.pytorch_export_model(model=quant_model_pp,
                                  save_model_path=modelPath,
                                  repr_dataset=representative_dataset)
        
        return modelPath