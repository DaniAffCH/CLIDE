from typing import Any, Tuple, List
from ultralytics.engine.results import Results
import torch
from dataclasses import dataclass
from clide.adapters.preprocess import preprocessImage

@dataclass
class MultiscaleBatchesResult:
    maskedImages: torch.Tensor
    masks: torch.Tensor

# Functor class 
class ImportanceDistiller:
    def __init__(self, gridSizes: List[Tuple[int, int]], probThresholds: List[float], numMasks: int, batchSize: int, normalization: str = "standard") -> None:
        self.gridSizes = gridSizes
        self.probThresholds = probThresholds
        self.numMasks = numMasks
        self.batchSize = batchSize
        self.onlineScaleEval = {gs: 0 for gs in gridSizes}
        self.onlineScaleElem = 0
        self.normalization = normalization

        assert numMasks % batchSize == 0
        assert (self.numMasks // self.batchSize) % len(self.gridSizes) == 0
        assert len(probThresholds) == len(self.gridSizes)

    def _box_iou(self, box1: torch.Tensor, box2: torch.Tensor, eps=1e-7):
        """
        Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py.

        Args:
            box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
            box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

        Returns:
            (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
        """
        # NOTE: Need .float() to get accurate iou values
        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
    
    def _compute_similarity_iou(self, ground_truth: Results, prediction: Results):
        gt_boxes = ground_truth.boxes.xyxy
        pred_boxes = prediction.boxes.xyxy

        gt_cls = ground_truth.boxes.cls
        pred_cls = prediction.boxes.cls

        pred_conf = prediction.boxes.conf

        n = len(gt_cls)

        iouMat = self._box_iou(gt_boxes, pred_boxes)

        if iouMat.shape[-1] == 0:
            return torch.tensor(0.0, device=gt_boxes.device)
        else:
            clsMatchMat = (gt_cls.unsqueeze(1) == pred_cls.unsqueeze(0)).float()
            confidenceMat = pred_conf.unsqueeze(0).repeat(n, 1)

            assert iouMat.shape == clsMatchMat.shape == confidenceMat.shape

            iouMat = iouMat * clsMatchMat * confidenceMat
            maxIouPerDet, _ = torch.max(iouMat, dim=-1) 
            return maxIouPerDet.mean()

    def _compute_batched_similarity_iou(self, ground_truth, predictions):
        return torch.stack([self._compute_similarity_iou(ground_truth, p) for p in predictions])
    
    @staticmethod
    def _generate_mask_batch(feature_size, grid_size, prob_thresh, batch_size):
        num_channels, image_h, image_w = feature_size  # CxHxW
        grid_w, grid_h = grid_size

        # Generate random mask for each channel in the grid size, including batch dimension
        # Shape: [batch_size, num_channels, grid_h, grid_w]
        mask = (torch.rand(batch_size, num_channels, grid_h, grid_w, device='cuda') < prob_thresh).float()

        # Resize each channel's mask independently to the upsampled size
        # Resulting shape: [batch_size, num_channels, up_h, up_w]
        mask = torch.nn.functional.interpolate(
            mask.view(batch_size * num_channels, 1, grid_h, grid_w),
            size=(image_h, image_w),
            mode='nearest'
        ).view(batch_size, num_channels, image_h, image_w)

        return mask
    
    def checkUnevenScales(self, threshold = 0.01):
        for scale, meanScore in self.onlineScaleEval.items():
            if meanScore < threshold:
                print(f"Scale {scale} produced a very little mean evaluation score {meanScore}. Consider increasing the probability.")    

    def __call__(self, model, image) -> Any:
        image = preprocessImage(image)
        model.removeHooks()
        originalOutput = model.model(image)[0]

        num_of_batches = self.numMasks // self.batchSize
        element_per_scale = num_of_batches // len(self.gridSizes)

        current_heatmap = None

        model.registerHooksForImportance(model.hookLayers)
    
        for i in range(0, self.numMasks, self.batchSize):
            idx = i // (element_per_scale * self.batchSize)

            # Create object that will be filled by the hook
            masks = torch.empty((self.batchSize, 1, 1, 1), device=model.device)

            model.bindMasksForImportance(masks, self.gridSizes[idx], self.probThresholds[idx])

            maskedOutputs = model.model([image]*self.batchSize)
            scores = self._compute_batched_similarity_iou(originalOutput, maskedOutputs)

            self.onlineScaleElem += 1
            self.onlineScaleEval[self.gridSizes[idx]] += (scores.mean(-1).cpu().item() - self.onlineScaleEval[self.gridSizes[idx]])/self.onlineScaleElem

            partial_heatmap = torch.sum(masks * scores.view(-1, 1, 1, 1), dim=0)

            if current_heatmap is None:
                current_heatmap = partial_heatmap
            else:
                current_heatmap = current_heatmap + partial_heatmap 

            model.unbindMasksForImportance() 
        
        model.removeHooks()

        match self.normalization:
            
            case "standard":
                current_heatmap = current_heatmap / current_heatmap.max()
            case "minmax":
                current_heatmap = (current_heatmap - current_heatmap.min()) / (current_heatmap.max() - current_heatmap.min()) 
            case _:
                raise AssertionError(f"{self.normalization} not supported")

        return current_heatmap

