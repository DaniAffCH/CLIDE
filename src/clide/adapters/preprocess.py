import torch
import numpy as np
from ultralytics.data.augment import LetterBox

def preprocessImage(image):
    lb = LetterBox()
    if isinstance(image, torch.Tensor):
        adjusted_image = lb(image=image.numpy())
        return torch.tensor(adjusted_image, dtype=torch.uint8).permute(2, 0, 1)

    elif isinstance(image, np.ndarray):
        adjusted_image = lb(image=image)
        return adjusted_image
        
