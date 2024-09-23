from torch.utils.data import Dataset
import io
from PIL import Image
from feda.managers.dataManager import DataManager
import torch
from ultralytics.data.augment import LetterBox
import numpy as np


class StubDataset(Dataset):
    def __init__(self, dataManager: DataManager):
        '''
        Initialize the dataset by fetching the dataset IDs from the DataManager.
        '''
        self.dataManager = dataManager
        self.dataset = self.dataManager.getAllIds()  # Fetch dataset IDs
        self.lb = LetterBox() # TODO: pass parameters!

    def __len__(self):
        '''
        Returns the size of the dataset.
        '''
        return len(self.dataset)

    def __getitem__(self, idx: int):
        '''
        Fetch an image and its annotation based on the index.
        '''
        image_id = self.dataset[idx]
        image_data, annotation = self.dataManager.getImage(image_id)
        
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        adjusted_image = self.lb(image=np.array(image))
        image_tensor = torch.tensor(adjusted_image, dtype=torch.uint8).permute(2, 0, 1)
        # TODO: return some kind of annotation?

        return {
            "img":image_tensor
        }