from torch.utils.data import Dataset
from typing import List
import io
from PIL import Image
from feda.managers.dataManager import DataManager
import torch
from ultralytics.data.augment import LetterBox
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

class StubDataset(Dataset):
    def __init__(self, dataManager: DataManager, data: List[str]):
        self.dataManager = dataManager
        self.data = data
        self.lb = LetterBox() # TODO set params

    def __len__(self):
        '''
        Returns the size of the dataset.
        '''
        return len(self.data)

    def __getitem__(self, idx: int):
        '''
        Fetch an image and its annotation based on the index.
        '''
        image_id = self.data[idx]
        image_data, annotation = self.dataManager.getImage(image_id)
        
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        adjusted_image = self.lb(image=np.array(image))
        image_tensor = torch.tensor(adjusted_image, dtype=torch.uint8).permute(2, 0, 1)
        # TODO: return some kind of annotation?
        return {
            "img":image_tensor,
            "ori_shape":torch.tensor([640,640]),
            "ratio_pad":torch.tensor([[1,1],[0,0]]),
            "img_id":image_id
        }

class StubDatasetFactory:
    def __init__(self, dataManager: DataManager, splitRatio: dict) -> None:
        assert len(splitRatio) <= 3, "Expected at most 3 keys in splitRatio for train, val, and optional test splits"
        assert "train" in splitRatio and "val" in splitRatio, "Missing 'train' or 'val' key in splitRatio"
        assert sum(splitRatio.values()) > 0.99, "The sum of split ratios must be approximately 1"
        if len(splitRatio) == 3:
            assert "test" in splitRatio, "Missing 'test' key in splitRatio when specifying 3 splits"
        self.dataManager = dataManager
        self.splitRatio = splitRatio

        self.updateData()
        

    def updateData(self):
        self.dataset = self.dataManager.getAllIds()  # Fetch dataset IDs
        
        random.shuffle(self.dataset)

        total_size = len(self.dataset)
        train_size = int(total_size * self.splitRatio['train'])
        val_size = int(total_size * self.splitRatio['val'])

        self.splits = {}
        self.splits['train'] = self.dataset[:train_size]
        self.splits['val'] = self.dataset[train_size:train_size + val_size]
        if 'test' in self.splitRatio:
            self.splits['test'] = self.dataset[train_size + val_size:]

        logger.info("Dataset updated")

        
    def __call__(self, split: str) -> StubDataset:
        assert split in self.splitRatio, "Invalid mode, must be one of: {}".format(list(self.splitRatio.keys()))
        return StubDataset(self.dataManager, self.splits[split])