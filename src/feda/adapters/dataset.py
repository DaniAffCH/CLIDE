from torch.utils.data import Dataset
import io
from PIL import Image
from feda.managers.dataManager import DataManager
from torchvision import transforms

class StubDataset(Dataset):
    def __init__(self, dataManager: DataManager):
        '''
        Initialize the dataset by fetching the dataset IDs from the DataManager.
        '''
        self.dataManager = dataManager
        self.dataset = self.dataManager.getAllIds()  # Fetch dataset IDs

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
        image_tensor = transforms.ToTensor()(image)  
        
        return image_tensor, annotation