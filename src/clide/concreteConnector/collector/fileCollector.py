from clide.abstractConnector.remoteCollector import RemoteCollector
from clide.managers.dataManager import DataManager
from overrides import override
import numpy as np
from typing import Tuple, Generator
import os
import logging
from pathlib import Path
from PIL import Image
import cv2
import time

logger = logging.getLogger(__name__)

class FileCollector(RemoteCollector):
    def __init__(self, 
                 address: str,
                 deviceName: str,
                 dataManager: DataManager,
                 imagePath: str,
                 frame_res: Tuple[int, int]
                 ) -> None:
        super().__init__(address, deviceName, dataManager)
        self.imagesPath = imagePath
        self._pathIter = None
        self._frame_res = frame_res

    @override
    def connect(self) -> bool:
        if not self.isAlive():
            raise AssertionError(f"{self.imagesPath}: No such directory")
        self._pathIter = iter(sorted(Path(self.imagesPath).iterdir(), key=os.path.getmtime))
        
        logger.info(f"Fake connection to {self.imagesPath} established.")

        self._isConnected = True
        return self._isConnected
    
    def _isImage(self, path: Path):
        return path.suffix.lower() in ['.png', '.jpg', '.jpeg']
        
    @override
    def _pollData(self) -> Generator[np.ndarray, None, None]:
        for imgPath in self._pathIter:
            if self._isImage(imgPath):
                img = np.array(Image.open(imgPath))
                img = cv2.resize(img, self._frame_res)
                yield img

        logger.info("Streaming images are over")

    @override
    def isAlive(self) -> bool:
        return os.path.exists(self.imagesPath) and os.path.isdir(self.imagesPath)
    
    @override
    def disconnect(self) -> bool:
        self._isConnected = False
        return self._isConnected
