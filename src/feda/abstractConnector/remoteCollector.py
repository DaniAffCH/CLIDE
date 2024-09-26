from feda.abstractConnector.remoteConnector import RemoteConnector
from feda.managers.dataManager import DataManager
from abc import ABC, abstractmethod
from typing import Generator
from overrides import override
import logging
import numpy as np

logger = logging.getLogger(__name__)

class RemoteCollector(RemoteConnector, ABC):
    def __init__(self, address: str, deviceName: str, dataManager: DataManager) -> None:
        self._dataManager = dataManager
        super().__init__(address, deviceName)

    @abstractmethod
    def _pollData() -> Generator[np.ndarray, None, None]:
        pass

    def _writeData(self, image: np.ndarray):
        self._dataManager.addImage(image)

    def _stopPolling(self) -> bool:
        return self._dataManager.stopCollecting()

    def poll(self) -> None:
        logger.info(f"Starting image streaming.")
        if not self._isConnected:
            logger.critical("Attempted to poll data without a device connection.")
            raise AssertionError("Device is not connected.")
        
        self._dataManager.startCollectionSession()
        for image in self._pollData():
            if self._stopPolling():
                return
            self._writeData(image)
