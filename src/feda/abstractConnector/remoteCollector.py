from feda.abstractConnector.remoteConnector import RemoteConnector
from feda.managers.dataManager import DataManager
from abc import ABC, abstractmethod

class RemoteCollector(RemoteConnector, ABC):
    def __init__(self, address: str, deviceName: str, dataManager: DataManager) -> None:
        self._dataManager = dataManager
        super().__init__(address, deviceName)

    @abstractmethod
    def poll() -> None:
        pass
