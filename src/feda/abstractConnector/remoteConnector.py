from abc import ABC, abstractmethod

class RemoteConnector(ABC):
    def __init__(self, address: str, deviceName: str) -> None:
        self.address = address
        self.deviceName = deviceName
        self._isConnected = False

    @abstractmethod
    def isAlive(self) -> bool:
        pass

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        pass