from feda.abstractConnector.remoteConnector import RemoteConnector
from feda.abstractModel.studentModel import StudentModel
from abc import ABC, abstractmethod

class RemoteDeployer(RemoteConnector, ABC):
    def __init__(self, address: str, deviceName: str, studentModel: StudentModel) -> None:
        self._studentModel = studentModel
        super().__init__(address, deviceName)

    @abstractmethod
    def deploy() -> None:
        pass
