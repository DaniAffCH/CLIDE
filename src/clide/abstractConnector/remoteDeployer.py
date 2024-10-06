from clide.abstractConnector.remoteConnector import RemoteConnector
from clide.abstractModel.studentModel import StudentModel
from abc import ABC, abstractmethod

class RemoteDeployer(RemoteConnector, ABC):
    def __init__(self, address: str, deviceName: str, studentModel: StudentModel) -> None:
        self._studentModel = studentModel
        super().__init__(address, deviceName)

    @abstractmethod
    def deploy(self, workingPath: str, modelPath: str) -> None:
        pass
