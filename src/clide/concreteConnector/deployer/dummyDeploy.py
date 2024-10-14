from clide.abstractConnector.remoteDeployer import RemoteDeployer
from clide.abstractModel.studentModel import StudentModel
import logging
from overrides import override

logger = logging.getLogger(__name__)

class DummyDeploy(RemoteDeployer):
    def __init__(self, address: str, deviceName: str, studentModel: StudentModel) -> None:
        super().__init__(address, deviceName, studentModel)

    @override
    def deploy(self, workingPath: str, modelPath: str) -> None:  
        logger.info(f"Dummy deployment!")

    @override
    def isAlive(self) -> bool:
        return True
    
    @override
    def connect(self) -> bool:
        self._isConnected = True
        return self._isConnected
    
    @override
    def disconnect(self) -> bool:
        self._isConnected = False
        return self._isConnected