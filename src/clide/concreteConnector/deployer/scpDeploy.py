from clide.abstractConnector.remoteDeployer import RemoteDeployer
from clide.concreteConnector.deployer.packager.sdsp import SDSPPackager
from clide.abstractModel.studentModel import StudentModel
import subprocess
import paramiko
from overrides import override
import logging
from scp import SCPClient

logger = logging.getLogger(__name__)

class SCPDeploy(RemoteDeployer):
    def __init__(self, packager: SDSPPackager,address: str, port: str, serial:str, deviceName: str, user: str, password: str, remoteModelPath: str, studentModel: StudentModel) -> None:
        super().__init__(address, deviceName, studentModel)
        self._port = port
        self._serial = serial
        self._user = user
        self._pass = password
        self._sshClient = None
        self._scpClient = None
        self._packager = packager
        self._remoteModelPath = remoteModelPath
        
    def _createSSHClient(server, port, user, password) -> paramiko.SSHClient:
        """ Create a SSH client to connect to a device 
        
        Parameters
        ----------
        - `server`: IP address of the device
        - `port`: ssh port to use for the connection 
        - `user`: device user that is going to connect
        - `password`: root password of the chosen user 

        Return
        ------
        `client`: the ready ssh client
        """
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(server, port, user, password)
        return client

    @override
    def connect(self) -> bool:
        if self.isAlive():
            logger.info(f"Device {self.address} is reachable")
        else:
            raise AssertionError(f"Device {self.address} is not reachable")

        logger.info(f"Establishing SCP connection to serial {self._serial} | {self._user}@{self.address}:{self._port}")
        self.sshClient = self._createSSHClient(self._port, self._user, self._pass)
        self._scpClient = SCPClient(self.sshClient.get_transport())
        logger.info(f"SCP connection established")
        self._isConnected = True
        return self._isConnected

    @override
    def deploy(self, workingPath: str, modelPath: str) -> None:  
        packagedModelPath = self._packager(workingPath, modelPath, self._serial)
        self._scpClient.put(packagedModelPath, self._remoteModelPath)
        logger.info(f"New model deployed successfully!")

    @override
    def isAlive(self) -> bool:
        try:
            subprocess.run(['ping', "-c", '1', self.address], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1)
            return True
        except subprocess.CalledProcessError:
            return False
        except subprocess.TimeoutExpired:
            return False
        
    def disconnect(self) -> bool:
        if self._scpClient: 
            self._scpClient.close()
            self._scpClient = None
        else:
            logger.warning("No SCP connection to close for SCP Deployer.")
    
        if self._sshClient:
            self._sshClient.close()
            self._sshClient = None
        else:
            logger.warning("No SSH connection to close for SCP Deployer.")
            

        self._isConnected = False
        return self._isConnected
    