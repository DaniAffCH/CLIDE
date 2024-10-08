import subprocess
import os
import base64
import logging
import uuid
import requests
import zipfile

logger = logging.getLogger(__name__)

class SDSPPackager:
    def __init__(self, sdspPath: str, inputPersistency: bool, authClient: str, authSecret: str, tenantId: str, packagerVersion: str):
        self._uniDirName = "packModel"
        self._convertedModelName = "convModel"
        self._uniMinVersion = "1.11.0"
        self._sdspMinVersion = "3.15.3"
        self._versionFlag = "--version"
        self._authClient = authClient
        self._authSecret = authSecret
        self._tenantId = tenantId
        self._packagerVersion = packagerVersion

        self._sdspPath = sdspPath
        self._inputPersistency = inputPersistency
    
    @property
    def uniDirName(self): 
        return self._uniDirName

    @property
    def convertedModelName(self): 
        return self._convertedModelName
    
    def _check_version(self, command: str, version_flag: str, min_version: str) -> bool:
        try:
            result = subprocess.run([command, version_flag], shell=False, capture_output=True, text=True)
            if result.returncode == 0:
                installed_version = result.stdout.strip().split()[-1]
                if installed_version >= min_version:
                    return True
                else:
                    logger.error(f"{command} version {installed_version} is lower than the required version {min_version}")
                    return False
            else:
                logger.error(result.stderr)
                return False
        except FileNotFoundError:
            logger.error(f"{command} is not installed.")
            return False


    def _produceUniModel(self, outputPath: str, modelPath: str):
        inputModel = os.path.join(outputPath, modelPath)
        outputModel = os.path.join(outputPath, self.uniDirName)

        cmdBase = "uni-pytorch" 

        if not self._check_version(cmdBase, self._versionFlag, self._uniMinVersion):
            raise EnvironmentError(f"{cmdBase} is either not installed or the version is lower than {self._uniMinVersion}")

        command = [cmdBase, "-i", inputModel, "-o", outputModel]

        logger.info("Start converting in Uni model")
        result = subprocess.run(command, shell=False, capture_output=True, text=True)
        logger.info(result.stdout)
        if result.returncode != 0:
            logger.error(result.stderr)
            raise AssertionError(f"Command {command} failed!")
        

    def _convertModel(self, outputPath: str):
        inputModel = os.path.join(outputPath, self.uniDirName)
        outputModel = os.path.join(outputPath, self.convertedModelName)

        cmdBase = os.path.join(self._sdspPath, "sdspconv")

        if not self._check_version(cmdBase, self._versionFlag, self._sdspMinVersion):
            raise EnvironmentError(f"{cmdBase} is either not installed or the version is lower than {self._sdspMinVersion}")

        command = [cmdBase, "-n", inputModel, "-o", outputModel, "--input-persistency" if self._inputPersistency else "--no-input-persistency"]

        logger.info("Start packaging")
        result = subprocess.run(command, shell=False, capture_output=True, text=True)
        logger.info(result.stdout)
        if result.returncode != 0:
            logger.error(result.stderr)
            raise AssertionError(f"Command {command} failed!")
        
    def _getAccessToken(self) -> str:
        logger.info("Getting access token")
        auth_str = f"{self._authClient}:{self._authSecret}"
        auth_code = base64.b64encode(auth_str.encode()).decode()
        response = requests.post(
            url = f"https://auth.aitrios.sony-semicon.com/oauth2/default/v1/token",
            headers = {
                "Accept": "application/json",
                "Authorization": f"Basic {auth_code}",
                "Cache-Control": "no-cache",
                "Content-Type": "application/x-www-form-urlencoded"},
            data = {"grant_type": "client_credentials", "scope": "system"})
        
        if int(response.status_code) != 200:
            logger.error(response.json())
            raise AssertionError("Unable to retrieve access token")
        
        return response.json()["access_token"]

    def _packageModel(self, outputPath: str, serial: str):
        convertedModelPath = os.path.join(outputPath, self.convertedModelName)
        
        access_token = self._getAccessToken()

        logger.info("Uploading model")

        # Upload file
        packerOutPath = os.path.join(convertedModelPath, "packerOut.zip")
        response = requests.post(
            url = "https://conv-pack.aitrios.sony-semicon.com/api/v1/files",
            files = {
                "file": (os.path.basename(packerOutPath), 
                        open(packerOutPath, "rb").read(), "text/plain"),
                "type_code": (None, f"productAiModelConverted")},
            headers = {
                "Authorization": f"Bearer {access_token}",
                "tenant_id": f"{self._tenantId}"})
        if int(response.status_code) != 200:
            logger.error(response.json())
            raise AssertionError("Model Uploading failed")
        
        file_id = response.json()["file_info"]["id"]
        model_id = str(uuid.uuid4())

        # Import Model
        logger.info("Importing model")
        data_raw = {
            "model_id": model_id,
            "file_id": file_id,
            "input_format_param": [{
                    "ordinal": 0,
                    "format": "RGB"
            }]
        }

        response = requests.post(
            url = f"https://conv-pack.aitrios.sony-semicon.com/api/v1/models",
            headers = {
                "Content-Type": "application/json",
                "source-service": "marketplace",
                "tenant_id": f"{self._tenantId}",
                "Authorization": f"Bearer {access_token}",
                "parent_meta_field": "aitriosConsoleDeveloperEditionBasicPlus",
                "child_meta_field": "aitriosConsoleDeveloperEditionConveter"},
            json = data_raw)
        
        if int(response.status_code) != 200:
            logger.error(response.json())
            raise AssertionError("Model import failed")

        logger.info("Publishing model")
        
        key_generation = "ffff" if serial == "00000000000000000000000000000000" else "0001"
        response = requests.post(
            url = (f"https://conv-pack.aitrios.sony-semicon.com/api/v1/" +
                    f"models/{model_id}/model_publish"),
            headers = {
                "Content-Type": "application/json",
                "source-service": "marketplace",
                "tenant_id": f"{self._tenantId}",
                "Authorization": f"Bearer {access_token}",
                "parent_meta_field": "aitriosConsoleDeveloperEditionBasicPlus",
                "child_meta_field": "aitriosConsoleDeveloperEditionPackager"},
            json = {
                "device_id": serial,
                "key_generation": key_generation,
                "packager_version": self._packagerVersion})
        
        if int(response.status_code) != 200:
            logger.error(response.json())
            raise AssertionError("Failed to publish the model")

        transaction_id = response.json()["transaction_id"]

        # Busy polling 
        while True:
            response = requests.get(
                url = (f"https://conv-pack.aitrios.sony-semicon.com/api/v1/" +
                        f"model_publish/{transaction_id}/status?include_publish_url=true"),
                headers = {
                    "Content-Type": "application/json",
                    "source-service": "marketplace",
                    "tenant_id": f"{self._tenantId}",
                    "Authorization": f"Bearer {access_token}"})
            status = response.json()["status"]
            publish_url = response.json().get("publish_url", "")
            if status == "Publish complete":
                break

        # Download Packaged Model
        logger.info("Downloading packaged model")
        file_name = f"{serial}.zip"
        response = requests.get(publish_url)
        with open(os.path.join(outputPath, file_name), "wb") as f:
            f.write(response.content)

    def __call__(self, outputPath: str, modelPath: str, serial: str) -> str:
        self._produceUniModel(outputPath, modelPath)
        self._convertModel(outputPath)
        self._packageModel(outputPath, serial)

        logger.info(f"Model {modelPath} packaged successfully")

        zipPath = os.path.join(outputPath, f"{serial}.zip")

        with zipfile.ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall(outputPath)
        
        fpk_files = [f for f in os.listdir(outputPath) if f.endswith('.fpk')]
        
        if not fpk_files:
            raise FileNotFoundError("No .fpk file found in the zip archive.")
        if len(fpk_files) > 0:
            return AssertionError("Too many .fpk files found in the zip archive")
        
        return os.path.join(outputPath, fpk_files[0])

    