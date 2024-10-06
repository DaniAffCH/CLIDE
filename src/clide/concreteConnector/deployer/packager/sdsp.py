from typing import Any
import subprocess
import os
import logging

logger = logging.getLogger(__name__)

class SDSPPackager:
    def __init__(self, inputPersistency: bool):
        self._uniDirName = "packModel"
        self._convertedModelName = "convModel"
        self._inputPersistency = inputPersistency
    
    @property
    def uniDirName(self): 
        return self._uniDirName

    @property
    def convertedModelName(self): 
        return self._convertedModelName
    
    def _produceUniModel(self, outputPath: str, modelPath: str):
        inputModel = os.path.join(outputPath, modelPath)
        outputModel = os.path.join(outputPath, self.uniDirName)

        # TODO: check if it is installed!

        command = ["uni-pytorch", "-i", inputModel, "-o", outputModel]

        result = subprocess.run(command, shell=False, capture_output=True, text=True)
        logger.info(result.stdout)
        if result.returncode != 0:
            logger.error(result.stderr)
            raise AssertionError(f"Command {command} failed!")
        

    def _convertModel(self, outputPath: str):
        inputModel = os.path.join(outputPath, self.uniDirName)
        outputModel = os.path.join(outputPath, self.convertedModelName)

        # TODO: check if it is installed!

        command = ["sdspconv", "-n", inputModel, "-o", outputModel, "--input-persistency" if self._inputPersistency else " --no-input-persistency"]

        result = subprocess.run(command, shell=False, capture_output=True, text=True)
        logger.info(result.stdout)
        if result.returncode != 0:
            logger.error(result.stderr)
            raise AssertionError(f"Command {command} failed!")


    def __call__(self, outputPath: str, modelPath: str) -> str:
        self._produceUniModel(outputPath, modelPath)
        self._convertModel(outputPath)
        logger.info(f"Model {modelPath} converted successfully")
        return os.path.join(outputPath, self.convertedModelName)
