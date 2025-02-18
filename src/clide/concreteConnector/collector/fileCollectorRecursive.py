from clide.concreteConnector.collector.fileCollector import FileCollector
from overrides import override
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FileCollectorRecursive(FileCollector):
    @override
    def connect(self) -> bool:
        if not self.isAlive():
            raise AssertionError(f"{self.imagesPath}: No such directory")
        self._pathIter = iter(sorted(Path(self.imagesPath).rglob("*"), key=os.path.getmtime))
        
        self.ctr = 0
        logger.info(f"Fake connection to {self.imagesPath} established.")

        self._isConnected = True
        return self._isConnected