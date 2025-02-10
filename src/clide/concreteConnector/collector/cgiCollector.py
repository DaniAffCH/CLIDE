import urllib.error

import cv2 as cv
from clide.abstractConnector.remoteCollector import RemoteCollector
from clide.managers.dataManager import DataManager
from overrides import override
import numpy as np
from typing import Generator
import logging
import urllib

logger = logging.getLogger(__name__)

class CGICollector(RemoteCollector):
    def __init__(self, address: str, deviceName: str, dataManager: DataManager, bufferSize: int):
        super().__init__(address, deviceName, dataManager)
        self.lastTokenRefresh = None
        self.bufferSize = bufferSize
        self.currentCap = None
        
    @override
    def connect(self) -> bool:
        if self.isAlive():
            logger.info(f"Device {self.address} is reachable")
        else:
            raise AssertionError(f"Device {self.address} is not reachable")

        self.currentCap = self._getCap() 
        self._isConnected = True
        
        return self._isConnected
    
    def _getCap(self) -> cv.VideoCapture:
        cap = cv.VideoCapture(self.address)
        cap.set(cv.CAP_PROP_BUFFERSIZE, self.bufferSize)
        
        return cap 
    
    @override
    def _pollData(self) -> Generator[np.ndarray, None, None]:
        # TODO: error handling needed?
        assert(self._isConnected is not None)
        while True:
            ret, frame = self.currentCap.read()
            
            if not ret:
                # Token expired!
                logger.info("CAP closed!")
                self.currentCap = self._getCap()
                
            else:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                yield frame
    
    @override
    def isAlive(self) -> bool:
        try:
            retCode = urllib.request.urlopen(self.address).getcode()
            
            if retCode != 200:
                logger.warning(f"CGI connector, address {self.address} returned code {retCode}")
                return False
            else:
                return True
        
        except urllib.error.HTTPError as e:
             logger.error(e)
             return False
            
    
    @override
    def disconnect(self) -> bool:
        self._isConnected = False
        self.currentCap.release()
        return self._isConnected