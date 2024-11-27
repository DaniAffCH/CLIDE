import urllib.error

import cv2 as cv 
from clide.abstractConnector.remoteCollector import RemoteCollector
from clide.managers.dataManager import DataManager
from overrides import override
import numpy as np
from typing import Generator
import logging
import urllib
import requests
import time
import re

logger = logging.getLogger(__name__)

class M3U8Collector(RemoteCollector):
    def __init__(self, address: str, deviceName: str, dataManager: DataManager, streamEndpointTemplate: str, tokenRegex: str, bufferSize: int):
        super().__init__(address, deviceName, dataManager)
        self.lastTokenRefresh = None
        self.streamEndpointTemplate = streamEndpointTemplate
        self.header = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
        self.bufferSize = bufferSize
        self.tokenRegex = tokenRegex
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

    def _refreshToken(self) -> str:
        response = requests.get(self.address, headers=self.headers)
        
        assert(response == 200, f"Failed to load {self.address}. Error Code {response}")
        
        m3u8_match = re.search(r'livee.m3u8\?a=([\w\d]+)', response.text)
        if not m3u8_match:
            raise Exception("Failed to find the streaming URL.")

        token = m3u8_match.group(1)
        
        if self.lastTokenRefresh is not None:
            timeElapsed = time.time() - self.lastTokenRefresh
            logger.info(f"Token refreshed after {timeElapsed}s")
            self.lastTokenRefresh = time.time()
            
        return token
        
    
    def _getCap(self) -> cv.VideoCapture:
        token = self._refreshToken()
        
        streamEndpoint = self.streamEndpointTemplate.format(token)
        
        cap = cv.VideoCapture(streamEndpoint)
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
                logger.info("M3U8 Token expired!")
                self.currentCap = self._getCap()
                
            else:
                yield frame
    
    @override
    def isAlive(self) -> bool:
        try:
            retCode = urllib.request.urlopen(self.address).getcode()
            
            if retCode != 200:
                logger.warning(f"M3U8 connector, address {self.address} returned code {retCode}")
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