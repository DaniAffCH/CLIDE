from clide.abstractConnector.remoteCollector import RemoteCollector
from clide.managers.dataManager import DataManager
from overrides import override
import numpy as np
import time 
from typing import Tuple, Generator
import av
import logging
import subprocess

logger = logging.getLogger(__name__)

class RTSPCollector(RemoteCollector):
    def __init__(self, 
                 address: str,
                 deviceName: str,
                 dataManager: DataManager,
                 port: str,
                 user: str,
                 dev_token: str,
                 seek_period_s: int,
                 channel: str,
                 protocol: str,
                 timeout: float,
                 frame_res: Tuple[int, int]) -> None:
        super().__init__(address, deviceName, dataManager)
        self._port = port
        self._user = user
        self._dev_token = dev_token
        self._seek_period_s = seek_period_s
        self._tt_prev = None
        self._channel = channel
        self._protocol = protocol
        self._timeout = timeout
        self._frame_res = frame_res

    @override
    def connect(self) -> bool:
        
        if self.isAlive():
            logger.info(f"Device {self.address} is reachable")
        else:
            raise AssertionError(f"Device {self.address} is not reachable")

        while not self._isConnected:
            try:
                conn = (f"rtsp://{self._user}:{self._dev_token}@"
                        + f"{self.address}:{self._port}/"
                        + self._channel)
                logger.info(f"Attempting connection to RTSP streaming {conn}")
                container = av.open(conn, format="rtsp",
                    options={"rtsp_transport": self._protocol},
                    timeout=self._timeout)
            except Exception as e:
                logger.warning(f"Cannot connect: {e}")
                time.sleep(1)
                continue

            self._isConnected = True
            self._container = container

            self._sync_stream(check_diff=False, old_s=-1)

        logger.info(f"Connection to RTSP streaming established.")

        return self._isConnected

    def _sync_stream(self, check_diff: bool = False, old_s = None) -> bool:
        """ Synchronize the RTSP stream """
        running_s = 0
        sync = not check_diff
        if self._seek_period_s >= 0:
            if check_diff and self._tt_prev is not None:
                tt_prev = str(self._tt_prev)
                seconds = int(tt_prev.split(":")[-1].split(".")[0])
                minutes = int(tt_prev.split(":")[-2])
                hours = int(tt_prev.split(":")[0].split(" ")[-1])
                running_s = (hours * 60 * 60) + (minutes * 60) + seconds
                if running_s % self._seek_period_s == 0:
                    sync = True
            if sync and old_s != running_s:
                logger.info("Synchronizing RTSP stream.")
                self._container.seek(-1)
        self.rs = running_s
        return sync

    @override
    def _pollData(self) -> Generator[np.ndarray, None, None]:
        try:
            for frame in self._container.decode(video=0):

                width, height = self._frame_res                
                image = frame.to_rgb(width=width, height=height).to_ndarray()

                sync = self._sync_stream(check_diff=True, old_s=self.rs)

                yield image

        except av.InvalidDataError as ave:
            logger.warning("Received invalid av data.")
            yield np.empty(0)


    @override
    def isAlive(self) -> bool:
        try:
            subprocess.run(['ping', "-c", '1', self.address], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1)
            return True
        except subprocess.CalledProcessError:
            return False
        except subprocess.TimeoutExpired:
            return False
    
    @override
    def disconnect(self) -> bool:
        if self._container is not None:
            self._container.close()
        else:
            logger.warning("No container to close for RTSP collector.")
        self._container = None
        self._isConnected = False

        logger.info("RTSP connection closed.")

        return self._isConnected

