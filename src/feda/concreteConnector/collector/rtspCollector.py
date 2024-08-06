from feda.abstractConnector.remoteCollector import RemoteCollector
from feda.managers.dataManager import DataManager
from overrides import override
import numpy as np
import time 
from typing import Tuple, Generator
import warnings
import av
from dataclasses import dataclass

@dataclass
class RTSPParams:
    address: str
    deviceName: str
    dataManager: DataManager
    port: str
    user: str
    dev_token: str
    seek_period_s: int
    channel: str
    protocol: str
    timeout: float
    frame_res: Tuple[int, int]

class RTSPCollector(RemoteCollector):
    def __init__(self, params: RTSPParams) -> None:
        super().__init__(params.address, params.deviceName, params.dataManager)
        self._port = params.port
        self._user = params.user
        self._dev_token = params.dev_token
        self._seek_period_s = params.seek_period_s
        self._tt_prev = None
        self._channel = params.channel
        self._protocol = params.protocol
        self._timeout = params.timeout
        self._frame_res = params.frame_res

    @override
    def connect(self) -> bool:
        
        while not self._isConnected:
            try:
                conn = (f"rtsp://{self._user}:{self._dev_token}@"
                        + f"{self.address}:{self._port}/"
                        + self._channel)
                print("Connection at", conn)
                container = av.open(conn, format="rtsp",
                    options={"rtsp_transport": self._protocol},
                    timeout=self._timeout)
            except Exception as e:
                print("Cannot connect:", e)
                time.sleep(1)
                continue

            self._isConnected = True
            self._container = container

            self._sync_stream(check_diff=False, old_s=-1)

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
                print("Synchronizing RTSP stream ...")
                self._container.seek(-1)
        self.rs = running_s
        return sync

    def _pollData(self) -> Generator[np.ndarray, None, None]:

        try:
            for frame in self._container.decode(video=0):

                width, height = self._frame_res                
                image = frame.to_rgb(width=width, height=height).to_ndarray()

                sync = self._sync_stream(check_diff=True, old_s=self.rs)

                yield image

        except av.InvalidDataError as ave:
            warnings.warn("Received invalid av data")
            yield np.empty(0)

    def _writeData(self, image: np.ndarray):
        self._dataManager.addImage(image)

    @override
    def poll(self) -> None:
        if not self._isConnected:
            raise AssertionError("Device is not connected.")
        for image in self._pollData():
            self._writeData(image)

    @override
    def isAlive(self) -> bool:
        return True
    
    @override
    def disconnect(self) -> bool:
        return True
