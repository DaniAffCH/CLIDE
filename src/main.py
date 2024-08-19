from feda.concreteModel.teacher.florence2 import Florence2, FlorenceModelType
from feda.concreteModel.teacher.yolov8 import YoloV8, YoloV8ModelType
from feda.concreteModel.teacherPool import TeacherPool
from feda.concreteConnector.collector.rtspCollector import RTSPCollector, RTSPParams
from feda.managers.dataManager import DataManager
import humanfriendly
import logging
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example application with logging.')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help='Set the logging level')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)

    logger.info("Starting the application")

    teachers = [
        Florence2(FlorenceModelType.FLORENCE_2_BASE),
        YoloV8(YoloV8ModelType.YOLOV8X)
    ]

    teacherPool = TeacherPool(teachers)

    maxSize = "10 MB"
    maxSize = humanfriendly.parse_size(maxSize)

    dataManager = DataManager(dbUri="mongodb://localhost:27017",
                              dbName="testing_db",
                              teacherPool=teacherPool,
                              maxSize=maxSize)
    
    rtspParams = RTSPParams(address="daxpytip1dec23.ddns.net",
                            deviceName="DaxpyXXSettembre",
                            dataManager=dataManager, 
                            port="11001", 
                            user="root", 
                            dev_token="tina", 
                            seek_period_s=300, 
                            channel="ch0", 
                            protocol="tcp", 
                            timeout=10., 
                            frame_res=(1920, 1080))
    
    collector = RTSPCollector(rtspParams)
    collector.connect()
    collector.poll()