from feda.concreteModel.teacher.florence2 import Florence2, FlorenceModelType
from feda.concreteConnector.collector.rtspCollector import RTSPCollector, RTSPParams
from feda.managers.dataManager import DataManager

if __name__ == "__main__":
    m = Florence2(FlorenceModelType.FLORENCE_2_BASE)

    dataManager = DataManager(dbUri="mongodb://localhost:27017",
                              dbName="testing_db",
                              teacherModel=m,
                              maxSize=100000000)
    
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