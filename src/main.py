from feda.concreteModel.teacherPool import TeacherPool
from omegaconf import DictConfig, OmegaConf
from enum import Enum
import logging
import hydra

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    logging.basicConfig(level=getattr(logging, LogLevel[cfg.log_level].value), 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')

    logger = logging.getLogger(__name__)

    logger.info("Starting the application")

    teachers = [hydra.utils.instantiate(teacher) for teacher in cfg.teachers.values()]
    teacherPool = TeacherPool(teachers)

    student = hydra.utils.instantiate(cfg.student)

    dataManager = hydra.utils.instantiate(cfg.datamanager, teacherPool=teacherPool)
    
    collector = hydra.utils.instantiate(cfg.collector, dataManager=dataManager)
    
    collector.connect()

    # trainer must be recreated each time?
    trainer = hydra.utils.instantiate(cfg.trainer, studentModel=student, teacherPool=teacherPool, dataManager = dataManager, overrides={"workers":0, "patience":10})
    while True:
        collector.poll()
        trainer.train()

if __name__ == "__main__":
    main()