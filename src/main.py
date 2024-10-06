from clide.concreteModel.teacherPool import TeacherPool
from omegaconf import DictConfig, OmegaConf
from enum import Enum
import tempfile
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

    trainer = hydra.utils.instantiate(cfg.trainer, studentModel=student, teacherPool=teacherPool, dataManager = dataManager)
    quantizer = hydra.utils.instantiate(cfg.quantizer)
    deployer = hydra.utils.instantiate(cfg.deployer, studentModel=student)
    
    bestMetric = 0

    while True:
        collector.poll()
        trainer.train()

        if trainer.getResultMetric() > bestMetric:
            logger.info(f"Monitor Metric improved passing from {bestMetric} to {trainer.getResultMetric()}")
            bestMetric = trainer.getResultMetric()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # quantize
                qStudentPath = quantizer.quantize(student, temp_dir, dataset=trainer.build_dataset(None, mode="val"))
                # deploy
                deployer.deploy(temp_dir, qStudentPath)

if __name__ == "__main__":
    main()