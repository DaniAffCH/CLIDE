from clide.concreteModel.teacherPool import TeacherPool
from clide.managers.featureDistillationManager import FeatureDistillationManager
from omegaconf import DictConfig, OmegaConf
from enum import Enum
import tempfile
import logging
import hydra
import os

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

    dataManager = hydra.utils.instantiate(cfg.datamanager, teacherPool=teacherPool, useImportanceEstimation=cfg.trainer.useImportanceEstimation)
    featureDistiller = FeatureDistillationManager(student, teacherPool)
    
    collector = hydra.utils.instantiate(cfg.collector, dataManager=dataManager)
    collector.connect()

    quantizer = hydra.utils.instantiate(cfg.quantizer)
    deployer = hydra.utils.instantiate(cfg.deployer, studentModel=student)

    bestMetric = 0

    while True:
        collector.poll()

        trainer = hydra.utils.instantiate(cfg.trainer, studentModel=student, teacherPool=teacherPool, dataManager = dataManager, featureDistiller = featureDistiller)
        trainer.train()

        with tempfile.TemporaryDirectory() as temp_dir:
            if trainer.getResultMetric() > bestMetric:
                logger.info(f"Monitor Metric improved passing from {bestMetric} to {trainer.getResultMetric()}")
                bestMetric = trainer.getResultMetric()
                
                # Quantize
                #qStudentPath = quantizer.quantize(student, temp_dir, dataset=trainer.build_dataset(None, mode="val"))
                qStudentPath = temp_dir

                # Deploy
                deployer.connect()
                deployer.deploy(temp_dir, qStudentPath)
                deployer.disconnect()
            else:
                logger.info(f"Monitor Metric didn't improve. Best is {bestMetric}")

            # Resetting model
            student.saveWeights(os.path.join(temp_dir,"model_tmp.pth"))
            student = hydra.utils.instantiate(cfg.student)
            student.loadWeights(os.path.join(temp_dir,"model_tmp.pth"))
            featureDistiller.updateModel(student)


if __name__ == "__main__":
    main()