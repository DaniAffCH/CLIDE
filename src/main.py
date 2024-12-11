from clide.concreteModel.teacherPool import TeacherPool
from clide.managers.featureDistillationManager import FeatureDistillationManager
from clide.callbacks.callbackFactory import CallbackFactory
from omegaconf import DictConfig, OmegaConf
from enum import Enum
import tempfile
import logging
import hydra
import wandb

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

    # Initialize wandb
    wandb.init(
        project=cfg.logger.wandb.project, 
        entity=cfg.logger.wandb.entity, 
        config=OmegaConf.to_container(cfg, resolve=True), 
        name=cfg.logger.wandb.run_name
    )
    wandb.run.log_code(".")
    callbackFactory = CallbackFactory("wandb")
    callbacks = callbackFactory.callback_factory()

    teachers = [hydra.utils.instantiate(teacher) for teacher in cfg.teachers.values()]
    teacherPool = TeacherPool(teachers)

    student = hydra.utils.instantiate(cfg.student)

    dataManager = hydra.utils.instantiate(cfg.datamanager, teacherPool=teacherPool, useImportanceEstimation=cfg.trainer.useImportanceEstimation, _callbacks=callbacks)
    featureDistiller = FeatureDistillationManager(student, teacherPool)
    
    collector = hydra.utils.instantiate(cfg.collector, dataManager=dataManager)
    collector.connect()

    quantizer = hydra.utils.instantiate(cfg.quantizer)
    deployer = hydra.utils.instantiate(cfg.deployer, studentModel=student)

    bestMetric = 0
    sessionNumber = 0
    bestModelPath = None
    
    while True:
        for callbackHandler in callbacks["on_loop_start"]:
            callbackHandler(sessionNumber)
                
        collector.poll()

        trainer = hydra.utils.instantiate(cfg.trainer, studentModel=student, teacherPool=teacherPool, dataManager=dataManager, featureDistiller=featureDistiller, _callbacks=callbacks)
        trainer.train()
        
        currentMetric = trainer.getResultMetric()
        
        
        with tempfile.TemporaryDirectory() as temp_dir:
            if bestMetric > 0:
                # Load former best model and evaluate it on the current validation set
                bestMetric = trainer.validator(None, bestModelPath)["fitness"]

            if currentMetric > bestMetric:
                logger.info(f"Monitor Metric improved from {bestMetric} to {currentMetric}")
                bestMetric = currentMetric
                bestModelPath = trainer.best
                
                # Quantize
                # qStudentPath = quantizer.quantize(student, temp_dir, dataset=trainer.build_dataset(None, mode="val"))
                qStudentPath = temp_dir

                # Deploy
                deployer.connect()
                deployer.deploy(temp_dir, qStudentPath)
                deployer.disconnect()

                # Log artifact
                artifact = wandb.Artifact("best_model", type="model")
                artifact.add_file(bestModelPath)
                wandb.log_artifact(artifact)
            else:
                logger.info(f"Monitor Metric didn't improve. Best is {bestMetric}")

            # Resetting model
            student = hydra.utils.instantiate(cfg.student)
            student.loadWeights(bestModelPath)
            featureDistiller.updateModel(student)
            
            #|-_|
            baselineMetric = trainer.validator(None, "yolov8n.pt")["fitness"]
            
            for callbackHandler in callbacks["on_loop_finish"]:
                callbackHandler(bestMetric, currentMetric, baselineMetric, student, trainer.validator.dataloader)
                
            # Resetting model
            student = hydra.utils.instantiate(cfg.student)
            student.loadWeights(bestModelPath)
            featureDistiller.updateModel(student)
                
            sessionNumber += 1


if __name__ == "__main__":
    main()
