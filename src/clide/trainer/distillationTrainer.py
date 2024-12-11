from clide.abstractModel.studentModel import StudentModel
from clide.concreteModel.teacherPool import TeacherPool
from clide.managers.dataManager import DataManager
from clide.managers.featureDistillationManager import FeatureDistillationManager
from clide.adapters.dataset import StubDatasetFactory
from clide.adapters.ultralytics import batchedTeacherPredictions
from clide.validator.distillationValidator import UltralyticsValidator
from clide.trainer.loss import KDv8DetectionLoss
from ultralytics.utils import LOGGER, colorstr, RANK, TQDM, DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import autocast
from copy import copy
from torch import optim, nn
import torch.nn.functional as F
import numpy as np
import torch
import time
import warnings
import gc
from torch import distributed as dist
import math
from overrides import override
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class UltralyticsTrainer(DetectionTrainer):

    def __init__(self, studentModel: StudentModel, teacherPool: TeacherPool, featureDistiller: FeatureDistillationManager, dataManager: DataManager, splitRatio: Dict[str, float], distillationAlpha: float, useSoftLabels: bool, useFeatureDistillation: bool, useImportanceEstimation: bool, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        cfg.plots = False
        cfg.augment = False
        cfg.workers = 0

        # Disable lighting augmentation
        cfg.hsv_h = 0.
        cfg.hsv_s = 0.
        cfg.hsv_v = 0.

        # Disable augmentations
        cfg.translate = 0.
        cfg.scale = 0.
        cfg.mosaic = 0.
        cfg.fliplr=0.
        cfg.erasing=0.
        cfg.crop_fraction=0.
        
        cfg.model = studentModel.model.model.args["model"]
        cfg.device =  studentModel.model.model.args["device"]
        
        super().__init__(cfg, overrides)
        
        self.dataManager = dataManager
        self.teacherPool = teacherPool
        self.studentModel = studentModel
        self.datasetFactory = StubDatasetFactory(self.dataManager, splitRatio)
        self.studentModel.model.requires_grad_(True)
        self.distillationAlpha = distillationAlpha
        self.featureDistiller = featureDistiller
        self.useSoftLabels = useSoftLabels
        self.useFeatureDistillation = useFeatureDistillation
        self.useImportanceEstimation = useImportanceEstimation
        self.callbacks = _callbacks
        
        self.studentModel.model.model.args = self.args        
        self.set_criterion(self.studentModel)

        self.teacherModel = None
        self.reviewerModels = None
        
    def updateTeacherReviewers(self):
        self.teacherModel, self.reviewerModels = self.teacherPool.teacherReviewersSplit()
        logger.info(f"Teacher Model: {self.teacherModel.name}")
        logger.info(f"Reviewer Models: " + ", ".join([reviewerModel.name for reviewerModel in self.reviewerModels]))

    def updateDataset(self):
        self.datasetFactory.updateData()

    @override
    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )
        
        studentHooks = self.studentModel.hookLayers
        teacherHooks = self.teacherModel.hookLayers

        for s,t in zip(studentHooks, teacherHooks):
            adaptationLayer = self.featureDistiller.getAdaptationLayer(self.studentModel.name, self.teacherModel.name, s, t)
            for p in adaptationLayer.adaptationModule.parameters():
                g[0].append(p) # Add adaptation module

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
        )
        return optimizer

    def set_criterion(self, student: StudentModel):
        student.model.model.criterion = KDv8DetectionLoss(student.model.model, softLabels = self.useSoftLabels)

    @override
    def get_dataset(self):
        return "", ""
    
    @override
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return UltralyticsValidator(
            self.reviewerModels, self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    @override
    def build_dataset(self, img_path, mode="train", batch=None):  
        ds = self.datasetFactory(mode)
        logger.info(f"Built a dataset for {mode} containing {len(ds)} samples")          
        return ds
    
    @override
    def get_model(self, cfg=None, weights=None, verbose=False):
        return self.studentModel.model.model
    
    @override
    def set_model_attributes(self):
        self.model.args = self.args        

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        self.featureDistiller.updateAllHooks()
        self.updateTeacherReviewers()

        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()
            self.model.train() 
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                if len(self.loss_names) == 3:
                    self.loss_names += ("im_loss",) 
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with autocast(self.amp):
                
                    batch = self.preprocess_batch(batch)

                    teacher_pred = batchedTeacherPredictions(self.teacherModel, batch["img"])

                    batch["batch_idx"] = teacher_pred["batch_idx"]
                    batch["cls"] = teacher_pred["cls"]
                    batch["bboxes"] = teacher_pred["bboxes"]
                    batch["conf"] = teacher_pred["conf"]
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                    adaptedFeatures = self.featureDistiller.getAdaptedFeatures(self.studentModel.name, self.teacherModel.name)

                    teacherFeatures = adaptedFeatures["teacher"]
                    studentFeatures = adaptedFeatures["student"]

                    imitationLoss = torch.tensor(0.0, device=self.model.args.device, dtype=torch.float32)
                    
                    if self.useFeatureDistillation:
                        for (sf_key, sf), (tf_key, tf) in zip(studentFeatures.items(), teacherFeatures.items()):
                            assert sf_key == tf_key, "Keys do not match for student and teacher features"
                            
                            fmse = (sf - tf) ** 2
                            
                            if self.useImportanceEstimation:
                                importance = batch["importance_map"][self.teacherModel.name].to(sf.device)
                                fmse = importance * fmse
                            
                            imitationLoss = imitationLoss + fmse.mean()

                        imitationLoss = imitationLoss * self.distillationAlpha
                        
                    self.loss = self.loss + imitationLoss

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.shape) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                losses = torch.cat((losses, imitationLoss.detach().unsqueeze(0)))
                loss_len+=1
                if RANK in {-1, 0}:
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_len))
                        % (f"{epoch + 1}/{self.epochs}", mem, *losses, batch["cls"].shape[0], batch["img"].shape[-1])
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = Deployment(t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            gc.collect()
            torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            LOGGER.info(
                f"\n{epoch - self.start_epoch + 1} epochs completed in "
                f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
            )
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self.model.eval()
        gc.collect()
        torch.cuda.empty_cache()
        self.run_callbacks("teardown")

    def getResultMetric(self):
        return self.best_fitness