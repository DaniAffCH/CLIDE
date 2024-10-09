from clide.abstractModel.studentModel import StudentModel
from clide.concreteModel.teacherPool import TeacherPool
from clide.managers.dataManager import DataManager
from clide.adapters.dataset import StubDatasetFactory
from clide.adapters.ultralytics import batchedTeacherPredictions
from clide.validator.distillationValidator import UltralyticsValidator
from ultralytics.utils import LOGGER, colorstr, RANK, TQDM, DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import autocast
from copy import copy
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

    def __init__(self, studentModel: StudentModel, teacherPool: TeacherPool, dataManager: DataManager, splitRatio: Dict[str, float], cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
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
    
        super().__init__(cfg, overrides, _callbacks)
        
        self.dataManager = dataManager
        self.teacherPool = teacherPool
        self.studentModel = studentModel
        self.datasetFactory = StubDatasetFactory(self.dataManager, splitRatio)
        self.studentModel.model.requires_grad_(True)

        self.teacherModel = None
        self.reviewerModel = None
        
    def updateTeacherReviewer(self):
        self.teacherModel, self.reviewerModel = self.teacherPool.teacherReviewerSplit()
        logger.info(f"Teacher Model: {self.teacherModel.name}")
        logger.info(f"Reviewer Model: {self.reviewerModel.name}")

    def updateDataset(self):
        self.datasetFactory.updateData()

    @override
    def get_dataset(self):
        return "", ""
    
    @override
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return UltralyticsValidator(
            self.reviewerModel, self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    @override
    def build_dataset(self, img_path, mode="train", batch=None):            
        return self.datasetFactory(mode)
    
    @override
    def get_model(self, cfg=None, weights=None, verbose=True):
        return self.studentModel.model.model
    
    @override
    def set_model_attributes(self):
        self.model.args = self.args        

    def _do_train(self, world_size=1):
        
        """Train completed, evaluate and plot if specified by arguments."""
        self.updateDataset()
        self.updateTeacherReviewer()

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

                    teacher_pred = batchedTeacherPredictions(self.teacherModel ,batch["img"])

                    batch["batch_idx"] = teacher_pred["batch_idx"]
                    batch["cls"] = teacher_pred["cls"]
                    batch["bboxes"] = teacher_pred["bboxes"]

                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

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
        gc.collect()
        torch.cuda.empty_cache()
        self.run_callbacks("teardown")

    def getResultMetric(self):
        return self.metrics["metrics/mAP50-95(B)"]