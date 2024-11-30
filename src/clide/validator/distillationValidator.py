from ultralytics.models.yolo.detect import DetectionValidator
from clide.abstractModel.teacherModel import TeacherModel
from clide.adapters.ultralytics import batchedTeacherPredictions
from ultralytics.utils.torch_utils import select_device, smart_inference_mode, de_parallel
from ultralytics.utils import LOGGER, callbacks
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr
from typing import List
import json
import torch
from ultralytics.data import converter
from ultralytics.utils.metrics import ConfusionMatrix

class UltralyticsValidator(DetectionValidator):
    def __init__(self, reviewerModels: List[TeacherModel], dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        self.reviewerModels = reviewerModels
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        self.is_coco = False
        self.is_lvis = False
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(len(model.names)))
        self.args.save_json |= (self.is_coco or self.is_lvis) and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def preprocess(self, batch, reviewerModel: TeacherModel):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255

        reviewerPred = batchedTeacherPredictions(reviewerModel, batch["img"])

        batch["batch_idx"] = reviewerPred["batch_idx"]
        batch["cls"] = reviewerPred["cls"]
        batch["bboxes"] = reviewerPred["bboxes"]
        batch["conf"] = reviewerPred["conf"]

        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]

        return batch

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Executes validation process, running inference on dataloader and computing performance metrics."""
        self.training = trainer is not None            
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            #self.data = trainer.data TODO
            # force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            # self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            for reviewerModel in self.reviewerModels: 
                self.batch_i = batch_i

                # Preprocess
                with dt[0]:
                    batch = self.preprocess(batch, reviewerModel)

                # Inference
                with dt[1]:
                    preds = model(batch["img"], augment=augment)

                # Loss
                with dt[2]:
                    if self.training:
                        self.loss += model.loss(batch, preds)[1]

                # Postprocess
                with dt[3]:
                    preds = self.postprocess(preds)

                self.update_metrics(preds, batch)
                if self.args.plots and batch_i < 3:
                    self.plot_val_samples(batch, batch_i)
                    self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / (len(self.dataloader.dataset)*len(self.reviewerModels)) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image"
                % tuple(self.speed.values())
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats