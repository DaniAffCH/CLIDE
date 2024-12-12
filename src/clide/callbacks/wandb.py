from cProfile import label
import wandb
import numpy as np
from clide.abstractModel.studentModel import StudentModel

class WandbCallbacks:
    def __init__(self):
        self.step = 1
    def get_callbacks(self):
        """
        Returns the dictionary of wandb callbacks.
        """
        return {
            "on_pretrain_routine_start": [self.on_pretrain_routine_start],
            "on_pretrain_routine_end": [self.on_pretrain_routine_end],
            "on_train_start": [self.on_train_start],
            "on_train_epoch_start": [self.on_train_epoch_start],
            "on_train_batch_start": [self.on_train_batch_start],
            "optimizer_step": [self.optimizer_step],
            "on_before_zero_grad": [self.on_before_zero_grad],
            "on_train_batch_end": [self.on_train_batch_end],
            "on_train_epoch_end": [self.on_train_epoch_end],
            "on_fit_epoch_end": [self.on_fit_epoch_end],
            "on_model_save": [self.on_model_save],
            "on_train_end": [self.on_train_end],
            "on_params_update": [self.on_params_update],
            "teardown": [self.teardown],
            "on_val_start": [self.on_val_start],
            "on_val_batch_start": [self.on_val_batch_start],
            "on_val_batch_end": [self.on_val_batch_end],
            "on_val_end": [self.on_val_end],
            "on_predict_start": [self.on_predict_start],
            "on_predict_batch_start": [self.on_predict_batch_start],
            "on_predict_postprocess_end": [self.on_predict_postprocess_end],
            "on_predict_batch_end": [self.on_predict_batch_end],
            "on_predict_end": [self.on_predict_end],
            "on_export_start": [self.on_export_start],
            "on_export_end": [self.on_export_end],
            "on_data_update": [self.on_data_update],
            "on_loop_start": [self.on_loop_start],
            "on_loop_finish": [self.on_loop_finish]
        }
        
    # Trainer callbacks ---------------------------------------------------------------------------------------------------

    def on_pretrain_routine_start(self, trainer):
        """Called before the pretraining routine starts."""
        pass

    def on_pretrain_routine_end(self, trainer):
        """Called after the pretraining routine ends."""
        pass

    def on_train_start(self, trainer):
        """Called when the training starts."""
        pass

    def on_train_epoch_start(self, trainer):
        """Called at the start of each training epoch."""
        pass

    def on_train_batch_start(self, trainer):
        """Called at the start of each training batch."""
        pass

    def optimizer_step(self, trainer):
        """Called when the optimizer takes a step."""
        pass

    def on_before_zero_grad(self, trainer):
        """Called before the gradients are set to zero."""
        pass

    def on_train_batch_end(self, trainer):
        """Called at the end of each training batch."""
        pass

    def on_train_epoch_end(self, trainer):
        """Called at the end of each training epoch."""
        wandb.log(trainer.label_loss_items(trainer.tloss, prefix="train"), self.step)
        wandb.log(trainer.lr, self.step)

    def on_fit_epoch_end(self, trainer):
        """Called at the end of each fit epoch (train + val)."""
        wandb.log(trainer.metrics, self.step)
        self.step += 1

    def on_model_save(self, trainer):
        """Called when the model is saved."""
        pass

    def on_train_end(self, trainer):
        """Called when the training ends."""
        self.step += 1

    def on_params_update(self, trainer):
        """Called when the model parameters are updated."""
        pass

    def teardown(self, trainer):
        """Called during the teardown of the training process."""
        pass


    # Validator callbacks --------------------------------------------------------------------------------------------------

    def on_val_start(self, validator):
        """Called when the validation starts."""
        pass

    def on_val_batch_start(self, validator):
        """Called at the start of each validation batch."""
        pass

    def on_val_batch_end(self, validator):
        """Called at the end of each validation batch."""
        pass

    def on_val_end(self, validator):
        """Called when the validation ends."""
        pass


    # Predictor callbacks --------------------------------------------------------------------------------------------------

    def on_predict_start(self, predictor):
        """Called when the prediction starts."""
        pass

    def on_predict_batch_start(self, predictor):
        """Called at the start of each prediction batch."""
        pass

    def on_predict_batch_end(self, predictor):
        """Called at the end of each prediction batch."""
        pass

    def on_predict_postprocess_end(self, predictor):
        """Called after the post-processing of the prediction ends."""
        pass

    def on_predict_end(self, predictor):
        """Called when the prediction ends."""
        pass


    # Exporter callbacks ---------------------------------------------------------------------------------------------------

    def on_export_start(self, exporter):
        """Called when the model export starts."""
        pass

    def on_export_end(self, exporter):
        """Called when the model export ends."""
        pass

    # CLIDE callbacks ------------------------------------------------------------------------------------------------------
    def on_data_update(self, dataManager):
        num_elements = dataManager.getNumSamples()
        new_samples_ratio = dataManager._unusedRatio()
        
        memory_occupied_gb = dataManager._currentSize / (1024 ** 3)

        wandb.log({
            "Data Manager/Num Elements": num_elements,
            "Data Manager/Memory Occupied (GB)": memory_occupied_gb,
            "Data Manager/New Samples Ratio": new_samples_ratio,
            "Data Manager/New Samples Threshold": dataManager._unusedRatioThreshold
        }, self.step)
        
        self.step += 1
    
    def on_loop_start(self, sessionNumber):
        wandb.log({"CLIDE/session number": sessionNumber}, self.step)
        
    def on_loop_finish(self, bestMetric, current_metric, baselineMetric, model: StudentModel, dataloader, n=5):
        wandb.log({"CLIDE/best score": bestMetric}, self.step)
        wandb.log({"CLIDE/score": current_metric}, self.step)
        wandb.log({"CLIDE/baseline": baselineMetric}, self.step)
        
        batch = next(iter(dataloader))    
        images = batch["img"]
        
        n = min(n, images.shape[0])
        
        for i in range(n):
            npimg = np.ascontiguousarray(images[i].cpu().numpy().transpose(1, 2, 0))
            res = model.model(npimg)
            wandb.log({
                f"Images/image {i}": wandb.Image(res[0].plot(labels=False)),
            }, self.step)
        