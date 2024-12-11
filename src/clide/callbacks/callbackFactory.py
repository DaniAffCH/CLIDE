from clide.callbacks.wandb import WandbCallbacks

class CallbackFactory:
    def __init__(self, logger="wandb"):
        self.logger = logger

    def callback_factory(self):
        """
        Return a dictionary of callbacks for each logger.
        Currently supports only "wandb".
        """
        callbacks = {}
        if "wandb" in self.logger:
            wandb_callbacks = WandbCallbacks()
            callbacks.update(wandb_callbacks.get_callbacks())
        
        return callbacks