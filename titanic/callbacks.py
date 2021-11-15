from pytorch_tabnet.callbacks import Callback
from typing import Dict, Any
from dataclasses import dataclass
import wandb


@dataclass
class WandBCallback(Callback):
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        wandb.log(logs)


@dataclass
class ModelCheckpointer(Callback):
    path: str

    def on_train_end(self, logs: Dict[str, Any] = None):
        self.trainer.save_model(self.path)
