from pprint import pprint
from typing import Any, Dict

import pytorch_lightning as pl
import torch as t

from src.interface.pytorch.lightning.dataloaders import MNISTDataModule
from src.interface.pytorch.lightning.model import LightningSequential
from src.interface.pytorch.optimizer import Optimizer as TorchOptimizer


def train_sequential(model: t.nn.Module, optimizer: TorchOptimizer) -> Dict[str, Any]:
    mnist = MNISTDataModule()  # type: ignore
    lightning_model = LightningSequential(model, optimizer)
    trainer = pl.Trainer(gpus=int(t.cuda.is_available()), max_epochs=1, val_check_interval=0.25)
    trainer.fit(lightning_model, mnist)  # type: ignore
    results = trainer.test()[0]
    pprint(results)
    print(f"Testing Accuracy: {results['test_acc']}")
    return results
