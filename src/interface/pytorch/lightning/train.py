from typing import Any, Dict, cast

import pytorch_lightning as pl
import torch as t

from src.interface.pytorch.lightning.dataloaders import MNISTDataModule
from src.interface.pytorch.lightning.model import LightningSequential
from src.interface.pytorch.lightning.timer import Timer
from src.interface.pytorch.optimizer import Optimizer as TorchOptimizer


def train_sequential(
    model: t.nn.Module, optimizer: TorchOptimizer, fast_dev_run: bool = False
) -> Dict[str, Any]:
    EPOCHS = 1
    if EPOCHS == 1:
        VAL_CHECK_INTERVAL = 0.25
    elif EPOCHS < 5:
        VAL_CHECK_INTERVAL = 0.5
    else:
        VAL_CHECK_INTERVAL = 1.0
    mnist = MNISTDataModule(fast_dev_run=fast_dev_run)  # type: ignore
    lightning_model = LightningSequential(model, optimizer)
    if fast_dev_run:
        timer = Timer(duration=dict(seconds=5), interval="step")
        trainer = pl.Trainer(
            gpus=int(t.cuda.is_available()),
            max_epochs=1,
            check_val_every_n_epoch=2,
            max_steps=1,
            # callbacks=[timer],
            progress_bar_refresh_rate=0,
            weights_summary=None,
        )
    else:
        timer = Timer(duration=dict(minutes=2), interval="step")
        trainer = pl.Trainer(
            gpus=int(t.cuda.is_available()),
            max_epochs=EPOCHS,
            # val_check_interval=VAL_CHECK_INTERVAL,
            callbacks=[timer],
        )
    trainer.fit(lightning_model, mnist)  # type: ignore
    results = trainer.test(verbose=False)[0]
    print(f"Testing Accuracy: {results['test_acc']}")
    return cast(Dict[str, Any], results)
