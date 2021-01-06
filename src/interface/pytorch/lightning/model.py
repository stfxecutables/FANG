from typing import Tuple, no_type_check

import pytorch_lightning as pl
import torch as t
from torch import Tensor, nn
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from src.interface.pytorch.optimizer import Optimizer as TorchOptimizer


class LightningSequential(pl.LightningModule):
    def __init__(self, model: Module, optimizer: TorchOptimizer) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        # self.linear = Linear(512 * 20 * 20, 10)

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        # x = x.reshape(x.size(0), -1)
        # x = self.linear(x)
        return x

    @no_type_check
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        # training_step defined the train loop. It is independent of forward
        loss = self._forward(batch)[1]
        self.log("train_loss", loss)
        return loss

    @no_type_check
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        out, loss = self._forward(batch)
        self.log("val_loss", loss)

        acc = self._accuracy(batch, out)
        self.log("val_acc", acc, prog_bar=True)

    @no_type_check
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        out, loss = self._forward(batch)
        self.log("test_loss", loss)

        acc = self._accuracy(batch, out)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self) -> Optimizer:
        # self.optimizer.create(self.sequential.parameters())
        self.optimizer.create(self.parameters())
        return self.optimizer.torch  # type: ignore

    def _forward(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x, y = batch
        out = self(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, y)
        return out, loss

    def _accuracy(self, batch: Tuple[Tensor, Tensor], out: Tensor) -> Tensor:
        x, y = batch
        preds = out.max(axis=1)[1]  # type: ignore
        eq = (preds == y).detach().float()
        acc = t.mean(eq).cpu()
        return acc
