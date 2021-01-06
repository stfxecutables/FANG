import os
from pprint import pprint
from typing import Any, Optional, Tuple, no_type_check

import pytorch_lightning as pl
import torch as t
from torch import Tensor, nn
from torch.nn import Conv2d, Linear, ReLU
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

BATCH_SIZE = 128


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = BATCH_SIZE, seed: int = None) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        # download only
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage: Optional[str] = None) -> None:
        # transform
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transform)
        mnist_test = MNIST(os.getcwd(), train=False, download=False, transform=transform)

        # train/val split
        gen = t.Generator().manual_seed(self.seed) if self.seed is not None else None
        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000], gen)

        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)


class MnistCNN(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            Conv2d(1, out_channels=64, kernel_size=3, padding=0),
            ReLU(),
            Conv2d(64, out_channels=128, kernel_size=3, padding=0),
            ReLU(),
            Conv2d(128, out_channels=256, kernel_size=3, padding=0),
            ReLU(),
            Conv2d(256, out_channels=512, kernel_size=3, padding=0),
            ReLU(),
            # Softmax(10)
        )
        self.linear = Linear(512 * 20 * 20, 10)

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        x = self.sequential(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x

    @no_type_check
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        out = self(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, y)
        self.log("train_loss", loss)
        return loss

    @no_type_check
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        out = self(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, y)
        self.log("val_loss", loss)

        preds = out.max(axis=1)[1].detach()
        eq = (preds == y).detach().float()
        acc = t.mean(eq).cpu()
        self.log("val_acc", acc, prog_bar=True)

    @no_type_check
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        out = self(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, y)
        self.log("test_loss", loss)

        preds = out.max(axis=1)[1]
        eq = (preds == y).detach().float()
        acc = float(t.mean(eq).cpu().numpy())
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self) -> Optimizer:
        optimizer = t.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


dm = MNISTDataModule()  # type: ignore
model = MnistCNN()
trainer = pl.Trainer(gpus=1, max_epochs=1, val_check_interval=0.25)
trainer.fit(model, dm)  # type: ignore
result = trainer.test()[0]
pprint(result)

print(f"Testing Accuracy: {result['test_acc']}")
