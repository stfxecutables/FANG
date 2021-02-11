import os
import platform
from typing import Any, Optional

import pytorch_lightning as pl
import torch as t
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

BATCH_SIZE = 64
NUM_WORKERS = 0 if platform.system().lower() == "windows" else 4


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        seed: int = None,
        fast_dev_run: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.fast_dev_run: bool = fast_dev_run

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
        if self.fast_dev_run:
            mnist_train, _ = random_split(mnist_train, [5000, 50000], gen)

        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
