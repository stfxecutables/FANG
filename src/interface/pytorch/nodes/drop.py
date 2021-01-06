from __future__ import annotations  # noqa

import torch

from src.interface.layer import Layer


class Drop(Layer):
    ARGS = {"p": ("float", (0.0, 1.0))}


class Dropout(Drop):
    def create(self) -> None:
        if self.torch is not None:  # type: ignore
            return
        args = self._cleaned_args()
        self.torch = torch.nn.Dropout(**args)


class Dropout2d(Drop):
    def create(self) -> None:
        if self.torch is not None:  # type: ignore
            return
        args = self._cleaned_args()
        self.torch = torch.nn.Dropout2d(**args)


IMPLEMENTED = [Dropout, Dropout2d]
