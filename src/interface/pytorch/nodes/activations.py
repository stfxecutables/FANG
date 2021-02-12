from __future__ import annotations  # noqa

import torch

from src.interface.layer import Layer


class Activation(Layer):
    """Class for activation functions with no meaningful options"""


class ReLU(Activation):
    def create(self) -> None:
        if self.torch is not None:
            return
        self.torch = torch.nn.ReLU()


class Hardswish(Activation):
    def create(self) -> None:
        if self.torch is not None:
            return
        self.torch = torch.nn.Hardswish()


class LeakyReLU(Activation):
    ARGS = {"negative_slope": ("float", (0.0, 1.0))}  # 0.0 reduces to ReLU

    def create(self) -> None:
        if self.torch is not None:  # type: ignore
            return
        args = self._cleaned_args()
        self.torch = torch.nn.LeakyReLU(**args)


class PReLU(Activation):
    # NOTE: this should be modified to allow randomly setting the `a` argument
    def create(self) -> None:
        if self.torch is not None:  # type: ignore
            return
        self.torch = torch.nn.PReLU()


class ELU(Activation):
    ARGS = {"alpha": ("float", (0.0, 5.0))}  # 0.0 reduces to ReLU

    def create(self) -> None:
        if self.torch is not None:  # type: ignore
            return
        args = self._cleaned_args()
        self.torch = torch.nn.ELU(**args)


class Sigmoid(Activation):

    def create(self) -> None:
        if self.torch is not None:  # type: ignore
            return
        self.torch = torch.nn.Sigmoid()


class Tanh(Activation):

    def create(self) -> None:
        if self.torch is not None:  # type: ignore
            return
        self.torch = torch.nn.Tanh()


IMPLEMENTED = [ELU, Hardswish, LeakyReLU, PReLU, ReLU, Sigmoid, Tanh]
