from __future__ import annotations  # noqa

import torch

from src.interface.initializer import PyTorch
from src.interface.layer import Layer


class BatchNorm2d(Layer, PyTorch):
    MAX_MOMENTUM = 1.0
    ARGS = {"momentum": ("float", (0, MAX_MOMENTUM))}

    def create(self) -> None:
        """Make the actual Torch Layer object, and save it for later"""
        if self.torch is not None:  # type: ignore
            return
        args = self._cleaned_args(remove="out_channels")
        args["num_features"] = args["in_channels"]
        del [args["in_channels"]]
        self.torch = torch.nn.BatchNorm2d(**args)


class InstanceNorm2d(Layer, PyTorch):
    MAX_MOMENTUM = 1.0
    ARGS = {"momentum": ("float", (0, MAX_MOMENTUM)), "affine": ("bool", None)}  # type: ignore

    def create(self) -> None:
        """Make the actual Torch Layer object, and save it for later"""
        if self.torch is not None:  # type: ignore
            return
        args = self._cleaned_args(remove="out_channels")
        args["num_features"] = args["in_channels"]
        del [args["in_channels"]]
        self.torch = torch.nn.InstanceNorm2d(**args)


class LayerNorm(Layer, PyTorch):
    def __init__(self) -> None:
        raise NotImplementedError("TODO!")


class GroupNorm(Layer, PyTorch):
    def __init__(self) -> None:
        raise NotImplementedError(
            """
GroupNorm provides some challenges for FANG because if you try to use it in a REPL, you will see you
get errors if your number of groups don't perfectly divide the number of channels:

    >>> import torch as t
    >>> t.nn.GroupNorm(3, num_channels=2)(t.rand([1, 2, 3]))  # (batch, channel, height)
    RuntimeError:
        Expected number of channels in input to be divisible by num_groups, but got input of shape
        [1, 2, 3] and num_groups=3

If we don't require e.g. the number of output channels to either even, powers or 2, or other numbers
which are very un-prime (lots of whole-number divisors) then attempting to add a GroupNorm layer
will extremely often result in an error.

In general, it makes a great deal of sense to limit layer channel sizes (other than the input layer)
to factors of 2. Entire classes of problems just *go away* by doing this, with no real *meaningful*
losses on the space of architectures that is searched.
"""
        )


IMPLEMENTED = [BatchNorm2d, InstanceNorm2d]
