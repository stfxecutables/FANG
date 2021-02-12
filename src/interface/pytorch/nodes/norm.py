from __future__ import annotations  # noqa
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from copy import deepcopy
import torch
from torch.nn import Module
from src.interface.initializer import PyTorch
from src.interface.layer import Layer, ReshapingLayer


class BatchNorm2d(Layer, PyTorch):
    """
    A “momentum” argument allows you to control how much of the statistics from the previous mini batch to include
    when the update is calculated. By default, this is kept high with a value of 0.99. This can be set to 0.0 to only
    use statistics from the current mini-batch
    """

    MAX_MOMENTUM = 1.0
    ARGS = {"momentum": ("float", (0, MAX_MOMENTUM)), "affine": ("bool", None)}

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
    ARGS = {
        "momentum": ("float", (0, MAX_MOMENTUM)),
        "affine": ("bool", None),
        "track_running_stats": ("bool", None),
    }  # type: ignore

    def create(self) -> None:
        """Make the actual Torch Layer object, and save it for later"""
        if self.torch is not None:  # type: ignore
            return
        args = self._cleaned_args(remove="out_channels")
        args["num_features"] = args["in_channels"]
        del [args["in_channels"]]
        self.torch = torch.nn.InstanceNorm2d(**args)


class LayerNorm(Layer, PyTorch):
    ARGS = {
        "elementwise_affine": ("bool", None),  # type: ignore
        "normalized_shape_i": ("int", (0, 3)),
    }

    # def __init__(self, input_shape: Tuple[int, ...], normalized_shape: Tuple[int, ...]):
    #     super().__init__(input_shape)
    #     self.input_shape = input_shape
    #     self.normalized_shape = input_shape[0]

    def create(self) -> None:
        args = deepcopy(self.args.arg_values)  # gets dict
        i = args["normalized_shape_i"]
        args["normalized_shape"] = self.input_shape[i:]
        del args["normalized_shape_i"]
        del args["in_channels"]
        self.torch = torch.nn.LayerNorm(**args)

    def _output_shape(self) -> Tuple[int, ...]:
        pass


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


IMPLEMENTED = [BatchNorm2d, InstanceNorm2d, LayerNorm]
