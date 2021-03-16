from __future__ import annotations  # noqa
from typing import Any, Optional, Tuple

import torch
from torch.nn import Module

from src.interface.arguments import Arguments
from src.interface.initializer import PyTorch
from src.interface.layer import ReshapingLayer


class Padding(ReshapingLayer):
    """This class should abstract over the similar components of the Padding layers"""

    MAX_PAD = 5

    ARGS = {
        "padding": ("int", (1, MAX_PAD)),
        # "padding_tuple": ("tuple", ((0,MAX_PAD), (0,MAX_PAD), (0,MAX_PAD), (0,MAX_PAD))),
    }

    def __init__(self, input_shape: Tuple[int, ...], **layer_kwargs: Any):
        self.torch: Optional[Module] = None
        self.input_shape = input_shape
        self.in_channels = input_shape[0]

        # by default, `out_channels` is random initially
        # But, in terms of padding, the input and output channels should be same
        channel_argvals = {"in_channels": input_shape[0], "out_channels": input_shape[0]}
        # order below is IMPORTANT, want to update OUT with ARGS
        self.args = Arguments({**self.OUT, **self.ARGS})
        self.args.arg_values.update(channel_argvals)
        # we need to adjust random padding in case when image size is small, or get:
        # RuntimeError: Padding size should be less than the corresponding input dimension
        max_pad = min(input_shape[1:]) - 1
        current_pad = self.args.arg_values["padding"]
        self.args.arg_values["padding"] = min(max_pad, current_pad)
        self.out_channels = self.args.arg_values["out_channels"]
        self.output_shape = self._output_shape()

    def _output_shape(self) -> Tuple[int, ...]:
        # https://pytorch.org/docs/stable/generated/torch.nn.ZeroPad2d.html#torch.nn.ZeroPad2d
        H, W = self.input_shape[1], self.input_shape[2]
        P = self.args.arg_values["padding"]
        C_out = self.out_channels
        H_out = int(H + 2 * P)
        W_out = int(W + 2 * P)
        return C_out, H_out, W_out


class ZeroPadding(Padding, PyTorch):
    """Pads the input tensor boundaries with zero."""

    def create(self) -> None:
        if self.torch is not None:
            return
        args = self.args.clone().arg_values
        del args["out_channels"]
        del args["in_channels"]
        self.torch = torch.nn.ZeroPad2d(**args)


class ReplicationPadding(Padding, PyTorch):
    """Pads the input tensor using replication of the input boundary"""

    def create(self) -> None:
        if self.torch is not None:
            return
        args = self.args.clone().arg_values
        del args["out_channels"]
        del args["in_channels"]
        self.torch = torch.nn.ReplicationPad2d(**args)


class ReflectionPadding(Padding, PyTorch):
    """Pads the input tensor using the reflection of the input boundary"""

    def create(self) -> None:
        if self.torch is not None:
            return
        args = self.args.clone().arg_values
        del args["out_channels"]
        del args["in_channels"]
        self.torch = torch.nn.ReflectionPad2d(**args)


IMPLEMENTED = [ZeroPadding, ReplicationPadding, ReflectionPadding]
