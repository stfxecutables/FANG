from __future__ import annotations  # noqa

from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.nn import Module

from src.interface.arguments import Arguments
from src.interface.initializer import PyTorch
from src.interface.layer import Layer, ReshapingLayer


class Pool(ReshapingLayer):
    """This class should abstract over the similar components of the Pooling layers"""

    MAX_KERNEL = 3
    MAX_STRIDE = 3
    MAX_PAD = 6  # TODO: tie to kernel_size, string, padding, and dilation instead
    MAX_DILATE = 3

    ARGS = {
        "kernel_size": ("int", (1, MAX_KERNEL)),
        "stride": ("int", (1, MAX_STRIDE)),
        "padding": ("int", (0, MAX_PAD)),
        "dilation": ("int", (1, MAX_DILATE)),
    }

    def __init__(self, input_shape: Tuple[int, ...], **layer_kwargs: Any):
        self.torch: Optional[Module] = None
        self.input_shape = input_shape
        self.in_channels = input_shape[0]

        # by default, `out_channels` is random initially
        channel_argvals = {"in_channels": input_shape[0], "out_channels": input_shape[0]}
        self.args = Arguments({**self.OUT, **self.ARGS})
        self.args.arg_values.update(channel_argvals)

        # Handle some special cases relating to valid padding sizes i.e. the error:
        # RuntimeError: pad should be smaller than half of kernel size
        kernel = self.args.arg_values["kernel_size"]
        self.args.arg_values["padding"] = (
            int(np.random.randint(0, kernel // 2)) if kernel >= 2 else 0
        )

        # Handle some special cases relating to valid kernel sizes for small inputs
        # RuntimeError:
        #   Given input size: (1x28x28) calculated output size: (1x-1x-1). Output size is too small
        self.out_channels = self.args.arg_values["out_channels"]
        self.output_shape = self._output_shape()

    def _output_shape(self) -> Tuple[int, ...]:
        # NOTE: currently assumes padding, kernels, dilation are all symmetric (e.g. `(n,n)`)
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        H, W = self.input_shape[1], self.input_shape[2]
        K = self.args.arg_values["kernel_size"]
        D = self.args.arg_values["dilation"]
        P = self.args.arg_values["padding"]
        S = self.args.arg_values["stride"]
        C_out = self.out_channels
        H_out = int(np.floor(((H + 2 * P - D * (K - 1) - 1) / S) + 1))
        W_out = int(np.floor(((W + 2 * P - D * (K - 1) - 1) / S) + 1))
        return (C_out, H_out, W_out)


class MaxPool2d(Pool, PyTorch):
    def create(self) -> None:
        """Make the actual Torch Layer object, and save it for later"""
        if self.torch is not None:
            return
        args = self.args.clone().arg_values
        del args["out_channels"]
        del args["in_channels"]
        self.torch = torch.nn.MaxPool2d(**args)


class AveragePool2d(Pool, PyTorch):
    MAX_KERNEL = 3
    MAX_STRIDE = 3
    MAX_PAD = 6  # TODO: tie to kernel_size, string, padding, and dilation instead

    ARGS = {
        "kernel_size": ("int", (1, MAX_KERNEL)),
        "stride": ("int", (1, MAX_STRIDE)),
        "padding": ("int", (0, MAX_PAD)),
    }

    def create(self) -> None:
        """Make the actual Torch Layer object, and save it for later"""
        if self.torch is not None:
            return
        args = self.args.clone().arg_values
        del args["out_channels"]
        del args["in_channels"]
        self.torch = torch.nn.AvgPool2d(**args)

    def _output_shape(self) -> Tuple[int, ...]:
        # NOTE: currently assumes padding, kernels, dilation are all symmetric (e.g. `(n,n)`)
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        H, W = self.input_shape[1], self.input_shape[2]
        K = self.args.arg_values["kernel_size"]
        P = self.args.arg_values["padding"]
        S = self.args.arg_values["stride"]
        C_out = self.out_channels
        H_out = int(np.floor(((H + 2 * P - K) / S) + 1))
        W_out = int(np.floor(((W + 2 * P - K) / S) + 1))
        return (C_out, H_out, W_out)


class AdaptiveMaxPool2d(Layer, PyTorch):
    def __init__(self) -> None:
        raise NotImplementedError()


class AdaptiveAvgPool2d(Layer, PyTorch):
    def __init__(self) -> None:
        raise NotImplementedError()


class FractionalMaxPool2d(Layer, PyTorch):
    def __init__(self) -> None:
        raise NotImplementedError()


IMPLEMENTED = [MaxPool2d, AveragePool2d]
