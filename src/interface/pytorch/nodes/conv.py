from __future__ import annotations  # noqa

from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.nn import Module

from src.exceptions import VanishingError
from src.interface.arguments import Arguments
from src.interface.initializer import PyTorch
from src.interface.layer import ReshapingLayer


class Conv(ReshapingLayer):
    """As both Tensorflow and Torch convolution layers tend to be pretty similar in implementation,
    it makes sense to have another abstract base Conv class. For Torch, this is the base class for
    torch.nn.ConvNd, and torch.nn.ConvTransposeNd classes.

    Parameters
    ----------
    input_shape: Tuple[int, ...]
        Must be in a "channels first" format, i.e.:

            (n_channels, height) = (n_channels, sequence_length) for Conv1d
            (n_channels, height, width) ........................ for Conv2d
            (n_channels, height, width, depth) ................. for Conv3d

    We require this to calculate the output shape, which we require for use in the construction of
    subsequent layers.

    out_channels: int
        Number of output_channels. Corresponds to `filters` in Tensorflow.

    Properties
    ----------
    output_shape: Tuple[int, ...]
        The resulting output dimensions in "channels first" format, i.e.:

            (n_channels, height) = (n_channels, sequence_length) for Conv1d
            (n_channels, height, width) ........................ for Conv2d
            (n_channels, height, width, depth) ................. for Conv3d
    """

    MAX_OUT_CHANNELS = 512
    MAX_PAD = 5  # TODO: tie to kernel_size, string, padding, and dilation instead
    MAX_DILATE = 5
    MAX_KERNEL = 5
    MAX_STRIDE = 2
    PADDING_MODES = ("zeros", "reflect", "replicate", "circular")

    ARGS = {
        "out_channels": ("int", (1, MAX_OUT_CHANNELS)),  # will be over=written, just defaults
        "kernel_size": ("int", (1, MAX_KERNEL)),
        "stride": ("int", (1, MAX_STRIDE)),
        "padding": ("int", (0, MAX_PAD)),
        "dilation": ("int", (1, MAX_DILATE)),
        "padding_mode": ("enum", PADDING_MODES),
        # # we likely wish to ignore the below two for now. Why?
        # # "bias" should always be true (can always learn to set bias to zero)
        # # "groups" is solely to improve efficiency, and no one uses this otherwise
        # "bias": ("bool", None),
        # "groups": ("int", (1, 12)),
    }

    def __init__(self, input_shape: Tuple[int, ...], **layer_kwargs: Any):
        self.torch: Optional[Module] = None
        self.input_shape = input_shape
        self.in_channels = input_shape[0]

        # by default, `out_channels` is random initially
        channel_argvals = {"in_channels": input_shape[0]}
        # order below is IMPORTANT, want to update OUT with ARGS
        self.args = Arguments({**self.OUT, **self.ARGS})
        self.args.arg_values.update(channel_argvals)

        # Handle bad padding sizes given a kernel size:
        # pad should be smaller than half of kernel size, but got padW = 2, padH = 2, kW = 1, kH = 1
        P, K = self.args.arg_values["padding"], self.args.arg_values["kernel_size"]
        max_pad = K // 2
        if max_pad == 0:
            self.args.arg_values["padding"] = 0
        elif P >= max_pad:
            self.args.arg_values["padding"] = int(np.random.randint(0, max_pad))

        self.out_channels = self.args.arg_values["out_channels"]
        self.output_shape = self._output_shape()

    def _output_shape(self) -> Tuple[int, ...]:
        # NOTE: currently assumes padding, kernels, dilation are all symmetric (e.g. `(n,n)`)
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        H, W = self.input_shape[1], self.input_shape[2]
        D = self.args.arg_values["dilation"]
        K = self.args.arg_values["kernel_size"]
        P = self.args.arg_values["padding"]
        S = self.args.arg_values["stride"]
        C_out = self.out_channels
        H_out = int(np.floor(((H + 2 * P - D * (K - 1) - 1) / S) + 1))
        W_out = int(np.floor(((W + 2 * P - D * (K - 1) - 1) / S) + 1))
        return (C_out, H_out, W_out)


class Conv2d(Conv, PyTorch):
    def create(self) -> None:
        """Make the actual Torch Layer object, and save it for later"""
        if self.torch is not None:  # type: ignore
            return
        self.torch = torch.nn.Conv2d(**self.args.arg_values)


class UpConv2d(Conv, PyTorch):
    # NOTE: You need to be very careful to set small values here or you blow up your input sizes and
    # get a a huge / slow network that a priori we know will perform poorly
    MAX_OUT_PADDING = 5
    MAX_KERNEL = 2
    MAX_DILATE = 2
    MAX_STRIDE = 2
    KERNEL_SIZE = ("int", (1, MAX_KERNEL)) if MAX_KERNEL > 2 else ("enum", (2, 2))
    DILATION = ("int", (1, MAX_DILATE)) if MAX_DILATE > 2 else ("enum", (1, 2))
    STRIDE = ("int", (1, MAX_STRIDE)) if MAX_STRIDE > 2 else ("enum", (1, 2))

    ARGS = {
        "in_channels": ("int", (1, Conv.MAX_IN_CHANNELS)),  # will be over-written, just defaults
        "out_channels": ("int", (1, Conv.MAX_OUT_CHANNELS)),  # will be over=written, just defaults
        "kernel_size": KERNEL_SIZE,
        "stride": STRIDE,
        "padding": ("int", (0, Conv.MAX_PAD)),
        "dilation": DILATION,
        "output_padding": ("int", (0, MAX_OUT_PADDING)),
    }

    def __init__(self, input_shape: Tuple[int, ...], **layer_kwargs: Any):
        self.torch: Optional[Module] = None
        self.input_shape = input_shape
        self.in_channels = input_shape[0]

        self.args = Arguments({**self.OUT, **self.ARGS})
        channel_argvals = {"in_channels": input_shape[0]}
        self.args.arg_values.update(channel_argvals)

        # We need to handle some special cases relating to padding, since not all padding options
        # are actually valid. In particular, you will otherwise frequently see:
        # RuntimeError: output padding must be smaller than either stride or dilation
        stride = self.args.arg_values["stride"]
        dilation = self.args.arg_values["dilation"]
        max_valid_pad = np.max([stride, dilation]) - 1
        self.args.arg_values["output_padding"] = (
            int(np.random.randint(0, max_valid_pad + 1)) if max_valid_pad > 1 else 0
        )

        self.out_channels = self.args.arg_values["out_channels"]
        self.output_shape = self._output_shape()

    def _output_shape(self) -> Tuple[int, int, int]:
        # NOTE: we of course have to override this each time
        H, W = self.input_shape[1], self.input_shape[2]
        D = self.args.arg_values["dilation"]
        K = self.args.arg_values["kernel_size"]
        P = self.args.arg_values["padding"]
        S = self.args.arg_values["stride"]
        OP = self.args.arg_values["output_padding"]

        C_out = self.out_channels
        H_out = int((H - 1) * S - 2 * P + D * (K - 1) + OP + 1)
        W_out = int((W - 1) * S - 2 * P + D * (K - 1) + OP + 1)
        if H_out <= 0 or W_out <= 0:
            raise VanishingError("UpConv2d will produce negative-sized outputs.")
        return (C_out, H_out, W_out)

    def create(self) -> None:
        """Make the actual Torch Layer object, and save it for later"""
        if self.torch is not None:  # type: ignore
            return
        self.torch = torch.nn.ConvTranspose2d(**self.args.arg_values)


# IMPLEMENTED = [Conv2d, UpConv2d]
IMPLEMENTED = [Conv2d]
