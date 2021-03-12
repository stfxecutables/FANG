from __future__ import annotations  # noqa

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union, cast, no_type_check

import numpy as np
import torch as t
from torch import Tensor
from torch.nn import Conv2d, Flatten, Linear, MaxPool2d, Module, ModuleList
from typing_extensions import Literal

from src.interface.initializer import PyTorch

_Reduction = Literal["channels", "pooling", "1x1", "strided"]
Reduction = Union[None, _Reduction, List[_Reduction]]

VALID_REDUCTIONS = [
    ["channels", "pooling"],
    ["pooling", "channels"],
    ["1x1", "pooling"],
    ["pooling", "1x1"],
]


def _conv_output_shape(
    spatial_in: Tuple[int, int],
    C_out: int,
    kernel: int,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 0,
) -> Tuple[int, int, int]:
    # NOTE: currently assumes padding, kernels, dilation are all symmetric (e.g. `(n,n)`)
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    H, W = spatial_in
    D, K, P, S = dilation, kernel, padding, stride
    H_out = int(np.floor(((H + 2 * P - D * (K - 1) - 1) / S) + 1))
    W_out = int(np.floor(((W + 2 * P - D * (K - 1) - 1) / S) + 1))
    return (C_out, H_out, W_out)


class GlobalAveragePooling(Module):
    def __init__(self, input_shape: Tuple[int, ...], channel_dim: int = 0):
        super().__init__()
        self._channel_dim = channel_dim
        self._input_shape = input_shape
        output_shape = list(input_shape)
        output_shape[channel_dim] = 1
        self._output_shape = tuple(output_shape)
        # NOTE: during runtime we have a batch_size dimension in front!!!
        self._runtime_channel_dim = 1

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        return t.mean(x, dim=self._runtime_channel_dim, keepdim=True)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(input_shape={str(self._input_shape)}, reducing_dim={self._runtime_channel_dim})"

    __repr__ = __str__


class FinalLinear2d(Module):
    """Layer to handle the final 2d feature map for classification

    Parameters
    ----------
    input_shape: Tuple[int, ...]
        Input shape WITHOUT batch_size, (C, H, W).

    """

    def __init__(self, input_shape: Tuple[int, ...], n_classes: int, channel_dim: int = 0):
        super().__init__()
        self._channel_dim = channel_dim
        self._runtime_channel_dim = 1
        self._input_shape = input_shape
        self._n_classes = n_classes

        output_shape = list(input_shape)
        output_shape[channel_dim] = 1
        self._output_shape = tuple(output_shape)

        self.flatten = Flatten()
        self.linear = Linear(in_features=np.prod(input_shape), out_features=n_classes)

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.linear(x)
        return x


class Output(Module):
    def __init__(self, layers: List[Module]) -> None:
        super().__init__()
        self.layers = ModuleList(layers)

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ClassificationOutput(PyTorch):
    """Assumes task is `classification`. Not this layer CANNOT be evolved.

    Parameters
    ----------
    input_shape: Tuple[int, ...]
        Size incoming to this layer. Must be in a "channels first" format, i.e.:

            (n_channels, height) = (n_channels, sequence_length) for Conv1d
            (n_channels, height, width) ........................ for Conv2d
            (n_channels, height, width, depth) ................. for Conv3d

        We require this to calculate the output shape, which we require for use in the construction
        of subsequent layers.

    n_classes: int
        Number of classes for final linear layer

    reduction: "channels" | "pooling" | "1x1" | "strided" | None | List[str]
        How to reduce the incoming tensors to ensure reshaping does not create an inordinately
        expensive Linear layer:

        If "channels", use global average pooling along the channel dimension to reduce channel
        dimensions to one.

        If "pooling", use one MaxPooling layer (symmetric with kernel size 2 and stride 2) to reduce
        the size of the spatial dimensions by a factor of two.

        If "1x1", use a convolution with filter size 1 in all dimensions to reduce channels.
        Mutually exclusive with "channels". Currently unimplemented.

        If "strided", use a convolution with kernel of size 2, stride 2, and out_channels=1 to
        perform all reductions at once in a learnable manner. Cannot be used with other options.

        If None, perform immediately reshape the input without any reduction.

        If a List of the above, combine the above methods (subject to incompatiable combinations).
        Reductions will be performed in the order given in the list.
    """

    def __init__(
        self, input_shape: Tuple[int, ...], n_classes: int, reduction: Reduction = "channels"
    ) -> None:
        self.torch: Optional[Module] = None
        self.input_shape = input_shape
        self.in_channels = input_shape[0]
        self.output_shape = (n_classes,)
        self.out_channels = n_classes

        self.reductions = self._validate_reduction(reduction)
        self.constructor_args = dict(
            input_shape=input_shape, n_classes=n_classes, reduction=reduction
        )

    @staticmethod
    def _validate_reduction(reduction: Reduction) -> Optional[List[_Reduction]]:
        if reduction is None or reduction is [None]:
            return None
        reductions = reduction if isinstance(reduction, list) else [reduction]
        if None in reductions:
            raise ValueError("Can't have None and other reduction options.")
        if len(set(reductions)) != len(reductions):
            raise ValueError("List of `reduction` methods contains duplicates")
        cleaned = list(filter(lambda s: s in ["channels", "pooling", "1x1", "strided"], reductions))
        if len(cleaned) != len(reductions):
            raise ValueError("`reduction` list contains invalid values")
        if len(cleaned) > 1:
            if cleaned not in VALID_REDUCTIONS:
                raise ValueError("Invalid combination of reduction options.")
        return cleaned

    def create(self) -> None:
        """Build the Torch Module depending on the options"""
        layers = self.__build_layers()
        self.torch = cast(Module, Output(layers))

    def clone(self) -> ClassificationOutput:
        cloned = ClassificationOutput(**self.constructor_args)  # type: ignore
        # cloned.torch = self.torch  # don't do this for now
        cloned.input_shape = deepcopy(self.input_shape)
        cloned.in_channels = deepcopy(self.in_channels)
        cloned.output_shape = deepcopy(self.output_shape)
        cloned.out_channels = deepcopy(self.out_channels)
        cloned.reductions = deepcopy(self.reductions)
        return cloned

    def __build_layers(self) -> List[Module]:
        # Our options currently are:
        # - None
        # - length 1
        #    - [<an_option>]
        # - length 2
        #    - ["channels", "pooling"]
        #    - ["pooling", "channels"]
        #    - ["1x1", "pooling"]
        #    - ["pooling", "1x1"]
        layers: List[Module]
        if self.reductions is None:
            layers = [FinalLinear2d(self.input_shape, self.out_channels)]
        elif len(self.reductions) == 1:
            reducer, reducer_outshape = self.__build_reducer(self.reductions[0], self.input_shape)
            layers = [reducer, FinalLinear2d(reducer_outshape, self.out_channels)]
        elif len(self.reductions) == 2:
            if self.reductions not in VALID_REDUCTIONS:
                raise ValueError("Invalid reduction combo.")
            reducer0, reducer_outshape0 = self.__build_reducer(self.reductions[0], self.input_shape)
            reducer1, reducer_outshape1 = self.__build_reducer(
                self.reductions[1], reducer_outshape0
            )
            layers = [reducer0, reducer1, FinalLinear2d(reducer_outshape1, self.out_channels)]
        else:
            raise ValueError("Invalid reduction options [should be unreachable].")
        return layers

    def __build_reducer(
        self, reduction: Reduction, input_shape: Tuple[int, ...]
    ) -> Tuple[Module, Tuple[int, ...]]:
        """Build the specific reducing layer, and return it and its output shape

        Returns
        -------
        reducing_layer: Module

        out_shape: Tuple[int, ...]
            The shape the reducing layer output, (C, H, W), not including batch_size

        """
        if reduction == "channels":
            pooling: Module = GlobalAveragePooling(input_shape=input_shape)
            return pooling, pooling._output_shape
        elif reduction == "pooling":
            pooling = MaxPool2d(2, 2)
            spatial = (input_shape[1], input_shape[2])
            pool_outshape = _conv_output_shape(spatial, self.out_channels, 2, 2)
            return pooling, pool_outshape
        elif reduction == "1x1":
            if len(input_shape) != 3:
                raise NotImplementedError("Only 2D networks currently implemented.")
            reducer = Conv2d(in_channels=self.input_shape[0], out_channels=1, kernel_size=1)
            outshape = (1,) + input_shape[1:]
            return reducer, outshape
        elif reduction == "strided":
            in_ch = self.input_shape[0]
            spatial = (self.input_shape[1], self.input_shape[2])
            strider = Conv2d(in_channels=in_ch, out_channels=1, kernel_size=2, stride=2)
            strider_out = _conv_output_shape(spatial, 1, 2, 2)
            return strider, strider_out
        else:
            raise ValueError("Invalid reduction option.")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ClassificationOutput": {
                "created": self.torch is not None,
                "input_shape": self.input_shape,
                "in_channels": self.in_channels,
                "output_shape": self.output_shape,
                "out_channels": self.out_channels,
                "reductions": self.reductions,
                "constructor_args": self.constructor_args,
            }
        }

    def __str__(self) -> str:
        header = f"{self.__class__.__name__} interface"
        arg_info = []
        for argname, val in self.constructor_args.items():
            arg_info.append(f"{argname}={val}")
        arg_info = ", ".join(arg_info)
        arg_info = f"({arg_info})" if arg_info != "" else ""
        io_info = f"{self.input_shape} -> {self.output_shape}"
        info = f"{header}  {arg_info}\r\n   {io_info}"
        return info

    def __repr__(self) -> str:
        return str(self)
