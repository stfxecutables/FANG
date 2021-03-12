from __future__ import annotations  # noqa

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

# NOTE: If we were allowing both Tensorflow and Torch here, we would define
# Module = Union[torch.nn.Module, tensorflow.Module]
# Sequential = Union[torch.nn.Sequential, tensorflow.keras.Sequential]
from typing_extensions import Literal

from src.interface.arguments import ArgMutation, Arguments, ArgVal
from src.interface.evolver import Evolver
from src.interface.initializer import PyTorch

NetSize = Union[int, Tuple[int, int], Literal["random"]]  # something to think about as well
# the final output layers of the network can't be truly random, as they require an exact shape and
# activation for the task at hand
Data = Dict[str, Tensor]  # holds train, test, val


# Activations are special since they work for any data dimensionality
# Also note there are nearly 30 activations available in PyTorch: we use only a sampling here
ACTIVATIONS = [nn.LeakyReLU, nn.ReLU, nn.Tanh, nn.ELU, nn.Hardswish]
LAYERS_2D = {
    "conv": [nn.Conv2d, nn.ConvTranspose2d],
    "pool": [nn.MaxPool2d, nn.MaxUnpool2d, nn.AvgPool2d],
    "pad": [nn.ReflectionPad2d, nn.ReplicationPad2d, nn.ConstantPad2d],
    "norm": [nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d],
    "drop": [nn.Dropout2d],
    "linear": [nn.Linear, nn.Bilinear],
}

# just a helper function used later
def rand(maximum: int) -> int:
    return int(np.random.randint(1, maximum + 1))


class Layer(Evolver, PyTorch):
    """
    Parameters
    ----------
    input_shape: Tuple[int, ...]
        Must be in a "channels first" format, i.e.:

            (n_channels, height) = (n_channels, sequence_length) for Conv1d
            (n_channels, height, width) ........................ for Conv2d
            (n_channels, height, width, depth) ................. for Conv3d

    We require this to calculate the output shape, which we require for use in the construction of
    subsequent layers.

    Properties
    ----------
    input_shape: int = None
        The layer required input shape.

    output_shape: int = None
        The layer required input shape.
    """

    MAX_IN_CHANNELS = 512
    MAX_OUT_CHANNELS = 512
    ARGS = {}  # type: ignore # no args by default

    def __init__(self, input_shape: Tuple[int, ...], **layer_kwargs: Any):
        self.torch: Optional[Module] = None
        self.input_shape = self.output_shape = input_shape
        self.in_channels = self.out_channels = input_shape[0]

        # by default, `out_channels` is random initially
        channel_argvals = {"in_channels": input_shape[0]}
        # order below is IMPORTANT, want to update OUT with ARGS
        self.args = Arguments(self.ARGS)
        self.args.arg_values.update(channel_argvals)

    def as_dict(self) -> Dict[str, Any]:
        return {
            self.__class__.__name__: {
                "created": self.torch is not None,
                "input_shape": self.input_shape,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "output_shape": self.output_shape,
                "args": self.args.as_dict(),
            }
        }

    def clone(self) -> Layer:
        cloned = self.__class__(self.input_shape)
        cloned.args = self.args.clone()
        output_shape = getattr(self, "output_shape", None)
        if output_shape is not None:
            cloned.output_shape = output_shape
        return cloned

    def mutate(
        self, probability: float, method: ArgMutation = "random", mutate_channels: bool = False
    ) -> Layer:
        cloned = self.clone()
        cloned.args = cloned.args.mutate(
            probability=probability, method=method, mutate_channels=mutate_channels
        )
        return cloned

    def _cleaned_args(
        self, remove: Union[str, List[str]] = ["in_channels", "out_channels"]
    ) -> Dict[str, ArgVal]:
        args = self.args.clone().arg_values
        if isinstance(remove, str):
            remove = [remove]
        for argname in remove:
            if argname in args:
                del args[argname]
        return cast(Dict[str, ArgVal], args)

    def __str__(self) -> str:
        header = f"{self.__class__.__name__} interface"
        arg_infos = []
        for argname, val in self.args.arg_values.items():
            if argname == "in_channels":
                continue
            domain = self.args.arg_definitions[argname].domain
            kind = self.args.arg_definitions[argname].kind
            if kind == "enum":
                domain = str(domain).replace("(", "{").replace(")", "}")
            elif kind == "float":
                val = "{:1.2e}".format(val)  # type: ignore
                domain = str(domain).replace("(", "[")
            elif kind == "int":
                domain = str(domain).replace("(", "[")
            elif kind == "bool":
                val = "True" if val else "False"
                domain = str(domain).replace("(", "{").replace(")", "}")
            # info.append(f"   {argname:<15}: {val:<10} in {domain}")
            arg_infos.append(f"{argname}={val} in {domain}")
        arg_info = ", ".join(arg_infos)
        arg_info = f"({arg_info})" if arg_info != "" else ""
        io_info = f"{self.input_shape} -> {self.output_shape}"
        info = f"{header}  {arg_info}\r\n   {io_info}"
        return info

    __repr__ = __str__


class ReshapingLayer(Layer):
    """Layer to inherit from when the output shape is usually different than the input shape

    Parameters
    ----------
    input_shape: Tuple[int, ...]
        The needed input size.

    Properties
    ----------
    input_shape: int = None
        The layer required input shape.

    output_shape: int = None
        The layer required input shape.
    """

    MAX_IN_CHANNELS = 512
    MAX_OUT_CHANNELS = 512
    ARGS = {}  # type: ignore # no args by default
    OUT = {"out_channels": ("int", (1, MAX_OUT_CHANNELS))}  # will be over=written, just defaults

    def __init__(self, input_shape: Tuple[int, ...], **layer_kwargs: Any):
        self.torch: Optional[Module] = None
        self.input_shape = input_shape
        self.in_channels = input_shape[0]

        # by default, `out_channels` is random initially
        channel_argvals = {"in_channels": input_shape[0]}
        # order below is IMPORTANT, want to update OUT with ARGS
        self.args = Arguments({**self.OUT, **self.ARGS})
        self.args.arg_values.update(channel_argvals)

        self.out_channels = self.args.arg_values["out_channels"]
        self.output_shape = self._output_shape()

    @abstractmethod
    def _output_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError("ResizeLayer classes must implement the `_output_shape` method")


class Output(Layer):
    pass
