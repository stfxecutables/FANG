from __future__ import annotations  # noqa

from copy import deepcopy
from typing import Tuple

import torch

from src.interface.initializer import PyTorch
from src.interface.layer import ReshapingLayer


class Linear(ReshapingLayer, PyTorch):
    def _output_shape(self) -> Tuple[int, ...]:
        shape = deepcopy(self.input_shape)[1:]
        shape = (int(self.out_channels),) + shape
        return shape  # type: ignore

    def create(self) -> None:
        if self.torch is not None:  # type: ignore
            return
        # for some random reason the Linear layer args are named differently
        args = deepcopy(self.args.arg_values)
        args["in_features"] = args["in_channels"]
        args["out_features"] = args["out_channels"]
        del args["in_channels"]
        del args["out_channels"]
        self.torch = torch.nn.Linear(**args)


IMPLEMENTED = [Linear]
