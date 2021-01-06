from __future__ import annotations  # noqa

from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer as TorchOptimizer

from src.interface.arguments import ArgKind, ArgMutation, Arguments, Domain
from src.interface.evolver import Evolver
from src.interface.initializer import PyTorch


class Optimizer(Evolver, PyTorch):
    ARGS: Dict[str, Tuple[ArgKind, Domain]] = {}
    OPTIMIZER: Any = TorchOptimizer

    def __init__(self) -> None:
        super().__init__()  # calls Intializer __init__
        self.torch: Optional[TorchOptimizer] = None
        self.args = Arguments(self.ARGS)

    def clone(self) -> Optimizer:
        cloned = self.__class__()
        cloned.args = self.args.clone()
        return cloned

    def mutate(
        self, probability: float = 0.0, method: ArgMutation = "random", *args: Any, **kwargs: Any
    ) -> Optimizer:
        mutated = self.clone()
        mutated.args = mutated.args.mutate(probability=probability, method=method)
        return mutated

    def create(self, params: Iterable) -> None:
        if self.torch is not None:
            return
        self.torch = self.OPTIMIZER(params, **self.args)

    def __str__(self) -> str:
        header = f"{self.__class__.__name__} interface"
        arg_info = []

        for argname, val in self.args.arg_values.items():
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
            arg_info.append(f"{argname}={val} in {domain}")
        arg_info = ", ".join(arg_info)
        arg_info = f"({arg_info})" if arg_info != "" else ""
        info = f"{header}  {arg_info}"
        return info

    def __repr__(self) -> str:
        return str(self)


class Adam(Optimizer):
    OPTIMIZER: Any = torch.optim.Adam
    ARGS = {
        "lr": ("float", (1e-9, 0.01)),  # very hard to choose a range here
        "beta0": ("float", (0.8, 0.99)),
        "beta1": ("float", (0.995, 0.9999)),  # at least ensure beta1 > beta0
        "weight_decay": ("float", (0, 0.5)),  # also hard to say what is good here
    }

    def create(self, params: Iterable) -> None:
        args = deepcopy(self.args.arg_values)
        args["betas"] = (args["beta0"], args["beta1"])
        del args["beta0"]
        del args["beta1"]
        self.torch = self.OPTIMIZER(params, **args)


class AdamW(Adam):
    OPTIMIZER = torch.optim.AdamW


IMPLEMENTED = [Adam, AdamW]
