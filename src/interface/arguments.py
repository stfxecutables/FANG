from __future__ import annotations  # noqa

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
from typing_extensions import Literal

from src.interface.evolver import Evolver

"""Much of the work to be done interfacing with PyTorch or Tensorflow will be just calling to their
library functions with sane and reasonable arguments. For example, constructing random layers is
almost entirely just referencing a set of arguments and their valid values in order to create random
but valid *args and **kwargs values and name-value pairs.

In addition, arguments in PyTorch and Tensorflow can *usually* be divided into just a few simple
classes based on the range of inputs (domain):

1. Boolean Flags
2. Enum (finite set of enumerable values, e.g. strings, or set of flags)
3. Int
4. Float

So for example in Tensorflow there is `model.fit(..., verbose=<enum>)`, because there are only three
possible values to `verbose`, which are `(0, 1, 2)`. This looks like the same as an integer range
(and is in this case) but should be kept conceptually separate.

Later, we may want to implement a TupleArg class, because certain arguments (e.g. `kernel_size`)
allow for Tuple[int, ...] arguments. This could be a trivial later addition.
"""

ArgKind = Literal["bool", "enum", "int", "float"]
ArgMutation = Literal["perturb", "random"]
ArgVal = Union[bool, float, int, str]
BoolDomain = Tuple[bool, bool]
EnumDomain = Union[Tuple[str, ...], Tuple[int, ...]]
IntDomain = Tuple[int, int]
FloatDomain = Tuple[float, float]
Domain = Union[BoolDomain, EnumDomain, IntDomain, FloatDomain]


class ArgDef:
    """ArgDef class for defining the structure of a function argument. This makes some simplifying
    assumptions for now (see Notes).

    Parameters
    ----------
    name: str
        The argument name, which should should `eval` to the exact argument being encoded. E.g. the
        function:

            def foo(arg1: int = 3) -> None:
                pass

        should get `name="arg1"`.

    kind: Literal["bool", "enum", "int", "float"] = int
        The argument kind. Enums are defined above.

    domain: Domain = None
        If `kind="bool"`, you may leave this unspecified as None.
        if `kind="enum"`, domain should be a Tuple or List of int or str.
        if `kind="int"`, domain should be a Tuple of TWO ints (min, max), which specify the range
        `[min, max)`, i.e. the set `[min, min + 1, ..., max - 1]` that EXCLUDES `max`
        if `kind="float"`, domain should be a Tuple of TWO floats (min, max), which specify the
        range `[min, max)`, i.e. the set `min <= x < max` that EXCLUDES `max`

    Notes
    -----
    Currently, this doesn't allow encoding an argument like `kernel_size=[3, 5]`. However, it would
    be trivial to modify this function to do this, and such logic should be implemented *here*
    rather than at each layer.
    """

    def __init__(self, name: str, kind: ArgKind = "int", domain: Optional[Domain] = None) -> None:
        self.name = name
        self.dtype: Any = None
        self.kind, self.domain = self.validate_args(kind, domain)

    def clone(self) -> ArgDef:
        return deepcopy(self)  # below is safe since we only use basic Python types here

    def random_value(self) -> Union[bool, str, float, int]:
        """Returns a random valid parameter value of the appropriate kind / type."""
        if self.kind == "enum":
            return cast(ArgVal, np.random.choice(self.domain).tolist())
        elif self.kind == "bool":
            return bool(np.random.binomial(1, 0.5))
        elif self.kind == "int":
            return int(np.random.randint(self.domain[0], self.domain[1]))
        elif self.kind == "float":
            return float(np.random.uniform(self.domain[0], self.domain[1]))
        else:
            raise ValueError(f"Validation error. Invalid `self.kind`: {self.kind}")

    def validate_kind(self, kind: str) -> ArgKind:
        kind = kind.lower()
        for substr in ["enum", "bool", "int", "float"]:
            if substr in kind:
                return cast(ArgKind, substr)
        raise ValueError("Invalid argument kind. Must be one of: 'enum', 'bool', 'int', 'float'.")

    def validate_args(self, kind: str, domain: Optional[Domain]) -> Tuple[ArgKind, Domain]:
        kind = self.validate_kind(kind)
        if domain is None:
            if kind == "bool":
                self.dtype = bool
                return kind, (True, False)
            else:
                raise ValueError("`domain` must be specified for any argument kind except `bool`.")

        # yes, we are just using numpy to flatten...
        try:
            dom = np.array(domain)
        except BaseException as e:
            raise ValueError(
                "Can't interpret domain as simple iterable. Use a tuple or list."
            ) from e
        if dom.ndim > 1:
            raise ValueError("The `domain` cannot be a nested structure. Use a flat tuple or list")
        if len(dom) <= 1:
            raise ValueError("Domains must be at minimum two values.")

        domain = tuple(dom.tolist())

        if kind == "enum":
            for d in domain:
                if not (isinstance(d, str) or isinstance(d, int)):
                    raise ValueError("enum / flag types can be only either strings or ints.")
            self.dtype = Union[str, int]
            return kind, domain
        elif kind == "int":
            if not isinstance(domain[0], int) or len(domain) != 2:
                raise ValueError("`int` argument kind must be a Tuple of two ints (min, max).")
            return kind, domain
        elif kind == "float":
            if not isinstance(domain[0], float) or len(domain) != 2:
                raise ValueError("`float` argument kind must be a Tuple of two floats [min, max).")
            if not np.all(np.isfinite(domain)):
                raise ValueError("if `kind='float', domain must contain only finite values.")
            return kind, domain
        else:
            raise ValueError(f"Invalid kind: {kind}")


class Arguments(Evolver):
    """Just a helper class that lets you construct arguments and their ranges with simple literals.
    E.g. to construct the arguments for a function

        def random_digit_plus_noise(integer: int, noise: float, verbose: int) -> float:
            if integer < 1 or integer > 9: raise ValueError("Not a digit!")
            if noise > 5: raise ValueError("Too noisy!")
            if np.abd(verbose) > 2: raise ValueError("Too verbose!")

            if np.abs(verbose) <= 2: print(f"Verbosity level {verbose}")
            return integer + np.random.normal(0, noise)

    You just write:

        args = Arguments({
            "integer": ("int", (1, 10)),
            "noise": ("float", (0.0, 5.0)),
            "verbose": ("enum", (0, 1, 2)),
        })

    Convenient, no?

    Parameters
    ----------
    argsdict: Dict[str, Tuple[ArgKind, Domain]]

    Properties
    ----------
    arg_values: Dict[str, ArgVal]
        Dictionary indexed by argument names, with the actual random argument value as the value.

    arg_defintions: Dict[str, ArgDef]
        Dictionary indexed by argument names, with the ArgDef as value.

    """

    def __init__(self, argsdict: Dict[str, Tuple[ArgKind, Domain]]):
        super().__init__()
        self.arg_values: Dict[str, Optional[ArgVal]] = {}
        self.arg_definitions: Dict[str, ArgDef] = {}
        for argname, (kind, domain) in argsdict.items():
            self.arg_definitions[argname] = ArgDef(argname, kind, domain)
            self.arg_values[argname] = None
        self._randomize_argvals()

    def clone(self) -> Arguments:
        """Get an identical but indepenent copy of the Arguments"""
        cloned = Arguments({})
        cloned.arg_values = deepcopy(self.arg_values)
        cloned.arg_definitions = deepcopy(self.arg_definitions)
        return cloned

    def mutate(
        self, probability: float, method: ArgMutation = "random", mutate_channels: bool = False
    ) -> Arguments:
        """Summary

        Parameters
        ----------
        probability: float
            Probability of mutation *each argument*. Must be in `[0,1]`

        method: "random" | "perturb"
            If "random", generate a new random valid value for each argument.
            If "perturb", move float argument values by a random value within 10% of the arg range.

        mutate_channels: bool = False
            If False (default) do not mutate arguments with the name "in_channels" or "out_channels"

        Returns
        -------
        mutated: Arguments
            A clone with mutated arguments.
        """
        if method != "random":
            raise NotImplementedError("Currently only 'random' mutation available for Arguments.")
        cloned = self.clone()
        for argname in cloned.arg_values.keys():
            # only mutate IO arguments if explicitly asked for
            if argname in ["in_channels", "out_channels"] and not mutate_channels:
                continue
            # if we should mutate, replace that value with a new valid value
            if np.random.uniform() < probability:
                newval = cloned.arg_definitions[argname].random_value()
                cloned.arg_values[argname] = newval
        return cloned

    def _randomize_argvals(self) -> None:
        self.arg_values = {
            argname: arg.random_value()
            for argname, arg in self.arg_definitions.items()  # type: ignore
        }
