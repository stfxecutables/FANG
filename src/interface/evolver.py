from __future__ import annotations  # noqa

from abc import ABC, abstractmethod
from typing import Any


class Evolver(ABC):
    """Class for Layer, Individual, Trainer, Population to all inherit from

    Note
    ----
    We don't define crossover to be part of the general interface, because crossover makes sense
    only for Individuals."""

    @abstractmethod
    def clone(self, *args: Any, **kwargs: Any) -> Evolver:
        """If you think about it, you're probably not going to want to just alter the values of our
        EA objects in place, because Torch/Tensorflow will store all sorts of additional data in
        these objects (e.g. Layers). This data might be expensive to work with, and we may or may
        not want to pass over that data to clones. Thus we probably in general want to clone first
        and *then* mutate, crossover, or whatever, always working with the clones to avoid having
        all sorts of differently-intialized state to worry about.

        At bottom, all layers can be *specified* by a name (e.g. "Conv1d") and a dict of the
        argument names and values. Thus when we clone, at core we are just cloning these dicts and
        strings."""

    @abstractmethod
    def mutate(self, probability: float = 0.0, *args: Any, **kwargs: Any) -> Evolver:
        """For Layer and Trainer classes, this function will actually need implementation. But for
        the container classes (Individual, Population) this function will just loop over the units
        and call their respective `mutate` functions."""
