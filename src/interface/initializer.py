from __future__ import annotations  # noqa

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from tensorflow import Module as TFModule
from torch.nn import Module as TorchModule
from typing_extensions import Literal

Framework = Literal["pytorch", "torch", "tensorflow", "tf"]
Module = Union[TorchModule, TFModule]


class PyTorch(ABC):
    """Mixin for classes that need to keep initilization and passing around separate """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.torch: Optional[Module] = None

    @abstractmethod
    def create(self) -> None:
        """Create the actual Torch / Tensorflow / Python objects from their specifications"""


class Tensorflow(ABC):
    """Mixin for classes that need to keep initilization and passing around separate """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.tensorflow: Optional[Module] = None

    @abstractmethod
    def create(self) -> None:
        """Create the actual Torch / Tensorflow / Python objects from their specifications"""
