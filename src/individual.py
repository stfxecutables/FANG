from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, no_type_check, Type
from copy import deepcopy

import numpy as np
import torch
from torch import Tensor as TorchTensor
from torch.nn import Module as TorchModule
from typing_extensions import Literal
from src.interface.arguments import ArgMutation

from src.exceptions import VanishingError
from src.interface.initializer import Framework
from src.interface.layer import Layer
from src.interface.pytorch.lightning.train import train_sequential
from src.interface.pytorch.nodes.activations import IMPLEMENTED as IMPLEMENTED_ACTIVATIONS
from src.interface.pytorch.nodes.conv import IMPLEMENTED as IMPLEMENTED_CONV
from src.interface.pytorch.nodes.drop import IMPLEMENTED as IMPLEMENTED_DROP
from src.interface.pytorch.nodes.linear import IMPLEMENTED as IMPLEMENTED_LINEAR
from src.interface.pytorch.nodes.norm import IMPLEMENTED as IMPLEMENTED_NORM
from src.interface.pytorch.nodes.output import ClassificationOutput
from src.interface.pytorch.nodes.pad import IMPLEMENTED as IMPLEMENTED_PAD
from src.interface.pytorch.nodes.pool import IMPLEMENTED as IMPLEMENTED_POOL
from src.interface.pytorch.optimizer import IMPLEMENTED as IMPLEMENTED_OPTIMS
from src.interface.pytorch.optimizer import Optimizer as TorchOptimizer

TORCH_NODES: List[Layer] = [
    *IMPLEMENTED_ACTIVATIONS,
    *IMPLEMENTED_CONV,
    *IMPLEMENTED_DROP,
    *IMPLEMENTED_LINEAR,
    *IMPLEMENTED_NORM,
    *IMPLEMENTED_PAD,
    *IMPLEMENTED_POOL,
]

# To incorporate random Linear layers we would also need to build in reshaping layers
TORCH_NODES_2D: List[Layer] = [
    *IMPLEMENTED_ACTIVATIONS,
    *IMPLEMENTED_CONV,
    *IMPLEMENTED_DROP,
    *IMPLEMENTED_NORM,
    *IMPLEMENTED_PAD,
    *IMPLEMENTED_POOL,
]
assert None not in TORCH_NODES
assert None not in TORCH_NODES_2D

TORCH_OPTIMIZERS: List[TorchOptimizer] = IMPLEMENTED_OPTIMS


Task = Literal["classification", "regression", "segmentation", "seq2seq"]


# We are doing this because unfortunately wrapping our layers in a
# Sequential model makes debugging impossible.
class IndividualModel(TorchModule):
    def __init__(self, layers: List[TorchModule], interfaces: List[Layer]) -> None:
        super().__init__()
        # without below debugging becomes impossible later
        self.interfaces = interfaces
        # ModuleList is needed to ensure parameters can be found
        self.layers = torch.nn.ModuleList(layers)

    @no_type_check
    def forward(self, x: TorchTensor) -> TorchTensor:
        for layer, interface in zip(self.layers, self.interfaces):
            x = layer(x)
            if x.shape[2:] != interface.output_shape[1:]:
                raise RuntimeError(
                    f"Interface {interface} shape is out of sync with actual shape {x.shape}"
                )
        return x

    def clone(self) -> IndividualModel:
        layers = [deepcopy(layer) for layer in self.layers]
        interfaces = [interface.clone() for interface in self.interfaces]
        cloned = IndividualModel(layers, interfaces)
        return cloned

    def __str__(self) -> str:
        info = []
        for i, layer in enumerate(self.layers):
            info.append(f"({i}) {str(layer)}")
        return "\r\n".join(info)

    __repr__ = __str__


class Individual:
    """Build an Individual (a network + optimizer).

    Parameters
    ----------
    n_nodes: int
        How many nodes (NOT counting the input and output nodes).

    task: "classification" | "regression" | "segmentation"
        The kind of network to build.

    input_shape: Tuple[int, ...]
        Must be in a "channels first" format, i.e.:

            (n_channels, height) = (n_channels, sequence_length) for Conv1d
            (n_channels, height, width) ........................ for Conv2d
            (n_channels, height, width, depth) ................. for Conv3d

        We require this to calculate the output shape, which we require for use in the construction
        of subsequent layers.

    output_shape: Union[int, Tuple[int, ...]]
        If `task="classification"`, then `output_shape` should be an `int` equal to `n_classes`.

        Must be in a "channels first" format, i.e.:

            (n_channels, height) = (n_channels, sequence_length) for Conv1d
            (n_channels, height, width) ........................ for Conv2d
            (n_channels, height, width, depth) ................. for Conv3d

        We require this to set the final output layer.
        of subsequent layers.

    sequential: bool = True
        If True (default), generate a network where the computational graph is a straight line and
        there are no skip connections (i.e. every node except the input and output nodes have
        exactly one parent and one child).

        If False, first generate a sequential network, and then add random skip connections between
        some nodes. [Currently not implemented].

    activation_interval: int = 0
        If greater than zero, require a non-linear activation to be placed after
        `activation_interval` non-activation layers. E.g. if `activation_interval=2`, there can be
        at maximum two consecutive non-activation layers before a random activation layer is
        forcefully inserted. We may wish to require semi-regular activations because e.g. two Conv
        layers just compose to a single linear function, which is an inefficiency.

    framework: "pytorch" | "Tensorflow"
        Which framework to use for instantiating and training the models.

    Returns
    -------
    val1: Any
    """

    def __init__(
        self,
        n_nodes: int,
        task: Task,
        input_shape: Union[int, Tuple[int, ...]],
        output_shape: Union[int, Tuple[int, ...]],
        sequential: bool = True,
        activation_interval: int = 0,
        framework: Framework = "pytorch",
    ) -> None:
        if task.lower() != "classification":
            raise NotImplementedError("Currently only classification supported.")
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        if len(input_shape) != 3:
            raise NotImplementedError("Currently only 2D data with format (C, H, W) supported.")
        if framework not in ["pytorch", "torch"]:
            raise NotImplementedError("Currently only Torch is implemented.")
        if not sequential:
            raise NotImplementedError("Currently only Sequential models are supported.")
        output_shape = (output_shape,) if isinstance(output_shape, int) else output_shape

        self.n_nodes: int = n_nodes
        self.task: Task = task
        self.input_shape: Tuple[int, ...] = input_shape
        self.output_shape: Tuple[int, ...] = output_shape
        self.is_sequential: bool = sequential
        self.activation_interval: int = activation_interval
        self.framework: Framework = framework

        self.layers: List[Layer] = self.create_random_nodes()
        self.output_layer: Layer = self.create_output_layer(self.layers)
        self.torch_model: IndividualModel = self.realize_model()
        self.optimizer: TorchOptimizer = np.random.choice(TORCH_OPTIMIZERS)()

        self.fitness: Optional[float] = None

    def __copy__(self) -> str:
        # NOTE: DO NOT MODIFY THIS FUNCTION
        raise RuntimeError(
            "You may not copy Individual objects directly. Use `Individual.clone()` if you need "
            "a deep copy, or just assign a reference instead if you need a shallow copy."
        )

    def __deepcopy__(self, memo: Dict) -> None:
        # NOTE: DO NOT MODIFY THIS FUNCTION
        raise RuntimeError(
            "Individual objects cannot be copied with `deepcopy`. "
            "Use `Individual.clone()` instead."
        )

    def __str__(self) -> str:
        info = ["\r\n"]
        for i, layer in enumerate(self.layers):
            info.append(f"({i}) {str(layer)}")
            info.append("\r\n")
        info.append(f"({len(self.layers)}) {str(self.output_layer)}")
        info.append("\r\n")
        info.append(f"(Optimizer) {self.optimizer}")
        return "".join(info)

    __repr__ = __str__

    def __eq__(self, o: object) -> bool:
        # TODO: Do better than this!
        return str(self) == str(o)

    def evaluate_fitness(self) -> None:
        results = train_sequential(self.torch_model, self.optimizer)
        # Results is a dict with keys:
        # {'test_acc', 'test_loss', 'val_acc', 'val_loss'}
        self.fitness = results["test_acc"]

    def create_random_nodes(self) -> List[Layer]:
        """Select random layer interfaces from implemented interfaces, create actual instances of
        those interfaces (ensuring outputs have not shrunk to zero size), and additionally create
        the torch realizations
        """
        realized_layers: List[Layer] = []
        # + 1 for input node
        layers: List[Type[Layer]] = np.random.choice(
            TORCH_NODES_2D, size=self.n_nodes + 1, replace=True
        )
        prev: Layer = layers[0]

        # loop over the random selection of layers and make sure input and output shapes align
        for i, layer in enumerate(layers):
            # Special handling of input layer. This is in fact the most important step
            node = layer(input_shape=self.input_shape if i == 0 else prev.output_shape)
            for size in node.output_shape[1:]:
                if size <= 0:
                    raise VanishingError("Convolutional layers have reduced output to zero size.")
            # node.create()
            prev = node
            realized_layers.append(node)

        for i in range(len(realized_layers)):
            if realized_layers[i] is None:
                raise RuntimeError(f"Invalid `None` in realized_layers[{i}]: {realized_layers}")
        return realized_layers

    def create_output_layer(self, layers: List[Layer]) -> ClassificationOutput:
        if self.task == "classification":
            # The final layer is going to have some shape of the form (C, H, W, D). We need to get
            # this down to (n_classes,), where n_classes == self.output_shape[0] in this case. While
            # we *could* just reshape and then make a final linear layer, this is likely to produce
            # inordinately large final layers which will cause memory issues.
            #
            # Instead, let's first collapse the channels
            last_shape = layers[-1].output_shape
            output_layer = ClassificationOutput(
                input_shape=last_shape, n_classes=self.output_shape[0]
            )
            # output_layer.create()
        else:
            raise NotImplementedError()

        return output_layer

    def realize_model(self) -> IndividualModel:
        """Accesses `self.layers` and `self.output_layer`, call `create` methods, and then builds
        into a torch Module"""
        # create torch instances
        for layer in self.layers:
            layer.create()
        self.output_layer.create()

        torch_layers = tuple(map(lambda layer: layer.torch, self.layers))
        if None in torch_layers:
            print(self.layers)
            print(torch_layers)
            raise RuntimeError("Invalid `None` in layers. Maybe bad `create()` implementation?")
        output = self.output_layer.torch
        all_layers = [*torch_layers, output]
        interfaces = [*self.layers, self.output_layer]
        return IndividualModel(all_layers, interfaces)

    def clone(
        self, clone_fitness: bool = True, sequential: Literal["clone", "create"] = None
    ) -> Individual:
        clone: Individual = self.__class__.__new__(self.__class__)
        clone.n_nodes = self.n_nodes
        clone.task = self.task
        clone.input_shape = self.input_shape
        clone.output_shape = self.output_shape
        clone.is_sequential = self.is_sequential
        clone.activation_interval = self.activation_interval
        clone.framework = self.framework

        clone.layers = [layer.clone() for layer in self.layers]
        for layer in clone.layers:
            layer.create()
        clone.output_layer = self.output_layer.clone()
        if sequential == "clone":
            raise NotImplementedError("Cloning nn.Module is not implemented yet")
            # clone.sequential_model = self.sequential_model.clone()
        elif sequential == "create":
            clone.torch_model = clone.realize_model()
        elif sequential is None:
            pass
        else:
            raise ValueError("Valid options for `sequential` are 'clone', 'create' or `None`.")

        clone.optimizer = self.optimizer.clone()
        clone.fitness = self.fitness if clone_fitness else None
        return clone

    def mutate(
        self,
        prob: float = 0.1,
        method: ArgMutation = "random",
        add_layers: bool = False,
        swap_layers: bool = False,
        delete_layers: bool = False,
        optimizer: bool = False,
    ) -> Individual:
        mutated = self.mutate_parameters(prob)
        if add_layers:
            mutated = mutated.mutate_new_layer(prob)
        if swap_layers:
            mutated = mutated.mutate_swap_layer(prob)
        if delete_layers:
            mutated = mutated.mutate_delete_layer(prob)
        if optimizer:
            mutated.optimizer = mutated.optimizer.mutate(prob, method)
        mutated.output_layer = mutated.create_output_layer(mutated.layers)
        mutated.torch_model = mutated.realize_model()
        return mutated

    def mutate_parameters(self, prob: float = 0.1, method: ArgMutation = "random") -> Individual:
        """Change the parameters of the internal layers only

        Parameters
        ----------
        prob: float = 0.1
            The probability per parameter

        method: "perturb" | "random"
            The argument perturbation method

        Returns
        -------
        mutated: Individual
            The mutated individual
        """

        # NOTE: we take care to call `sequential=None` so that we don't waste time building a torch
        # Module before we are done other mutation steps
        mutated = self.clone(clone_fitness=False, sequential=None)
        mutated.layers = [layer.mutate(prob, method) for layer in mutated.layers]
        # we do not need to mutate the output layer, because mutation doesn't muck with I/O shapes
        return mutated

    def mutate_new_layer(self, prob: float = 0.1) -> Individual:
        """Add a new layer internally with probability `prob`, fixing input/output sizes of the
        layers connecting to where the new layer was inserted

        Parameters
        ----------
        prob: float = 0.1
            The probability per parameter

        Returns
        -------
        mutated: Individual
            The mutated individual
        """
        raise NotImplementedError()

    def mutate_delete_layer(self, prob: float = 0.1) -> Individual:
        """Delete a middle layer with probability `prob`, fixing input/output sizes of the layers
        that end up being joined after the deletion (plus sizes of subsequent layers at the
        insertion point)

        Parameters
        ----------
        prob: float = 0.1
            The probability per parameter

        Returns
        -------
        mutated: Individual
            The mutated individual
        """
        raise NotImplementedError()

    def mutate_swap_layer(self, prob: float = 0.1) -> Individual:
        """Swap the positions of  a middle layer with probability `prob`, fixing input/output sizes
        in the process

        Parameters
        ----------
        prob: float = 0.1
            The probability per parameter

        Returns
        -------
        mutated: Individual
            The mutated individual
        """
        raise NotImplementedError()

