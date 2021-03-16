from __future__ import annotations  # noqa

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union, no_type_check
from uuid import uuid1

import numpy as np
import torch
from torch import Tensor as TorchTensor
from torch.nn import Module as TorchModule
from typing_extensions import Literal

from src.exceptions import ShapingError, VanishingError
from src.interface.arguments import ArgMutation
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
        self.sanity_checked: bool = False

    @no_type_check
    def forward(self, x: TorchTensor) -> TorchTensor:
        if not self.sanity_checked:
            for layer, interface in zip(self.layers, self.interfaces):
                x = layer(x)
                if x.shape[2:] != interface.output_shape[1:]:
                    name = interface.__class__.__name__
                    raise ShapingError(
                        f"{name} interface is out of sync with Torch actual shape:\n"
                        f"    {name}.input_shape:   {interface.input_shape}\n"
                        f"    {name}.output_shape:  {interface.output_shape}\n"
                        f"    Torch Tensor x.shape: {tuple(x.shape[1:])}"
                    )
            self.sanity_checked
            return x
        for layer, interface in zip(self.layers, self.interfaces):
            x = layer(x)
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
        self.uuid: str = str(uuid1())

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

    def evaluate_fitness(self, fast_dev_run: bool = False) -> None:
        if self.fitness is None:
            results = train_sequential(self.torch_model, self.optimizer, fast_dev_run)
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
        ).tolist()
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

    def fix_input_output(self) -> None:
        prev: Layer = self.layers[0]
        for i, layer in enumerate(self.layers):
            # we need to correct the input_shape and output_shape
            if i == 0:
                continue
            self.layers[i] = layer.__class__(input_shape=prev.output_shape)
            for size in self.layers[i].output_shape[1:]:
                if size <= 0:
                    raise VanishingError("Convolutional layers have reduced output to zero size.")
            prev = self.layers[i]

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
            if not isinstance(layer, Layer):
                raise ValueError(f"Someone fucked up. {layer} is not a Layer.")
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
        for prop, value in self.__dict__.items():
            if prop in ["layers", "output_layer", "torch_model", "fitness"]:
                setattr(clone, prop, None)
            else:
                setattr(clone, prop, value)

        clone.layers = [layer.clone() for layer in self.layers]
        clone.optimizer = self.optimizer.clone()
        clone.output_layer = self.output_layer.clone()
        # unfortunately, python deepcopy does not work with floats, because if you write e.g.
        #
        #                   x = 1.0; y = deepcopy(x)
        #                   print(x is y)
        #
        # you see "True". This is trash behaviour, and in general the behaviour of `deepcopy`
        # for floats is not what anyone really wants. Even:
        #
        #                   x = 1.0; y = float(x)
        #                   print(x is y)
        #
        # still results in a `True`. So we force a new float with multiplication by 1.0...
        if self.fitness is not None:
            # eps = sys.float_info.epsilon
            # f = float(self.fitness) + eps - eps
            f = float(self.fitness) * 1.0  # force a copy...
            clone.fitness = f if clone_fitness else None
        if sequential == "clone":
            raise NotImplementedError("Cloning nn.Module is not implemented yet")
            # clone.sequential_model = self.sequential_model.clone()
        elif sequential == "create":
            for layer in clone.layers:
                layer.create()
            clone.torch_model = clone.realize_model()
            return clone
        elif sequential is None:
            return clone
        else:
            raise ValueError("Valid options for `sequential` are 'clone', 'create' or `None`.")

    def mutate(
        self,
        prob: float = 0.1,
        method: ArgMutation = "random",
        add_layers: bool = False,
        swap_layers: bool = False,
        delete_layers: bool = False,
        optimizer: bool = False,
    ) -> Optional[Individual]:
        mutated = self.mutate_parameters(prob)
        if add_layers:
            mutated = mutated.mutate_new_layer(prob)
        if swap_layers:
            mutated = mutated.mutate_swap_layer(prob)
        if delete_layers:
            mutated = mutated.mutate_delete_layer(prob)
        if optimizer:
            mutated.optimizer = mutated.optimizer.mutate(prob, method)
        try:
            mutated.fix_input_output()
            mutated.output_layer = mutated.create_output_layer(mutated.layers)
            mutated.torch_model = mutated.realize_model()
            return mutated
        except VanishingError:
            return None

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

        # get length of layers to get a random insertion point range
        n_layers = len(self.layers)
        position = np.random.randint(low=1, high=n_layers)
        prev = self.layers[position - 1]
        mutated = self.clone(clone_fitness=False, sequential=None)
        layer_constructor: Layer = np.random.choice(TORCH_NODES_2D, size=1)[0]
        layer = layer_constructor(input_shape=prev.output_shape)
        # next layer may now be in an inconsistent state, must be fixed later!
        mutated.layers.insert(position, layer)
        return mutated

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
        deletion_point = np.random.randint(low=1, high=len(self.layers))
        mutated = self.clone(clone_fitness=False, sequential=None)
        mutated.layers.pop(deletion_point)
        return mutated

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
        # pick random layers (except for first/input layer) and swap them
        idx1, idx2 = np.random.choice(range(1, len(self.layers)), 2).tolist()
        mutated = self.clone(clone_fitness=False, sequential=None)
        mutated.layers[idx1], mutated.layers[idx2] = mutated.layers[idx2], mutated.layers[idx1]
        return mutated

    def as_dict(self) -> Dict[str, Any]:
        serializable: Dict[str, Any] = {"Individual": {}}
        for attr in self.__dict__.keys():
            if attr in ["torch_model", "optimizer"]:  # need to be saved with torch.save
                continue
            elif attr in ["output_layer"]:  # to to call as_dict()
                serializable["Individual"][attr] = getattr(self, attr).as_dict()
            elif attr in ["layers"]:  # to to call as_dict()
                layers = getattr(self, attr)
                layer_dicts = [layer.as_dict() for layer in layers]
                serializable["Individual"][attr] = layer_dicts
            else:
                serializable["Individual"][attr] = getattr(self, attr)
        return serializable

    def save(self, dir: Path) -> None:
        if not Path(dir).is_dir():
            raise ValueError("`dir` must be a directory.")
        jsonfile = dir / f"{self.uuid}.json"
        torchfile = dir / f"{self.uuid}.pt"
        txtfile = dir / f"{self.uuid}.txt"
        with open(jsonfile, "w") as file:
            json.dump(self.as_dict(), file, check_circular=True, indent=2, sort_keys=True)
        with open(txtfile, "w") as file:
            file.write(str(self))
            file.write("\n")
        torch.save(self.torch_model, str(torchfile))
