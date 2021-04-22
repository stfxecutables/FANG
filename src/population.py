from __future__ import annotations

import traceback
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from typing_extensions import Literal

from src.crossover import cross_individuals
from src.exceptions import VanishingError
from src.individual import Individual, Task
from src.interface.arguments import ArgMutation
from src.interface.evolver import Evolver
from src.interface.initializer import Framework

PairingMethod = Literal["random", "best", "best2", "weighted-random"]


class Population(Evolver):
    """Utility class for storing a set of individuals and providing common methods.

    Parameters
    ----------
    individuals: Union[List[Individual], int]
        The individuals to be held in the population.

        If an integer `n`, construct a new population with `n` individuals according to the
        remaining arguments.

        If a List[Individual], then simply use those individuals, and ignore remaining arguments.

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

    min_activation_spacing: int = 1
        Sets the minimum number of layers between activations when generating random individuals.
        E.g. if greater than zero, require `activation_spacing` non-activation layers to be included
        before adding an activation layer. Not recommended to set larger than 3. Helpful to prevent
        long chains of nothing but un-trainable activation functions.

    max_activation_spacing: int = 3
        Forces an activation function to be inserted after `max_activation_spacing` non-activation
        layers have been randomly selected when creatings a new random individual.


    framework: "pytorch" | "Tensorflow"
        Which framework to use for instantiating and training the models.

    attempts_per_individual: int = 3
        Random network generation can produce networks which, due to convolutions, produce feature
        maps with zero or negative size. When this happens, a `VanishingError` is raised. However,
        with sufficient attempts, you can usually ensure a valid network is created. The value set
        here determines how many `VanishingError`s are allowed before a genuine error is raised and
        execution is halted. It is important to not allow infinite attempts here since otherwise we
        can get stuck endlessly attempting to generate new individuals.

    fast_dev_run: bool = False
        If True, then training is with only 5000 samples of the MNIST data rather than the full
        55000. For testing basic behaviour more quickly.

    Notes
    -----
    This class is ultimately not necessary, but is logically obvious and will help make the code
    much easier to read and more functionally pure (always allowing the copying and creation of
    Population classes). It also works as a sensible base for a `Generation` class later.
    """

    def __init__(
        self,
        individuals: Union[List[Individual], int],
        n_nodes: int = None,
        task: Task = None,
        input_shape: Union[int, Tuple[int, ...]] = None,
        output_shape: Union[int, Tuple[int, ...]] = None,
        sequential: bool = True,
        min_activation_spacing: int = 1,
        max_activation_spacing: int = 3,
        framework: Framework = "pytorch",
        attempts_per_individual: int = 3,
        fast_dev_run: bool = False,
    ) -> None:
        # NOTE: YOU SHOULD NOT NEED TO MODIFY THIS FUNCTION
        self.fast_dev_run = fast_dev_run

        if isinstance(individuals, list):
            self.individuals: List[Individual] = self.__validate_individuals(individuals)
            self.fitnesses: List[Optional[float]] = list(
                map(lambda ind: ind.fitness, self.individuals)  # type: ignore
            )
            ind = individuals[0]
            self.input_shape = ind.input_shape
            self.output_shape = ind.output_shape
            self.task = ind.task
            self.is_sequential = ind.is_sequential
            self.framework = ind.framework
            self.min_activation_spacing = min_activation_spacing
            self.max_activation_spacing = max_activation_spacing
            return

        if task is None:
            raise ValueError("Must specify type of network in `task`.")

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.task = task
        self.is_sequential = sequential
        self.framework = framework
        self.min_activation_spacing = min_activation_spacing
        self.max_activation_spacing = max_activation_spacing
        self.fitnesses = []
        self.individuals = []
        for _ in range(int(individuals)):
            for _ in range(int(attempts_per_individual)):
                try:
                    ind = Individual(
                        n_nodes=n_nodes,
                        task=task,
                        input_shape=input_shape,
                        output_shape=output_shape,
                        sequential=sequential,
                        min_activation_spacing=min_activation_spacing,
                        max_activation_spacing=max_activation_spacing,
                        framework=framework,
                    )
                except VanishingError:  # we only want to ignore known error types
                    continue
                # except Exception as e:
                #     raise RuntimeError("Unexpected error:") from e
                self.individuals.append(ind)
                break

    def __len__(self) -> int:
        # NOTE: DO NOT MODIFY THIS FUNCTION
        return len(self.individuals)

    def __iter__(self) -> Iterator[Individual]:
        # NOTE: DO NOT MODIFY THIS FUNCTION
        return iter(self.individuals)

    def __getitem__(self, i: int) -> Individual:
        # NOTE: DO NOT MODIFY THIS FUNCTION
        return self.individuals[i]

    def __str__(self) -> str:
        # NOTE: YOU SHOULD NOT NEED TO MODIFY THIS FUNCTION
        e = "evaluated" if self.fitnesses is not None else "unevaluated"  # type: ignore
        return f"Population of size {len(self.individuals)} (fitnesses {e})."

    def __copy__(self) -> str:
        # NOTE: DO NOT MODIFY THIS FUNCTION
        pop = Population(0)
        for key, val in self.__dict__.items():
            setattr(pop, key, val)
        return pop

    def __deepcopy__(self, memo: Dict) -> None:
        # NOTE: DO NOT MODIFY THIS FUNCTION
        raise RuntimeError(
            "Population objects cannot be copied with `deepcopy`. "
            "Use `Population.clone()` instead."
        )

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Population):
            return NotImplemented
        for i1, i2 in zip(self.individuals, o.individuals):
            if i1 != i2:
                return False
        return True

    __repr__ = __str__

    @staticmethod
    def __validate_individuals(individuals: List[Individual]) -> List[Individual]:
        if len(individuals) == 0:
            raise RuntimeError("We've run out of individuals!")
        if None in individuals:
            raise RuntimeError("One of our individuals is `None`")
        input_shape = individuals[0].input_shape
        output_shape = individuals[0].output_shape
        task = individuals[0].task
        is_sequential = individuals[0].is_sequential
        framework = individuals[0].framework
        for individual in individuals:
            if individual.input_shape != input_shape:
                raise ValueError("All `Individual`s in a `Population` must share input shapes.")
            if individual.output_shape != output_shape:
                raise ValueError("All `Individual`s in a `Population` must share output shapes.")
            if individual.task != task:
                raise ValueError(
                    "All `Individual`s in a `Population` must perform identical tasks."
                )
            if individual.is_sequential != is_sequential:
                raise ValueError(
                    "Currently all `Individual`s in a `Population` must all be sequential."
                )
            if individual.framework != framework:
                raise ValueError(
                    "All `Individual`s in a `Population` must be implemented in the same framework."
                )
        return individuals

    def evaluate_fitnesses(self) -> None:
        # NOTE: YOU SHOULD NOT NEED TO MODIFY THIS FUNCTION
        for individual in self.individuals:
            try:  # we want this to be fail-proof
                individual.evaluate_fitness(self.fast_dev_run)
            except Exception as e:
                print(traceback.format_exc())
                individual.fitness = -1.0
        self.fitnesses = np.array(
            list(map(lambda individual: individual.fitness, self.individuals))  # type: ignore
        )

    # NOTE: You must indeed modify (implement) all functions below!

    def clone(
        self, clone_fitness: bool = True, sequential: Literal["clone", "create"] = None
    ) -> Population:
        """Copies each individual and returns a new population. """
        args = dict(clone_fitness=clone_fitness, sequential=sequential)
        clones = [ind.clone(**args) for ind in self]
        pop = Population(clones)
        if clone_fitness:
            pop.fitnesses = deepcopy(self.fitnesses)
        return pop

    def mutate(
        self,
        prob: float,
        method: ArgMutation = "random",
        add_layers: bool = False,
        swap_layers: bool = False,
        delete_layers: bool = False,
        optimizer: bool = False,
    ) -> Population:
        """Calls the `.mutate` method of each individual with relevant arguments passed in here, and
        returns a new population of those mutated individuals.

        Parameters
        ----------
        probability: float
            The probability of mutation.

        Returns
        -------
        population: Population
            A new `Population` object of of the same length, but with each individual mutated with
            probability `probability`.

        """
        args = dict(
            prob=prob,
            method=method,
            add_layers=add_layers,
            swap_layers=swap_layers,
            delete_layers=delete_layers,
            optimizer=optimizer,
        )
        mutateds = []
        for ind in self:
            mutated: Optional[Individual] = None
            count = 0
            while mutated is None and count < 100:
                mutated = ind.mutate(**args)
                count += 1
            mutateds.append(mutated)
        if None in mutateds:
            raise VanishingError("Couldn't get a mutation to produce a valid size.")
        return Population(mutateds, fast_dev_run=self.fast_dev_run)

    def select_best(self, n: int) -> List[Individual]:
        """Gets the fitnesses of each individual and selects the top n individuals by fitness, in
        descending order (highest fitness first).

        Parameters
        ----------
        n: int
            How many individuals to select

        Returns
        -------
        best: List[Individual]
            A List of the Individuals (references, not clones) with the best fitnesses.

        """

        if n > len(self.fitnesses):
            raise RuntimeError(
                f"Only {len(self.fitnesses)} individuals to choose from, but trying to choose {n}."
            )
        if len(self.fitnesses) == 0:
            raise RuntimeError("Not enough individuals.")

        idx_sort = np.argsort(-np.array(self.fitnesses))
        best = (np.array(self.individuals)[idx_sort]).tolist()
        return best[:n]  # type: ignore

    def get_crossover_pairs(
        self, n_pairs: int, method: PairingMethod
    ) -> List[Tuple[Individual, Individual]]:
        """Generate `n_pairs` pairings of individuals to be used in crossover

        Parameters
        ----------
        n_pairs: int
            How many pairings to create.

        method: "random" | "best" | "weighted-random"
            If "random", just randomly pair off individuals.

            If "best", and len(self) is N, generate the N pairs of individuals where each pair
            includes the individual with the highest fitness

            If "weighted-random", randomly select pairs, but where the probability of selecting an
            individual is scaled to the fitness.

        Returns
        -------
        pairs: List[Tuple[Individual, Individual]]
            A list of tuples of Individuals

        Notes
        -----
        The crossover pairs should NOT be clones, and should indeed just be references to the
        original individuals.
        """
        if method == "random":
            pairs = []
            for _ in range(n_pairs):
                pairs.append(tuple(np.random.choice(self.individuals, size=2, replace=False)))
            return pairs
        elif method == "best2":
            best_pairs = []
            for _ in range(n_pairs):
                best_pairs.append(tuple(self.select_best(n=2)))
            return best_pairs

        elif method == "weighted":
            raise NotImplementedError()
        else:
            raise ValueError("Invalid pairing method.")

    def crossover(self, method: PairingMethod = "random") -> Population:
        """Generate crossover pairs, perform crossover, and select a random amount for the offspring

        Parameters
        ----------
        method: "random" | "best" | "weighted-random"
            If "random", just randomly pair off individuals.

            If "best", and len(self) is N, generate the N pairs of individuals where each pair
            includes the individual with the highest fitness

            If "weighted-random", randomly select pairs, but where the probability of selecting an
            individual is scaled to the fitness.

        Returns
        -------
        crossed: Population
            A `Population` with the same number of individuals as the generating population.
        """
        pairs = self.get_crossover_pairs(n_pairs=len(self), method=method)
        crosses = []
        for ind1, ind2 in pairs:
            crosses.extend(cross_individuals(ind1, ind2))
        offspring = np.random.choice(crosses, size=len(self), replace=False).tolist()
        return Population(
            offspring,
            task=self.task,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            sequential=self.is_sequential,
        )

    def save(self, dir: Path) -> None:
        dir = Path(dir)
        if not dir.is_dir():
            raise ValueError("`dir` must be a directory.")
        for individual in self:
            individual.save(dir)
