from __future__ import annotations  # noqa

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union, no_type_check

from src.hall_of_fame import HallOfFame
from src.individual import Individual, Task
from src.interface.arguments import ArgMutation
from src.interface.initializer import Framework
from src.population import Population


class State(Enum):
    INITIALIZED = 0
    EVALUATED = 1
    SURVIVED = 2
    REPOPULATED = 3
    SAVED = 4
    MUTATED = 5
    CROSSED = 6
    REPRODUCED = 7


class Generation:
    """Class to manage Populations and the basic evolutionary step. Is implemented (very loosely) as
    a finite state machine.

    Parameters
    ----------
    population: Population
        The starting population. For the very first generation, this will be a population of random
        individuals. For subseqeuent generations the constructor is not called, because subsequent
        generations are produced via `Generation.next()` or `next(Generation)` and etc..

    size: int = 10
        The number of individuals to be held in the population.

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

    activation_interval: int = 0
        If greater than zero, require a non-linear activation to be placed after
        `activation_interval` non-activation layers. E.g. if `activation_interval=2`, there can be
        at maximum two consecutive non-activation layers before a random activation layer is
        forcefully inserted. We may wish to require semi-regular activations because e.g. two Conv
        layers just compose to a single linear function, which is an inefficiency.

    framework: "pytorch" | "Tensorflow"
        Which framework to use for instantiating and training the models.

    attempts_per_individual: int = 3
        Random network generation can produce networks which, due to convolutions, produce feature
        maps with zero or negative size. When this happens, a `VanishingError` is raised. However,
        with sufficient attempts, you can usually ensure a valid network is created. The value set
        here determines how many `VanishingError`s are allowed before a genuine error is raised and
        execution is halted. It is important to not allow infinite attempts here since otherwise we
        can get stuck endlessly attempting to generate new individuals.

    Attributes
    ----------
    state: "INITIALIZED" | "EVALUATED" | "SURVIVED" | "MUTATED" | "CROSSED" | "REPRODUCED"
        The state of the generation.

    progenitors: Population
        The original `Individual` objects

    survivors: Optional[Population]
        After evaluating fitness (e.g. in state `EVALUATED`), the `Population` of `Individual`s with
        fitness above `survival_threshold`.

    mutated: Optional[Population]
        The mutated survivors.

    crossed: Optional[Population]
        The crossed, mutated individuals.

    offspring: Optional[Population]
        The `self.size` `Individual`s remaining after mutation and crossover.

    hall_of_fame: List[Individual]
        A list of the *architectures* so far that have had the highest fitnesses. Size determined by
        `hall_size`. such that `len(self.hall_of_fame)` is always <= `hall_size`, and where
        `self.hall_size[i].fitness >= self.hall_size[i+1].fitness`. When a generation produces a new
        individual with higher fitness than the fitness of the lowest-fitness individual in the hall
        of fame, that individual is bumped out of the hall of fame. That is, the hall of fame always
        contains the `hall_size` individuals with the highest fitnesses across all generations.

    Notes
    -----
    There are six stages for a generation. We can think of this as a very simple finite state
    machine:

        1. initialized <--
        2. evaluated      |
        3. survived       |
        4. mutated        |
        5. crossed        |
        6. reproduced  -->
    """

    def __init__(
        self,
        size: int = 10,
        n_nodes: int = None,
        task: Task = None,
        input_shape: Union[int, Tuple[int, ...]] = None,
        output_shape: Union[int, Tuple[int, ...]] = None,
        survival_threshold: float = 0.0,
        mutation_probability: float = 0.1,
        crossover: bool = False,
        mutation_method: ArgMutation = "random",
        add_layers: bool = False,
        swap_layers: bool = False,
        delete_layers: bool = False,
        mutate_optimizer: bool = False,
        hall_size: int = 10,
        hall_dir: Path = None,
        sequential: bool = True,
        activation_interval: int = 0,
        framework: Framework = "pytorch",
        attempts_per_individual: int = 3,
        attempts_per_generation: int = 1,
        fast_dev_run: bool = False,
    ):
        self.state: State = State.INITIALIZED
        self.survival_threshold: float = float(survival_threshold)
        self.do_crossover: bool = crossover

        self.mutation_prob: float = mutation_probability
        self.mutation_method: ArgMutation = mutation_method
        self.add_layers: bool = add_layers
        self.swap_layers: bool = swap_layers
        self.delete_layers: bool = delete_layers
        self.mutate_optimizer: bool = mutate_optimizer

        self.hall_size: int = int(hall_size)
        self.hall: HallOfFame = HallOfFame(self.hall_size)
        self.hall_dir = hall_dir
        self.survivors: Optional[Population] = None
        self.mutated: Optional[Population] = None
        self.crossed: Optional[Population] = None
        self.offspring: Optional[Population] = None

        self.size = size
        self.n_nodes = n_nodes
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.task: Task = task
        self.is_sequential: bool = sequential
        self.framework: Framework = framework
        self.activation_interval = activation_interval
        self.attempts_per_individual = attempts_per_individual
        self.attempts_per_generation = attempts_per_generation
        self.attempts = 0
        self.fast_dev_run: bool = fast_dev_run

        self.progenitors = Population(
            individuals=self.size,
            n_nodes=self.n_nodes,
            task=self.task,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            sequential=self.is_sequential,
            activation_interval=self.activation_interval,
            framework=self.framework,
            attempts_per_individual=self.attempts_per_individual,
            fast_dev_run=self.fast_dev_run,
        )

    @classmethod
    def from_population(
        cls: Type[Generation],
        population: Population,
        survival_threshold: float = 0.0,
        mutation_probability: float = 0.1,
        crossover: bool = False,
        mutation_method: ArgMutation = "random",
        add_layers: bool = False,
        swap_layers: bool = False,
        delete_layers: bool = False,
        mutate_optimizer: bool = False,
        hall_size: int = 10,
        hall_dir: Path = None,
        activation_interval: int = 0,
        attempts_per_individual: int = 10,
        attempts_per_generation: int = 5,
        fast_dev_run: bool = False,
    ) -> Generation:
        self: Generation = cls.__new__(cls)

        self.survival_threshold = float(survival_threshold) if not fast_dev_run else 0.0
        self.do_crossover = crossover

        self.mutation_prob = mutation_probability
        self.mutation_method = mutation_method
        self.add_layers = add_layers
        self.swap_layers = swap_layers
        self.delete_layers = delete_layers
        self.mutate_optimizer = mutate_optimizer

        self.hall_size = int(hall_size)
        self.hall = HallOfFame(self.hall_size)
        self.hall_dir = hall_dir
        self.survivors = None
        self.mutated = None
        self.crossed = None
        self.offspring = None

        self.state = State.INITIALIZED
        self.size = len(population)
        self.progenitors = population
        self.input_shape = population.input_shape
        self.output_shape = population.output_shape
        self.task = population.task
        self.is_sequential = population.is_sequential
        self.framework = population.framework
        self.activation_interval = activation_interval
        self.attempts_per_individual = attempts_per_individual
        self.attempts_per_generation = attempts_per_generation
        self.attempts = 0
        self.fast_dev_run = fast_dev_run
        # self.fast_dev_run = population.fast_dev_run  # ensure we override this
        return self

    @no_type_check
    def __str__(self) -> str:
        info = f"Generation (stage={self.state})."
        if self.state == State.INITIALIZED:
            return f"{info} {len(self.progenitors)} progenitors."
        elif self.state == State.EVALUATED:
            return f"{info} {len(self.progenitors)} evaluated `Individual`s."
        elif self.state == State.SURVIVED:
            return (
                f"{info} {len(self.survivors)} survivors for threshold={self.survival_threshold}."
            )
        elif self.state == State.SAVED:
            return (
                f"{info} {len(self.survivors)} survivors, {len(self.hall)} in Hall of Fame."
            )  # type: ignore # noqa
        elif self.state == State.MUTATED:
            return f"{info} {len(self.mutated)} mutated `Individual`s."  # type: ignore
        elif self.state == State.CROSSED:
            return f"{info} {len(self.crossed)} crossed `Individual`s."  # type: ignore
        elif self.state == State.REPRODUCED:
            return f"{info} {len(self.offspring)} surviving offspring."  # type: ignore
        else:
            raise ValueError("Impossible `Generation` state.")

    __repr__ = __str__

    def next(self, survivor_dir: Path = None) -> Generation:
        """Perform the entire set of evolutionary steps:

        1. Compute fitnesses for the starting population (progenitors)
        2. Select survivors based on a threshold
        3. Save survivors into hall of fame and to disk, for resuming later
        4. Perform mutation (and optionally crossover)
        5. Return the offspring of above as a fresh Generation

        """
        self.evaluate_fitnesses()
        self.get_survivors()
        self.repopulate()
        self.save_progress(survivor_dir)
        self.mutate_survivors()
        if self.do_crossover:
            self.cross()
        self.set_offspring()
        next_gen = Generation.from_population(
            self.offspring,
            survival_threshold=self.survival_threshold,
            mutation_probability=self.mutation_prob,
            crossover=self.do_crossover,
            mutation_method=self.mutation_method,
            add_layers=self.add_layers,
            swap_layers=self.swap_layers,
            delete_layers=self.delete_layers,
            mutate_optimizer=self.mutate_optimizer,
            hall_size=self.hall_size,
            hall_dir=self.hall_dir,
            activation_interval=self.activation_interval,
            attempts_per_individual=self.attempts_per_individual,
            attempts_per_generation=self.attempts_per_generation,
            fast_dev_run=self.fast_dev_run,
        )
        next_gen.hall = self.hall
        return next_gen

    def evaluate_fitnesses(self) -> None:
        assert self.state == State.INITIALIZED
        self.progenitors.evaluate_fitnesses()
        self.state = State.EVALUATED

    def get_survivors(self) -> None:
        assert self.state == State.EVALUATED
        self.survivors = self.filter_by_fitness()
        self.state = State.SURVIVED

    def repopulate(self) -> None:
        assert self.state == State.SURVIVED
        assert self.survivors is not None
        if len(self.survivors) < self.size:
            print("Not enough surviving individuals to reproduce. Repopulating...")
            individuals = [ind for ind in self.survivors]
            while len(individuals) < self.size:
                if self.attempts >= self.attempts_per_generation:
                    att = self.attempts_per_generation
                    raise RuntimeError(
                        f"Exceeded maximum attempts ({att}) to generate a viable population.\n"
                        f"Consider increasing `attempts_per_generation` or `attempts_per_individual`,\n"
                        f"or changing options related to activation function frequency."
                    )
                print(f"Repopulation attempt {self.attempts}:")
                self.state = State.INITIALIZED
                args = self._args()
                args.update(dict(individuals=self.size - len(individuals)))
                new = Population(**args)
                new.evaluate_fitnesses()
                new_survivors = [ind for ind in new if ind.fitness >= self.survival_threshold]
                individuals.extend(new_survivors)
                self.attempts += 1
            self.survivors = Population(individuals, fast_dev_run=self.fast_dev_run)
        self.state = State.REPOPULATED

    def save_progress(self, survivor_dir: Path = None) -> None:
        """Update the hall of fame, save the hall, and
        individuals with the `hall_size` highest fitnesses across both the previous hall and
        current survivors.
        """
        assert self.state == State.REPOPULATED
        assert self.survivors is not None

        self.hall.update(self.survivors)
        if self.hall_dir is not None:
            self.hall.save(self.hall_dir)
        if survivor_dir is not None:
            self.survivors.save(survivor_dir)
        self.state = State.SAVED

    def mutate_survivors(self) -> None:
        assert self.state == State.SAVED
        assert self.survivors is not None

        mutation_options = dict(
            prob=self.mutation_prob,
            method=self.mutation_method,
            add_layers=self.add_layers,
            swap_layers=self.swap_layers,
            delete_layers=self.delete_layers,
            optimizer=self.mutate_optimizer,
        )
        # for individual in self.survivors:
        #     mutated.append(individual.mutate(**mutation_options))
        # self.mutated = Population(mutated, fast_dev_run=self.fast_dev_run)
        self.mutated = self.survivors.mutate(**mutation_options)
        self.mutated.fast_dev_run = self.fast_dev_run
        self.state = State.MUTATED

    def cross(self) -> None:
        assert self.state == State.MUTATED
        """From mutated survivors, get crossover pairs and perform crossover"""
        self.crossed = self.mutated.crossover()  # type: ignore
        self.crossed.fast_dev_run = self.fast_dev_run
        self.state = State.CROSSED

    def set_offspring(self) -> None:
        """This is really just book-keeping so we have a clear, `reproduced` state and single source
        of offspring"""
        if self.state == State.MUTATED and not self.do_crossover:
            self.offspring = self.mutated
        elif self.state == State.CROSSED and self.do_crossover:
            self.offspring = self.crossed
        else:
            raise ValueError("Invalid state.")
        self.state = State.REPRODUCED

    def filter_by_fitness(self) -> Population:
        """Get a population of individuals from self.progenitors that all have fitness above
        `self.survival_threshold`. Must return *references*, and not copies."""
        survivors = list(
            filter(lambda ind: ind.fitness >= self.survival_threshold, self.progenitors)
        )
        return Population(survivors, fast_dev_run=self.fast_dev_run)

    def _args(self) -> Dict[str, Any]:
        return dict(
            individuals=self.size,
            n_nodes=self.n_nodes,
            task=self.task,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            sequential=self.is_sequential,
            activation_interval=self.activation_interval,
            framework=self.framework,
            attempts_per_individual=self.attempts_per_individual,
            fast_dev_run=self.fast_dev_run,
        )
