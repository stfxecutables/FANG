<<<<<<< HEAD
from pathlib import Path
from typing import Any, Union

import numpy as np

from src.exceptions import VanishingError
from src.individual import Individual
from src.interface.population import population

LOGFILE = Path(__file__).resolve().parent / "test_individual_outputs.txt"


def test_individual(capsys: Any) -> None:
    # SEED = 2
    # np.random.seed(SEED)
    # random.seed(SEED)

    MAX_ATTEMPTS = 2
    vanishings = 0
    descs = []
    torch_descs = []
    fitnesses = []
    errs = []
    pop = population()

    def log(ind: Union[Individual, str]) -> None:
        if isinstance(ind, Individual):
            fitnesses.append(ind.fitness)
            descs.append(str(ind))
            torch_descs.append(str(ind.sequential_model))
        elif isinstance(ind, str):
            fitnesses.append(None)
            descs.append("Error")
            torch_descs.append("Error")
            errs.append(ind)
        else:
            raise ValueError()

    for i in range(MAX_ATTEMPTS):
        try:
            ind = Individual(
                n_nodes=10, task="classification", input_shape=(1, 28, 28), output_shape=10
            )
        except VanishingError:
            vanishings += 1
            fitnesses.append(0)
            continue
        except RuntimeError as e:
            if "mat1 dim 1 must match mat2 dim 0" in str(e):
                with capsys.disabled():
                    print(ind)
                    print(ind.sequential_model)
            raise e
        try:
            with capsys.disabled():
                print(ind)
                print(ind.sequential_model)
                ind.evaluate_fitness()
                log(ind)
        except RuntimeError as e:
            if "Output size is too small" in str(e):
                vanishings += 1
                log(str(e))
            elif "Kernel size can't be greater than actual input size" in str(e):
                vanishings += 1
                log(str(e))
            elif "Padding size should be less than the corresponding input dimension" in str(e):
                vanishings += 1
                log(str(e))
            elif "mat1 dim 1 must match mat2 dim 0" in str(e):
                with capsys.disabled():
                    print(ind)
                    print(ind.sequential_model)
                    log(str(e))
                raise e
            else:
                raise e

    with capsys.disabled():
        messages = [f"Number of networks where size shrunk to zero: {vanishings}\r\n"]
        messages.append("Fitnesses of networks after 1 epoch (accuracies):\r\n")
        # population_N = []
        num = 2
        population_N = pop.create_population(Individual, fitnesses)
        parent = pop.selection_unique(population_N, num)
        # populate.append(population)
        print("parent")
        print(parent)

        for i, fitness in enumerate(fitnesses):
            messages.append(f"Model {i}: {str(float(np.round(fitness, 3)))}")
        for i, (desc, fitness) in enumerate(zip(descs, fitnesses)):
            messages.append("\r\n\r\n")
            messages.append("=" * 80)
            messages.append(f"Model {i}: {str(float(np.round(fitness, 3)))}")
            messages.append("=" * 80)
            messages.append(desc)

        messages.append("\r\nErrors:")
        for err in errs:
            messages.append(err)

        with open(LOGFILE, "w") as logfile:
            logfile.write("\r\n".join(messages))
        for message in messages:
            print(message)
=======
from __future__ import annotations
from copy import deepcopy
from src.exceptions import VanishingError
from src.individual import Individual
from src.interface.evolver import Evolver

from typing import Any, List, Tuple
from typing_extensions import Literal

import pytest
import numpy as np
from src.population import Population
from src.individual import Individual
from test.utils import get_pop


@pytest.mark.spec
@pytest.mark.fast
class TestPopulation:
    def test_init_from_individuals(self) -> None:
        individuals: List[Individual] = []
        errors = 0
        while len(individuals) < 10 and errors < 100:
            try:
                individuals.append(
                    Individual(
                        n_nodes=np.random.randint(7, 13),
                        task="classification",
                        input_shape=(1, 28, 28),
                        output_shape=10,
                    )
                )
            except VanishingError:
                errors += 1

        pop = Population(individuals)
        assert len(pop) == 10

    def test_init_from_int(self) -> None:
        pop = get_pop(10)
        assert len(pop) == 10

    def test_clone(self) -> None:
        pop = get_pop(10)
        clone = pop.clone()
        assert isinstance(clone, Population)
        assert len(pop) == len(clone)
        for i in range(len(pop)):
            # NOT A GOOD TEST LONG TERM...
            # TODO: implement __eq__ for all core classes
            assert isinstance(clone[i], Individual)
            assert str(clone[i]) == str(pop[i])
            assert clone[i].fitness == pop[i].fitness
            try:
                assert clone[i].fitness is not pop[i].fitness
                assert clone[i] is not pop[i]
            except AssertionError as e:
                raise RuntimeError(
                    "When copying objects, you need to make sure you actually copy, e.g. using "
                    "`from copy import deepcopy` so that new objects are allocated in memory. "
                    "You cannot just do `new = old`. "
                ) from e

    def test_mutate(self) -> None:
        pop = get_pop(10)
        mutated = pop.mutate(0.9)
        assert len(mutated) == len(pop)
        assert mutated is not pop
        assert mutated.individuals is not pop.individuals

        for i in range(len(pop)):
            # NOT A GOOD TEST LONG TERM...
            # TODO: implement __eq__ for all core classes
            assert str(mutated[i]) != str(pop[i])
            assert mutated[i].fitness is None
            try:
                assert mutated[i] is not pop[i]
            except AssertionError as e:
                raise RuntimeError(
                    "When creating a mutation, you cannot mutate an object in place. "
                    "You must mutate a clone only."
                ) from e

    def test_select_best(self) -> None:
        for n in range(10):
            pop = get_pop(10)
            best = pop.select_best(n)
            assert len(best) == n
            for i in range(len(pop) - 1):
                assert pop[i].fitness >= pop[i + 1].fitness
            for ind in best:
                for orig in pop:
                    if str(ind) == str(orig):
                        try:
                            assert ind is orig
                        except AssertionError as e:
                            raise RuntimeError(
                                "When getting the best members, you should not clone."
                            ) from e

    def test_get_crossover_pairs(self) -> None:
        for n in range(10):
            pop = get_pop(10)
            pairs = pop.get_crossover_pairs(n, method="random")
            assert isinstance(pairs, list)
            assert len(pairs) == n
            for pair in pairs:
                assert isinstance(pair, tuple)
                ind1, ind2 = pair
                assert isinstance(ind1, Individual)
                assert isinstance(ind2, Individual)
                assert ind1 in pop.individuals  # make sure we got references
                assert ind2 in pop.individuals
                assert ind1 != ind2

    def test_crossover(self) -> None:
        for n in range(10):
            pop = get_pop(10)
            crossed = pop.crossover()
            assert isinstance(crossed, Population)
            assert len(crossed) == len(pop)
            for cross in crossed:
                assert cross not in pop.individuals
                for ind in pop:
                    # impossible for crossover to create offspring identical to parents
                    assert ind != cross


@pytest.mark.fast
class TestPopulationTests:
    def test_test_get_best(self) -> None:
        pop = get_pop(10)
        fitnesses = np.linspace(0.95, 0.05, 10)
        for ind, fitness in zip(pop, fitnesses):
            ind.fitness = fitness
        for n in range(10):
            best = pop.individuals[:n]
            assert len(best) == n
            for i in range(len(pop) - 1):
                assert pop[i].fitness >= pop[i + 1].fitness
            for ind in best:
                for orig in pop:
                    if str(ind) == str(orig):
                        try:
                            assert ind is orig
                        except AssertionError as e:
                            raise RuntimeError(
                                "When getting the best members, you should not clone."
                            ) from e

    def test_test_mutate(self) -> None:
        pop = get_pop(10)
        mutated = get_pop(10)
        assert len(mutated) == len(pop)
        assert mutated is not pop
        assert mutated.individuals is not pop.individuals

        for i in range(len(pop)):
            # NOT A GOOD TEST LONG TERM...
            # TODO: implement __eq__ for all core classes
            assert str(mutated[i]) != str(pop[i])
            assert mutated[i].fitness is None
            try:
                assert mutated[i] is not pop[i]
            except AssertionError as e:
                raise RuntimeError(
                    "When creating a mutation, you cannot mutate an object in place. "
                    "You must mutate a clone only."
                ) from e
>>>>>>> origin/master
