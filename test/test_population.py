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
        clone = pop.clone(clone_fitness=True, sequential="create")
        assert isinstance(clone, Population)
        assert len(pop) == len(clone)
        for orig, cloned in zip(pop, clone):
            # NOT A GOOD TEST LONG TERM...
            # TODO: implement __eq__ for all core classes
            assert isinstance(cloned, Individual)
            assert str(cloned) == str(orig)
            assert cloned.fitness == orig.fitness
            try:
                assert cloned.fitness is not orig.fitness
                assert cloned is not orig
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
            for individual in pop:
                individual.fitness = np.random.uniform(0, 1)
            pop.fitnesses = list(map(lambda ind: ind.fitness, pop))
            best = pop.select_best(n)
            assert len(best) == n
            for i in range(len(best) - 1):
                assert best[i].fitness >= best[i + 1].fitness
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
