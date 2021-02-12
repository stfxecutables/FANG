from src.generation import Generation
from src.generation import State
import pytest
import numpy as np

from test.utils import get_pop
from typing import Any


def get_gen(size: int = 10) -> Generation:
    return Generation(
        size=size,
        n_nodes=10,
        task="classification",
        input_shape=(1, 28, 28),
        output_shape=10,
        attempts_per_individual=50,
        attempts_per_generation=50,
        fast_dev_run=True,
    )


@pytest.mark.spec
class TestGeneration:
    def test_init(self) -> None:
        get_gen()

    def test_init_from(self) -> None:
        pop = get_pop(10, attempts=50)
        Generation.from_population(pop, attempts_per_individual=50, attempts_per_generation=50)

    def test_evaluate_fitnesses(self, capsys: Any) -> None:
        gen = get_gen(2)
        with capsys.disabled():
            gen.evaluate_fitnesses()
        for ind in gen.progenitors:
            assert ind.fitness is not None

    def test_get_survivors(self, capsys: Any) -> None:
        gen = get_gen(2)
        for ind in gen.progenitors:
            ind.fitness = np.random.uniform(0, 1)
        gen.state = State.EVALUATED
        with capsys.disabled():
            gen.get_survivors()
        assert gen.state == State.SURVIVED

    def test_mutate_survivors(self, capsys: Any) -> None:
        gen = get_gen(2)
        for ind in gen.progenitors:
            ind.fitness = np.random.uniform(0, 1)
        gen.survivors = gen.progenitors
        gen.state = State.SURVIVED
        with capsys.disabled():
            gen.mutate_survivors()
        assert gen.state == State.MUTATED

    def test_crossover(self, capsys: Any) -> None:
        gen = get_gen(2)
        for ind in gen.progenitors:
            ind.fitness = np.random.uniform(0, 1)
        gen.mutated = gen.survivors = gen.progenitors
        gen.state = State.MUTATED
        with capsys.disabled():
            gen.cross()
        assert gen.state == State.CROSSED
