from src.generation import Generation
from src.generation import State
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
import numpy as np

from test.utils import get_pop
from typing import Any


def get_gen(size: int = 10, fast_dev_run: bool = True, **kwargs: Any) -> Generation:
    return Generation(
        size=size,
        n_nodes=10,
        task="classification",
        input_shape=(1, 28, 28),
        output_shape=10,
        attempts_per_individual=50,
        attempts_per_generation=50,
        fast_dev_run=fast_dev_run,
        **kwargs,
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
        gen.evaluate_fitnesses()
        for ind in gen.progenitors:
            assert ind.fitness is not None

    def test_get_survivors(self, capsys: Any) -> None:
        gen = get_gen(10)
        gen.survival_threshold = 0.5
        for i, ind in enumerate(gen.progenitors):
            if i < 5:
                ind.fitness = np.random.uniform(0.8, 1)
            else:
                ind.fitness = 0.0
        gen.progenitors.fitnesses = [ind.fitness for ind in gen.progenitors]
        gen.state = State.EVALUATED
        gen.get_survivors()
        assert gen.state == State.SURVIVED
        assert len(gen.survivors) == 5

    def test_repopulate(self, capsys: Any) -> None:
        gen = get_gen(10)
        gen.survival_threshold = 0.2
        for i, ind in enumerate(gen.progenitors):
            if i < 7:
                ind.fitness = np.random.uniform(0.8, 1)
            else:
                ind.fitness = 0.0
        gen.progenitors.fitnesses = [ind.fitness for ind in gen.progenitors]
        gen.state = State.EVALUATED
        gen.get_survivors()
        gen.repopulate()
        assert len(gen.survivors) == gen.size

    def test_mutate_survivors(self, capsys: Any) -> None:
        gen = get_gen(2)
        for ind in gen.progenitors:
            ind.fitness = np.random.uniform(0, 1)
        gen.survivors = gen.progenitors
        gen.state = State.SAVED
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


def test_next(capsys: Any, full_run: bool, generations: int, mutation_prob: float) -> None:
    gen = get_gen(
        10,
        fast_dev_run=not full_run,
        add_layers=True,
        swap_layers=True,
        delete_layers=True,
        survival_threshold=0.1 if not full_run else 0.6,
        mutation_probability=mutation_prob,
        max_activation_spacing=2,
        crossover=True,
        # 10, fast_dev_run=not full_run, add_layers=True
    )
    tmpdir = TemporaryDirectory()
    path = Path(tmpdir.name)
    with capsys.disabled():
        print("\n{:=^80}".format("  Beginning evolution...  "))
        for i in range(generations):
            print("{:-^80}".format(f"  Generation {i + 1}  "))
            gen = gen.next(survivor_dir=path)
            gen.fast_dev_run = not full_run
            print("Hall of Fame Fitnesses:")
            print(np.round(gen.hall.fitnesses(), 3))
            print("Best Hall of Fame model:")
            print(gen.hall.best())
