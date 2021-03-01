import pytest

import numpy as np
from tempfile import TemporaryDirectory
from src.generation import Generation, HallOfFame
from test.utils import get_pop


@pytest.mark.spec
class TestHallOfFame:
    def test_init(self) -> None:
        for i in np.random.randint(1, 10, 10):
            hall = HallOfFame(i)
            assert hall.size == i
            assert len(hall.hall) == 0

    def test_update(self) -> None:
        start = get_pop(11, attempts=50)
        start_fits = np.random.uniform(0, 1, 11)
        for i, ind in enumerate(start):
            ind.fitness = start_fits[i]
        start.fitnesses = list(map(lambda ind: ind.fitness, start))
        start_bests = sorted(start, key=lambda ind: ind.fitness, reverse=True)[:10]
        start_best_fitnesses = list(map(lambda ind: ind.fitness, start_bests))

        # test start
        hall = HallOfFame(10)
        hall.update(start)
        assert len(hall) == 10
        assert sorted(start_best_fitnesses) == sorted(hall.fitnesses())
        for ind in start_bests:
            assert ind in hall.hall

        # test a real update
        survivors = get_pop(10, attempts=50)
        survive_fits = np.random.uniform(0, 1, 10)
        for i, ind in enumerate(survivors):
            ind.fitness = survive_fits[i]
        hall.update(survivors)

        assert len(hall) == 10
        fitnesses = sorted(start_best_fitnesses + survive_fits, reverse=True)[:10]
        assert sorted(hall.fitnesses()) == sorted(fitnesses)

    def test_save(self) -> None:
        hall = HallOfFame(10)
        start = get_pop(10, attempts=50)
        for ind in start:
            ind.fitness = np.random.uniform(0, 1)
        hall.update(start)

        tmpdir = TemporaryDirectory()
        try:
            hall.save(tmpdir)
        finally:
            tmpdir.cleanup()

