import pytest
from src.individual import Individual
from src.crossover import cross_individuals
from test.utils import get_individual


@pytest.mark.spec
def test_crossover_sanity() -> None:
    for n in range(10):
        ind1 = get_individual(50)
        ind2 = get_individual(50)
        crosses = cross_individuals(ind1, ind2)
        assert len(crosses) == 2
        for ind in crosses:
            assert isinstance(ind, Individual)
