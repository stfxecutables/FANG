import pytest

from itertools import combinations
from test.utils import get_individual


@pytest.mark.spec
class TestMutation:
    @pytest.mark.fast
    def test_params(self) -> None:
        for _ in range(10):
            ind = get_individual(50)
            mutated = ind.mutate(
                prob=0.5, method="random", add_layers=False, swap_layers=False, delete_layers=False
            )
            assert ind != mutated

    @pytest.mark.fast
    def test_add(self) -> None:
        for _ in range(10):
            ind = get_individual(50)
            mutated = ind.mutate(
                prob=0.99, method="random", add_layers=True, swap_layers=False, delete_layers=False
            )
            assert ind != mutated

    @pytest.mark.fast
    def test_swap(self) -> None:
        for _ in range(10):
            ind = get_individual(50)
            mutated = ind.mutate(
                prob=0.99, method="random", add_layers=False, swap_layers=True, delete_layers=False
            )
            assert ind != mutated

    @pytest.mark.fast
    def test_delete(self) -> None:
        for _ in range(10):
            ind = get_individual(50)
            # input_output_fix = fix_input_output()
            mutated = ind.mutate(
                prob=0.99, method="random", add_layers=False, swap_layers=False, delete_layers=True
            )
            assert ind != mutated

    def test_all_permutations(self) -> None:
        options = list(combinations([True, True, True, False, False, False], 3))
        for opt in options:
            args = dict(add_layers=opt[0], swap_layers=opt[1], delete_layers=opt[2])
            for _ in range(10):
                ind = get_individual(50)
                mutated = ind.mutate(prob=0.5, method="random", **args)
                assert ind != mutated
