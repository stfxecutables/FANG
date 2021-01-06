from typing import Any

from src.interface.pytorch.optimizer import Adam, AdamW, Optimizer


def use_evolver_methods(optimizer: Optimizer) -> None:
    optim = optimizer()
    print(optim)
    print("Cloned:")
    print(optim.clone())
    print("Mutated:")
    print(optim.mutate(probability=0.4))


class TestOptimizers:
    def test_Adam(self, capsys: Any) -> None:
        use_evolver_methods(Adam)

    def test_AdamW(self, capsys: Any) -> None:
        use_evolver_methods(AdamW)
