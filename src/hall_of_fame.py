from __future__ import annotations
from enum import Enum

from src.interface.arguments import ArgMutation
from src.interface.pytorch.optimizer import Optimizer
from src.individual import Individual
from src.population import Population
from src.interface.evolver import Evolver
from src.individual import Individual, Task
from src.interface.evolver import Evolver
from src.interface.initializer import Framework
from src.exceptions import VanishingError

from typing import Any, Dict, List, Optional, Tuple, Union, Type
from typing import cast, no_type_check
from typing_extensions import Literal

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from numpy import ndarray
from pandas import DataFrame, Series


class HallOfFame:
    """Class for storing the best individuals seen over an entire run

    Parameters
    ----------
    size: int
        The maximum allowed number of individuals allowed in the hall.
    """

    def __init__(self, size: int = 10) -> None:
        self.size = int(size)
        self.hall: List[Individual] = []

    def __len__(self) -> int:
        return len(self.hall)

    def fitnesses(self) -> List[float]:
        return list(map(lambda ind: ind.fitness, self.hall))

    def update(self, survivors: Population) -> None:
        """Compare the individuals in `survivors` to those in `self.hall`, and update `self.hall` to
        include the individuals with the `size` highest-fitness individuals from both

        Parameters
        ----------
        survivors: List[Individual]
            List of individuals that have survived selection.
        """

        fitness1 = self.fitnesses()

        print(fitness1)
        survivors.evaluate_fitnesses()
        fitness2 = sorted(survivors.fitnesses.tolist(), reverse=True)
        print(fitness2)
        best_survivors = list(survivors.select_best(self.size))

        if len(self.hall) == 0:
            self.hall = best_survivors
            # fitness2 = fitness2[: self.size]
        else:
            fitness2.extend(fitness1)
            print("Fitness2 Again")
            print(fitness2)

            self.hall.extend(best_survivors)
            idx_sort = sorted(range(len(fitness2)), key=lambda k: fitness2[: k - 1])[:-1]
            print("IDX SORT")
            print(idx_sort)
            fitness2 = [fitness2[i] for i in idx_sort][: self.size]
            self.hall = [self.hall[i] for i in idx_sort][: self.size]
        fitness2 = survivors.fitnesses.tolist()
        print(fitness2)
        for i, ind in enumerate(self.hall):
            ind.fitness = fitness2[i]

        return None

        raise NotImplementedError()

    def save(self, directory: Path) -> None:
        """Save the hall of fame individuals to `directory`.

        Parameters
        ----------
        directory: Path
            Location to save models. Must be a directory (folder)

        Note
        ----
        We need to make a strong distinction between saving the architecture and saving the trained
        model.
        """
        assert directory.is_dir()
        raise NotImplementedError()
