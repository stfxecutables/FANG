from __future__ import annotations  # noqa

from pathlib import Path
from typing import List

from src.individual import Individual
from src.population import Population


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
        return list(map(lambda ind: ind.fitness, self.hall))  # type: ignore

    def update(self, survivors: Population) -> None:
        """Compare the individuals in `survivors` to those in `self.hall`, and update `self.hall` to
        include the individuals with the `size` highest-fitness individuals from both

        Parameters
        ----------
        survivors: List[Individual]
            List of individuals that have survived selection.
        """

        self.hall.extend(survivors.individuals)
        self.hall.sort(key=lambda ind: ind.fitness, reverse=True)
        self.hall = self.hall[: self.size]

    def save(self, dir: Path) -> None:
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
        assert dir.is_dir()
        for individual in self.hall:
            individual.save(dir)
