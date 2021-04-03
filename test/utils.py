from typing import Any, Dict

from src.exceptions import VanishingError
from src.individual import Individual
from src.population import Population
from copy import deepcopy

import numpy as np


def get_individual(attempts: int = 10, **kwargs: Dict[str, Any]) -> Individual:
    ind = None
    for i in range(attempts):
        try:
            return Individual(
                n_nodes=10, task="classification", input_shape=(1, 28, 28), output_shape=10, **kwargs
            )
        except VanishingError:
            continue
        except RuntimeError as e:
            if "mat1 dim 1 must match mat2 dim 0" in str(e):
                continue
            else:
                raise e
    return ind


def get_pop(n_individuals: int = 10, attempts: int = 10) -> Population:
    pop = Population(
        individuals=n_individuals,
        n_nodes=np.random.randint(7, 13),
        task="classification",
        input_shape=(1, 28, 28),
        output_shape=10,
        attempts_per_individual=attempts,
    )
    pop.fitnesses = np.random.uniform(0, 1, n_individuals).tolist()
    for ind, f in zip(pop, pop.fitnesses):
        ind.fitness = deepcopy(f)
    return pop
