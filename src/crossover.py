from src.individual import Individual
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import cast, no_type_check
from typing_extensions import Literal


def cross_individuals(
    ind1: Individual, ind2: Individual, n_pairs: int = 1
) -> List[Tuple[Individual, Individual]]:
    """Return crossovers of `ind1` and `ind2`. Ensures the crossed individuals are all valid
    (because the probability of a random cross being valid is very small).

    Parameters
    ----------
    ind1: Individual
        The first individual to cross

    ind2: Individual
        The second individual to cross

    n_pairs: int = 1
        The number of pairs of crosses to return.

    Returns
    -------
    crosses: List[Tuple[Individual, Individual]]
        The list of pairs of crossed individuals
    """
    raise NotImplementedError()
