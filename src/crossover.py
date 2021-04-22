from src.individual import Individual
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import cast, no_type_check
from typing_extensions import Literal
import numpy as np
from src.exceptions import VanishingError


def cross_individuals(ind1: Individual, ind2: Individual) -> Tuple[Individual, Individual]:
    """Return crossovers of `ind1` and `ind2`. Ensures the crossed individuals are all valid
    (because the probability of a random cross being valid is very small).

    Parameters
    ----------
    ind1: Individual
        The first individual to cross

    ind2: Individual
        The second individual to cross


    Returns
    -------
    crosses: Tuple[Individual, Individual]
        The list of pairs of crossed individuals

        1) Find a Corssover point at random (take it randomly - donot take the crossover point to be at center for now)
        2) If the layers are of Uniform length then divide the layers positions into 2 parts based on the
        crossover point
        For ex: if L1 - 010|010 and
        L2 - 000|101
        then L1 = L11 + L12 and L2 = L21 + L22

        Then the new generated off springs would be
        child_1 = L21 + L12
        child_2 = L11 + L22

        Things to take care of:
        1. Make sure you are doing crossover only when the layers are uniform
        2. When doing crossover handle the input and o/p sizes of the crossover layers
    """

    crossover_point1 = np.random.randint(low=1, high=len(ind1.layers))
    crossover_point2 = np.random.randint(low=1, high=len(ind2.layers))

    clone1 = ind1.clone(clone_fitness=False, sequential=None)
    clone2 = ind2.clone(clone_fitness=False, sequential=None)
    ind11, ind12 = clone1.layers[0:crossover_point1], clone1.layers[crossover_point1:]
    ind21, ind22 = clone2.layers[0:crossover_point2], clone2.layers[crossover_point2:]

    clone1.layers = ind21 + ind12
    clone2.layers = ind11 + ind22

    try:
        clone1.fix_input_output()
        clone2.fix_input_output()
    except VanishingError:
        print("Cannot resolve input/output sizes. Generating new cross.")
        return cross_individuals(ind1, ind2)  # type: ignore

    # now we need to rebuild the output layers to handle the new shapes
    clone1.output_layer = clone1.create_output_layer(clone1.layers)
    clone2.output_layer = clone2.create_output_layer(clone2.layers)

    # and finally we need to build the actual Torch/Tensorflow model
    clone1.torch_model = clone1.realize_model()
    clone2.torch_model = clone2.realize_model()
    return clone1, clone2
