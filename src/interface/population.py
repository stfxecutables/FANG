from random import random
import numpy as np
import torch
from torch import Tensor as TorchTensor
from torch.nn import Module as TorchModule
from typing_extensions import Literal


class population:
    # take the consideration of fitness in the obj creation of a selection obj
    # def __init__(self):

    def select(self, fitness_set):
        raise NotImplementedError()

    # def reproduce(self, selected_set):
    # mutation + crossover implementation
    #

    # def next(self, reproduced_set):
    # create a new population
