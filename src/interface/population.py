from random import random
import numpy as np
import torch
from torch import Tensor as TorchTensor
from torch.nn import Module as TorchModule
from typing_extensions import Literal


class population:
    #take the consideration of fitness in the obj creation of a selection obj
    #def __init__(self):

    def select(self, fitness_set):
        #function to select parents
        sum = 0
        #normalize the fitness
        for element in fitness_set:
            sum = sum + element
        for element in fitness_set:
            element = element/sum
        #return the chromosome with the first highest value more than R
        while(1==1):
            R = random()
            for element in fitness_set:
                if element> R:
                    return fitness_set.index(element)

    
    #def reproduce(self, selected_set):
        #mutation + crossover implementation
        # 
    
    
    #def next(self, reproduced_set):
        #create a new population
        

    

