import numpy as np
import random
import statistics


class population:
    """Build a population from individuals.
    Pick up the best individuals from population according to fitness.
    Best fit individuals are selected as parents.
    """

    def create_population(self, Individual, fitnesses):
        population = []
        n = 3
        for i in range(len(fitnesses)):
            population.append("Model {}".format(i))
        population_zip = zip(fitnesses, population)
        population_sort = sorted(population_zip, reverse=True)
        fitnesses, population = zip(*population_sort)
        return population[:n]

    def selection_unique(self, population_N, num):
        parents = random.sample(population_N, num)
        # return np.unique(parents, axis=None)
        return parents


#     raise NotImplementedError()
