import numpy as np


class Chromosome:
    def __init__(self,genes,val_fitness_function):
        self.genes = np.asarray(genes)
        self.val_fitness_function = val_fitness_function
