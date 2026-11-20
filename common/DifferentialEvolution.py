import numpy as np
from Chromosome import Chromosome




class DifferentialEvolution:
    def __init__(self, population_size,fitness_function,no_genes,upper_genes_limit,lower_genes_limit,max_generations,f = 0.8,cr = 0.9,seed = 42):


        self.population_size = population_size
        self.fitness_function = fitness_function
        self.no_genes = no_genes
        self.upper_genes_limit = np.asarray(upper_genes_limit, dtype=float)
        self.lower_genes_limit = np.asarray(lower_genes_limit, dtype=float)
        self.F = f
        self.CR = cr
        self.max_generations = max_generations

        self.random_seed = np.random.default_rng(seed)

        self.population = self.initialize_population()






    def initialize_population(self):
        population = []
        for i in range(self.population_size):
            genes = self.lower_genes_limit + self.random_seed.random(self.no_genes) *(self.upper_genes_limit - self.lower_genes_limit)
            population.append(Chromosome(genes, self.compute_fitness(genes)))
        return population

    #curs 5 slide 14 (?)
    def selection(self,minimize = True):

        for i in range(self.population_size):
            mutant = self.mutate(i)
            trial = self.crossover(self.population[i].genes, mutant)

            f_trial = self.compute_fitness(trial)
            f_current = self.population[i].val_fitness_function

            if minimize:
                if f_trial <= f_current:
                    self.population[i] = Chromosome(trial, f_trial)
            else:
                if f_trial >= f_current:
                    self.population[i] = Chromosome(trial, f_trial)





    #asta e incrucisarea binomiala din cursu 5 slide 10
    def crossover(self,individ,mutant):
        trial_vector = individ.copy()
        k = self.random_seed.integers(0,self.no_genes)
        for j in range(self.no_genes) :
            if self.random_seed.random() < self.CR or j == k :
                trial_vector[j] = mutant[j]
        return trial_vector



    def mutate(self,individ):

        candidates = [k for k in range(self.population_size) if k != individ ]

        r1, r2, r3 = self.random_seed.choice(candidates, 3, replace=False)

        v_i = self.population[r1].genes + self.F*(self.population[r2].genes - self.population[r3].genes)

        v_i_clipped = np.clip(v_i,self.lower_genes_limit,self.upper_genes_limit)

        return v_i_clipped



    def compute_fitness(self,chromosome):
        return self.fitness_function(chromosome)


    def evolve(self,minimize = True):
        history = []
        for generation in range(self.max_generations):
            self.selection(minimize)

            fitness_vals = [x.val_fitness_function for x in self.population]

            best_idx = np.argmin(fitness_vals) if minimize else np.argmax(fitness_vals)

            history.append(fitness_vals[best_idx])

        return self.population[best_idx].genes, fitness_vals[best_idx], history
