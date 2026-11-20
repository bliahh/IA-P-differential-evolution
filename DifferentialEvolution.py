import numpy as np

class DifferentialEvolution:
    def __init__(self, population_size,fitness_function,no_genes,upper_genes_limit,lower_genes_limit,no_chromosoms,max_generations,f = 0.8,cr = 0.9,seed = 42):


        self.population_size = population_size
        self.fitness_function = fitness_function
        self.no_genes = no_genes
        self.upper_genes_limit = np.asarray(upper_genes_limit, dtype=float)
        self.lower_genes_limit = np.asarray(lower_genes_limit, dtype=float)
        self.no_chromosoms = no_chromosoms
        self.F = f
        self.CR = cr
        self.max_generations = max_generations

        self.random_seed = np.random.default_rng(seed)

        self.population = self.initialize_population()

        self.fitness = np.array([self.fitness_function(x) for x in self.population])





    def initialize_population(self):
        return self.lower_genes_limit + self.random_seed.random((self.population_size, self.no_genes)) *(self.upper_genes_limit - self.lower_genes_limit)

    #curs 5 slide 14 (?)
    def selection(self,minimize = True):

        for i in range(self.population_size):
            mutant = self.mutate(i)
            trial = self.crossover(self.population[i], mutant)

            f_trial = self.compute_fitness(trial)
            f_current = self.compute_fitness(self.population[i])

            if minimize:
                if f_trial <= f_current:
                    self.population[i] = trial
            else:
                if f_trial >= f_current:
                    self.population[i] = trial





    #asta e incrucisarea binomiala din cursu 5 slide 10
    def crossover(self,individ,mutant):
        trial_vector = individ.copy()
        k = self.random_seed.integers(0,self.no_genes)
        for j in range(self.no_genes) :
            if np.random.random() < self.CR or j == k :
                trial_vector[j] = mutant[j]
        return trial_vector



    def mutate(self,individ):

        candidates = [k for k in range(self.population_size) if k != individ ]

        r1, r2, r3 = self.random_seed.choice(candidates, 3, replace=False)

        v_i = self.population[r1] + self.F*(self.population[r2] - self.population[r3])

        v_i_clipped = np.clip(v_i,self.lower_genes_limit,self.upper_genes_limit)

        return v_i_clipped



    def compute_fitness(self,chromosome):
        return self.fitness_function(chromosome)


    def evolve(self,minimize = True):
        history = []
        for generation in range(self.max_generations):
            self.selection(minimize)

            fitness_vals = [self.compute_fitness(x) for x in self.population]

            best_idx = np.argmin(fitness_vals) if minimize else np.argmax(fitness_vals)

            history.append(fitness_vals[best_idx])

        return self.population[best_idx], fitness_vals[best_idx], history

