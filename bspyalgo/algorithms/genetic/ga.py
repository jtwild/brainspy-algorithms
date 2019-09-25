# -*- coding: utf-8 -*-
""" Contains class implementing the Genetic Algorithm for all SkyNEt platforms.
Created on Thu May 16 18:16:36 2019
@author: HCRuiz and A. Uitzetter
"""

import time
import random
import numpy as np

from bspyalgo.algorithms.genetic.core.fitness import choose_fitness_function
from bspyalgo.algorithms.genetic.core.evaluation import choose_evaluation_function
from bspyalgo.utils.io import create_directory_timestamp
from bspyalgo.interface.interface_manager import get_interface
from bspyalgo.utils.io import save
# TODO: Implement Plotter


class GA:
    '''This is a class implementing the genetic algorithm (GA).
    The methods implement the GA regardless of the platform being optimized,
    i.e. it can be used with the chip, the model or the physical simulations.

    ---------------------------------------------------------------------------

    Argument : config dictionary.

    The configuration dictionary must contain the following:
      - genes : number of genes in each individual
      - generange
      - genomes : number of individuals in the population
      - partition : a list with the partition for the different operations on the population
      - mutation_rate : rate of mutation applied to genes
      - Arguments for GenWaveform:
          o lengths : defines the input lengths (in ms) of the targets
          o slopes : defines the slopes (in ms) between targets

    Notes:
        * All other methods serve as default settings but they can be modified as well by
        the user via inheritance.
        * Requires args to GenWaveform because the method Evolve requires targets,
        so one single instance of GA can handle multiple tasks.
        *   The get_platform() function gets an instance of the platform used
        to evaluate the genomes in the population
        * The get_fitness() function gets the fitness function used as a score in GA

    '''

    def __init__(self, config_dict):
        self.load_configs(config_dict)

        # Internal parameters and variables
        self._next_state = None

    def get_torch_model_path(self):
        return self.config_dict['ga_evaluation_configs']['torch_model_path']

    def load_configs(self, config_dict):
        self.config_dict = config_dict
        self.save_path = config_dict['results_path']
        self.save_dir = config_dict['experiment_name']

        self.load_hyperparameters(config_dict)
        self.load_functions(config_dict)
        self.stop_thr = config_dict['stop_threshold']

    def load_functions(self, config_dict):
        self.evaluator = choose_evaluation_function(config_dict['ga_evaluation_configs'])
        self.fitness_function = choose_fitness_function(config_dict['hyperparameters']['fitness_function_type'])

    def load_hyperparameters(self, config_dict):
        # Define GA hyper-parameters
        self.generange = config_dict['hyperparameters']['generange']   # Voltage range of CVs      # Nr of individuals in population
        self.partition = config_dict['hyperparameters']['partition']   # Partitions of population

        self.mutationrate = config_dict['hyperparameters']['mutationrate']
        self.seed = config_dict['hyperparameters']['seed']
        self.generations = config_dict['hyperparameters']['epochs']

        self.genes = len(self.generange)
        self.genomes = sum(self.partition)
        self.config_dict['hyperparameters']['genes'] = self.genes
        self.config_dict['hyperparameters']['genomes'] = self.genomes
# %% Method implementing evolution

    def optimize(self, inputs, targets):
        self.interface = get_interface(inputs, targets, self.config_dict)
        return self.train()

    def train(self):
        np.random.seed(seed=self.seed)
        inputs_wfm, target_wfm = self.interface.reset(self.config_dict['hyperparameters'])

        self.pool = np.zeros((self.genomes, self.genes))
        self.opposite_pool = np.zeros((self.genomes, self.genes))
        for i in range(0, self.genes):
            self.pool[:, i] = np.random.uniform(self.generange[i][0], self.generange[i][1], size=(self.genomes,))

        # Evolution loop
        for gen in range(self.generations):
            start = time.time()

            self.outputs = self.evaluator.evaluate_population(inputs_wfm, self.pool, target_wfm)
            self.fitness = self.fitness_function(self.outputs, target_wfm)

            # Status print
            max_fit = max(self.fitness)
            print(f"Highest fitness: {max_fit}")

            self.interface.update({'generation': gen, 'genes': self.pool, 'outputs': self.outputs, 'fitness': self.fitness})
            if gen % 5 == 0:
                # Save generation
                print('--- checkpoint ---')
                self.save_results()

            end = time.time()
            print("Generation nr. " + str(gen + 1) + " completed; took " + str(end - start) + " sec.")
            if self.stop_condition(max_fit):
                print('--- final saving ---')
                self.save_results()
                break
            # Evolve to the next generation
            self.next_gen(gen)

        # Get best results
        self.interface.judge()
        return

    def stop_condition(self, max_fit):
        best = self.outputs[self.fitness == max_fit][0]
        corr = self.interface.corr(best)
        print(f"Correlation of fittest genome: {corr}")
        if corr >= self.stop_thr:
            print(f'Very high correlation achieved, evolution will stop! \
                  (correlaton threshold set to {self.stop_thr})')
        return corr >= self.stop_thr

    def save_results(self):
        save_directory = create_directory_timestamp(self.save_path, self.save_dir)
        save(mode='configs', path=save_directory, filename='configs.json', data=self.config_dict)
        save(mode='pickle', path=save_directory, filename='result.pickle', data=self.interface.results)
# %% Step to next generation

    def next_gen(self, gen):
        # Sort genePool based on fitness
        indices = np.argsort(self.fitness)
        indices = indices[::-1]
        self.pool = self.pool[indices]
        self.fitness = self.fitness[indices]
        # Copy the current pool
        self.newpool = self.pool.copy()
        # Determine which genomes are chosen to generate offspring
        # Note: twice as much parents are selected as there are genomes to be generated
        chosen = self.universal_sampling()
        # Generate offspring by means of crossover.
        # The crossover method returns 1 genome from 2 parents
        for i in range(0, len(chosen), 2):
            index_newpool = int(i / 2 + sum(self.partition[:1]))
            if chosen[i] == chosen[i + 1]:
                if chosen[i] == 0:
                    chosen[i] = chosen[i] + 1
                else:
                    chosen[i] = chosen[i] - 1
            # The individual with the highest fitness score is given as input first
            if chosen[i] < chosen[i + 1]:
                self.newpool[index_newpool, :] = self.crossover_blxab(self.pool[chosen[i], :], self.pool[chosen[i + 1], :])
            else:
                self.newpool[index_newpool, :] = self.crossover_blxab(self.pool[chosen[i + 1], :], self.pool[chosen[i], :])
        # The mutation rate is updated based on the generation counter
        self.update_mutation(gen)
        # Every genome, except the partition[0] genomes are mutated
        self.mutation()
        self.remove_duplicates()
        self.pool = self.newpool.copy()

# %%
#    ##########################################################################
#    ##################### Methods defining evolution #########################
#    ##########################################################################
# ------------------------------------------------------------------------------

    def universal_sampling(self):
        '''
        Sampling method: Stochastic universal sampling returns the chosen 'parents'
        '''
        no_genomes = 2 * self.partition[1]
        chosen = []
        probabilities = self.linear_rank()
        for i in range(1, len(self.fitness)):
            probabilities[i] = probabilities[i] + probabilities[i - 1]
        distance = 1 / (no_genomes)
        start = random.random() * distance
        for n in range(no_genomes):
            pointer = start + n * distance
            for i in range(len(self.fitness)):
                if pointer < probabilities[0]:
                    chosen.append(0)
                    break
                elif pointer < probabilities[i] and pointer >= probabilities[i - 1]:
                    chosen.append(i)
                    break
        chosen = random.sample(chosen, len(chosen))
        return chosen

    def linear_rank(self):
        '''
        Assigning probabilities: Linear ranking scheme used for stochastic universal sampling method.
        It returns an array with the probability that a genome will be chosen.
        The first probability corresponds to the genome with the highest fitness etc.
        '''
        maximum = 1.5
        rank = np.arange(self.genomes) + 1
        minimum = 2 - maximum
        probability = (minimum + ((maximum - minimum) * (rank - 1) / (self.genomes - 1))) / self.genomes
        return probability[::-1]

    def crossover_blxab(self, parent1, parent2):
        '''
        Crossover method: Blend alpha beta crossover returns a new genome (voltage combination)
        from two parents. Here, parent 1 has a higher fitness than parent 2
        '''

        alpha = 0.6
        beta = 0.4
        maximum = np.maximum(parent1, parent2)
        minimum = np.minimum(parent1, parent2)
        diff_maxmin = (maximum - minimum)
        offspring = np.zeros((parent1.shape))
        for i in range(len(parent1)):
            if parent1[i] > parent2[i]:
                offspring[i] = np.random.uniform(minimum[i] - diff_maxmin[i] * beta, maximum[i] + diff_maxmin[i] * alpha)
            else:
                offspring[i] = np.random.uniform(minimum[i] - diff_maxmin[i] * alpha, maximum[i] + diff_maxmin[i] * beta)
        for i in range(0, self.genes):
            if offspring[i] < self.generange[i][0]:
                offspring[i] = self.generange[i][0]
            if offspring[i] > self.generange[i][1]:
                offspring[i] = self.generange[i][1]
        return offspring

    def update_mutation(self, gen):
        '''
        Dynamic parameter control of mutation rate: This formula updates the mutation
        rate based on the generation counter
        '''
        pm_inv = 2 + 5 / (self.generations - 1) * gen
        self.mutationrate = 0.625 / pm_inv

    def mutation(self):
        '''
        Mutate all genes but the first partition[0] with a triangular
        distribution in generange with mode=gene to be mutated.
        '''
        np.random.seed(seed=None)
        mask = np.random.choice([0, 1], size=self.pool[self.partition[0]:].shape,
                                p=[1 - self.mutationrate, self.mutationrate])
        mutatedpool = np.zeros((self.genomes - self.partition[0], self.genes))

        for i in range(0, self.genes):
            if self.generange[i][0] == self.generange[i][1]:
                mutatedpool[:, i] = self.generange[i][0] * np.ones(mutatedpool[:, i].shape)
            else:
                mutatedpool[:, i] = np.random.triangular(self.generange[i][0], self.newpool[self.partition[0]:, i], self.generange[i][1])
        self.newpool[self.partition[0]:] = ((np.ones(self.newpool[self.partition[0]:].shape) - mask) * self.newpool[self.partition[0]:] + mask * mutatedpool)

    def remove_duplicates(self):
        np.random.seed(seed=None)
        '''
        Check the entire pool for any duplicate genomes and replace them by
        the genome put through a triangular distribution
        '''
        for i in range(self.genomes):
            for j in range(self.genomes):
                if(j != i and np.array_equal(self.newpool[i], self.newpool[j])):
                    for k in range(0, self.genes):
                        if self.generange[k][0] != self.generange[k][1]:
                            self.newpool[j][k] = np.random.triangular(self.generange[k][0], self.newpool[j][k], self.generange[k][1])
                        else:
                            self.newpool[j][k] = self.generange[k][0]

    # Methods required for evaluating the opposite pool

    def opposite(self):
        '''
        Define opposite pool
        '''
        opposite_pool = np.zeros((self.genomes, self.genes))
        for i in range(0, self.genes):
            opposite_pool[:, i] = self.generange[i][0] + self.generange[i][1] - self.pool[:, i]
        self.opposite_pool = opposite_pool

    def set_new_pool(self, indices):
        '''
        After evaluating the opposite pool, set the new pool.
        '''
        for k in range(len(indices)):
            if indices[k][0]:
                self.pool[k, :] = self.opposite_pool[k, :]
