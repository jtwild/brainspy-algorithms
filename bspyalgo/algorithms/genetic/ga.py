# -*- coding: utf-8 -*-
""" Contains class implementing the Genetic Algorithm for all SkyNEt platforms.
Created on Thu May 16 18:16:36 2019
@author: HCRuiz and A. Uitzetter
"""

import os
import random
import numpy as np

from tqdm import trange

from bspyalgo.algorithms.genetic.core.fitness import choose_fitness_function
from bspyalgo.utils.io import create_directory, create_directory_timestamp, save
from bspyalgo.algorithms.genetic.core.trafo import get_trafo
from bspyalgo.algorithms.genetic.core.data import GAData
from bspyproc.bspyproc import get_processor
from bspyproc.utils.waveform import generate_slopped_plato
from bspyproc.utils.control import merge_inputs_and_control_voltages_in_numpy, get_control_voltage_indices
from bspyalgo.utils.io import create_directory, save

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

    def __init__(self, configs, is_main=False):
        self.init_configs(configs, is_main)
        # Internal parameters and variables
        self._next_state = None

    def init_configs(self, configs):
        self.configs = configs
        self.init_processor(configs['processor'])
        self.init_hyperparameters(configs['hyperparameters'])
        self.init_checkpoint_configs(configs['checkpoints'])
        self.default_output_dir = 'reproducibility'

    def init_dirs(self, base_dir, is_main=False):
        if is_main:
            base_dir = create_directory_timestamp(base_dir,'genetic_algorithm_data')
        else:
            base_dir = os.path.join(base_dir, 'genetic_algorithm_data')
            create_directory(base_dir)
        self.default_output_dir = os.path.join(base_dir, 'reproducibility')
        create_directory(self.default_output_dir)
        self.default_checkpoints_dir = os.path.join(base_dir, 'checkpoints')
        create_directory(self.default_checkpoints_dir)

    def init_processor(self, configs):
        self.input_electrode_no = configs['input_electrode_no']
        self.input_indices = configs['input_indices']
        self.nr_control_genes = self.input_electrode_no - len(self.input_indices)
        self.control_voltage_indices = get_control_voltage_indices(self.input_indices, self.input_electrode_no)
        self.processor = get_processor(configs)
        self.load_control_voltage_configs(configs)
        self.clipvalue = configs['waveform']['output_clipping_value'] * self.processor.get_amplification_value()  # 3.55

    def load_control_voltage_configs(self, configs):
        if configs['platform'] == 'hardware':
            self.base_slopped_plato = generate_slopped_plato(configs['waveform']['slope_lengths'], configs['shape'])
            self.get_control_voltages = self.get_safety_formatted_control_voltages
        else:
            self.get_control_voltages = self.get_regular_control_voltages

    def init_hyperparameters(self, configs):
        # Define GA hyper-parameters
        self.generange = configs['generange']   # Voltage range of CVs      # Nr of individuals in population
        self.partition = configs['partition']   # Partitions of population

        # self.mutationrate = configs['mutationrate']
        self.seed = configs['seed']
        self.generations = configs['epochs']

        self.genes = len(self.generange)
        self.genomes = sum(self.partition)
        self.configs['hyperparameters']['genes'] = self.genes
        self.configs['hyperparameters']['genomes'] = self.genomes
        self.stop_thr = configs['stop_threshold']

        self.fitness_function = choose_fitness_function(configs['fitness_function_type'])
        self.load_trafo(configs['transformation'])

    def load_trafo(self, configs):
        self.gene_trafo_index = configs['gene_trafo_index']
        if self.gene_trafo_index is not None:
            self._input_trafo = get_trafo(configs['trafo_function'])
        else:
            self._input_trafo = lambda x, y: x  # define trafo as identity

    def init_checkpoint_configs(self, configs):
        self.use_checkpoints = configs['use_checkpoints']
        self.checkpoint_frequency = configs['save_interval']

    def get_torch_model_path(self):
        return self.configs['ga_evaluation_configs']['torch_model_path']

# %% Method implementing evolution
# TODO: Implement feeding the validation_data and mask as optional kwargs

    def optimize(self, inputs, targets, validation_data=(None, None), mask=None, save_data=True):
        '''
            inputs = The inputs of the algorithm. They need to be in numpy. The GA also requires the input to be a waveform.
            targets = The targets to which the algorithm will try to fit the inputs. They need to be in numpy.
            validation_data = In some cases, it is required to provide the validation data in the form of (training_data, validation_data)
            mask = In cases where the input is a waveform, the mask helps filtering the slopes of the waveform
        '''

        np.random.seed(seed=self.seed)
        if (validation_data[0] is not None) and (validation_data[1] is not None):
            print('======= WARNING: Validation data is not processed in GA =======')

        self.data = GAData(inputs, targets, mask, self.configs['hyperparameters'])
        self.pool = np.zeros((self.genomes, self.genes))
        self.opposite_pool = np.zeros((self.genomes, self.genes))
        for i in range(0, self.genes):
            self.pool[:, i] = np.random.uniform(self.generange[i][0], self.generange[i][1], size=(self.genomes,))

        # Evolution loop
        looper = trange(self.generations, desc='Initialising', leave=False)
        for gen in looper:

            self.outputs = self.evaluate_population(inputs, self.pool, self.data.results['targets'])
            self.fitness = self.fitness_function(self.outputs[:, self.data.results['mask']],
                                                 self.data.results['targets'][self.data.results['mask']],
                                                 clipvalue=self.clipvalue)

            self.data.update({'generation': gen, 'genes': self.pool, 'outputs': self.outputs, 'fitness': self.fitness})
            looper.set_description(self.data.get_description(gen))  # , end - start))

            if self.check_threshold(save_data):
                break

            if (self.use_checkpoints is True and gen % self.checkpoint_frequency == 0):
                save(mode='pickle', file_path=os.path.join(self.default_checkpoints_dir, 'result.pickle'), data=self.data.results)

            self.next_gen(gen)

        self.save_results(save_data)
        return self.data

    def evaluate_population(self, inputs_wfm, gene_pool, target_wfm):
        '''Optimisation function of the platform '''
        genomes = len(gene_pool)
        output_popul = np.zeros((genomes,) + (len(inputs_wfm), 1))
        for j in range(genomes):

            control_voltage_genes = self.get_control_voltages(gene_pool[j], len(inputs_wfm))  # , gene_pool[j, self.gene_trafo_index]
            inputs_without_offset_and_scale = self._input_trafo(inputs_wfm, gene_pool[j, self.gene_trafo_index])

            output_popul[j] = self.processor.get_output(merge_inputs_and_control_voltages_in_numpy(inputs_without_offset_and_scale, control_voltage_genes, self.input_indices, self.control_voltage_indices))
        return output_popul

    def get_regular_control_voltages(self, gene_pool, input_length):
        return np.broadcast_to(gene_pool, (input_length, len(gene_pool)))

    def get_safety_formatted_control_voltages(self, gene_pool, input_length):
        control_voltages = np.empty([input_length, len(gene_pool)])
        for i in range(len(gene_pool)):
            control_voltages[:, i] = self.base_slopped_plato * gene_pool[i]
        return control_voltages

    def save_results(self, save_data):
        if save_data:
            save(mode='pickle', file_path=os.path.join(self.default_output_dir, 'results.pickle'), data=self.data.results)
            save(mode='configs', file_path=os.path.join(self.default_output_dir, 'configs.json'), data=self.configs)
# %% Step to next generation

    def check_threshold(self, save_data):
        if self.data.results['correlation'] >= self.stop_thr:
            print(f"  STOPPED: Correlation {self.data.results['correlation']} reached {self.stop_thr}. ")
            self.save_results(save_data)
            return True
        return False

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

    def close(self):
        """
        Experiments in hardware require that the connection with the drivers is closed.
        This method helps closing this connection when necessary.
        """
        try:
            self.processor.close_tasks()
        except AttributeError:
            print('There is no closing function for the current processor configuration. Skipping.')
