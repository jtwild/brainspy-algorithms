# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:14:23 2019

@author: HCRuiz
"""
import os
import pickle
import numpy as np
from bspyalgo.utils.io import create_directory_timestamp
from bspyalgo.utils.io import save


class GAObserver:

    def __init__(self, config_dict):
        self.subject = None
        self.config_dict = config_dict

    def update(self, next_sate):
        gen = next_sate['generation']
        self.gene_array[gen, :, :] = next_sate['genes']
        self.output_array[gen, :, :] = next_sate['outputs']
        self.fitness_array[gen, :] = next_sate['fitness']
        if gen % 5 == 0:
            # Save generation
            print('--- checkpoint ---')
            self.save_results()

    def reset(self):
        # Define placeholders
        self.gene_array = np.zeros((self.subject.generations, self.subject.genomes, self.subject.genes))
        self.output_array = np.zeros((self.subject.generations, self.subject.genomes, len(self.subject.target_wfm)))
        self.fitness_array = -np.inf * np.ones((self.subject.generations, self.subject.genomes))
        # Initialize save directory
        self.save_directory = create_directory_timestamp(
            self.subject.savepath,
            self.subject.dirname)
        # Save experiment configurations
        self.config_dict['target'] = self.subject.target_wfm
        self.config_dict['inputs'] = self.subject.inputs_wfm
        self.config_dict['mask'] = self.subject.filter_array
        with open(os.path.join(self.save_directory, 'configs.pkl'), 'wb') as f:
            pickle.dump(self.config_dict, f)

    def judge(self):
        max_fitness = np.max(self.fitness_array)
        ind = np.unravel_index(np.argmax(self.fitness_array, axis=None), self.fitness_array.shape)
        best_genome = self.gene_array[ind]
        best_output = self.output_array[ind]

        return max_fitness, best_genome, best_output

    def save_results(self):
        save(mode='numpy', configs=self.config_dict, path=self.save_directory + '/ga_results/', filename='result',
             gene_array=self.gene_array,
             output_array=self.output_array,
             fitness_array=self.fitness_array,
             mask=self.subject.filter_array)
