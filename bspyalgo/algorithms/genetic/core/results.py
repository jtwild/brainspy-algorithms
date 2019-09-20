# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:14:23 2019

@author: HCRuiz
"""
# import os
# import pickle
import numpy as np

from bspyalgo.algorithms.genetic.core.waveforms import WaveformManager
from bspyalgo.algorithms.genetic.core.classifier import perceptron


class GAResults:

    def __init__(self, inputs, targets, waveform_configs):
        waveform_mgr = WaveformManager(waveform_configs)
        self.results = {}
        self.results['targets'] = waveform_mgr.waveform(targets)
        self.results['inputs'], self.results['mask'] = waveform_mgr.input_waveform(inputs)

    def update(self, next_sate):
        gen = next_sate['generation']
        self.results['gene_array'][gen, :, :] = next_sate['genes']
        self.results['output_array'][gen, :, :] = next_sate['outputs']
        self.results['fitness_array'][gen, :] = next_sate['fitness']

    def reset(self, hyperparams):
        # Define placeholders
        self.results['gene_array'] = np.zeros((hyperparams['epochs'], hyperparams['genomes'], hyperparams['genes']))
        self.results['output_array'] = np.zeros((hyperparams['epochs'], hyperparams['genomes'], len(self.results['targets'])))
        self.results['fitness_array'] = -np.inf * np.ones((hyperparams['epochs'], hyperparams['genomes']))
        return self.results['inputs'], self.results['targets']

    def judge(self):
        max_fitness = np.max(self.results['fitness_array'])
        ind = np.unravel_index(np.argmax(self.results['fitness_array'], axis=None), self.results['fitness_array'].shape)
        best_genome = self.results['gene_array'][ind]
        best_output = self.results['output_array'][ind]
        best_corr = self.corr(best_output)

        y = best_output[self.results['mask']][:, np.newaxis]
        trgt = self.results['targets'][self.results['mask']][:, np.newaxis]
        accuracy, _, _ = perceptron(y, trgt)
        return self.process_results(best_output, max_fitness, best_corr, best_genome, accuracy)

    def corr(self, x):
        x = x[self.results['mask']][np.newaxis, :]
        y = self.results['targets'][self.results['mask']][np.newaxis, :]
        return np.corrcoef(np.concatenate((x, y), axis=0))[0, 1]
        # return self.results['corr']

    def process_results(self, best_output, max_fitness, best_corr, best_genome, accuracy):  # print(best_output.shape,self.target_wfm.shape)
        print(f'\n========================= BEST SOLUTION =======================')
        print('Fitness: ', max_fitness)
        print('Correlation: ', best_corr)
        print(f'Genome:\n {best_genome}')

        print('Accuracy: ', accuracy)
        print('===============================================================')

        self.results['max_fitness'] = max_fitness
        self.results['best_genome'] = best_genome
        self.results['best_output'] = best_output
        self.results['accuracy'] = accuracy
        return {'best_genome': best_genome, 'best_output': best_output, 'max_fitness': max_fitness, 'accuracy': accuracy}
