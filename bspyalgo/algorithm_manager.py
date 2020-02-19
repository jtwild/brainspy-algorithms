# -*- coding: utf-8 -*-
"""Contains the platforms used in all SkyNEt experiments to be optimized by Genetic Algo.

The classes in Platform must have a method self.evaluate() which takes as arguments
the inputs inputs_wfm, the gene pool and the targets target_wfm. It must return
outputs as numpy array of shape (len(pool), inputs_wfm.shape[-1]).

Created on Wed Aug 21 11:34:14 2019

@author: HCRuiz
"""
# import logging
from bspyalgo.algorithms.genetic.ga import GA
from bspyalgo.algorithms.gradient.gd import GD
from bspyalgo.utils.io import load_configs

# TODO: Add chip platform
# TODO: Add simulation platform
# TODO: Target wave form as argument can be left out if output dimension is known internally


def get_algorithm(configs):
    if(isinstance(configs, str)):       # Enable to load configs as a path to configurations or as a dictionary
        configs = load_configs(configs)

    if configs['algorithm'] == 'genetic':
        return GA(configs)
    elif configs['algorithm'] == 'gradient_descent':
        return get_gd(configs)
    else:
        raise NotImplementedError(f"Algorithm {configs['algorithm']} is not recognised. Please try again with 'genetic' or 'gradient_descent'")


def get_gd(configs):
    if configs['processor']['platform'] == 'hardware':
        raise NotImplementedError('Hardware platform not implemented')
        # TODO: Implement the lock in algorithm class
    elif configs['processor']['platform'] == 'simulation':
        return GD(configs)
    else:
        raise NotImplementedError('Platform not implemented')

