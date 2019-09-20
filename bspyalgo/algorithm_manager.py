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

# TODO: Add chip platform
# TODO: Add simulation platform
# TODO: Target wave form as argument can be left out if output dimension is known internally

# %% Chip platform to measure the current output from
# voltage configurations of disordered NE systems


def get_algorithm(algorithm_type):

    # logging.basicConfig(filename='myapp.log', level=logging.INFO)
    if algorithm_type == 'genetic':
        configs = load_configs('./configs/ga/ga_configs_template.json')
        return GA(configs)
    elif algorithm_type == 'gradient_descent':
        configs = load_configs('./configs/gd/gd_configs_template.json')
        raise NotImplementedError(f"The gradient descent algorithm has still not been implemented.")
    else:
        raise NotImplementedError(f"Algorithm {configs['algorithm']} is not recognised. Please try again with 'genetic' or 'gradient_descent'")


# %% MAIN
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from bspyalgo.utils.pytorch import TorchUtils
    from bspyalgo.utils.io import load_configs

    TorchUtils.set_force_cpu(True)

    INPUTS = [[-1., 0.4, -1., 0.4, -0.8, 0.2], [-1., -1., 0.4, 0.4, 0., 0.], [-1., 1.4, -0.2, 0.1, -0.33, 0.2]]
    TARGETS = [1, 1, 0, 0, 1, 1]

    RESULT = get_algorithm('genetic').optimize(INPUTS, TARGETS)

    plt.figure()
    plt.plot(RESULT['best_output'])
    plt.title(f'Best output for target {TARGETS}')
    plt.show()
