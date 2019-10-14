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
from bspyalgo.algorithms.gradient.gd import get_gd
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


# %% MAIN
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # Get device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create data
    x = 0.5 * np.random.randn(10, 2)
    INPUTS = torch.Tensor(x).to(DEVICE)
    TARGETS = torch.Tensor(5. * np.ones((10, 1))).to(DEVICE)
    x = 0.5 * np.random.randn(4, 2)
    INPUTS_VAL = torch.Tensor(x).to(DEVICE)
    TARGETS_VAL = torch.Tensor(5. * np.ones((4, 1))).to(DEVICE)

    ALGO = get_algorithm('./configs/gd/gd_configs_template.json')
    DATA = ALGO.optimize(INPUTS, TARGETS, validation_data=(INPUTS_VAL, TARGETS_VAL))

    plt.figure()
    plt.plot(DATA.results['performance_history'])
    plt.title("Loss per epoch")
    plt.legend(["Training", "Validation"])
    plt.show()
