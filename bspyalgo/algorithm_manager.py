# -*- coding: utf-8 -*-
"""Contains the platforms used in all SkyNEt experiments to be optimized by Genetic Algo.

The classes in Platform must have a method self.evaluate() which takes as arguments
the inputs inputs_wfm, the gene pool and the targets target_wfm. It must return
outputs as numpy array of shape (len(pool), inputs_wfm.shape[-1]).

Created on Wed Aug 21 11:34:14 2019

@author: HCRuiz
"""
# import logging
import matplotlib.pyplot as plt
from bspyalgo.algorithms.genetic.ga import GA
from bspyalgo.algorithms.gradient.gd import get_gd
from bspyalgo.utils.io import load_configs

# TODO: Add chip platform
# TODO: Add simulation platform
# TODO: Target wave form as argument can be left out if output dimension is known internally

# %% Chip platform to measure the current output from
# voltage configurations of disordered NE systems


def get_algorithm(algorithm_type, configs_dir):

    # logging.basicConfig(filename='myapp.log', level=logging.INFO)
    if algorithm_type == 'genetic':
        configs = load_configs(configs_dir)
        return GA(configs)
    elif algorithm_type == 'gradient_descent':
        configs = load_configs(configs_dir)
        return get_gd(configs)
    else:
        raise NotImplementedError(f"Algorithm {configs['algorithm']} is not recognised. Please try again with 'genetic' or 'gradient_descent'")


# %% MAIN
if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # from bspyalgo.utils.pytorch import TorchUtils

    # import numpy as np
    # TorchUtils.set_force_cpu(True)

    # INPUTS = [[-1., 0.4, -1., 0.4, -0.8, 0.2], [-1., -1., 0.4, 0.4, 0., 0.], [-1., 1.4, -0.2, 0.1, -0.33, 0.2]]
    # INPUTS = np.array(INPUTS)
    # TARGETS = [1, 1, 0, 0, 1, 1]
    # TARGETS = np.array(TARGETS)

    # RESULT = get_algorithm('gradient_descent', './configs/gd/gd_configs_template.json').optimize(INPUTS, TARGETS)

    # plt.figure()
    # plt.plot(RESULT['best_output'])
    # plt.title(f'Best output for target {TARGETS}')
    # plt.show()
    import numpy as np
    import torch

    # Get device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Make config dict for GD
    SGD_HYPERPARAMETERS = {}
    SGD_HYPERPARAMETERS['nr_epochs'] = 3000
    SGD_HYPERPARAMETERS['batch_size'] = 128
    SGD_HYPERPARAMETERS['learning_rate'] = 1e-4
    SGD_HYPERPARAMETERS['save_interval'] = 10
    SGD_HYPERPARAMETERS['seed'] = 33
    SGD_HYPERPARAMETERS['betas'] = (0.9, 0.99)

    SGD_MODEL_CONFIGS = {}
    SGD_MODEL_CONFIGS['input_indices'] = [0, 1]
    SGD_MODEL_CONFIGS['torch_model_path'] = r'tmp/NN_model/checkpoint3000_02-07-23h47m.pt'

    SGD_CONFIGS = {}
    SGD_CONFIGS['platform'] = 'simulation'
    SGD_CONFIGS['get_network'] = 'dnpu'
    SGD_CONFIGS['results_path'] = r'tmp/NN_test/'
    SGD_CONFIGS['experiment_name'] = 'TEST'
    SGD_CONFIGS['hyperparameters'] = SGD_HYPERPARAMETERS
    SGD_CONFIGS['model_configs'] = SGD_MODEL_CONFIGS

    # Create data
    x = 0.5 * np.random.randn(10, len(SGD_MODEL_CONFIGS['input_indices']))
    inp_train = torch.Tensor(x).to(DEVICE)
    t_train = torch.Tensor(5. * np.ones((10, 1))).to(DEVICE)
    x = 0.5 * np.random.randn(4, len(SGD_MODEL_CONFIGS['input_indices']))
    inp_val = torch.Tensor(x).to(DEVICE)
    t_val = torch.Tensor(5. * np.ones((4, 1))).to(DEVICE)

    INPUTS, TARGETS = (inp_train, inp_val), (t_train, t_val)

    # RESULTS = get_gd(SGD_CONFIGS).optimize(INPUTS, TARGETS)
    RESULTS = get_algorithm('gradient_descent', './configs/gd/gd_configs_template.json').optimize(INPUTS, TARGETS)

    plt.figure()
    plt.plot(RESULTS['costs'])
    plt.title("Loss per epoch")
    plt.legend(["Training", "Validation"])
    plt.show()
