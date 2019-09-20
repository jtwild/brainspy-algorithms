'''
Library that handles saving data results from the execution of the algorithm.
'''
import os
import time
import json
import codecs

import numpy as np


def save(mode, configs, path, filename, **kwargs):
    create_directory(path)
    save_configs(configs, os.path.join(path, 'configs.json'))
    if mode != 'configs':
        if mode == 'numpy':
            np.savez(os.path.join(path, filename), **kwargs)
        elif mode == 'torch':
            raise NotImplementedError(f"Saving results for torch has still not been implemented")
        else:
            raise NotImplementedError(f"Mode {mode} is not recognised. Please choose a value between 'numpy', 'torch' and 'configs'.")


def load_configs(file):
    object_text = codecs.open(file, 'r', encoding='utf-8').read()
    return json.loads(object_text)


def save_configs(configs, file):
    for key in configs:
        if type(configs[key]) is np.ndarray:
            configs[key] = configs[key].tolist()
    json.dump(configs, open(file, 'w'))


def create_directory(path):
    '''
    This function checks if there exists a directory filepath+datetime_name.
    If not it will create it and return this path.
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def create_directory_timestamp(path, name):
    datetime = time.strftime("%Y_%m_%d_%H%M%S")
    path = path + datetime + '_' + name
    return create_directory(path)


if __name__ == '__main__':
    import numpy as np

    GA_EVALUATION_CONFIGS = {}

    GA_EVALUATION_CONFIGS['platform'] = 'simulation'
    GA_EVALUATION_CONFIGS['simulation_type'] = 'neural_network'
    GA_EVALUATION_CONFIGS['torch_model_path'] = r'/home/unai/Documents/3-programming/boron-doped-silicon-chip-simulation/checkpoint3000_02-07-23h47m.pt'
    # platform['torch_model_path'] = r'D:\UTWENTE\PROJECTS\DARWIN\Data\Mark\MSE_n_d10w90_200ep_lr1e-3_b1024_b1b2_0.90.75.pt'
    # platform['torch_model_path'] = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/Devices/Marks_Data/April_2019/MSE_n_d10w90_200ep_lr1e-3_b1024_b1b2_0.90.75.pt'

    GA_EVALUATION_CONFIGS['input_indices'] = [0, 5, 6]  # indices of NN input
    GA_EVALUATION_CONFIGS['control_indices'] = np.arange(4).tolist()  # indices of gene array
    GA_EVALUATION_CONFIGS['amplification'] = 10.

    # Parameters to define target waveforms
    GA_WAVEFORM_CONFIGS = {}
    GA_WAVEFORM_CONFIGS['lengths'] = 80     # Length of data in the waveform
    GA_WAVEFORM_CONFIGS['slopes'] = 0        # Length of ramping from one value to the next

    GA_HYPERPARAMETERS = {}

    # Voltage range of CVs in V
    GA_HYPERPARAMETERS['generange'] = [[-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-0.7, 0.3], [-0.7, 0.3], [1, 1]]
    GA_HYPERPARAMETERS['partition'] = [5] * 5  # Partitions of population
    GA_HYPERPARAMETERS['mutationrate'] = 0.1
    GA_HYPERPARAMETERS['epochs'] = 100
    GA_HYPERPARAMETERS['fitness_function_type'] = 'corrsig_fit'
    GA_HYPERPARAMETERS['seed'] = None
    # Parameters to define the fitness function and the evaluation method (hardware or simulation dependent)

    GA_CONFIGS = {}

    GA_CONFIGS['save_path'] = r'./tmp/output/genetic_algorithm/'
    GA_CONFIGS['directory'] = 'OPTIMIZATION_TEST'

    GA_CONFIGS['hyperparameters'] = GA_HYPERPARAMETERS
    GA_CONFIGS['waveform_configs'] = GA_WAVEFORM_CONFIGS
    GA_CONFIGS['ga_evaluation_configs'] = GA_EVALUATION_CONFIGS    # Dictionary containing all variables for the platform

    save_configs(GA_CONFIGS, './configs/ga_configs.json')
