# -*- coding: utf-8 -*-
"""Contains the platforms used in all SkyNEt experiments to be optimized by Genetic Algo.

The classes in Platform must have a method self.evaluate() which takes as arguments
the inputs inputs_wfm, the gene pool and the targets target_wfm. It must return
outputs as numpy array of shape (len(pool), inputs_wfm.shape[-1]).

Created on Wed Aug 21 11:34:14 2019

@author: HCRuiz
"""

from bspyalgo.algorithms.genetic.ga import GA

# TODO: Add chip platform
# TODO: Add simulation platform
# TODO: Target wave form as argument can be left out if output dimension is known internally

# %% Chip platform to measure the current output from
# voltage configurations of disordered NE systems


def get_algorithm(configs):
    if configs['algorithm'] == 'genetic':
        return GA(configs['ga_configs'])
    elif configs['algorithm'] == 'gradient_descent':
        raise NotImplementedError(f"The gradient descent algorithm has still not been implemented.")
    else:
        raise NotImplementedError(f"Algorithm {configs['algorithm']} is not recognised. Please try again with 'genetic' or 'gradient_descent'")


# %% MAIN
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from bspyalgo.utils.pytorch import TorchUtils
    from bspyalgo.utils.configs import load_configs


#    from bspyalgo.algorithm_manager import get_algorithm

    TorchUtils.set_force_cpu(True)
    # Define platform
    # GA_EVALUATION_CONFIGS = {}
    # GA_EVALUATION_CONFIGS['platform'] = 'simulation'
    # GA_EVALUATION_CONFIGS['simulation_type'] = 'neural_network'
    # # platform['path2NN'] = r'D:\UTWENTE\PROJECTS\DARWIN\Data\Mark\MSE_n_d10w90_200ep_lr1e-3_b1024_b1b2_0.90.75.pt'
    # # platform['path2NN'] = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/Devices/Marks_Data/April_2019/MSE_n_d10w90_200ep_lr1e-3_b1024_b1b2_0.90.75.pt'
    # GA_EVALUATION_CONFIGS['torch_model_path'] = r'/home/unai/Documents/3-programming/boron-doped-silicon-chip-simulation/checkpoint3000_02-07-23h47m.pt'
    # GA_EVALUATION_CONFIGS['amplification'] = 10.
    # GA_EVALUATION_CONFIGS['input_indices'] = [0, 5, 6]  # indices of NN input
    # GA_EVALUATION_CONFIGS['control_indices'] = np.arange(4).tolist()  # indices of gene array

    # GA_CONFIGS = {}
    # GA_CONFIGS['partition'] = [5] * 5  # Partitions of population
    # # Voltage range of CVs in V
    # GA_CONFIGS['generange'] = [[-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-0.7, 0.3], [-0.7, 0.3], [1, 1]]
    # GA_CONFIGS['genes'] = len(GA_CONFIGS['generange'])    # Nr of genes
    # GA_CONFIGS['genomes'] = sum(GA_CONFIGS['partition'])  # Nr of individuals in population
    # GA_CONFIGS['mutationrate'] = 0.1

    # # Parameters to define target waveforms
    # GA_CONFIGS['lengths'] = 80     # Length of data in the waveform
    # GA_CONFIGS['slopes'] = 0        # Length of ramping from one value to the next
    # # Parameters to define task
    # GA_CONFIGS['fitness_function_type'] = 'corrsig_fit'  # 'corr_fit'

    # GA_CONFIGS['ga_evaluation_configs'] = GA_EVALUATION_CONFIGS    # Dictionary containing all variables for the platform

    # ALGORITHM_CONFIGS = {}
    # ALGORITHM_CONFIGS['algorithm'] = 'genetic'
    # ALGORITHM_CONFIGS['ga_configs'] = GA_CONFIGS

   # obj_text = codecs.open('conf.json', 'r', encoding='utf-8').read()
   # ALGORITHM_CONFIGS = json.loads(obj_text)

    ALGORITHM = get_algorithm(load_configs())

    INPUTS = [[-1., 0.4, -1., 0.4, -0.8, 0.2], [-1., -1., 0.4, 0.4, 0., 0.], [-1., 1.4, -0.2, 0.1, -0.33, 0.2]]
    TARGETS = [1, 1, 0, 0, 1, 1]

    BEST_GENOME, BEST_OUTPUT, MAX_FITNESS, ACCURACY = ALGORITHM.optimize(INPUTS, TARGETS)

    plt.figure()
    plt.plot(BEST_OUTPUT)
    plt.title(f'Best output for target {TARGETS}')
    plt.show()

    # json.dump(ALGORITHM_CONFIGS, open('conf.json', 'w'))
    # with io.open('data.yaml', 'w', encoding='utf8') as outfile:
    #     yaml.dump(ALGORITHM_CONFIGS, outfile, default_flow_style=False, allow_unicode=True)
