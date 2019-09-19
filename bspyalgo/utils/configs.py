import json
import codecs


def load_configs(file):
    object_text = codecs.open(file, 'r', encoding='utf-8').read()
    return json.loads(object_text)


def save_configs(configs, file):
    json.dump(configs, open(file, 'w'))


if __name__ == '__main__':
    import numpy as np

    GA_EVALUATION_CONFIGS = {}
    GA_EVALUATION_CONFIGS['platform'] = 'simulation'
    GA_EVALUATION_CONFIGS['simulation_type'] = 'neural_network'
    # platform['torch_model_path'] = r'D:\UTWENTE\PROJECTS\DARWIN\Data\Mark\MSE_n_d10w90_200ep_lr1e-3_b1024_b1b2_0.90.75.pt'
    # platform['torch_model_path'] = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/Devices/Marks_Data/April_2019/MSE_n_d10w90_200ep_lr1e-3_b1024_b1b2_0.90.75.pt'
    GA_EVALUATION_CONFIGS['torch_model_path'] = r'/home/unai/Documents/3-programming/boron-doped-silicon-chip-simulation/checkpoint3000_02-07-23h47m.pt'
    GA_EVALUATION_CONFIGS['amplification'] = 10.
    GA_EVALUATION_CONFIGS['input_indices'] = [0, 5, 6]  # indices of NN input
    GA_EVALUATION_CONFIGS['control_indices'] = np.arange(4).tolist()  # indices of gene array

    GA_CONFIGS = {}
    GA_CONFIGS['partition'] = [5] * 5  # Partitions of population
    # Voltage range of CVs in V
    GA_CONFIGS['generange'] = [[-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-0.7, 0.3], [-0.7, 0.3], [1, 1]]
    # GA_CONFIGS['genes'] = len(GA_CONFIGS['generange'])    # Nr of genes
    # GA_CONFIGS['genomes'] = sum(GA_CONFIGS['partition'])  # Nr of individuals in population
    GA_CONFIGS['mutationrate'] = 0.1

    # Parameters to define target waveforms
    GA_CONFIGS['lengths'] = 80     # Length of data in the waveform
    GA_CONFIGS['slopes'] = 0        # Length of ramping from one value to the next
    # Parameters to define task
    GA_CONFIGS['fitness_function_type'] = 'corrsig_fit'  # 'corr_fit'

    GA_CONFIGS['ga_evaluation_configs'] = GA_EVALUATION_CONFIGS    # Dictionary containing all variables for the platform

    save_configs(GA_CONFIGS, './configs/ga_configs.json')
