import importlib
import numpy as np

from bspyalgo.utils.pytorch import TorchModel


def choose_evaluation_function(configs):
    '''Gets the fitness function used in GA from the module FitnessFunctions
    The fitness functions must take two arguments, the outputs of the black-box and the target
    and must return a numpy array of scores of size len(outputs).
    '''
    if configs['platform'] == 'hardware':
        return HardwareEvaluator(configs)
    elif configs['platform'] == 'simulation':
        if configs['simulation_type'] == 'neural_network':
            return NeuralNetworkSimulationEvaluator(configs)
        elif configs['simulation_type'] == 'kinetic_monte_carlo':
            return KineticMonteCarloEvaluator(configs)
        else:
            raise NotImplementedError(f"{configs['simulation_type']} 'simulation_type' configuration is not recognised. The simulation type has to be defined as 'neural_network' or 'kinetic_monte_carlo'. ")
        return NeuralNetworkSimulationEvaluator
    else:
        raise NotImplementedError(f"Platform {configs['platform']} is not recognised. The platform has to be either 'hardware' or 'simulation'")


class HardwareEvaluator:
    '''Platform which connects to a single boron-doped silicon chip through
    digital-to-analog and analog-to-digital converters. '''

    def __init__(self, platform_dict):
        pass

    def evaluate_population(self, inputs_wfm, gene_pool, target_wfm):
        '''Optimisation function of the platform '''
        pass


class NeuralNetworkSimulationEvaluator:
    '''Platform which simulates a single boron-doped silicon chip using a
    torch-based neural network. '''

    def __init__(self, evaluation_configs):
        # Import required packages
        self.torch = importlib.import_module('torch')
        # self.staNNet = importlib.import_module('SkyNEt.modules.Nets.staNNet').staNNet

        # Initialize NN
        # self.net = self.staNNet(platform_dict['path2NN'])
        self.torch_model = TorchModel()
        self.torch_model.load_model(evaluation_configs['torch_model_path'])
        self.torch_model.model.eval()
        # Set parameters
        self.amplification = self.torch_model.info_dict['amplification']

        self.nn_input_dim = len(self.torch_model.info_dict['amplitude'])
        self.input_indices = evaluation_configs['input_indices']
        self.nr_control_genes = self.nn_input_dim - len(self.input_indices)

        print(f'Initializing NN platform with {self.nr_control_genes} control genes')
        self.control_indices = np.delete(np.arange(self.nn_input_dim), self.input_indices)
        print(f'Input indices chosen : {self.input_indices} \n Control indices chosen: {self.control_indices}')

        if evaluation_configs.__contains__('trafo_index'):
            self.trafo_indx = evaluation_configs['trafo_index']
            self.trafo = evaluation_configs['trafo']  # explicitly define the trafo func
        else:
            self.trafo_indx = None
            self.trafo = lambda x, y: x  # define trafo as identity

    def evaluate_population(self, inputs_wfm, gene_pool, target_wfm):
        '''Optimisation function of the platform '''
        genomes = len(gene_pool)
        output_popul = np.zeros((genomes, target_wfm.shape[-1]))

        for j in range(genomes):
            # Feed input to NN
            # target_wfm.shape, genePool.shape --> (time-steps,) , (nr-genomes,nr-genes)
            control_voltage_genes = np.ones_like(target_wfm)[:, np.newaxis] * gene_pool[j, self.control_indices, np.newaxis].T
            control_voltage_genes_index = np.delete(np.arange(self.nn_input_dim), self.input_indices)
            # g.shape,x.shape --> (time-steps,nr-CVs) , (input-dim, time-steps)
            x_dummy = np.empty((control_voltage_genes.shape[0], self.nn_input_dim))  # dims of input (time-steps)xD_in
            # Set the input scaling
            # inputs_wfm.shape -> (nr-inputs,nr-time-steps)
            x_dummy[:, self.input_indices] = self.trafo(inputs_wfm, gene_pool[j, self.trafo_indx]).T
            x_dummy[:, control_voltage_genes_index] = control_voltage_genes

            output = self.torch_model.inference_in_nanoamperes(x_dummy)

            output_popul[j] = output[:, 0]

        return output_popul


class KineticMonteCarloEvaluator:
    '''Platform which connects to a single boron-doped silicon chip through
    digital-to-analog and analog-to-digital converters. '''

    def __init__(self, platform_dict):
        pass

    def evaluate_population(self, inputs_wfm, gene_pool, target_wfm):
        '''Optimisation function of the platform '''
        pass
