import numpy as np

from bspyalgo.interface.waveforms import WaveformManager
from bspyalgo.algorithms.genetic.core.classifier import perceptron
from bspyalgo.utils.pytorch import TorchUtils


def get_interface(inputs, targets, configs):
    if configs['algorithm'] == 'genetic':
        return GAInterface(inputs, targets, configs['waveform_configs'])
    elif configs['algorithm'] == 'gradient_descent':
        raise NotImplementedError(f"Algorithm {configs['algorithm']} is not recognised. The interface for gradient descent has not yet been implemented'")
    else:
        raise NotImplementedError(f"Interface for algorithm {configs['algorithm']} is not recognised. Please try again with 'genetic' or 'gradient_descent'. ")


class GDInterfaceData:
    def __init__(self, inputs, targets, waveform_configs):
        assert len(inputs[0]) == len(targets), f'No. of input data {len(inputs)} does not match no. of targets {len(targets)}'
        inputs[0] = TorchUtils.get_tensor_from_list(inputs[0])
        targets[0] = TorchUtils.get_tensor_from_list(targets[0])
        inputs[1] = TorchUtils.get_tensor_from_list(inputs[1])
        targets[1] = TorchUtils.get_tensor_from_list(targets[1])

        self.data = [(inputs[0], targets[0]), (inputs[1], targets[1])]
        waveform_mgr = WaveformManager(waveform_configs)
        self.results = {}
        self.results['targets'] = waveform_mgr.waveform(targets)
        self.results['inputs'], self.results['mask'] = waveform_mgr.input_waveform(inputs)


class GAInterface:
    def __init__(self, inputs, targets, waveform_configs):
        assert len(inputs[0]) == len(targets), f'No. of input data {len(inputs)} does not match no. of targets {len(targets)}'
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
        self.process_results(best_output, max_fitness, best_corr, best_genome, accuracy)

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
        # return {'best_genome': best_genome, 'best_output': best_output, 'max_fitness': max_fitness, 'accuracy': accuracy}
