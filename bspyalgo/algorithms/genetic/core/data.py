import numpy as np


class GAData:
    def __init__(self, inputs, targets, mask, hyperparams):  # , waveform_configs):
        assert len(inputs[0]) == len(targets), f'No. of input data {len(inputs)} does not match no. of targets {len(targets)}'
        self.results = {}
        self.results['inputs'] = inputs
        self.results['targets'] = targets
        self.results['mask'] = mask
        self.reset(hyperparams)

    def update(self, next_sate):
        gen = next_sate['generation']
        self.results['control_voltage_array'][gen, :, :] = next_sate['genes']
        self.results['output_current_array'][gen, :, :] = next_sate['outputs']
        self.results['fitness_array'][gen, :] = next_sate['fitness']

    def reset(self, hyperparams):
        # Define placeholders
        self.results['control_voltage_array'] = np.zeros((hyperparams['epochs'], hyperparams['genomes'],
                                                          hyperparams['genes']))
        self.results['output_current_array'] = np.zeros((hyperparams['epochs'], hyperparams['genomes'])
                                                        + self.results['targets'].shape)
        self.results['fitness_array'] = -np.inf * np.ones((hyperparams['epochs'], hyperparams['genomes']))
        # return self.results['inputs'], self.results['targets']

    def judge(self):
        ind = np.unravel_index(np.argmax(self.results['fitness_array'], axis=None), self.results['fitness_array'].shape)
        self.results['best_output'] = self.results['output_current_array'][ind]
        self.results['best_control_voltage'] = self.results['control_voltage_array'][ind]
        self.results['max_fitness'] = np.max(self.results['fitness_array'])
        self.print_results()

    def print_results(self):  # print(best_output.shape,self.target_wfm.shape)
        print(f'\n========================= BEST SOLUTION =======================')
        print('Performance: ', self.results['max_fitness'])
        print(f"Control voltages:\n {self.results['best_control_voltage']}")
        print('===============================================================')
