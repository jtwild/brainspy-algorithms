import numpy as np
from bspyalgo.utils.performance import corr_coeff


class GAData:
    def __init__(self, inputs, targets, mask, hyperparams):  # , waveform_configs):
        if targets != None:
            # If targets are supplied
            assert len(inputs) == len(targets), f'No. of input data {len(inputs)} does not match no. of targets {len(targets)}'
        self.results = {}
        self.results['inputs'] = inputs
        self.results['targets'] = targets
        if mask is None or len(mask) <= 1:
            mask = np.ones(inputs.shape[0], dtype=bool)
        self.results['mask'] = mask
        self.reset(hyperparams)

    def update(self, current_state):
        gen = current_state['generation']
        self.results['control_voltage_array'][gen, :, :] = current_state['genes']
        self.results['fitness_array'][gen, :] = current_state['fitness']
        self.results['best_output'] = current_state['outputs'][current_state['fitness'] == np.nanmax(current_state['fitness'])][0]
        self.results['best_performance'] = np.nanmax(current_state['fitness'])
        self.results['performance_history'] = np.nanmax(self.results['fitness_array'], axis=1)
        self.results['output_current_array'][gen, :, :] = current_state['outputs']
        self.results['correlation'] = corr_coeff(self.results['best_output'][self.results['mask']].T, self.results['targets'][self.results['mask']].T)

    def reset(self, hyperparams):
        # Define placeholders
        self.results['control_voltage_array'] = np.zeros((hyperparams['epochs'], hyperparams['genomes'],
                                                          hyperparams['genes']))
        self.results['output_current_array'] = np.zeros((hyperparams['epochs'], hyperparams['genomes']) + (len(self.results['inputs']), 1))
        self.results['fitness_array'] = -np.inf * np.ones((hyperparams['epochs'], hyperparams['genomes']))
        # return self.results['inputs'], self.results['targets']

    def judge(self):
        ind = np.unravel_index(np.argmax(self.results['fitness_array'], axis=None), self.results['fitness_array'].shape)
        self.results['best_output'] = self.results['output_current_array'][ind]
        self.results['control_voltages'] = self.results['control_voltage_array'][ind]
        self.results['best_performance'] = np.max(self.results['fitness_array'])
        # self.print_results()

    def get_description(self, gen):
        return "  Gen: " + str(gen + 1) + ". Max fitness: " + str(self.results['best_performance']) + ". Corr: " + str(self.results['correlation'])

    def print_results(self):  # print(best_output.shape,self.target_wfm.shape)
        print(f'\n========================= BEST SOLUTION =======================')
        print('Max fitness: ', self.results['best_performance'])
        print(f"Best control voltages:\n {self.results['control_voltages']}")
        print('===============================================================')
