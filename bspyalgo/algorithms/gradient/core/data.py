import numpy as np
from bspyproc.processors.simulation.dopanet import DNPU


class GDData:
    def __init__(self, inputs, targets, nr_epochs, processor, validation_data=(None, None)):
        assert len(inputs) == len(targets), f'No. of input data {len(inputs)} does not match no. of targets {len(targets)}'
        self.results = {}
        self.results['inputs'] = inputs
        self.results['targets'] = targets
        self.nr_epochs = nr_epochs
        self.results['processor'] = processor
        if validation_data[0] is not None and validation_data[1] is not None:
            assert len(validation_data[0]) == len(validation_data[1]), f'No. of validation input data {len(validation_data[0])} does not match no. of validation targets {len(validation_data[1])}'
            self.results['inputs_val'] = validation_data[0]
            self.results['targets_val'] = validation_data[1]
            self.results['performance_history'] = np.zeros((self.nr_epochs, 2))
            print('VALIDATION DATA IS AVAILABLE')
        else:
            self.results['performance_history'] = np.zeros((self.nr_epochs, 1))

    # TODO: Create an update function to store the history of the control voltages and output
    # def update(self, next_sate):

    def judge(self):
        self.results['best_performance'] = self.results['performance_history'][-1]
        if isinstance(self.results['processor'], DNPU):
            self.results['control_voltages'] = list(self.results['processor'].parameters())[0]
        self.print_results()

    def print_results(self):  # print(best_output.shape,self.target_wfm.shape)
        print(f'\n========================= BEST SOLUTION =======================')
        print('Performance: ', self.results['final_cost'])
        if isinstance(self.results['processor'], DNPU):
            print(f"Control voltages:\n {self.results['best_control_voltage']}")
        print('===============================================================')
