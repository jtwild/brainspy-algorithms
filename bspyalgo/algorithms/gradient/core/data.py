import numpy as np
from bspyproc.processors.simulation.dopanet import DNPU
from bspyproc.utils.pytorch import TorchUtils


class GDData:
    def __init__(self, inputs, targets, nr_epochs, processor, validation_data=(None, None), mask=None):
        assert len(inputs) == len(targets), f'No. of input data {len(inputs)} does not match no. of targets {len(targets)}'
        self.results = {}
        self.results['inputs'] = inputs
        self.results['targets'] = targets
        self.nr_epochs = nr_epochs
        self.results['processor'] = processor
        if mask is None or len(mask) <= 1:
            mask = np.ones(targets.shape[0], dtype=bool)
        self.results['mask'] = mask
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
    def set_result_as_numpy(self, result_key, result):
        self.results[result_key] = TorchUtils.get_numpy_from_tensor(result)

    def judge(self):
        self.results['best_performance'] = self.results['performance_history'][-1]
        if self.results['processor'].configs['platform'] == 'simulation' and self.results['processor'].configs['network_type'] == 'dnpu':
            self.set_result_as_numpy('control_voltages', self.results['processor'].get_control_voltages())
        # self.print_results()

    def print_results(self):  # print(best_output.shape,self.target_wfm.shape)
        print(f'\n========================= BEST SOLUTION =======================')
        print('Final cost: ', self.results['best_performance'])
        if isinstance(self.results['processor'], DNPU):
            print(f"Best control voltages:\n {self.results['control_voltages']}")
        print('===============================================================')
