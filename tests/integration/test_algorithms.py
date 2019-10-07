'''Test all algorithms for all available processors
in a scenario where the optimizers must be exchangeble.'''

import numpy as np
from bspyalgo.utils.performance import perceptron
from bspyalgo.algorithm_manager import get_algorithm
from bspyproc.utils.pytorch import TorchUtils as torchutils
# Load Data
XNOR = np.load('tests/inputs/XNOR_validation.npz')
# Keys: ['inputs', 'targets', 'mask', 'inputs_val', 'targets_val']
INPUTS = torchutils.get_tensor_from_numpy(XNOR['inputs'].T)
TARGETS = torchutils.get_tensor_from_numpy(XNOR['targets'])
INPUTS_VAL = torchutils.get_tensor_from_numpy(XNOR['inputs_val'].T)
TARGETS_VAL = torchutils.get_tensor_from_numpy(XNOR['targets_val'])


def task_to_solve(algorithm, validation=True):
    found = False
    for run in range(4):
        if validation:
            data = algorithm.optimize(INPUTS, TARGETS, validation_data=(INPUTS_VAL, TARGETS_VAL))
            OUTPUTS = data.results['best_output']
            accuracy, _, _ = perceptron(OUTPUTS.data.numpy(), TARGETS_VAL.data.numpy())
        else:
            data = algorithm.optimize(INPUTS, TARGETS)
            OUTPUTS = data.results['best_output']
            accuracy, _, _ = perceptron(OUTPUTS.data.numpy(), TARGETS.data.numpy())
        print(f'accuracy in {run} is {accuracy}')
        if accuracy > 0.95:
            found = True
            break
    assert found, f'Gate not found; accuracy was {accuracy}'

# Tests


def test_gd_dnpu():
    gd_dnpu = get_algorithm('./configs/gd/gd_configs_template.json')
    task_to_solve(gd_dnpu)


# def test_gd_nn():
#     gd_nn = get_algorithm('./configs/gd/nn_training_configs_template.json')
#     task_to_solve(gd_nn)


# def test_ga_devicemodel():
#     ga_devicemodel = get_algorithm('./configs/ga/ga_configs_template.json')
#     task_to_solve(ga_devicemodel)


# def test_ga_device():
#     ga_device = get_algorithm('./configs/ga/ga_device_configs_template.json')
#     task_to_solve(ga_device)


if __name__ == '__main__':
    test_gd_dnpu()
