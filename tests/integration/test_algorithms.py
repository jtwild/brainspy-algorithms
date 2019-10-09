'''Test all algorithms for all available processors
in a scenario where the optimizers must be exchangeble.'''

import numpy as np
import matplotlib.pyplot as plt
from bspyalgo.utils.performance import perceptron
from bspyalgo.algorithm_manager import get_algorithm
from bspyproc.utils.pytorch import TorchUtils as torchutils
import torch
# Load Data
XNOR = np.load('tests/inputs/XNOR_validation.npz')
# Keys: ['inputs', 'targets', 'mask', 'inputs_val', 'targets_val']


def task_to_solve(algorithm, INPUTS, TARGETS, INPUTS_VAL, TARGETS_VAL,
                  validation=False, mask=False, plot=False):
    found = False
    for run in range(4):
        if validation:
            data = algorithm.optimize(INPUTS, TARGETS, validation_data=(INPUTS_VAL, TARGETS_VAL), mask=mask)
        else:
            data = algorithm.optimize(INPUTS, TARGETS, mask=mask)

        OUTPUTS = data.results['best_output']
        if type(OUTPUTS) is torch.Tensor:
            accuracy, _, _ = perceptron(OUTPUTS.data.numpy(), TARGETS.data.numpy())
        else:
            accuracy, _, _ = perceptron(OUTPUTS, TARGETS)

        print(f'accuracy in {run} is {accuracy}')
        if accuracy > 0.95:
            found = True
            break
    assert found, f'Gate not found; accuracy was {accuracy}'
    if plot:
        plt.figure()
        plt.plot(OUTPUTS, 'r')
        plt.plot(TARGETS, 'k')
        plt.legend(['output', 'targets'])
        plt.show()

# Tests


def test_gd_dnpu():
    INPUTS = torchutils.get_tensor_from_numpy(XNOR['inputs'].T)
    TARGETS = torchutils.get_tensor_from_numpy(XNOR['targets'])
    INPUTS_VAL = torchutils.get_tensor_from_numpy(XNOR['inputs_val'].T)
    TARGETS_VAL = torchutils.get_tensor_from_numpy(XNOR['targets_val'])
    gd_dnpu = get_algorithm('./configs/gd/gd_configs_template.json')
    task_to_solve(gd_dnpu, INPUTS, TARGETS, INPUTS_VAL, TARGETS_VAL)


def test_gd_nn():
    INPUTS = torchutils.get_tensor_from_numpy(XNOR['inputs'].T)
    TARGETS = torchutils.get_tensor_from_numpy(XNOR['targets'])
    INPUTS_VAL = torchutils.get_tensor_from_numpy(XNOR['inputs_val'].T)
    TARGETS_VAL = torchutils.get_tensor_from_numpy(XNOR['targets_val'])
    gd_nn = get_algorithm('./configs/gd/nnmodel_configs_template.json')
    task_to_solve(gd_nn, INPUTS, TARGETS, INPUTS_VAL, TARGETS_VAL)


def test_ga_devicemodel():
    INPUTS = XNOR['inputs']
    TARGETS = XNOR['targets']
    INPUTS_VAL = XNOR['inputs_val']
    TARGETS_VAL = XNOR['targets_val']
    ga_devicemodel = get_algorithm('./configs/ga/ga_configs_template.json')
    task_to_solve(ga_devicemodel, INPUTS, TARGETS, INPUTS_VAL, TARGETS_VAL, mask=XNOR['mask'])


# def test_ga_device():
    # INPUTS = XNOR['inputs']
    # TARGETS = XNOR['targets']
    # INPUTS_VAL = XNOR['inputs_val']
    # TARGETS_VAL = XNOR['targets_val']
#     ga_device = get_algorithm('./configs/ga/ga_device_configs_template.json')
#     task_to_solve(ga_device,INPUTS,TARGETS,INPUTS_VAL,TARGETS_VAL)


if __name__ == '__main__':
    # test_ga_devicemodel()
    # test_gd_dnpu()
    test_gd_nn()
