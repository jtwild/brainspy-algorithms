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
                  validation=False, mask=None, plot=False):
    found = False
    for run in range(10):
        if validation:
            data = algorithm.optimize(INPUTS, TARGETS, validation_data=(INPUTS_VAL, TARGETS_VAL), mask=mask)
            OUTPUTS = data.results['best_output']
            accuracy = get_accuracy(OUTPUTS, TARGETS_VAL)
        else:
            data = algorithm.optimize(INPUTS, TARGETS, mask=mask)
            OUTPUTS = data.results['best_output']
            accuracy = get_accuracy(OUTPUTS, TARGETS)

        print(f'accuracy in {run} is {accuracy}')
        if accuracy > 0.95:
            found = True
            break
    assert found, f'Gate not found; accuracy was {accuracy}'

    if type(OUTPUTS) is torch.Tensor:
        OUTPUTS = OUTPUTS.cpu().numpy()
    if type(TARGETS) is torch.Tensor:
        if validation:
            TARGETS = TARGETS_VAL.cpu().numpy()
        else:
            TARGETS = TARGETS.cpu().numpy()
    if plot:
        plt.figure()
        plt.plot(OUTPUTS, 'r')
        plt.plot(TARGETS, 'ok')
        plt.legend(['output', 'targets'])
        plt.show()


def get_accuracy(x, targets):
    if isinstance(x, torch.Tensor):
        x = x.cpu().data.numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().data.numpy()
    accuracy, _, _ = perceptron(x, targets)
    return accuracy

# Tests


def test_gd_dnpu(validation=True, plot=False):
    INPUTS = torchutils.get_tensor_from_numpy(XNOR['inputs'].T)
    TARGETS = torchutils.get_tensor_from_numpy(XNOR['targets'])
    INPUTS_VAL = torchutils.get_tensor_from_numpy(XNOR['inputs_val'].T)
    TARGETS_VAL = torchutils.get_tensor_from_numpy(XNOR['targets_val'])
    gd_dnpu = get_algorithm('./configs/gd/configs_template_dpnu.json')
    task_to_solve(gd_dnpu, INPUTS, TARGETS, INPUTS_VAL, TARGETS_VAL,
                  validation=validation, plot=plot)


def test_gd_nn(validation=True, plot=False):
    INPUTS = torchutils.get_tensor_from_numpy(XNOR['inputs'].T)
    TARGETS = torchutils.get_tensor_from_numpy(XNOR['targets'])
    INPUTS_VAL = torchutils.get_tensor_from_numpy(XNOR['inputs_val'].T)
    TARGETS_VAL = torchutils.get_tensor_from_numpy(XNOR['targets_val'])
    gd_nn = get_algorithm('./configs/gd/configs_template_nn_model.json')
    task_to_solve(gd_nn, INPUTS, TARGETS, INPUTS_VAL, TARGETS_VAL,
                  validation=validation, plot=plot)


def test_ga(validation=False, plot=False):
    INPUTS = XNOR['inputs'].T
    TARGETS = XNOR['targets']
    INPUTS_VAL = XNOR['inputs_val']
    TARGETS_VAL = XNOR['targets_val']
    ga_devicemodel = get_algorithm('./configs/ga/configs_template.json')
    task_to_solve(ga_devicemodel, INPUTS, TARGETS, INPUTS_VAL, TARGETS_VAL, mask=XNOR['mask'],
                  validation=validation, plot=plot)


# def test_ga_device():
    # INPUTS = XNOR['inputs']
    # TARGETS = XNOR['targets']
    # INPUTS_VAL = XNOR['inputs_val']
    # TARGETS_VAL = XNOR['targets_val']
#     ga_device = get_algorithm('./configs/ga/ga_device_configs_template.json')
#     task_to_solve(ga_device,INPUTS,TARGETS,INPUTS_VAL,TARGETS_VAL)


if __name__ == '__main__':
    # torchutils.force_cpu = True
    # test_gd_dnpu(plot=True)
    # test_gd_nn(plot=True)
    test_ga(plot=True)
