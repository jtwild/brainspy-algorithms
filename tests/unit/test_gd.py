'''Test gradient descent algorithm for all available processors.'''

import numpy as np
from bspyalgo.utils.performance import perceptron
from bspyalgo.algorithm_manager import get_algorithm
from bspyproc.utils.pytorch import TorchUtils as tu
# Load Data
XNOR = np.load('tests/inputs/XNOR_validation.npz')
# Keys: ['inputs', 'targets', 'mask', 'inputs_val', 'targets_val']
INPUTS, TARGETS = tu.get_tensor_from_numpy(XNOR['inputs'].T), tu.get_tensor_from_numpy(XNOR['targets'].T)
INPUTS_VAL, TARGETS_VAL = tu.get_tensor_from_numpy(XNOR['inputs_val'].T), tu.get_tensor_from_numpy(XNOR['targets_val'].T)


def test_dnpu_with_validation():
    found = False
    gd = get_algorithm('./configs/gd/gd_configs_template.json')
    for _ in range(3):
        data = gd.optimize(INPUTS, TARGETS, validation_data=(INPUTS_VAL, TARGETS_VAL))
        OUTPUTS = data.results['best_output']
        accuracy, _, _ = perceptron(OUTPUTS, TARGETS_VAL)
        if accuracy > 0.95:
            found = True
    assert found


def test_dnpu_without_validation():
    found = False
    gd = get_algorithm('./configs/gd/gd_configs_template.json')
    for _ in range(3):
        data = gd.optimize(INPUTS, TARGETS)
        OUTPUTS = data.results['best_output']
        accuracy, _, _ = perceptron(OUTPUTS, TARGETS_VAL)
        if accuracy > 0.95:
            found = True
    assert found


# def test_nn_model():
#     pass

if __name__ == '__main__':
    test_dnpu_with_validation()
