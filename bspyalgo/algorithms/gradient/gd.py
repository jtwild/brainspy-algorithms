import torch
from bspyalgo.algorithms.gradient.core.sgd_torch import trainer


def get_gd(configs):
    if configs['platform'] == 'hardware':
        raise NotImplementedError('Hardware platform not implemented')

    elif configs['platform'] == 'simulation':
        from bspyalgo.utils.pytorch import TorchModel
        if configs['get_network'] == 'build':
            network = TorchModel()
            network.build(configs['model_configs'])
        elif configs['get_network'] == 'load':
            network = TorchModel()
            network.load_model(configs['model_configs']['torch_model_path'])
        elif configs['get_network'] == 'dnpu':
            from bspyalgo.algorithms.gradient.core.dopanet import DNPU
            network = DNPU(configs['model_configs']['input_indices'], path=configs['model_configs']['torch_model_path'])
        return GD(configs['hyperparameters'], network)
    else:
        raise NotImplementedError('Platform not implemented')


class GD:
    """
    Trains a neural network given data.
    Inputs and targets is assumed to be partitioned in training and validation sets s.t.
    data is passed to trainer in the form
            data = [(inputs[0], targets[0]), (inputs[1], targets[1])]

    @author: hruiz
    """

    def __init__(self, configs, network):
        self.network = network
        self.config_dict = configs
        self.loss_fn = torch.nn.MSELoss()

    def optimize(self, inputs, targets):
        """Wraps trainer function in sgd_torch for use in algorithm_manager.
        """
        data = [(inputs[0], targets[0]), (inputs[1], targets[1])]
        C = trainer(data, self.network, self.config_dict, loss_fn=self.loss_fn)
        return {'costs': C}


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    # Get device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Make config dict for GD
    SGD_HYPERPARAMETERS = {}
    SGD_HYPERPARAMETERS['nr_epochs'] = 3000
    SGD_HYPERPARAMETERS['batch_size'] = 128
    SGD_HYPERPARAMETERS['learning_rate'] = 1e-4
    SGD_HYPERPARAMETERS['save_interval'] = 10
    SGD_HYPERPARAMETERS['seed'] = 33
    SGD_HYPERPARAMETERS['betas'] = (0.9, 0.99)

    SGD_MODEL_CONFIGS = {}
    SGD_MODEL_CONFIGS['input_indices'] = [0, 1]
    SGD_MODEL_CONFIGS['torch_model_path'] = r'tmp/NN_model/checkpoint3000_02-07-23h47m.pt'

    SGD_CONFIGS = {}
    SGD_CONFIGS['platform'] = 'simulation'
    SGD_CONFIGS['get_network'] = 'dnpu'
    SGD_CONFIGS['results_path'] = r'tmp/NN_test/'
    SGD_CONFIGS['experiment_name'] = 'TEST'
    SGD_CONFIGS['hyperparameters'] = SGD_HYPERPARAMETERS
    SGD_CONFIGS['model_configs'] = SGD_MODEL_CONFIGS

    # Create data
    x = 0.5 * np.random.randn(10, len(SGD_MODEL_CONFIGS['input_indices']))
    inp_train = torch.Tensor(x).to(DEVICE)
    t_train = torch.Tensor(5. * np.ones((10, 1))).to(DEVICE)
    x = 0.5 * np.random.randn(4, len(SGD_MODEL_CONFIGS['input_indices']))
    inp_val = torch.Tensor(x).to(DEVICE)
    t_val = torch.Tensor(5. * np.ones((4, 1))).to(DEVICE)

    INPUTS, TARGETS = (inp_train, inp_val), (t_train, t_val)

    RESULTS = get_gd(SGD_CONFIGS).optimize(INPUTS, TARGETS)

    plt.figure()
    plt.plot(RESULTS['costs'])
    plt.title("Loss per epoch")
    plt.legend(["Training", "Validation"])
    plt.show()
