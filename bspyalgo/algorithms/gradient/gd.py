import torch
from bspyalgo.algorithms.gradient.core.sgd_torch import trainer


def get_gd(configs):
    if configs['platform'] == 'hardware':
        raise NotImplementedError('Hardware platform not implemented')

    elif configs['platform'] == 'simulation':
        if configs['get_network'] == 'build':
            raise NotImplementedError('Network not implemented')
        elif configs['get_network'] == 'load':
            from bspyalgo.utils.pytorch import TorchModel
            network = TorchModel()
            network.load_model(configs['path_to_network'])
        elif configs['get_network'] == 'dnpu':
            from bspyalgo.algorithms.gradient.core.dopanet import DNPU
            network = DNPU(configs['in_list'], path=configs['path_to_network'])
        return GD(configs, network)
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

        return trainer(data, self.network, self.config_dict, loss_fn=self.loss_fn)
