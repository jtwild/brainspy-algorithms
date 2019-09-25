import torch
import numpy as np
from bspyalgo.utils.pytorch import TorchUtils
from bspyalgo.utils.io import save, create_directory_timestamp
from bspyalgo.utils.pytorch import TorchModel


def get_gd(configs):
    if configs['platform'] == 'hardware':
        raise NotImplementedError('Hardware platform not implemented')
    elif configs['platform'] == 'simulation':
        network = get_neural_network_model(configs)
        return GD(configs['hyperparameters'], network)
    else:
        raise NotImplementedError('Platform not implemented')


def get_neural_network_model(configs):
    if configs['get_network'] == 'build':
        network = TorchModel()
        network.build(configs['model_configs'])
    elif configs['get_network'] == 'load':
        network = TorchModel()
        network.load_model(configs['model_configs']['torch_model_path'])
    elif configs['get_network'] == 'dnpu':
        from bspyalgo.algorithms.gradient.core.dopanet import DNPU
        network = DNPU(configs['model_configs']['input_indices'], path=configs['model_configs']['torch_model_path'])
    else:
        raise NotImplementedError('Specified neural network simulation in "get_network" configurations is not available. Try a value between "build", "load" or "dnpu".')
    return network


class GD:
    """
    Trains a neural network given data.
    Inputs and targets is assumed to be partitioned in training and validation sets s.t.
    data is passed to trainer in the form
            data = [(inputs[0], targets[0]), (inputs[1], targets[1])]

    @author: hruiz
    """

    def __init__(self, config_dict, network, loss_fn=torch.nn.MSELoss()):
        self.network = network
        self.loss_fn = loss_fn
        self.load_configs(config_dict)

    def load_configs(self, config_dict):
        self.config_dict = config_dict
        # set configurations
        if "seed" in self.config_dict.keys():
            torch.manual_seed(self.config_dict['seed'])
            print('The torch RNG is seeded with ', self.config_dict['seed'])

        if "betas" in self.config_dict.keys():
            self.optimizer = torch.optim.Adam(self.network.parameters(),
                                              lr=self.config_dict['learning_rate'],
                                              betas=self.config_dict["betas"])
            print("Set betas to values: ", {self.config_dict["betas"]})
        else:
            self.optimizer = torch.optim.Adam(self.network.parameters(),
                                              lr=self.config_dict['learning_rate'])
        print('Prediction using ADAM optimizer')
        if 'results_path' in self.config_dict.keys():
            self.dir_path = create_directory_timestamp(self.config_dict['results_path'], self.config_dict['experiment_name'])
        else:
            self.dir_path = None

    def get_torch_model_path(self):
        return self.config_dict['model_configs']['torch_model_path']

    def optimize(self, inputs, targets):
        """Wraps trainer function in sgd_torch for use in algorithm_manager.
        """
        inputs = TorchUtils.get_tensor_from_list(inputs)
        targets = TorchUtils.get_tensor_from_list(targets)
        data = [(inputs[0], targets[0]), (inputs[1], targets[1])]

        return {'costs': self.trainer(data)}

    def trainer(self, data):
        # Define variables
        x_train, y_train = data[0]
        x_val, y_val = data[1]
        costs = np.zeros((self.config_dict['nr_epochs'], 2))  # training and validation costs per epoch

        for epoch in range(self.config_dict['nr_epochs']):

            self.network.train()
            permutation = torch.randperm(x_train.size()[0])  # Permute indices

            for mb in range(0, len(permutation), self.config_dict['batch_size']):
                self.train_step(x_train, y_train, permutation, mb)

            costs[epoch, 0] = self.evaluate_training_error(x_val, x_train, y_train)
            costs[epoch, 1] = self.evaluate_validation_error(x_val, y_val)
            self.close_epoch(epoch, costs)

        if self.dir_path:
            self.save_results('trained_network.pt')
        return costs

    def train_step(self, x_train, y_train, permutation, mb):
        # Get y_pred
        indices = permutation[mb:mb + self.config_dict['batch_size']]
        x_mb = x_train[indices]
        y_pred = self.network(x_mb)

        # GD step
        if 'regularizer' in dir(self.network):
            loss = self.loss_fn(y_pred, y_train[indices]) + self.network.regularizer()
        else:
            loss = self.loss_fn(y_pred, y_train[indices])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def close_epoch(self, epoch, costs):
        if self.dir_path and (epoch + 1) % SGD_CONFIGS['save_interval'] == 0:
            self.save_results(f'checkpoint_epoch{epoch}.pt')
        if epoch % 10 == 0:
            print('Epoch:', epoch,
                  'Val. Error:', costs[epoch, 1],
                  'Training Error:', costs[epoch, 0])

    def evaluate_validation_error(self, x_val, y_val):
        # Evaluate Validation error
        prediction = self.network(x_val)
        return self.loss_fn(prediction, y_val).item()

    def evaluate_training_error(self, x_val, x_train, y_train):
        # Evaluate training error
        self.config_dictnetwork.eval()
        samples = len(x_val)
        get_indices = torch.randperm(len(x_train))[:samples]
        x_sampled = x_train[get_indices]
        prediction = self.network(x_sampled)
        target = y_train[get_indices]
        return self.loss_fn(prediction, target).item()

    def save_results(self, filename):
        save('configs', self.dir_path, f'configs.json', data=self.config_dict)
        save('torch', self.dir_path, filename, data=self.network)


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
    SGD_MODEL_CONFIGS['torch_model_path'] = r'tmp/input/models/nn_test/checkpoint3000_02-07-23h47m.pt'

    SGD_CONFIGS = {}
    SGD_CONFIGS['platform'] = 'simulation'
    SGD_CONFIGS['get_network'] = 'load'
    SGD_CONFIGS['results_path'] = r'tmp/output/NN_test/'
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
   #  RESULTS =

    plt.figure()
    plt.plot(RESULTS['costs'])
    plt.title("Loss per epoch")
    plt.legend(["Training", "Validation"])
    plt.show()
