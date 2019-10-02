import torch
import numpy as np
from bspyalgo.utils.pytorch import TorchUtils
from bspyalgo.utils.io import save, create_directory_timestamp
from bspyalgo.utils.pytorch import TorchModel
from bspyalgo.interface.interface_manager import get_interface
from bspyalgo.algorithms.gradient.core.data import GDData
from bspyproc.processors.processor_mgr import get_processor


def get_gd(configs):
    if configs['platform'] == 'hardware':
        raise NotImplementedError('Hardware platform not implemented')
        # TODO: Implement the lock in algorithm class
    elif configs['platform'] == 'simulation':
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

    def __init__(self, config_dict, loss_fn=torch.nn.MSELoss()):
        self.processor_configs = config_dict['processor']

        self.loss_fn = loss_fn
        self.load_configs(config_dict)

    def reset_processor(self):
        self.processor = get_processor(self.processor_configs)
        if 'regularizer' in dir(self.processor):
            self.loss_function = loss_with_regularizer
        else:
            self.loss_function = loss_fn

    def load_configs(self, config_dict):
        self.config_dict = config_dict
        # set configurations
        if "seed" in self.config_dict.keys():
            torch.manual_seed(self.config_dict['seed'])
            print('The torch RNG is seeded with ', self.config_dict['seed'])

        if "betas" in self.config_dict.keys():
            self.optimizer = torch.optim.Adam(self.processor.parameters(),
                                              lr=self.config_dict['learning_rate'],
                                              betas=self.config_dict["betas"])
            print("Set betas to values: ", {self.config_dict["betas"]})
        else:
            self.optimizer = torch.optim.Adam(self.processor.parameters(),
                                              lr=self.config_dict['learning_rate'])
        print('Prediction using ADAM optimizer')
        if 'results_path' in self.config_dict.keys():
            self.dir_path = create_directory_timestamp(self.config_dict['results_path'], self.config_dict['experiment_name'])
        else:
            self.dir_path = None

    def loss_with_regularizer(self, y_pred, y_train):
        return self.loss_fn(y_pred, y_train) + self.processor.regularizer()

    def get_torch_model_path(self):
        return self.config_dict['model_configs']['torch_model_path']

    def optimize(self, inputs, targets, validation_data=(None, None)):
        """Wraps trainer function in sgd_torch for use in algorithm_manager.
        """

        self.reset_processor()
        data = GDData(inputs, targets, validation_data)
        if(validation_data is not (None, None)):
            data = self.sgd_trainer(data)
        else:
            data = self.optimize_without_validation(data)
        if self.dir_path:
            self.save_results('trained_network.pt')
        return data
        # Define variables

    def post_process(self):
        self.data.judge()
        return self.data.results

    def sgd_train(self, data):
        x_train = data.results['inputs']
        y_train = data.results['targets']
        x_val = data.results['inputs_val']
        y_val = data.results['targets_val']
        for epoch in range(self.config_dict['nr_epochs']):
            self.train_step(x_train, y_train)
            data.results['performance_history'][epoch, 0], prediction_training = self.evaluate_training_error(x_val, x_train, y_train)
            data.results['performance_history'][epoch, 1], prediction_validation = self.evaluate_validation_error(x_val, y_val)
            if self.dir_path and (epoch + 1) % SGD_CONFIGS['save_interval'] == 0:
                save('torch', self.dir_path, f'checkpoint_epoch{epoch}.pt', data=self.processor)
            if epoch % 10 == 0:
                print('Epoch:', epoch,
                      'Training Error:', data.results['performance_history'][epoch, 0],
                      'Val. Error:', data.results['performance_history'][epoch, 1])
        data.results['best_output'] = prediction_validation
        data.results['best_output_training'] = prediction_training
        return data

    def sgd_train_without_validation(self, data):
        # Define variables
        prediction = None
        # costs = np.zeros((self.config_dict['nr_epochs'], 2))  # training and validation costs per epoch
        for epoch in range(self.config_dict['nr_epochs']):
            self.train_step(data.results['inputs'])
            with torch.no_grad:
                prediction = self.processor(data.results['inputs'])
                data.results['performance_history'][epoch] = self.loss_fn(prediction, data.results['targets']).item()
            if self.dir_path and (epoch + 1) % SGD_CONFIGS['save_interval'] == 0:
                save('torch', self.dir_path, f'checkpoint_epoch{epoch}.pt', data=self.processor)
            if epoch % 10 == 0:
                print('Epoch:', epoch, 'Training Error:', data.results['performance_history'][epoch])
        data.results['best_output'] = prediction
        return data

    def train_step(self, x_train, y_train):
        self.processor.train()
        permutation = torch.randperm(x_train.size()[0])  # Permute indices

        for mb in range(0, len(permutation), self.config_dict['batch_size']):
            self.minibatch_step(x_train, y_train, permutation, mb)

    def minibatch_step(self, x_train, y_train, permutation, mb):
        # Get y_pred
        indices = permutation[mb:mb + self.config_dict['batch_size']]
        x_mb = x_train[indices]
        y_pred = self.processor(x_mb)

        loss = self.loss_function(y_pred, y_train[indices])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate_validation_error(self, x_val, y_val):
        # Evaluate Validation error
        with torch.no_grad:
            prediction = self.processor(x_val)
        return self.loss_fn(prediction, y_val).item(), prediction

    def evaluate_training_error(self, x_val, x_train, y_train):
        # Evaluate training error
        self.config_dictnetwork.eval()
        samples = len(x_val)
        get_indices = torch.randperm(len(x_train))[:samples]
        x_sampled = x_train[get_indices]
        with torch.no_grad:
            prediction = self.processor(x_sampled)
        target = y_train[get_indices]
        return self.loss_fn(prediction, target).item(), prediction

    def save_results(self, filename):
        save('configs', self.dir_path, f'configs.json', data=self.config_dict)
        save('torch', self.dir_path, filename, data=self.processor)


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
