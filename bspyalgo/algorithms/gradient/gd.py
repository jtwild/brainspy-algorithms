# TODO: ''' '''
import torch

from bspyproc.processors.processor_mgr import get_processor
from bspyalgo.utils.io import save, create_directory_timestamp
from bspyalgo.algorithms.gradient.core.data import GDData


def get_gd(configs):
    if configs['processor']['platform'] == 'hardware':
        raise NotImplementedError('Hardware platform not implemented')
        # TODO: Implement the lock in algorithm class
    elif configs['processor']['platform'] == 'simulation':
        return GD(configs)
    else:
        raise NotImplementedError('Platform not implemented')


class GD:
    """
    Trains a neural network given data.
    Inputs and targets is assumed to be partitioned in training and validation sets.
    If saving is needed use key in config file : "results_path": "tmp/output/models/nn_test/"
    @author: hruiz
    """

    def __init__(self, configs, loss_fn=torch.nn.MSELoss()):
        self.configs = configs
        self.hyperparams = configs["hyperparameters"]
        self.loss_fn = loss_fn
        self.reset_processor()

    def reset_processor(self):

        self.processor = get_processor(self.configs["processor"])
        self.load_configs()
        if 'regularizer' in dir(self.processor):
            self.loss_function = self.loss_with_regularizer
        else:
            self.loss_function = self.loss_fn

    def load_configs(self):

        # set configurations
        if "seed" in self.hyperparams.keys():
            torch.manual_seed(self.hyperparams['seed'])
            print('The torch RNG is seeded with ', self.hyperparams['seed'])

        if "betas" in self.hyperparams.keys():
            self.optimizer = torch.optim.Adam(self.processor.parameters(),
                                              lr=self.hyperparams['learning_rate'],
                                              betas=self.hyperparams["betas"])
            print("Set betas to values: ", {self.hyperparams["betas"]})
        else:
            self.optimizer = torch.optim.Adam(self.processor.parameters(),
                                              lr=self.hyperparams['learning_rate'])
        print('Prediction using ADAM optimizer')
        if 'results_path' in self.configs.keys():
            self.dir_path = create_directory_timestamp(self.configs['results_path'], self.configs['experiment_name'])
        else:
            self.dir_path = None

    def loss_with_regularizer(self, y_pred, y_train):
        return self.loss_fn(y_pred, y_train) + self.processor.regularizer()

    def optimize(self, inputs, targets, validation_data=(None, None)):
        """Wraps trainer function in sgd_torch for use in algorithm_manager.
        """

        self.reset_processor()
        data = GDData(inputs, targets, self.hyperparams['nr_epochs'], self.processor, validation_data)
        if validation_data[0] is not None and validation_data[1] is not None:
            data = self.sgd_train_with_validation(data)
        else:
            data = self.sgd_train_without_validation(data)
        if self.dir_path:
            self.save_results('trained_network.pt')
        return data

    def sgd_train_with_validation(self, data):
        x_train = data.results['inputs']
        y_train = data.results['targets']
        x_val = data.results['inputs_val']
        y_val = data.results['targets_val']
        for epoch in range(self.hyperparams['nr_epochs']):
            self.train_step(x_train, y_train)
            data.results['performance_history'][epoch, 0], prediction_training = self.evaluate_training_error(x_val, x_train, y_train)
            data.results['performance_history'][epoch, 1], prediction_validation = self.evaluate_validation_error(x_val, y_val)
            if self.dir_path and (epoch + 1) % self.hyperparams['save_interval'] == 0:
                save('torch', self.dir_path, f'checkpoint_epoch{epoch}.pt', data=self.processor)
            if epoch % 10 == 0:
                print('Epoch:', epoch,
                      'Training Error:', data.results['performance_history'][epoch, 0],
                      'Val. Error:', data.results['performance_history'][epoch, 1])
        data.results['best_output'] = prediction_validation
        data.results['best_output_training'] = prediction_training
        return data

    def sgd_train_without_validation(self, data):
        x_train = data.results['inputs']
        y_train = data.results['targets']
        for epoch in range(self.hyperparams['nr_epochs']):
            self.train_step(x_train, y_train)
            with torch.no_grad():
                prediction = self.processor(data.results['inputs'])
                data.results['performance_history'][epoch] = self.loss_fn(prediction, data.results['targets']).item()
            if self.dir_path and (epoch + 1) % self.hyperparams['save_interval'] == 0:
                save('torch', self.dir_path, f'checkpoint_epoch{epoch}.pt', data=self.processor)
            if epoch % 10 == 0:
                print('Epoch:', epoch, 'Training Error:', data.results['performance_history'][epoch])
        data.results['best_output'] = prediction
        return data

    def train_step(self, x_train, y_train):
        self.processor.train()
        permutation = torch.randperm(x_train.size()[0])  # Permute indices

        for mb in range(0, len(permutation), self.hyperparams['batch_size']):
            self.minibatch_step(x_train, y_train, permutation, mb)

    def minibatch_step(self, x_train, y_train, permutation, mb):
        # Get y_pred
        indices = permutation[mb:mb + self.hyperparams['batch_size']]
        x_mb = x_train[indices]
        y_pred = self.processor(x_mb)

        loss = self.loss_function(y_pred, y_train[indices])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate_validation_error(self, x_val, y_val):
        # Evaluate Validation error
        with torch.no_grad():
            prediction = self.processor(x_val)
        return self.loss_fn(prediction, y_val).item(), prediction

    def evaluate_training_error(self, x_val, x_train, y_train):
        # Evaluate training error
        self.processor.eval()
        samples = len(x_val)
        get_indices = torch.randperm(len(x_train))[:samples]
        x_sampled = x_train[get_indices]
        with torch.no_grad():
            prediction = self.processor(x_sampled)
        target = y_train[get_indices]
        return self.loss_fn(prediction, target).item(), prediction

    def save_results(self, filename):
        save('configs', self.dir_path, f'configs.json', data=self.hyperparams)
        save('torch', self.dir_path, filename, data=self.processor)
