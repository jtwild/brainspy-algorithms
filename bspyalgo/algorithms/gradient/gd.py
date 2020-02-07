# TODO: ''' '''
import torch
import os
from tqdm import trange
from bspyproc.bspyproc import get_processor
from bspyalgo.utils.io import save, create_directory_timestamp
from bspyalgo.algorithms.gradient.core.data import GDData
from bspyalgo.algorithms.gradient.core.losses import choose_loss_function


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

        if 'loss_function' in self.hyperparams.keys():
            self.loss_fn = choose_loss_function(self.hyperparams['loss_function'])
        else:
            self.loss_fn = loss_fn
        self.init_processor()

    def init_processor(self):
        self.processor = get_processor(self.configs["processor"])

        self.load_configs()
        if 'regularizer' in dir(self.processor):
            self.loss_function = self.loss_with_regularizer
        else:
            self.loss_function = self.loss_fn

    def reset(self):
        self.init_optimizer()
        self.processor.reset()

    def load_configs(self):

        # set configurations
        if "seed" in self.hyperparams.keys():
            torch.manual_seed(self.hyperparams['seed'])
            print('The torch RNG is seeded with ', self.hyperparams['seed'])

        self.init_optimizer()
        print('Prediction using ADAM optimizer')
        if 'experiment_name' not in self.configs:
            self.configs['experiment_name'] = 'experiment'
        if 'results_path' in self.configs:
            self.dir_path = create_directory_timestamp(self.configs['results_path'], self.configs['experiment_name'])
        else:
            self.dir_path = create_directory_timestamp(os.path.join('tmp', 'dump'), self.configs['experiment_name'])

    def init_optimizer(self):
        if "betas" in self.hyperparams.keys():
            self.optimizer = torch.optim.Adam(self.processor.parameters(),
                                              lr=self.hyperparams['learning_rate'],
                                              betas=self.hyperparams["betas"]
                                              )
            print("Set betas to values from the config file: ")
            print(*self.hyperparams["betas"], sep=", ")
        else:
            self.optimizer = torch.optim.Adam(self.processor.parameters(),
                                              lr=self.hyperparams['learning_rate'])

    def loss_with_regularizer(self, y_pred, y_train):
        return self.loss_fn(y_pred, y_train) + self.processor.regularizer()


# TODO: Implement feeding the validation_data and mask as optional kwargs


    def optimize(self, inputs, targets, validation_data=(None, None), data_info=None, mask=None):
        """Wraps trainer function in sgd_torch for use in algorithm_manager.
        """
        assert isinstance(inputs, torch.Tensor), f"Inputs must be torch.Tensor, they are {type(inputs)}"
        assert isinstance(targets, torch.Tensor), f"Targets must be torch.Tensor, they are {type(targets)}"

        self.reset()
        if data_info is not None:
            self.processor.info['data_info'] = data_info
        data = GDData(inputs, targets, self.hyperparams['nr_epochs'], self.processor, validation_data, mask=mask)
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
        looper = trange(self.hyperparams['nr_epochs'], desc='Initialising')
        for epoch in looper:

            self.train_step(x_train, y_train)
            # with torch.no_grad():
            data.results['performance_history'][epoch, 0], prediction_training = self.evaluate_training_error(x_val, x_train, y_train)
            data.results['performance_history'][epoch, 1], prediction_validation = self.evaluate_validation_error(x_val, y_val)
            if self.dir_path and (epoch + 1) % self.configs['checkpoints']['save_interval'] == 0:
                save('torch', self.dir_path, f'checkpoint_epoch{epoch}.pt', data=self.processor)
        #    if epoch % self.hyperparams['save_interval'] == 0:
            training_error = data.results['performance_history'][epoch, 0]
            validation_error = data.results['performance_history'][epoch, 1]
            description = ' Epoch: ' + str(epoch) + ' Training Error:' + str(training_error) + ' Val. Error:' + str(validation_error)
            looper.set_description(description)
            if training_error <= self.hyperparams['stop_threshold'] and validation_error <= self.hyperparams['stop_threshold']:
                print(f"Reached threshold error {self.hyperparams['stop_threshold'] } for training and validation. Stopping")
                break
        data.set_result_as_numpy('best_output', prediction_validation)
        data.set_result_as_numpy('best_output_training', prediction_training)
        return data

    def sgd_train_without_validation(self, data):
        x_train = data.results['inputs']
        y_train = data.results['targets']
        looper = trange(self.hyperparams['nr_epochs'], desc='Initialising')
        for epoch in looper:
            # self.processor.train()
            self.train_step(x_train, y_train)
            # self.processor.eval()
            with torch.no_grad():
                prediction = self.processor(data.results['inputs'])
                data.results['performance_history'][epoch] = self.loss_fn(prediction, y_train).item()  # data.results['targets']).item()
            if self.configs['checkpoints']['use_checkpoints'] is True and (self.dir_path and (epoch + 1) % self.configs['checkpoints']['save_interval'] == 0):
                save('torch', self.dir_path, f'checkpoint_epoch{epoch}.pt', data=self.processor)
            # if epoch % self.hyperparams['save_interval'] == 0:
            error = data.results['performance_history'][epoch]
            description = ' Epoch: ' + str(epoch) + ' Training Error:' + str(error)
            looper.set_description(description)
            if error <= self.hyperparams['stop_threshold']:
                print(f"Reached threshold error {self.hyperparams['stop_threshold']}. Stopping")
                break
        data.set_result_as_numpy('best_output', prediction)
        return data

    def train_step(self, x_train, y_train):
        self.processor.train()
        permutation = torch.randperm(x_train.size()[0])  # Permute indices

        for mb in range(0, len(permutation), self.hyperparams['batch_size']):
            self.minibatch_step(x_train, y_train, permutation[mb:mb + self.hyperparams['batch_size']])

    def minibatch_step(self, x_train, y_train, indices):
        # Get y_pred
        x_mb = x_train[indices]
        y_pred = self.processor(x_mb)

        loss = self.loss_function(y_pred, y_train[indices])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate_validation_error(self, x_val, y_val):
        # Evaluate Validation error
        # with torch.no_grad():
        self.processor.eval()
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
        save('configs', self.dir_path, f'configs.json', timestamp=False, data=self.hyperparams)
        save('torch', self.dir_path, filename, timestamp=False, data=self.processor)

    def save_smg_configs_dict(self):
        self.processor.info['smg_configs'] = self.configs

    def save_model_mse(self, mse):
        self.processor.info['data_info']['mse'] = mse
        self.processor.configs['noise'] = True
        if self.dir_path:
            self.save_results('trained_network.pt')
