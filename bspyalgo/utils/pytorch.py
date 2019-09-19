""" The accelerator class enables to statically access the accelerator
(CUDA or CPU) that is used in the computer. The aim is to support both platforms seemlessly. """

import torch
import torch.nn as nn


class TorchUtils:
    """ A class to consistently manage declarations of torch variables for CUDA and CPU. """
    force_cpu = False
    data_type = 'float'

    @staticmethod
    def set_force_cpu(force):
        """ Enable setting the force CPU option for computers with an old CUDA version,
        where torch detects that there is cuda, but the version is too old to be compatible. """
        TorchUtils.force_cpu = force

    @staticmethod
    def set_data_type(data_type):
        """ Enable setting the force CPU option for computers with an old CUDA version,
        where torch detects that there is cuda, but the version is too old to be compatible. """
        TorchUtils.data_type = data_type

    @staticmethod
    def get_accelerator_type():
        """ Consistently returns the accelerator type for torch. """
        if torch.cuda.is_available() and not TorchUtils.force_cpu:
            return 'cuda'
        return 'cpu'

    @staticmethod
    def get_data_type():
        """It consistently returns the adequate data format for either CPU or CUDA.
        When the input force_cpu is activated, it will only create variables for the """
        if TorchUtils.get_accelerator_type() == 'cuda':
            return TorchUtils._get_cuda_data_type()
        return TorchUtils._get_cpu_data_type()

    @staticmethod
    def _get_cuda_data_type():
        if TorchUtils.data_type == 'float':
            return torch.cuda.FloatTensor
        if TorchUtils.data_type == 'long':
            return torch.cuda.LongTensor

    @staticmethod
    def _get_cpu_data_type():
        if TorchUtils.data_type == 'float':
            return torch.FloatTensor
        if TorchUtils.data_type == 'long':
            return torch.LongTensor
        # _ANS = type.__func__()
        # _ANS = data_type.__func__()

    @staticmethod
    def get_tensor_from_list(data):
        """Enables to create a torch variable with a consistent accelerator type and data type."""
        return torch.Tensor(
            data.type(TorchUtils.get_data_type())).to(device=TorchUtils.get_accelerator_type())

    # _ANS = format_torch.__func__()

    @staticmethod
    def get_tensor_from_numpy(data):
        """Enables to create a torch variable from numpy with a consistent accelerator type and
        data type."""
        return TorchUtils.get_tensor_from_list(torch.from_numpy(data))

    @staticmethod
    def get_numpy_from_tensor(data):
        return data.detach().numpy()


class TorchModel:
    """
        The TorchModel class is used to manage together a torch model and its state dictionary. The usage is expected to be as follows
        mymodel = TorchModel()
        mymodel.load_model('my_path/my_model.pt')
        mymodel.model
        mymodel.state_dict
    """

    def load_model(self, data_dir):
        """Loads a pytorch model from a directory string."""
        self.state_dict = torch.load(data_dir, map_location=TorchUtils.get_accelerator_type())
        self.info_dict = self._info_consistency_check(self.state_dict['info'])
        del self.state_dict['info']

        self._build_model(self.info_dict)
        self.model.load_state_dict(self.state_dict)

        if TorchUtils.get_accelerator_type() == 'cuda':
            self.model.cuda()

    def inference_in_nanoamperes(self, inputs):
        outputs = self.inference_as_numpy(inputs)
        return outputs * self.info_dict['amplification']

    def inference_as_numpy(self, inputs):
        inputs_torch = TorchUtils.get_tensor_from_numpy(inputs)
        outputs = self.model(inputs_torch)
        return TorchUtils.get_numpy_from_tensor(outputs)

    def _info_consistency_check(self, model_info):
        """ It checks if the model info follows the expected standards.
        If it does not follow the standards, it forces the model to
        follow them and throws an exception. """
        # if type(model_info['activation']) is str:
        #    model_info['activation'] = nn.ReLU()
        if 'D_in' not in model_info:
            model_info['D_in'] = len(model_info['offset'])
            print('WARNING: The model loaded does not define the input dimension as expected. Changed it to default value: %d.' % len(model_info['offset']))
        if 'D_out' not in model_info:
            model_info['D_out'] = 1
            print('WARNING: The model loaded does not define the output dimension as expected. Changed it to default value: %d.' % 1)
        if 'hidden_sizes' not in model_info:
            model_info['hidden_sizes'] = [90] * 6
            print('WARNING: The model loaded does not define the input dimension as expected. Changed it to default value: %d.' % 90)
        return model_info

    def _get_activation(self, activation):
        if type(activation) is str:
            return nn.ReLU()
        return activation

    def _build_model(self, model_info):

        hidden_sizes = model_info['hidden_sizes']
        input_layer = nn.Linear(model_info['D_in'], hidden_sizes[0])
        activ_function = self._get_activation(model_info['activation'])
        output_layer = nn.Linear(hidden_sizes[-1], model_info['D_out'])
        modules = [input_layer, activ_function]

        hidden_layers = zip(hidden_sizes[: -1], hidden_sizes[1:])
        for h_1, h_2 in hidden_layers:
            hidden_layer = nn.Linear(h_1, h_2)
            modules.append(hidden_layer)
            modules.append(activ_function)

        modules.append(output_layer)
        self.model = nn.Sequential(*modules)

        print('Model built with the following modules: \n', modules)
