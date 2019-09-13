import torch
from torch.autograd import Variable

""" The accelerator class enables to statically access the accelerator (CUDA or CPU) that is used in the computer. The aim is to support both platforms seemlessly. """


class TorchUtils:
    """ A class to consistently manage declarations of torch variables for CUDA and CPU. """
    force_cpu = False

    @staticmethod
    def set_force_cpu(force):
        TorchUtils.force_cpu = force

    @staticmethod
    def get_accelerator_type():
        if torch.cuda.is_available() and not TorchUtils.force_cpu:
            return 'cuda'
        else:
            return 'cpu'

    @staticmethod
    def get_data_type():
        """It consistently returns the adequate data format for either CPU or CUDA. When the input force_cpu is activated, it will only create variables for the """
        if TorchUtils.get_accelerator_type() == 'cuda':
            return torch.cuda.FloatTensor
        else:
            return torch.FloatTensor

    # _ANS = type.__func__()
    # _ANS = data_type.__func__()

    @staticmethod
    def format_torch(data):
        """Enables to create a torch variable with a consistent accelerator type and data type."""
        return Variable(data.type(TorchUtils.get_data_type())).to(device=TorchUtils.get_accelerator_type())

    # _ANS = format_torch.__func__()

    @staticmethod
    def format_numpy(data):
        """Enables to create a torch variable from numpy with a consistent accelerator type and 
        data type."""
        return TorchUtils.format_torch(torch.from_numpy(data))

    @staticmethod
    def load_torch_model(self, data_dir, eval=True):
        """Loads a pytorch model from a directory string."""
        model = torch.load(data_dir, map_location=TorchUtils.get_accelerator_type())
        if eval:
            model.eval()

    def load_torch_model(self, data_dir):

        print('Loading the model from ' + data_dir)
        self.ttype = torch.FloatTensor
        if torch.cuda.is_available():
            state_dic = torch.load(data_dir)
            self.ttype = torch.cuda.FloatTensor
        else:
            state_dic = torch.load(data_dir, map_location='cpu')

        # move info key from state_dic to self
        self.info = state_dic['info']
        print(f'Meta-info: \n {self.info.keys()}')
        state_dic.pop('info')

        self.D_in = self.info['D_in']
        self.D_out = self.info['D_out']
        self.hidden_sizes = self.info['hidden_sizes']

        self._contruct_model()
        self.model.load_state_dict(state_dic)

        if isinstance(list(net.model.parameters())[-1], torch.FloatTensor):
            self.itype = torch.LongTensor
        else:
            self.itype = torch.cuda.LongTensor
            self.model.cuda()
        self.model.eval()
