import torch
from torch.autograd import Variable

"""
    The accelerator class enables to statically access the accelerator
    (CUDA or CPU) that is used in the computer. The aim is to support
    both platforms seemlessly.
"""
@staticmethod
def accelerator_type():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


@staticmethod
def data_type(force_cpu=False):
    if torch.cuda.is_available() and not force_cpu:
        return torch.cuda.FloatTensor
    else:
        return torch.FloatTensor

# _ANS = type.__func__()
# _ANS = data_type.__func__()


@staticmethod
def format_torch(data):
    return Variable(data.type(data_type())).to(device=accelerator_type())

# _ANS = format_torch.__func__()


@staticmethod
def format_numpy(data):
    return format_torch(torch.from_numpy(data))
