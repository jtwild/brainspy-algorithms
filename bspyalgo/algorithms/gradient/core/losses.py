import torch
from bspyproc.utils.pytorch import TorchUtils


def choose_loss_function(loss_fn_name):
    '''Gets the fitness function used in GD from the module losses.
    The loss functions must take two arguments, the outputs of the black-box and the target
    and must return a torch array of scores of size len(outputs).
    '''
    if loss_fn_name == 'corrsig':
        return corrsig
    elif loss_fn_name == 'sqrt_corrsig':
        return sqrt_corrsig
    elif loss_fn_name == 'fisher':
        return fisher
    elif loss_fn_name == 'fisher_added_corr':
        return fisher_added_corr
    elif loss_fn_name == 'fisher_multipled_corr':
        return fisher_multipled_corr
    elif loss_fn_name == 'bce':
        bce = BCELossWithSigmoid()
        bce.cuda(TorchUtils.get_accelerator_type()).to(TorchUtils.data_type)
        return bce
    else:
        raise NotImplementedError(f"Loss function {loss_fn_name} is not recognized!")


class BCELossWithSigmoid(torch.nn.BCELoss):

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', activation=torch.nn.Sigmoid()):
        super(BCELossWithSigmoid, self).__init__(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction)
        self.activation = activation

    def forward(self, output, target):
        return super(BCELossWithSigmoid, self).forward(self.activation(output[:, 0]), target)


def corrsig(output, target):
    corr = torch.mean((output - torch.mean(output)) * (target - torch.mean(target))) / \
        (torch.std(output) * torch.std(target) + 1e-10)
    x_high_min = torch.min(output[(target == 1)])
    x_low_max = torch.max(output[(target == 0)])
    delta = x_high_min - x_low_max
    return (1.1 - corr) / torch.sigmoid((delta - 5) / 3)


def sqrt_corrsig(output, target):
    corr = torch.mean((output - torch.mean(output)) * (target - torch.mean(target))) / \
        (torch.std(output) * torch.std(target) + 1e-10)
    x_high_min = torch.min(output[(target == 1)])
    x_low_max = torch.max(output[(target == 0)])
    delta = x_high_min - x_low_max
    # 5/3 works for 0.2 V gap
    return (1. - corr)**(1 / 2) / torch.sigmoid((delta - 2) / 5)


def fisher(output, target):
    '''Separates classes irrespective of assignments.
    Reliable, but insensitive to actual classes'''
    x_high = output[(target == 1)]
    x_low = output[(target == 0)]
    m0, m1 = torch.mean(x_low), torch.mean(x_high)
    s0, s1 = torch.var(x_low), torch.var(x_high)
    mean_separation = (m1 - m0)**2
    return -mean_separation / (s0 + s1)


def fisher_added_corr(output, target):
    x_high = output[(target == 1)]
    x_low = output[(target == 0)]
    m0, m1 = torch.mean(x_low), torch.mean(x_high)
    s0, s1 = torch.var(x_low), torch.var(x_high)
    mean_separation = (m1 - m0)**2
    corr = torch.mean((output - torch.mean(output)) * (target - torch.mean(target))) / \
        (torch.std(output) * torch.std(target) + 1e-10)
    return (1 - corr) - 0.5 * mean_separation / (s0 + s1)


def fisher_multipled_corr(output, target):
    x_high = output[(target == 1)]
    x_low = output[(target == 0)]
    m0, m1 = torch.mean(x_low), torch.mean(x_high)
    s0, s1 = torch.var(x_low), torch.var(x_high)
    mean_separation = (m1 - m0)**2
    corr = torch.mean((output - torch.mean(output)) * (target - torch.mean(target))) / \
        (torch.std(output) * torch.std(target) + 1e-10)
    return (1 - corr) * (s0 + s1) / mean_separation
