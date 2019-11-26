import torch


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
    else:
        raise NotImplementedError(f"Loss function {loss_fn_name} is not recognized!")


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
