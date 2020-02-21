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
    # Below is the sigmoid_distance loss function used to train outputs far away from each other
    elif loss_fn_name == 'sigmoid_distance':
        return sigmoid_distance
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


def sigmoid_distance(outputs, target=None):
    # Sigmoid distance: a squeshed version of a sum of all internal distances between points.
    if target != None:
        raise Warning('This loss function does not use target values. Target ignored.')
    # Expecting a torch.tensor with a list of single outputs
    # Then we can just transpose it and subtract from the original tensor to obtain all distances.
    # The diagonal will all have distance zero, which puts all values of 0.5 on the diagonal and causes a offset, but offsets are not a problem
    # The sigmoid is shifted 0.5 downwards to set its zero point correctly. Onyl positive values are used in its argument.
    #TODO: Scale the sigmoid to prefer sepeartion upto... X nA.
    return -1*torch.mean( torch.sigmoid( torch.abs(outputs - outputs.T) /5 ) - 0.5 )
    #return torch.mean( torch.tanh( 1/ (torch.abs(outputs - outputs.T) +1e-10/2) ) )
    #return torch.zeros(1)


#  Testing a specific loss function
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    num_tests = 40
    loss = torch.zeros([num_tests])
    num_samples = 2**3
    spacing = torch.logspace(0.1,1.6,num_tests)

    for i in range(num_tests):
        limit = spacing[i] * num_samples
        outputs_temp = torch.linspace(0,limit, num_samples)
        outputs = torch.zeros([1,num_samples])
        for j in range(num_samples):
            outputs[0,j] = outputs_temp[j]
        targets = []
        loss[i] = sigmoid_distance(outputs)

    plt.plot(spacing, loss)
    plt.ylabel('Loss')
    plt.xlabel('Spacing between points')