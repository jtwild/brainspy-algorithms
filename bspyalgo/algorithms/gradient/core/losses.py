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
    elif loss_fn_name == 'entropy':
        return entropy
    elif loss_fn_name == 'entropy_abs':
        return entropy_abs
    elif loss_fn_name == 'entropy_hard_boundaries':
        return entropy_hard_boundaries
    elif loss_fn_name == 'entropy_distance':
        return entropy_distance
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


# Loss functions below are added by Jochem specifically for the patch filter. They do not require targets.
def sigmoid_distance(outputs, target=None):
    # Sigmoid distance: a squeshed version of a sum of all internal distances between points.
    if target != None:
        raise Warning('This loss function does not use target values. Target ignored.')
    # Expecting a torch.tensor with a list of single outputs
    # Then we can just transpose it and subtract from the original tensor to obtain all distances.
    # The diagonal will all have distance zero, which puts all values of 0.5 on the diagonal and causes a offset, but offsets are not a problem
    # The sigmoid is shifted 0.5 downwards to set its zero point correctly. Onyl positive values are used in its argument.
    #TODO: Scale the sigmoid to prefer sepeartion upto... X nA.
    #scale = outputs.max() - outputs.min()  # normalizing scale, to prevent promotion of complete outward descent.
    #return -1*torch.mean( torch.sigmoid( torch.abs( (outputs - outputs.T) / scale)  *5 ) - 0.5 ) -torch.sigmoid( scale/100 )
    return -1*torch.mean( torch.sigmoid( torch.abs( (outputs - outputs.T) / 2 ) - 0.5 ) )
    #return torch.mean( torch.tanh( 1/ (torch.abs(outputs - outputs.T) +1e-10/2) ) )
    #return torch.zeros(1)

def entropy(outputs, target=None, return_intervals=False):
    # Entropy E of a set of points S:
    # E(S) = sum_{all x element of S} P(x) * -log2(P(x))
    # Entropy is maximized by an even distribution.
    # Warning: scaling the output in the sigmoid_distance loss functions had unwanted results: current range was reduced drasitcally.
    # The edge points have badly defined nearest neighbours, so for the two edgepoints, the single closest neighbour is taken twice.
    outputs_sorted = outputs.sort(dim=0)[0]  # we need the sorted array for adjacent points.
    dist = torch.abs((outputs_sorted - outputs_sorted.T))

    # Now we need the values next to the diagonal, to determine the intervals. Edge points have only a single determined closest neighbour
    # For selection, we build an index array
    indices = [[], []]
    indices[0].append([0, 0])  # first edge point
    indices[1].append([1, 1])  # first edge point gets twice single neighbour
    for i in range(1, len(outputs_sorted)-1):
        indices[0].append([i, i])  # select row
        indices[1].append([i-1, i+1])  # select columns, next to diagonal.
    i+=1  # for the last edge point
    indices[0].append([i,i]) # the last edge point
    indices[1].append([i-1,i-1])  # twice the same value.

    # Determine intervals sepearint points.
    interval = torch.sum(dist[indices], dim=1, keepdims=True)  # relative interval between ajacent neihbour points. Should add to one.
    # Normalize to one, for the entropy calculation
    interval_norm = interval / torch.sum(interval, dim=0, keepdims=True)
    entropy = torch.sum( -interval_norm * torch.log(interval_norm) )
    # We want to maximize entropy, so minimize -1*entropy
    if return_intervals:
        # This part is used by entropy_distance loss function
        return -entropy, interval, interval_norm
    else:
        return -entropy

def entropy_abs(outputs, target=None):
    if target != None:
        raise Warning('This loss function does not use target values. Target ignored.')
    intervals = entropy(outputs, return_intervals = True)[1]
    fixed_distance = 50     # Optimal results: I_i = I_fixed/ e
    return torch.sum( intervals * torch.log(intervals) )

def entropy_hard_boundaries(outputs, target = None, boundaries=[-100, 0], use_softmax = False):
    if target != None:
        raise Warning('This loss function does not use target values. Target ignored.')
    # First we sort the output, and clip the output to a fixed interval.
    outputs_sorted = outputs.sort(dim=0)[0]
    outputs_clamped = outputs_sorted.clamp(boundaries[0], boundaries[1])

    # THen we prepare two tensors which we subtract from each other to calculate nearest neighbour distances.
    boundaries = torch.tensor( boundaries, dtype=outputs_sorted.dtype)
    boundary_low = boundaries[0].unsqueeze(0).unsqueeze(1)
    boundary_high = boundaries[1].unsqueeze(0).unsqueeze(1)
    outputs_highside = torch.cat( (outputs_clamped, boundary_high), dim=0)
    outputs_lowside = torch.cat( (boundary_low, outputs_clamped), dim=0)

    # Most intervals are multiplied by 0.5 because they are shared between two neighbours
    # The first and last interval do not get divided bu two because they are not shared
    multiplier = 0.5*torch.ones_like(outputs_highside)
    multiplier[0] = 1
    multiplier[-1] = 1

    # Determine the intervals between the points
    dist = (outputs_highside - outputs_lowside) * multiplier
    intervals = dist[1:] + dist[:-1]
    #dist_lowside = (outputs_highside - outputs_lowside) * multiplier
    #dist_highside = (outputs_lowside - outputs_highside) * multiplier
    #intervals = dist_highside[1:] + dist_lowside[:-1]   # The last point of dist_high is useless, just like the first poiint of distance low.

    if use_softmax:
        #raise Warning('Softmax is not yet tested.')
        return torch.sum( torch.nn.functional.softmax(intervals, dim=0) * torch.nn.functional.log_softmax(intervals, dim=0) )
        #return torch.nn.functional.cross_entropy(intervals[:,0], intervals[:,0])
    # WHat about torch.nn.functional.cross_entropy(input, target) with targe=input=intervals ? According to documentation, this combines log_softmax and nll_loss = negative log likelyhood
    else:
        intervals_norm = intervals / (boundary_high - boundary_low)  # boundary_high - boundary low determines the interval.
        return torch.sum( intervals_norm * torch.log(intervals_norm+1e-10) )
    #outputs_clipped = -torch.relu( -(torch.relu(outputs_sorted - boundary_low ) + boundary_low) + boundary_high ) + boundary_high
    #outputs_w_boundaries = torch.cat( (boundary_low, outputs_sorted, boundary_high), dim=0)
    #intervals = entropy(outputs_w_boundaries, return_intervals=True)[1] [1:-1] /

def entropy_distance(outputs, target=None):
    # A combination of entropy and sigmoid distance, all taken on the intervals. Multiplied which each other to promote both
    # a large distance absolute and relative
    if target != None:
        raise Warning('This loss function does not use target values. Target ignored.')
    interval, interval_norm = entropy(outputs, return_intervals=True)[1:]
    #TODO: update scaling of sigmoidu
    return torch.mean( (torch.sigmoid(interval/2)-0.5) * interval_norm * torch.log(interval_norm))


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