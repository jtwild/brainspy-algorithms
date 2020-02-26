#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:42:27 2018

@author: hruiz
"""

import numpy as np
import torch.nn as nn
import torch


def decision(data, targets, lrn_rate=0.007, max_iters=100, validation=False):

    if validation:
        n_total = len(data)
        assert n_total > 10, "Not enough data, we assume you have at least 10 points"
        n_val = int(n_total * 0.1)
        shuffle = np.random.permutation(n_total)
        indices_train = shuffle[n_val:]
        indices_val = shuffle[:n_val]
        x_train = torch.tensor(data[indices_train])
        t_train = torch.tensor(targets[indices_train])
        x_val = torch.tensor(data[indices_val])
        t_val = torch.tensor(targets[indices_val])
    else:
        x_train = torch.tensor(data)
        t_train = torch.tensor(targets)
        x_val = torch.tensor(data)
        t_val = torch.tensor(targets)

    node = nn.Linear(1, 1)
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(node.parameters(), lr=lrn_rate)
    best_accuracy = -1
    for epoch in range(max_iters):
        shuffle_data = torch.randperm(len(x_train))
        # TODO: add batcher
        for x_i, t_i in zip(x_train[shuffle_data], t_train[shuffle_data]):
            optimizer.zero_grad()
            y_i = node(x_i)
            cost = loss(y_i, t_i)
            cost.backward()
            optimizer.step()
        with torch.no_grad():
            y = node(x_val)
            labels = y > 0.
            correct_labeled = torch.sum(labels == t_val).detach().numpy()
            acc = 100. * correct_labeled / len(t_val)
            if acc > best_accuracy:
                best_accuracy = acc
                with torch.no_grad():
                    w, b = [p.detach().numpy() for p in node.parameters()]
                    decision_boundary = -b / w
                    predicted_class = node(torch.tensor(data)).detach().numpy() > 0.
        print(f'Epoch: {epoch}  Accuracy {acc}, loss: {cost.item()}')

    return best_accuracy, predicted_class, decision_boundary


def perceptron(input_waveform, target_waveform, tolerance=0.01, max_iter=200):
    # Assumes that the waveform input_waveform and the target_waveform have the shape (n_total,1)
    # Normalize the data; it is assumed that the target_waveform has binary values
    input_waveform = (input_waveform - np.mean(input_waveform)) / np.std(input_waveform)
    n_total = len(input_waveform)
    weights_ = np.random.randn(2, 1)  # np.zeros((2,1))
    inp = np.concatenate([np.ones_like(input_waveform), input_waveform], axis=1)
    shuffle = np.random.permutation(len(inp))

    x = inp[shuffle]
    y = target_waveform[shuffle]

    error = np.inf
    j = 0
    while (error > tolerance) and (j < max_iter):

        for i in range(len(x)):
            a = np.dot(weights_.T, x[i])
            delta = y[i] - f(a)
            weights_ = weights_ - delta * x[i][:, np.newaxis]

        predict = np.array(list(map(f, np.dot(x, weights_))))
        predict = predict[:, np.newaxis]
        error = np.mean(np.abs(y - predict))
        j += 1
#        print('Prediction Error: ',error, ' in ', j,' iters')

    buffer = np.zeros_like(y)
    buffer[y == predict] = 1
    n_correct = np.sum(buffer)
    accuracy_ = n_correct / n_total

#    print('Fraction of iterations used: ', j/max_iter)
#    pdb.set_trace()
    corrcoef = np.corrcoef(y.T, x[:, 1].T)[0, 1]
    if accuracy_ > 0.9 and corrcoef < 0:
        print('Weight is negative', weights_[0], ' and correlation also: ', corrcoef)
        accuracy_ = 0.
        print('Accuracy is set to zero!')

    return accuracy_, weights_, (shuffle, predict)


def f(x):
    return float(x < 0)


def corr_coeff(x, y):
    return np.corrcoef(np.concatenate((x, y), axis=0))[0, 1]

# TODO: use data object to get the accuracy (see corr_coeff above)


def accuracy(best_output, target_waveforms, mask):
    y = best_output[mask][:, np.newaxis]
    trgt = target_waveforms[mask][:, np.newaxis]
    accuracy, _, _ = perceptron(y, trgt)
    return accuracy


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import pickle as pkl

    data_dict = pkl.load(open("tmp/input/best_output_ring_example.pkl", 'rb'))
    data = data_dict['best_output']
    targets = np.zeros_like(data)
    targets[int(len(data) / 2):] = 1
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    accuracy, predicted_labels, threshold = decision(data, targets, validation=False)

    plt.figure()
    plt.title(f'Accuracy: {accuracy:.2f} %')
    plt.plot(data, label='Norm. Data')
    plt.plot(predicted_labels, '.', label='Predicted labels')
    plt.plot(targets, 'g', label='Targets')
    plt.plot(np.arange(len(predicted_labels)),
             np.ones_like(predicted_labels) * threshold, 'k:', label='Threshold')
    plt.legend()
    plt.show()
