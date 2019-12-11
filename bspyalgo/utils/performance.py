#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:42:27 2018

@author: hruiz
"""

import numpy as np


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
    # XOR as target_waveform
    target_waveform = np.zeros((800, 1))
    target_waveform[200:600] = 1

    # Create wave form
    noise = 0.05
    output = np.zeros((800, 1))
    output[200:600] = 1  # XOR
#    output[600:] = 1.75
    input_waveform = output + noise * np.random.randn(len(target_waveform), 1)

    accuracy, weights, predicted = perceptron(input_waveform, target_waveform)

    plt.figure()
    plt.plot(target_waveform)
    plt.plot(input_waveform, '.')
    plt.plot(np.arange(len(target_waveform)), (-weights[0] / weights[1]) * np.ones_like(target_waveform))
    plt.plot(predicted[0], predicted[1], 'xk')
    plt.show()

    nr_examples = 100
    accuracy = np.zeros((nr_examples,))
    for l in range(nr_examples):
        accuracy[l], weights, _ = perceptron(input_waveform, target_waveform)
        print(f'Prediction Accuracy: {accuracy[l]} and weights:{weights}')

    plt.figure()
    plt.hist(accuracy, 100)
    plt.show()
