#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:42:27 2018

@author: hruiz
"""

import numpy as np
import torch.nn as nn
import torch

from tqdm import trange

def decision(data, targets, lrn_rate=0.007, max_iters=100, validation=False, verbose=True):

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
    looper = trange(max_iters, desc='Calculating accuracy')
    for epoch in looper:
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
        if verbose:
            looper.set_description(f' Epoch: {epoch}  Accuracy {acc}, loss: {cost.item()}')

    return best_accuracy, predicted_class, decision_boundary


def perceptron(input_waveform, target_waveform, plot=None):
    # Assumes that the input_waveform and the target_waveform have the shape (n_total,1)
    # Normalizes the data; it is assumed that the target_waveform has binary values
    input_waveform = (input_waveform - np.mean(input_waveform, axis=0)) / np.std(input_waveform, axis=0)
    _accuracy, predicted_labels, threshold = decision(input_waveform, target_waveform, verbose=False)
    if plot:
        plt.figure()
        plt.title(f'Accuracy: {_accuracy:.2f} %')
        plt.plot(input_waveform, label='Norm. Waveform')
        plt.plot(predicted_labels, '.', label='Predicted labels')
        plt.plot(target_waveform, 'g', label='Targets')
        plt.plot(np.arange(len(predicted_labels)),
                 np.ones_like(predicted_labels) * threshold, 'k:', label='Threshold')
        plt.legend()
        if plot == 'show':
            plt.show()
        else:
            plt.savefig(plot)
            plt.close()
    return _accuracy, predicted_labels, threshold


def corr_coeff(x, y):
    return np.corrcoef(np.concatenate((x, y), axis=0))[0, 1]

# TODO: use data object to get the accuracy (see corr_coeff above)


def accuracy(best_output, target_waveforms, mask):
    y = best_output[mask][:, np.newaxis]
    trgt = target_waveforms[mask][:, np.newaxis]
    acc, _, _ = perceptron(y, trgt)
    return acc


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import pickle as pkl

    data_dict = pkl.load(open("tmp/input/best_output_ring_example.pkl", 'rb'))
    BEST_OUTPUT = data_dict['best_output']
    TARGETS = np.zeros_like(BEST_OUTPUT)
    TARGETS[int(len(BEST_OUTPUT) / 2):] = 1
    ACCURACY, LABELS, THRESHOLD = perceptron(BEST_OUTPUT, TARGETS, plot='show')

    MASK = np.ones_like(TARGETS, dtype=bool)
    ACC = accuracy(BEST_OUTPUT, TARGETS, MASK)
    print('Accuracy for best output: {ACC}')
