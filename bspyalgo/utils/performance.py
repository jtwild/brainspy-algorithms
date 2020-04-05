#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:42:27 2018

@author: hruiz
"""
from __future__ import generator_stop
import numpy as np
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from more_itertools import grouper
from tqdm import trange
from bspyproc.utils.pytorch import TorchUtils


def batch_generator(nr_samples, batch):
    batches = grouper(np.random.permutation(nr_samples), batch)
    while True:
        try:
            indices = list(next(batches))
            if None in indices:
                indices = [index for index in indices if index is not None]
            yield torch.tensor(indices, dtype=torch.int64)
        except StopIteration:
            return


def decision(data, targets, lrn_rate=0.007, mini_batch=8, max_iters=100, validation=False, verbose=True):

    if validation:
        n_total = len(data)
        assert n_total > 10, "Not enough data, we assume you have at least 10 points"
        n_val = int(n_total * 0.1)
        shuffle = np.random.permutation(n_total)
        indices_train = shuffle[n_val:]
        indices_val = shuffle[:n_val]
        x_train = torch.tensor(data[indices_train], dtype=TorchUtils.data_type)
        t_train = torch.tensor(targets[indices_train], dtype=TorchUtils.data_type)
        x_val = torch.tensor(data[indices_val], dtype=TorchUtils.data_type)
        t_val = torch.tensor(targets[indices_val], dtype=TorchUtils.data_type)
    else:
        x_train = torch.tensor(data, dtype=TorchUtils.data_type)
        t_train = torch.tensor(targets, dtype=TorchUtils.data_type)
        x_val = torch.tensor(data, dtype=TorchUtils.data_type)
        t_val = torch.tensor(targets, dtype=TorchUtils.data_type)

    node = nn.Linear(1, 1).type(TorchUtils.data_type)
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(node.parameters(), lr=lrn_rate, betas=(0.999, 0.999))
    best_accuracy = -1
    looper = trange(max_iters, desc='Calculating accuracy')
    for epoch in looper:
        for mb in batch_generator(len(x_train), mini_batch):
            x_i, t_i = x_train[mb], t_train[mb]
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
                    predicted_class = node(torch.tensor(data, dtype=TorchUtils.data_type)).detach().numpy() > 0.
        if verbose:
            looper.set_description(f' Epoch: {epoch}  Accuracy {acc}, loss: {cost.item()}')

    return best_accuracy, predicted_class, decision_boundary


def perceptron(input_waveform, target_waveform, plot=None):
    # Assumes that the input_waveform and the target_waveform have the shape (n_total,1)
    # Normalizes the data; it is assumed that the target_waveform has binary values
    input_waveform = (input_waveform - np.mean(input_waveform, axis=0)) / np.std(input_waveform, axis=0)
    _accuracy, predicted_labels, threshold = decision(input_waveform, target_waveform)
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


def accuracy(best_output, target_waveforms, plot=None):
    if len(best_output.shape) == 1:
        y = best_output[:, np.newaxis]
    else:
        y = best_output
    if len(target_waveforms.shape) == 1:
        trgt = target_waveforms[:, np.newaxis]
    else:
        trgt = target_waveforms
    acc, _, _ = perceptron(y, trgt, plot=plot)
    return acc


if __name__ == '__main__':

    import pickle as pkl

    data_dict = pkl.load(open("tmp/input/best_output_ring_example.pkl", 'rb'))
    BEST_OUTPUT = data_dict['best_output']
    TARGETS = np.zeros_like(BEST_OUTPUT)
    TARGETS[int(len(BEST_OUTPUT) / 2):] = 1
    ACCURACY, LABELS, THRESHOLD = perceptron(BEST_OUTPUT, TARGETS, plot='show')

    MASK = np.ones_like(TARGETS, dtype=bool)
    ACC = accuracy(BEST_OUTPUT, TARGETS)
    print(f'Accuracy for best output: {ACC}')
