#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:36:36 2018
Generate a piecewise linear wave form with general amplitudes and intervals.
@author: hruiz
"""
import numpy as np


def generate_waveform(amplitudes, lengths, slopes=0):
    '''Generates a waveform with constant intervals of value amplitudes[i]
    for interval i of length[i]. The slopes argument is the number of points of the slope.'''
    wave = []

    amplitudes.append(0)
    if type(slopes) is int:
        slopes = [slopes] * len(amplitudes)
    if type(lengths) is int:
        lengths = [lengths] * (len(amplitudes) - 1)
    lengths.append(0)

    if len(amplitudes) == len(lengths) == len(slopes):
        wave += np.linspace(0, amplitudes[0], slopes[0]).tolist()
        for i in range(len(amplitudes) - 1):
            wave += [amplitudes[i]] * lengths[i]
            wave += np.linspace(amplitudes[i], amplitudes[i + 1], slopes[i]).tolist()
    else:
        assert False, 'Assignment of amplitudes and lengths/slopes is not unique!'

    return wave


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    amplitudes, lengths = [0, 3, 1, -1, 1, 0], 100
    wave = generate_waveform(amplitudes, lengths, slopes=30)
    print(len(wave))
    plt.figure()
    plt.plot(wave)
    plt.show()
