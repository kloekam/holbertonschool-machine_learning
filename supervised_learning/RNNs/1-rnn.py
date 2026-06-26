#!/usr/bin/env python3
"""RNN forward propagation module."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation for a simple RNN."""
    t, m, i = X.shape
    h = h_0.shape[1]

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    Y_list = []

    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        Y_list.append(y)

    Y = np.array(Y_list)

    return H, Y
