#!/usr/bin/env python3
"""Deep RNN forward propagation module."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN."""
    t, m, _ = X.shape
    l, _, h = h_0.shape

    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    outputs = []

    for step in range(t):
        x_input = X[step]
        for layer in range(l):
            h_next, y = rnn_cells[layer].forward(H[step, layer], x_input)
            H[step + 1, layer] = h_next
            x_input = h_next
        outputs.append(y)

    Y = np.array(outputs)

    return H, Y
