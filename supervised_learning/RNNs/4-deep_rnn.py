#!/usr/bin/env python3
"""Deep RNN forward propagation module."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN."""
    t, m, _ = X.shape
    l, _, h = h_0.shape

    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    Y = None

    for step in range(t):
        x_input = X[step]
        for layer in range(l):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x_input)
            H[step + 1, layer] = h_next
            x_input = h_next
        Y = y if Y is None else np.concatenate(
            [Y, y[np.newaxis]], axis=0
        )

    Y_out = np.zeros((t, m, y.shape[-1]))
    for step in range(t):
        x_input = X[step]
        for layer in range(l):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x_input)
            x_input = h_next
        Y_out[step] = y

    return H, Y_out
