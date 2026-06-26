#!/usr/bin/env python3
"""RNN cell module."""
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """Initialize the RNN cell."""
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) with previous hidden state
            x_t: numpy.ndarray of shape (m, i) with input data

        Returns:
            h_next, y
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(h_x, self.Wh) + self.bh)

        y_linear = np.matmul(h_next, self.Wy) + self.by
        y_exp = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = y_exp / np.sum(y_exp, axis=1, keepdims=True)

        return h_next, y
