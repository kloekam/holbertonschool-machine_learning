#!/usr/bin/env python3
"""GRU cell module."""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit cell."""

    def __init__(self, i, h, o):
        """Initialize the GRU cell."""
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) with previous hidden state
            x_t: numpy.ndarray of shape (m, i) with input data

        Returns:
            h_next: the next hidden state
            y: the output of the cell
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)

        z = self._sigmoid(np.matmul(h_x, self.Wz) + self.bz)
        r = self._sigmoid(np.matmul(h_x, self.Wr) + self.br)

        rh_x = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(rh_x, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_tilde

        y_linear = np.matmul(h_next, self.Wy) + self.by
        y_exp = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = y_exp / np.sum(y_exp, axis=1, keepdims=True)

        return h_next, y

    def _sigmoid(self, x):
        """Apply sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
