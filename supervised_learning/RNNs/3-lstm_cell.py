#!/usr/bin/env python3
"""LSTM cell module."""
import numpy as np


class LSTMCell:
    """Represents a Long Short-Term Memory (LSTM) unit."""

    def __init__(self, i, h, o):
        """Initialize the LSTM cell."""
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) with previous hidden state
            c_prev: numpy.ndarray of shape (m, h) with previous cell state
            x_t: numpy.ndarray of shape (m, i) with input data

        Returns:
            h_next: the next hidden state
            c_next: the next cell state
            y: the output of the cell
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)

        f = self._sigmoid(np.matmul(h_x, self.Wf) + self.bf)
        u = self._sigmoid(np.matmul(h_x, self.Wu) + self.bu)
        c_tilde = np.tanh(np.matmul(h_x, self.Wc) + self.bc)
        o = self._sigmoid(np.matmul(h_x, self.Wo) + self.bo)

        c_next = f * c_prev + u * c_tilde
        h_next = o * np.tanh(c_next)

        y_linear = np.matmul(h_next, self.Wy) + self.by
        y_exp = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = y_exp / np.sum(y_exp, axis=1, keepdims=True)

        return h_next, c_next, y

    def _sigmoid(self, x):
        """Apply sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
