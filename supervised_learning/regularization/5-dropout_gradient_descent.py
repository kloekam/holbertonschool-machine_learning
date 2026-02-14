#!/usr/bin/env python3
"""Regularization Module"""

import numpy as np


def dropout_gradient_descent(Y,
                             weights,
                             cache,
                             alpha,
                             keep_prob,
                             L):
    """
    A function that updates the weights
    of a neural network with Dropout regularization
    using gradient descent
    """
    m = Y.shape[1]

    dZ = cache["A" + str(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]
        b = weights["b" + str(i)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights["W" + str(i)] = W - alpha * dW
        weights["b" + str(i)] = b - alpha * db

        if i > 1:
            dA_prev = np.matmul(W.T, dZ)

            D_prev = cache["D" + str(i - 1)]
            dA_prev = dA_prev * D_prev
            dA_prev = dA_prev / keep_prob

            A_prev = cache["A" + str(i - 1)]
            dZ = dA_prev * (1 - np.power(A_prev, 2))
